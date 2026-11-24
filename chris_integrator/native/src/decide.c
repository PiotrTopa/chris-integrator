/* decide.c: the decision layer's numerics (chris_integrator/decide): for the car against every object, over the
 * object's predicted futures (modes with a position covariance per time) and the car's own futures:
 *
 *   the collision probability over the horizon       quasi-Monte Carlo over the two position distributions and
 *                                                    the object's extents (samples coherent in time: one draw is
 *                                                    one whole future), or the exact Gaussian integral over the
 *                                                    Minkowski sum of the two footprints per time (Owen's T per
 *                                                    edge), the horizon's figure its largest (a lower bound)
 *   the acceleration required to avoid               braking, steering left / right, and combinations on the
 *                                                    friction circle: the smallest constant demand (with its
 *                                                    build-up) that keeps the car's footprint clear of the
 *                                                    object's (inflated by k sigma) over the horizon
 *   the threat numbers                               required / available (brake, steer, combined)
 *   time-to-brake, time-to-steer                     the last onset of full braking / steering that still avoids
 *   time to collision, time headway, separation      on the mean futures
 *   a level per pair and per object                  from per-class thresholds (entry and exit sets), weighted by
 *                                                    the modes' probabilities (unconditioned and conditioned)
 *
 * and, around that evaluation, the frame (decide_frame, below): the radar message's objects and the futures of its
 * time in, the records of the `decide` message out.
 *
 * It is included by chris_native.c (one translation unit) and uses its buffer helpers; its functions are
 * DECIDE_METHODS (the end of this file). Frame: the car frame of the
 * output time held fixed on the ground (x forward, y left, m); the car starts at the origin, heading 0.
 *
 * decide_eval(t[N], ego[E, 1+3N], ohdr[M, 10 | 12], modes[K, 2+5N], cfg[NC], thr[S*2*4*3], out_pair[K*E, PF], out_obj[M, OF])
 *   ego rows      p, x[N], y[N], v[N]
 *   ohdr rows     first mode, number of modes, len, wid, sd_len, sd_wid, hdg (NaN: unknown), sd_hdg, threshold set,
 *                 has_ctr (0 | 1) [, ctr_dx, ctr_dy]: with has_ctr 1 (12 columns) the futures' reference point to the
 *                 footprint's centre at the first time, car axes (the radar's `ctr` less its `x_m`, `y_m`); else
 *                 the centre is taken half the footprint's depth further along the line of sight (cfg DC_REF)
 *   modes rows    p0 (unconditioned), p (conditioned), x[N], y[N], sx[N], sy[N], rxy[N] (correlation)
 *   thr           per set, per entry/exit, per level 1..4: tn_min, tm_max, mass_min
 *
 * Exactness: plain IEEE arithmetic in the order written, libm's exp, sqrt, sin, cos, atan2, erfc, log; the Monte
 * Carlo draws are a fixed Halton sequence through the normal quantile, so results are deterministic; golden data
 * hold them to rounding (tests/test_decide_native.py).
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DK_MAXN 512              /* time samples of a future */
#define DK_MAXSIM 2048           /* steps of a simulated horizon */
#define DK_MAXS 4096             /* Monte Carlo samples */
#define DK_OH 10                 /* the object header's columns; 12 with the centre's offset */
#define DK_OHC 12
#define DK_HMOVE 1.0             /* m/s: a mode's travel over a second that sets the heading */
#define DK_PF 24
#define DK_OF 18

enum {
    DC_FRONT, DC_REAR, DC_HALFW, DC_V0, DC_A0, DC_ALONG, DC_ALAT, DC_TREACT, DC_TAUB, DC_TAUS, DC_DT, DC_NMAX,
    DC_KSIG, DC_MLONG, DC_MLAT, DC_METHOD, DC_NSAMP, DC_PRUNE, DC_SDEL, DC_SDET, DC_NDIRS, DC_AFAC, DC_TOLA,
    DC_TOLT, DC_REF, DC_OTHERS, DC_PMIN, DC_HMAX, DC_HEXT, DC_ECAPL, DC_ECAPT, DC_ICAPL, DC_ICAPT, DC_TMHIT, DC_NC
};

/* pair fields */
enum {
    DP_PC, DP_TCON, DP_TTC, DP_TTCI, DP_AB, DP_AL, DP_AR, DP_AC, DP_DIR, DP_AVL, DP_AVT, DP_BTN, DP_STN, DP_TN,
    DP_TTB, DP_TTS, DP_TM, DP_LE, DP_LX, DP_SEP, DP_THW, DP_W0, DP_WC, DP_COURSE
};
/* object fields */
enum {
    DO_PC, DO_PCC, DO_LE, DO_LX, DO_ME, DO_MX = DO_ME + 4, DO_BEST = DO_MX + 4, DO_WHY, DO_LC, DO_TTC, DO_THW,
    DO_SEP
};

typedef struct {
    double cx, cy, c, s, hl, hw;
} DkRect;

/* separation of two oriented rectangles along their four axes: the largest gap (< 0: they overlap); a lower bound
 * of the distance between them */
static double dk_sep(const DkRect *a, const DkRect *b)
{
    double dx = b->cx - a->cx, dy = b->cy - a->cy;
    double best = -1e300;
    const DkRect *r[2] = {a, b};
    for (int q = 0; q < 2; q++) {
        double ax[2][2] = {{r[q]->c, r[q]->s}, {-r[q]->s, r[q]->c}};
        for (int k = 0; k < 2; k++) {
            double ux = ax[k][0], uy = ax[k][1];
            double d = fabs(dx * ux + dy * uy);
            double pa = a->hl * fabs(a->c * ux + a->s * uy) + a->hw * fabs(-a->s * ux + a->c * uy);
            double pb = b->hl * fabs(b->c * ux + b->s * uy) + b->hw * fabs(-b->s * ux + b->c * uy);
            double g = d - (pa + pb);
            if (g > best)
                best = g;
        }
    }
    return best;
}

/* ---- the standard normal quantile (Acklam, one Halley step on erfc) and the fixed draws -------------------- */
static double dk_qnorm(double p)
{
    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                                1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                                6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                                -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                                3.754408661907416e+00};
    double x, q, r;
    if (p < 0.02425) {
        q = sqrt(-2.0 * log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 1.0 - 0.02425) {
        q = p - 0.5;
        r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        q = sqrt(-2.0 * log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    double e = 0.5 * erfc(-x / sqrt(2.0)) - p;
    double u = e * sqrt(2.0 * M_PI) * exp(x * x / 2.0);
    return x - u / (1.0 + x * u / 2.0);
}

static double dk_halton(long i, int base)
{
    double f = 1.0, r = 0.0;
    while (i > 0) {
        f /= base;
        r += f * (double)(i % base);
        i /= base;
    }
    return r;
}

static double DK_Z[DK_MAXS][5];
static int dk_z_n = 0;

static void dk_draws(int n)
{
    static const int base[5] = {2, 3, 5, 7, 11};
    if (n <= dk_z_n)
        return;
    for (int i = dk_z_n; i < n; i++)
        for (int k = 0; k < 5; k++)
            DK_Z[i][k] = dk_qnorm(dk_halton(i + 1, base[k]));
    dk_z_n = n;
}

/* ---- the Gaussian integral over a convex polygon (whitened: the standard normal; Owen's T per edge) ---------- */
static const double DK_GLX[6] = {-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
                                 0.2386191860831969, 0.6612093864662645, 0.9324695142031521};
static const double DK_GLW[6] = {0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                                 0.4679139345726910, 0.3607615730481386, 0.1713244923791704};


static double dk_gl6(double h2, double a, double b)
{
    double mid = 0.5 * (a + b), w = b - a, s = 0.0;
    for (int g = 0; g < 6; g++) {
        double cu = cos(mid + 0.5 * w * DK_GLX[g]);
        s += DK_GLW[g] * exp(-h2 / (2.0 * cu * cu));
    }
    return 0.5 * w * s;
}

/* adaptive: a part is halved until its two halves agree with it to tol */
static double dk_gl_adapt(double h2, double a, double b, double whole, double tol, int depth)
{
    double m = 0.5 * (a + b);
    double l = dk_gl6(h2, a, m), r = dk_gl6(h2, m, b);
    if (depth <= 0 || fabs(l + r - whole) <= tol)
        return l + r;
    return dk_gl_adapt(h2, a, m, l, 0.5 * tol, depth - 1) + dk_gl_adapt(h2, m, b, r, 0.5 * tol, depth - 1);
}

/* int_{u1}^{u2} exp(-h^2 / (2 cos^2 u)) du, |u| < pi/2: adaptive 6-point Gauss-Legendre, split at 0 */
static double dk_edge_int(double h2, double u1, double u2)
{
    double sgn = 1.0;
    if (u2 < u1) {
        double t = u1;
        u1 = u2;
        u2 = t;
        sgn = -1.0;
    }
    double tot = 0.0;
    double cuts[4] = {u1, u2, 0, 0};
    int nc = 2;
    if (u1 < 0.0 && u2 > 0.0) {
        cuts[1] = 0.0;
        cuts[2] = u2;
        nc = 3;
    }
    for (int p = 0; p + 1 < nc; p++) {
        double a = cuts[p], b = cuts[p + 1];
        tot += dk_gl_adapt(h2, a, b, dk_gl6(h2, a, b), 1e-12, 24);
    }
    return sgn * tot;
}

/* P(Z in polygon) for Z standard normal, polygon w[n] (closed implicitly), any orientation */
static double dk_poly_prob(const double (*w)[2], int n)
{
    double tot = 0.0;
    for (int e = 0; e < n; e++) {
        double ax = w[e][0], ay = w[e][1], bx = w[(e + 1) % n][0], by = w[(e + 1) % n][1];
        double ex = bx - ax, ey = by - ay;
        double L = sqrt(ex * ex + ey * ey);
        if (!(L > 1e-15))
            continue;
        double cr = ax * by - ay * bx;          /* twice the signed area of (0, a, b) */
        double h = fabs(cr) / L;                /* distance of the edge's line from the origin */
        double ta = atan2(ay, ax), tb = atan2(by, bx);
        double dth = tb - ta;
        while (dth > M_PI)
            dth -= 2.0 * M_PI;
        while (dth < -M_PI)
            dth += 2.0 * M_PI;
        if (h < 1e-12) {                        /* the edge's line passes through the origin: no area */
            continue;
        }
        /* the foot of the perpendicular: angle phi; u = theta - phi in (-pi/2, pi/2) over the edge */
        double nx = cr > 0.0 ? ey / L : -ey / L, ny = cr > 0.0 ? -ex / L : ex / L;
        double phi = atan2(ny, nx);
        double ua = ta - phi, ub = ua + dth;
        while (ua > M_PI)
            ua -= 2.0 * M_PI, ub -= 2.0 * M_PI;
        while (ua < -M_PI)
            ua += 2.0 * M_PI, ub += 2.0 * M_PI;
        double I = dk_edge_int(h * h, ua, ub);
        tot += (dth - I) / (2.0 * M_PI);
    }
    if (tot < 0.0)
        tot = -tot;                              /* a clockwise polygon */
    return tot > 1.0 ? 1.0 : tot;
}

/* convex hull (monotone chain) of n points into hull, returns its size */
static int dk_hull(double (*p)[2], int n, double (*hull)[2])
{
    for (int i = 1; i < n; i++) {                /* insertion sort by x, then y */
        double x = p[i][0], y = p[i][1];
        int j = i - 1;
        while (j >= 0 && (p[j][0] > x || (p[j][0] == x && p[j][1] > y))) {
            p[j + 1][0] = p[j][0];
            p[j + 1][1] = p[j][1];
            j--;
        }
        p[j + 1][0] = x;
        p[j + 1][1] = y;
    }
    int k = 0;
    for (int i = 0; i < n; i++) {
        while (k >= 2 && (hull[k - 1][0] - hull[k - 2][0]) * (p[i][1] - hull[k - 2][1]) -
                                 (hull[k - 1][1] - hull[k - 2][1]) * (p[i][0] - hull[k - 2][0]) <= 0.0)
            k--;
        hull[k][0] = p[i][0];
        hull[k][1] = p[i][1];
        k++;
    }
    for (int i = n - 2, lo = k + 1; i >= 0; i--) {
        while (k >= lo && (hull[k - 1][0] - hull[k - 2][0]) * (p[i][1] - hull[k - 2][1]) -
                                  (hull[k - 1][1] - hull[k - 2][1]) * (p[i][0] - hull[k - 2][0]) <= 0.0)
            k--;
        hull[k][0] = p[i][0];
        hull[k][1] = p[i][1];
        k++;
    }
    return k - 1;
}

/* P(the object's centre ~ N(mu, S) puts its footprint onto the car's): the Gaussian over the Minkowski sum of the
 * two rectangles about the car's centre */
static double dk_inst_prob(const DkRect *eg, const DkRect *ob, double mx, double my, double sxx, double sxy, double syy)
{
    double p[16][2], hull[33][2];
    double ec[4][2], oc[4][2];
    for (int q = 0; q < 4; q++) {
        double a = (q == 0 || q == 3) ? 1.0 : -1.0, b = (q < 2) ? 1.0 : -1.0;
        ec[q][0] = eg->cx + a * eg->hl * eg->c - b * eg->hw * eg->s;
        ec[q][1] = eg->cy + a * eg->hl * eg->s + b * eg->hw * eg->c;
        oc[q][0] = a * ob->hl * ob->c - b * ob->hw * ob->s;
        oc[q][1] = a * ob->hl * ob->s + b * ob->hw * ob->c;
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            p[4 * i + j][0] = ec[i][0] + oc[j][0];
            p[4 * i + j][1] = ec[i][1] + oc[j][1];
        }
    int n = dk_hull(p, 16, hull);
    double l11 = sqrt(sxx);
    if (!(l11 > 1e-9))
        l11 = 1e-9;
    double l21 = sxy / l11, r = syy - l21 * l21;
    double l22 = r > 1e-18 ? sqrt(r) : 1e-9;
    double w[16][2];
    for (int i = 0; i < n; i++) {
        double dx = hull[i][0] - mx, dy = hull[i][1] - my;
        double z1 = dx / l11;
        w[i][0] = z1;
        w[i][1] = (dy - l21 * z1) / l22;
    }
    return dk_poly_prob((const double(*)[2])w, n);
}

/* ---- the futures at the simulation's resolution --------------------------------------------------------- */
typedef struct {
    int n;                      /* steps of the simulation: the horizon and its extension */
    int nh;                     /* steps of the horizon */
    double dt;
    const double *t;            /* the futures' grid */
    int N;
} DkGrid;

static void dk_interp_idx(const DkGrid *g, double tau, int *i0, double *f)
{
    const double *t = g->t;
    int N = g->N;
    if (tau <= t[0]) {
        *i0 = 0;
        *f = 0.0;
        return;
    }
    if (tau >= t[N - 1]) {                       /* beyond: on along the last interval */
        *i0 = N - 2;
        *f = (tau - t[N - 2]) / (t[N - 1] - t[N - 2]);
        return;
    }
    int lo = 0, hi = N - 1;
    while (hi - lo > 1) {
        int m = (lo + hi) / 2;
        if (t[m] <= tau)
            lo = m;
        else
            hi = m;
    }
    *i0 = lo;
    *f = (tau - t[lo]) / (t[lo + 1] - t[lo]);
}

typedef struct {               /* one ego mode, sampled: pose and the path by arc length */
    double *x, *y, *th, *v, *s;   /* per step */
    double *ps, *px, *py, *pth;   /* the path: per grid point arc length, position, heading */
    int np;
} DkEgo;

typedef struct {               /* one object mode, sampled */
    double *cx, *cy, *c, *s;      /* centre and heading */
    double *sxx, *sxy, *syy;      /* the object's covariance */
    double hl, hw, sdl, sdw, sdh; /* half extents and the extents' 1 sigma */
} DkObj;

/* the car's own covariance of position at time tau: along and across its heading (a demand it holds with some
 * error: 1 sigma a t^2 / 2, bounded) */
static void dk_ego_cov(const double *cfg, double tau, double th, double *sxx, double *sxy, double *syy)
{
    double a = 0.5 * cfg[DC_SDEL] * tau * tau, b = 0.5 * cfg[DC_SDET] * tau * tau;
    if (a > cfg[DC_ECAPL])
        a = cfg[DC_ECAPL];
    if (b > cfg[DC_ECAPT])
        b = cfg[DC_ECAPT];
    double c = cos(th), s = sin(th);
    *sxx = a * a * c * c + b * b * s * s;
    *syy = a * a * s * s + b * b * c * c;
    *sxy = (a * a - b * b) * c * s;
}

static void dk_ego_rect(const double *cfg, double x, double y, double th, DkRect *r)
{
    double off = 0.5 * (cfg[DC_FRONT] - cfg[DC_REAR]);
    r->c = cos(th);
    r->s = sin(th);
    r->cx = x + off * r->c;
    r->cy = y + off * r->s;
    r->hl = 0.5 * (cfg[DC_FRONT] + cfg[DC_REAR]);
    r->hw = cfg[DC_HALFW];
}

/* the point of the ego path at arc length s: position and heading (extrapolated beyond its ends) */
static void dk_path_at(const DkEgo *e, double s, double *x, double *y, double *th, int *hint)
{
    int n = e->np;
    if (n < 2 || e->ps[n - 1] < 1e-6) {
        *th = e->pth[0];
        *x = e->px[0] + s * cos(*th);
        *y = e->py[0] + s * sin(*th);
        return;
    }
    if (s <= 0.0) {
        *th = e->pth[0];
        *x = e->px[0] + s * cos(*th);
        *y = e->py[0] + s * sin(*th);
        return;
    }
    if (s >= e->ps[n - 1]) {
        *th = e->pth[n - 1];
        double d = s - e->ps[n - 1];
        *x = e->px[n - 1] + d * cos(*th);
        *y = e->py[n - 1] + d * sin(*th);
        return;
    }
    int lo = *hint;
    if (lo < 0 || lo > n - 2 || e->ps[lo] > s)
        lo = 0;
    while (lo < n - 2 && e->ps[lo + 1] <= s)
        lo++;
    *hint = lo;
    double seg = e->ps[lo + 1] - e->ps[lo];
    double f = seg > 1e-9 ? (s - e->ps[lo]) / seg : 0.0;
    *x = e->px[lo] + f * (e->px[lo + 1] - e->px[lo]);
    *y = e->py[lo] + f * (e->py[lo + 1] - e->py[lo]);
    /* the heading of the segment (a polyline: piecewise constant, blended over the segment's ends) */
    double h0 = e->pth[lo], h1 = e->pth[lo + 1], dh = h1 - h0;
    while (dh > M_PI)
        dh -= 2.0 * M_PI;
    while (dh < -M_PI)
        dh += 2.0 * M_PI;
    *th = h0 + f * dh;
}

/* the object's rectangle at step i, inflated by k sigma along its axes (+ margins), with the car's covariance; the
 * inflation (spread and margins together) scaled by infl in [0, 1] (0: the rectangle as it is) */
static void dk_obj_rect(const double *cfg, const DkObj *o, int i, double esxx, double esxy, double esyy, double infl,
                        DkRect *r)
{
    r->cx = o->cx[i];
    r->cy = o->cy[i];
    r->c = o->c[i];
    r->s = o->s[i];
    r->hl = o->hl;
    r->hw = o->hw;
    if (infl > 0.0) {
        double sxx = o->sxx[i] + esxx, sxy = o->sxy[i] + esxy, syy = o->syy[i] + esyy;
        double c = r->c, s = r->s;
        double va = c * c * sxx + 2.0 * c * s * sxy + s * s * syy;
        double vc = s * s * sxx - 2.0 * c * s * sxy + c * c * syy;
        double k = cfg[DC_KSIG];
        double ia = k * sqrt(va > 0.0 ? va + 0.25 * o->sdl * o->sdl : 0.25 * o->sdl * o->sdl);
        double ic = k * sqrt(vc > 0.0 ? vc + 0.25 * o->sdw * o->sdw : 0.25 * o->sdw * o->sdw);
        /* bounded: a poorly ranged object's spread is the collision probability's matter, not a wall */
        r->hl += infl * ((ia < cfg[DC_ICAPL] ? ia : cfg[DC_ICAPL]) + cfg[DC_MLONG]);
        r->hw += infl * ((ic < cfg[DC_ICAPT] ? ic : cfg[DC_ICAPT]) + cfg[DC_MLAT]);
    }
}

typedef struct {
    const double *cfg;
    DkGrid g;
    DkEgo *eg;                  /* E */
    DkObj *ob;                  /* K */
    int E, K;
    const int *top;             /* per object, its most probable mode (the others' check) */
    const int *mode_obj;        /* per mode, its object */
    int M;
    DkRect *tgt;                /* the judged pair's object, inflated, per step (dk_pair_prep) */
    int i_course;               /* the first step of contact of the mode as it is (inflated); n: none */
    DkRect *oth;                /* per object, its most probable mode with the margins, per step */
} DkCtx;

/* the judged object mode's footprint per step, inflated by k sigma of its own spread (the threat numbers and the
 * collision course: what the car must keep clear of; the car's own spread enters the collision probability only).
 *
 * A margin larger than the gap the car has now (a car parked beside the path, passed 0.4 m off its side with a
 * spread of 1 m) cannot be kept by any demand and says nothing about a collision to come: the inflation is scaled
 * down (the same share over the whole horizon) until the inflated footprint is clear of the car now; none where the
 * footprint as it is touches the car now. */
static void dk_pair_prep(DkCtx *C, int j, int k)
{
    const DkEgo *e = &C->eg[j];
    DkRect er, r;
    dk_ego_rect(C->cfg, e->x[0], e->y[0], e->th[0], &er);
    double f = 1.0;
    dk_obj_rect(C->cfg, &C->ob[k], 0, 0.0, 0.0, 0.0, 1.0, &r);
    if (dk_sep(&er, &r) < 0.0) {
        dk_obj_rect(C->cfg, &C->ob[k], 0, 0.0, 0.0, 0.0, 0.0, &r);
        if (dk_sep(&er, &r) < 0.0)
            f = 0.0;
        else {                                  /* the largest share that keeps it clear now */
            double lo = 0.0, hi = 1.0;
            for (int q = 0; q < 12; q++) {
                double m = 0.5 * (lo + hi);
                dk_obj_rect(C->cfg, &C->ob[k], 0, 0.0, 0.0, 0.0, m, &r);
                if (dk_sep(&er, &r) < 0.0)
                    hi = m;
                else
                    lo = m;
            }
            f = lo;
        }
    }
    for (int i = 0; i < C->g.n; i++)
        dk_obj_rect(C->cfg, &C->ob[k], i, 0.0, 0.0, 0.0, f, &C->tgt[i]);
}

/* simulate the car on ego mode j: its own motion until ts, then a constant demand (along: deceleration a_long
 * >= 0 built up over tau_b; across: a_lat, left positive, built up over tau_s); 1 when its footprint stays clear of
 * mode k (inflated) and (steering only, `others`) of the other objects' most probable modes; t_hit the first contact */
static int dk_clear(const DkCtx *C, int j, int k, double ts, double a_long, double a_lat, double *t_hit)
{
    const double *cfg = C->cfg;
    const DkEgo *e = &C->eg[j];
    int n = C->g.n;
    double dt = C->g.dt;
    double tb = cfg[DC_TAUB], tsr = cfg[DC_TAUS], nmax = cfg[DC_NMAX];
    int i0 = (int)floor(ts / dt + 1e-9);
    if (i0 < 0)
        i0 = 0;
    double s = 0.0, v = 0.0, nn = 0.0, nd = 0.0;
    int others = cfg[DC_OTHERS] > 0.5 && a_lat != 0.0;
    int hint = 0, still = 0;
    double x = 0.0, y = 0.0, th = 0.0;
    double Re = 0.5 * (cfg[DC_FRONT] + cfg[DC_REAR]) + cfg[DC_HALFW];
    for (int i = 0; i < n; i++) {
        double tau = i * dt;
        if (i <= i0) {
            if (i < C->i_course) {               /* the mode as it is: clear before its contact (dk_first_contact) */
                s = e->s[i];
                v = e->v[i];
                continue;
            }
            x = e->x[i];
            y = e->y[i];
            th = e->th[i];
            s = e->s[i];
            v = e->v[i];
        } else {
            double ap = (e->v[i] - e->v[i - 1]) / dt;        /* the mode's own acceleration */
            double rb = tb > 0.0 ? (tau - ts) / tb : 1.0, rs = tsr > 0.0 ? (tau - ts) / tsr : 1.0;
            if (rb > 1.0)
                rb = 1.0;
            if (rs > 1.0)
                rs = 1.0;
            double acc = (1.0 - rb) * ap - rb * a_long;
            if (a_long == 0.0 && a_lat != 0.0)
                acc = ap;                                     /* steering alone: the speed as the mode has it */
            double v1 = v + acc * dt;
            if (v1 < 0.0)
                v1 = 0.0;
            s += 0.5 * (v + v1) * dt;
            v = v1;
            nd += rs * a_lat * dt;
            nn += nd * dt;
            if (nn > nmax) {
                nn = nmax;
                nd = 0.0;
            } else if (nn < -nmax) {
                nn = -nmax;
                nd = 0.0;
            }
            if (!(still && v == 0.0 && nd == 0.0)) {     /* standing with no sideways motion: the pose holds */
                double px, py, pth;
                dk_path_at(e, s, &px, &py, &pth, &hint);
                x = px - nn * sin(pth);
                y = py + nn * cos(pth);
                th = pth + atan2(nd, v > 0.5 ? v : 0.5);
                still = (v == 0.0 && nd == 0.0);
            }
        }
        const DkRect *tg = &C->tgt[i];
        double ox = 0.5 * (cfg[DC_FRONT] - cfg[DC_REAR]);
        double ecx = x + ox * cos(th), ecy = y + ox * sin(th);
        double ddx = tg->cx - ecx, ddy = tg->cy - ecy, RR = Re + tg->hl + tg->hw;
        DkRect er;
        int near = ddx * ddx + ddy * ddy <= RR * RR;
        if (near || others)
            dk_ego_rect(cfg, x, y, th, &er);
        if (near && dk_sep(&er, tg) < 0.0) {
            if (t_hit)
                *t_hit = tau;
            return 0;
        }
        if (others && i > i0) {
            for (int q = 0; q < C->M; q++) {
                int kk = C->top[q];
                if (kk < 0 || q == C->mode_obj[k])
                    continue;
                const DkRect *r2p = &C->oth[(size_t)q * C->g.n + i];
                DkRect r2 = *r2p;
                double dx = r2.cx - er.cx, dy = r2.cy - er.cy, R = er.hl + er.hw + r2.hl + r2.hw;
                if (dx * dx + dy * dy > R * R)
                    continue;
                if (dk_sep(&er, &r2) < 0.0) {
                    if (t_hit)
                        *t_hit = tau;
                    return 0;
                }
            }
        }
    }
    return 1;
}

/* the smallest demand A in [0, amax] along direction (cos phi: braking, sin phi: left) that keeps clear from now;
 * inf when even amax does not */
static double dk_required(const DkCtx *C, int j, int k, double phi, double amax)
{
    double cl = cos(phi), sl = sin(phi);
    if (fabs(cl) < 1e-12)
        cl = 0.0;
    if (fabs(sl) < 1e-12)
        sl = 0.0;
    if (dk_clear(C, j, k, 0.0, 0.0 * cl, 0.0, NULL) && cl >= 0.0 && sl == 0.0)
        return 0.0;
    if (!dk_clear(C, j, k, 0.0, amax * cl, amax * sl, NULL))
        return INFINITY;
    double lo = 0.0, hi = amax, tol = C->cfg[DC_TOLA];
    while (hi - lo > tol) {
        double m = 0.5 * (lo + hi);
        if (dk_clear(C, j, k, 0.0, m * cl, m * sl, NULL))
            hi = m;
        else
            lo = m;
    }
    return hi;
}

/* as dk_required, the demand hi known to keep clear */
static double dk_required_hi(const DkCtx *C, int j, int k, double phi, double hi)
{
    double cl = cos(phi), sl = sin(phi), lo = 0.0, tol = C->cfg[DC_TOLA];
    while (hi - lo > tol) {
        double m = 0.5 * (lo + hi);
        if (dk_clear(C, j, k, 0.0, m * cl, m * sl, NULL))
            hi = m;
        else
            lo = m;
    }
    return hi;
}

/* the latest onset in [0, tmax] of the demand (a_long, a_lat) that still keeps clear; -1 when none does */
static double dk_latest(const DkCtx *C, int j, int k, double a_long, double a_lat, double tmax)
{
    if (!dk_clear(C, j, k, 0.0, a_long, a_lat, NULL))
        return -1.0;
    double lo = 0.0, hi = tmax, tol = C->cfg[DC_TOLT];
    if (C->i_course < C->g.n) {                  /* an onset at the contact is too late: search before it */
        hi = C->i_course * C->g.dt;
    } else if (dk_clear(C, j, k, hi, a_long, a_lat, NULL))
        return hi;
    while (hi - lo > tol) {
        double m = 0.5 * (lo + hi);
        if (dk_clear(C, j, k, m, a_long, a_lat, NULL))
            lo = m;
        else
            hi = m;
    }
    return lo;
}

/* the collision probability over the horizon of ego mode j against object mode k */
static double dk_pc(const DkCtx *C, int j, int k, double *t_contact)
{
    const double *cfg = C->cfg;
    const DkEgo *e = &C->eg[j];
    const DkObj *o = &C->ob[k];
    int n = C->g.nh;
    double dt = C->g.dt;
    *t_contact = NAN;
    /* prune: the centres' distance beyond both bounding circles by prune_sigma of the largest spread at every step */
    double Ro0 = sqrt((o->hl + 3.0 * o->sdl) * (o->hl + 3.0 * o->sdl) + (o->hw + 3.0 * o->sdw) * (o->hw + 3.0 * o->sdw));
    double He = 0.5 * (cfg[DC_FRONT] + cfg[DC_REAR]), Re = sqrt(He * He + cfg[DC_HALFW] * cfg[DC_HALFW]);
    int any = 0;
    static double L11[DK_MAXSIM], L21[DK_MAXSIM], L22[DK_MAXSIM];
    static DkRect ER[DK_MAXSIM];
    static unsigned char live[DK_MAXSIM];
    for (int i = 0; i < n; i++) {
        double esxx, esxy, esyy;
        dk_ego_cov(cfg, i * dt, e->th[i], &esxx, &esxy, &esyy);
        double sxx = o->sxx[i] + esxx, sxy = o->sxy[i] + esxy, syy = o->syy[i] + esyy;
        dk_ego_rect(cfg, e->x[i], e->y[i], e->th[i], &ER[i]);
        double tr = 0.5 * (sxx + syy), dd = sqrt(0.25 * (sxx - syy) * (sxx - syy) + sxy * sxy);
        double lmax = tr + dd;
        double dx = o->cx[i] - ER[i].cx, dy = o->cy[i] - ER[i].cy;
        double gap = sqrt(dx * dx + dy * dy) - (Ro0 + Re);
        live[i] = !(gap > cfg[DC_PRUNE] * sqrt(lmax > 0.0 ? lmax : 0.0));
        if (live[i])
            any = 1;
        double l11 = sqrt(sxx > 1e-12 ? sxx : 1e-12);
        double l21 = sxy / l11, r = syy - l21 * l21;
        L11[i] = l11;
        L21[i] = l21;
        L22[i] = sqrt(r > 1e-12 ? r : 1e-12);
    }
    if (!any)
        return 0.0;
    if (cfg[DC_METHOD] > 0.5) {               /* the exact Gaussian per step; the horizon: its largest */
        double best = 0.0;
        static double pin[DK_MAXSIM];
        for (int i = 0; i < n; i++) {
            pin[i] = 0.0;
            if (!live[i])
                continue;
            DkRect orr;
            dk_obj_rect(cfg, o, i, 0.0, 0.0, 0.0, 0.0, &orr);
            double esxx, esxy, esyy;
            dk_ego_cov(cfg, i * dt, e->th[i], &esxx, &esxy, &esyy);
            double c = orr.c, s = orr.s, a2 = 0.25 * o->sdl * o->sdl, b2 = 0.25 * o->sdw * o->sdw;
            double sxx = o->sxx[i] + esxx + a2 * c * c + b2 * s * s;
            double syy = o->syy[i] + esyy + a2 * s * s + b2 * c * c;
            double sxy = o->sxy[i] + esxy + (a2 - b2) * c * s;
            double p = dk_inst_prob(&ER[i], &orr, o->cx[i], o->cy[i], sxx, sxy, syy);
            pin[i] = p;
            if (p > best)
                best = p;
        }
        for (int i = 0; i < n && best > 0.0; i++)        /* contact: the first time at half the peak */
            if (pin[i] >= 0.5 * best) {
                *t_contact = i * dt;
                break;
            }
        return best;
    }
    int ns = (int)cfg[DC_NSAMP];
    if (ns < 1)
        ns = 1;
    if (ns > DK_MAXS)
        ns = DK_MAXS;
    dk_draws(ns);
    int hits = 0;
    static double th_hit[DK_MAXS];
    for (int q = 0; q < ns; q++) {
        const double *z = DK_Z[q];
        double hl = o->hl + 0.5 * o->sdl * z[2], hw = o->hw + 0.5 * o->sdw * z[3];
        if (hl < 0.05)
            hl = 0.05;
        if (hw < 0.05)
            hw = 0.05;
        double dh = o->sdh * z[4], cd = cos(dh), sd = sin(dh);
        double R = sqrt(hl * hl + hw * hw) + Re;
        for (int i = 0; i < n; i++) {
            if (!live[i])
                continue;
            double ex = L11[i] * z[0], ey = L21[i] * z[0] + L22[i] * z[1];
            DkRect orr;
            orr.cx = o->cx[i] + ex;
            orr.cy = o->cy[i] + ey;
            double dx = orr.cx - ER[i].cx, dy = orr.cy - ER[i].cy;
            if (dx * dx + dy * dy > R * R)
                continue;
            orr.c = o->c[i] * cd - o->s[i] * sd;
            orr.s = o->s[i] * cd + o->c[i] * sd;
            orr.hl = hl;
            orr.hw = hw;
            if (dk_sep(&ER[i], &orr) < 0.0) {
                th_hit[hits++] = i * dt;
                break;
            }
        }
    }
    if (hits) {                                   /* the median first contact of the colliding draws */
        for (int a = 1; a < hits; a++) {
            double x = th_hit[a];
            int b = a - 1;
            while (b >= 0 && th_hit[b] > x) {
                th_hit[b + 1] = th_hit[b];
                b--;
            }
            th_hit[b + 1] = x;
        }
        *t_contact = hits % 2 ? th_hit[hits / 2] : 0.5 * (th_hit[hits / 2 - 1] + th_hit[hits / 2]);
    }
    return (double)hits / ns;
}

/* first contact and the least separation on the mean futures as they are (inflated or not) */
static double dk_first_contact(const DkCtx *C, int j, int k, int infl, double *sep_min)
{
    const DkEgo *e = &C->eg[j];
    int n = C->g.nh;
    double best = 1e300, tc = NAN;
    for (int i = 0; i < n; i++) {
        DkRect er, orr;
        dk_ego_rect(C->cfg, e->x[i], e->y[i], e->th[i], &er);
        if (infl)
            orr = C->tgt[i];
        else
            dk_obj_rect(C->cfg, &C->ob[k], i, 0.0, 0.0, 0.0, 0.0, &orr);
        double sp = dk_sep(&er, &orr);
        if (sp < best)
            best = sp;
        if (sp < 0.0 && isnan(tc)) {
            tc = i * C->g.dt;
            if (!sep_min)
                break;
        }
    }
    if (sep_min)
        *sep_min = best > 0.0 ? best : 0.0;
    return tc;
}

/* time headway of object mode k behind ... ahead of the car on ego mode j at t = 0 (NaN: not a vehicle ahead in
 * the car's lane going its way) */
static double dk_thw(const DkCtx *C, int j, int k)
{
    const double *cfg = C->cfg;
    const DkObj *o = &C->ob[k];
    const DkEgo *e = &C->eg[j];
    double v0 = cfg[DC_V0];
    if (!(v0 > 1.0) || C->g.nh < 2)
        return NAN;
    int i1 = (int)ceil(0.5 / C->g.dt);
    if (i1 >= C->g.nh)
        i1 = C->g.nh - 1;
    double c = cos(e->th[0]), s = sin(e->th[0]);
    double along = (o->cx[0] - e->x[0]) * c + (o->cy[0] - e->y[0]) * s;
    double across = -(o->cx[0] - e->x[0]) * s + (o->cy[0] - e->y[0]) * c;
    double vo = ((o->cx[i1] - o->cx[0]) * c + (o->cy[i1] - o->cy[0]) * s) / (i1 * C->g.dt);
    if (along <= 0.0 || fabs(across) > cfg[DC_HALFW] + o->hw + 0.3 || vo < 1.0)
        return NAN;
    double gap = along - o->hl - cfg[DC_FRONT];
    return (gap > 0.0 ? gap : 0.0) / v0;
}

static int dk_level(const double *thr, int set, int exit, double course, double tn, double tm)
{
    if (!(course > 0.5))
        return 0;
    const double *b = thr + ((set * 2 + exit) * 4) * 3;
    int L = 0;
    for (int l = 0; l < 4; l++) {
        double tnmin = b[3 * l], tmmax = b[3 * l + 1];
        if (tn >= tnmin || tm <= tmmax)
            L = l + 1;
    }
    return L;
}

/* the evaluation of one frame (the arrays of the file's head; N >= 2, E >= 1): 0 and the steps of the horizon and
 * of the simulation, or -1 with an exception set */
static int dk_eval(const double *t, int N, const double *EG, int E, const double *OH, int ohw, int M, const double *MD,
                   int K, const double *cfg, const double *thr, int nsets, double *PO, double *OO, int *nh_out, int *n_out)
{
    int rc = -1;
    double *buf = NULL;
    int *ibuf = NULL;
    DkEgo *eg = NULL;
    DkObj *ob = NULL;
    DkRect *rects = NULL;
    for (int i = 1; i < N; i++)
        if (!(t[i] > t[i - 1])) {
            PyErr_SetString(PyExc_ValueError, "decide_eval: the times must increase");
            goto done;
        }
    double dt = cfg[DC_DT] > 1e-3 ? cfg[DC_DT] : 1e-3;
    double T = t[N - 1] - t[0];
    if (cfg[DC_HMAX] > 0.0 && cfg[DC_HMAX] < T)
        T = cfg[DC_HMAX];
    int nh = (int)floor(T / dt + 1e-9) + 1;
    if (nh > DK_MAXSIM)
        nh = DK_MAXSIM;
    if (nh < 2)
        nh = 2;
    int n = nh + (int)floor((cfg[DC_HEXT] > 0.0 ? cfg[DC_HEXT] : 0.0) / dt + 1e-9);
    if (n > DK_MAXSIM)
        n = DK_MAXSIM;
    /* storage: per ego mode 5 n + 4 N, per object mode 7 n */
    size_t need = (size_t)E * (5 * n + 4 * N) + (size_t)K * 7 * n;
    buf = (double *)PyMem_Malloc((need ? need : 1) * sizeof(double));
    ibuf = (int *)PyMem_Malloc((size_t)(M + K + 1) * sizeof(int));
    eg = (DkEgo *)PyMem_Malloc((size_t)E * sizeof(DkEgo));
    ob = (DkObj *)PyMem_Malloc((size_t)(K ? K : 1) * sizeof(DkObj));
    rects = (DkRect *)PyMem_Malloc((size_t)(M + 1) * n * sizeof(DkRect));
    if (!buf || !ibuf || !eg || !ob || !rects) {
        PyErr_NoMemory();
        goto done;
    }
    double *w = buf;
    DkGrid g = {n, nh, dt, t, N};
    /* the ego modes */
    for (int j = 0; j < E; j++) {
        const double *row = EG + (size_t)j * (1 + 3 * N);
        const double *X = row + 1, *Y = row + 1 + N, *V = row + 1 + 2 * N;
        DkEgo *e = &eg[j];
        e->x = w; e->y = w + n; e->th = w + 2 * n; e->v = w + 3 * n; e->s = w + 4 * n; w += 5 * n;
        e->ps = w; e->px = w + N; e->py = w + 2 * N; e->pth = w + 3 * N; w += 4 * N;
        e->np = N;
        double s = 0.0, h = 0.0;
        int seen = 0;
        for (int i = 0; i < N; i++) {
            e->px[i] = X[i];
            e->py[i] = Y[i];
            if (i > 0) {
                double dx = X[i] - X[i - 1], dy = Y[i] - Y[i - 1], d = sqrt(dx * dx + dy * dy);
                s += d;
                if (d > 0.05) {
                    h = atan2(dy, dx);
                    if (!seen) {
                        for (int q = 0; q < i; q++)
                            e->pth[q] = h;
                        seen = 1;
                    }
                }
            }
            e->ps[i] = s;
            e->pth[i] = h;
        }
        for (int i = 0; i < n; i++) {
            double tau = t[0] + i * dt, f;
            int i0;
            dk_interp_idx(&g, tau, &i0, &f);
            e->x[i] = X[i0] + f * (X[i0 + 1] - X[i0]);
            e->y[i] = Y[i0] + f * (Y[i0 + 1] - Y[i0]);
            e->v[i] = V[i0] + f * (V[i0 + 1] - V[i0]);
            if (e->v[i] < 0.0)
                e->v[i] = 0.0;
            e->s[i] = e->ps[i0] + f * (e->ps[i0 + 1] - e->ps[i0]);
            double dx = X[i0 + 1] - X[i0], dy = Y[i0 + 1] - Y[i0];
            e->th[i] = (dx * dx + dy * dy > 0.0025) ? atan2(dy, dx) : e->pth[i0];
        }
    }
    /* the object modes */
    int *mode_obj = ibuf, *top = ibuf + K;
    for (int m = 0; m < M; m++) {
        const double *hd = OH + (size_t)m * ohw;
        int k0 = (int)hd[0], nk = (int)hd[1];
        top[m] = -1;
        double bestp = -1.0;
        for (int k = k0; k < k0 + nk && k < K; k++) {
            if (k < 0)
                continue;
            mode_obj[k] = m;
            const double *row = MD + (size_t)k * (2 + 5 * N);
            if (row[0] > bestp) {
                bestp = row[0];
                top[m] = k;
            }
            const double *X = row + 2, *Y = row + 2 + N, *SX = row + 2 + 2 * N, *SY = row + 2 + 3 * N, *RR = row + 2 + 4 * N;
            DkObj *o = &ob[k];
            o->cx = w; o->cy = w + n; o->c = w + 2 * n; o->s = w + 3 * n; o->sxx = w + 4 * n; o->sxy = w + 5 * n;
            o->syy = w + 6 * n; w += 7 * n;
            double L = hd[2] > 0.1 ? hd[2] : 0.1, W = hd[3] > 0.1 ? hd[3] : 0.1;
            o->hl = 0.5 * L;
            o->hw = 0.5 * W;
            o->sdl = hd[4] > 0.0 ? hd[4] : 0.0;
            o->sdw = hd[5] > 0.0 ? hd[5] : 0.0;
            o->sdh = isfinite(hd[7]) && hd[7] > 0.0 ? hd[7] : 0.0;
            double h0 = isfinite(hd[6]) ? hd[6] : 0.0, hprev = h0;
            /* the heading per step: now the stated one (the rectangle the radar measured; the offset of its centre
             * is taken in that frame); later the direction of the mode's travel over the next second where it
             * travels at least DK_HMOVE m/s on average, else held. The rectangle's axis, not its front: a step's
             * heading is taken modulo a half turn nearest the last. Before 2026-11-23 every grid interval's own
             * direction counted from 0.6 m/s on, now included: a mode starting 0.4 m beside the radar's point (the
             * first quarter second, a prepended "now") turned a car parked beside the path across it, its centre
             * offset swung by 90 degrees onto the path (parked_street object 400, move_off, pc 1.0). */
            double offb_x = 0.0, offb_y = 0.0;           /* the reference point to the centre, in the body frame */
            for (int i = 0; i < n; i++) {
                double tau = t[0] + i * dt, f;
                int i0;
                dk_interp_idx(&g, tau, &i0, &f);
                double x = X[i0] + f * (X[i0 + 1] - X[i0]), y = Y[i0] + f * (Y[i0 + 1] - Y[i0]);
                double tau2 = tau + 1.0 < t[N - 1] ? tau + 1.0 : t[N - 1], hm = NAN;
                if (tau2 - tau > 0.2) {
                    double f2;
                    int j0;
                    dk_interp_idx(&g, tau2, &j0, &f2);
                    double mx = X[j0] + f2 * (X[j0 + 1] - X[j0]) - x, my = Y[j0] + f2 * (Y[j0 + 1] - Y[j0]) - y;
                    if (mx * mx + my * my > DK_HMOVE * DK_HMOVE * (tau2 - tau) * (tau2 - tau))
                        hm = atan2(my, mx);
                }
                double h;
                if (i == 0)
                    h = isfinite(hd[6]) ? hd[6] : (isfinite(hm) ? hm : h0);
                else {
                    h = isfinite(hm) ? hm : hprev;
                    double dh = remainder(h - hprev, 2.0 * M_PI);
                    if (fabs(dh) > 0.5 * M_PI)
                        h = remainder(h + M_PI, 2.0 * M_PI);
                }
                hprev = h;
                double c = cos(h), s = sin(h);
                if (i == 0 && cfg[DC_REF] > 0.5 && ohw == DK_OHC && hd[9] > 0.5 && isfinite(hd[10]) &&
                    isfinite(hd[11])) {
                    /* the radar said where the centre is: its offset from the reference point, held in the body
                     * frame (it turns with the object) */
                    offb_x = hd[10] * c + hd[11] * s;
                    offb_y = -hd[10] * s + hd[11] * c;
                } else if (i == 0 && cfg[DC_REF] > 0.5) {
                    /* the radar's point is the nearest of the footprint as the camera sees it: the centre lies
                     * half the footprint's extent further along the line of sight */
                    double lx = x - eg[0].x[0], ly = y - eg[0].y[0], ll = sqrt(lx * lx + ly * ly);
                    if (ll > 1e-6) {
                        lx /= ll;
                        ly /= ll;
                        double sup = o->hl * fabs(lx * c + ly * s) + o->hw * fabs(-lx * s + ly * c);
                        double ox = sup * lx, oy = sup * ly;
                        offb_x = ox * c + oy * s;
                        offb_y = -ox * s + oy * c;
                    }
                }
                o->cx[i] = x + offb_x * c - offb_y * s;
                o->cy[i] = y + offb_x * s + offb_y * c;
                o->c[i] = c;
                o->s[i] = s;
                double sx = SX[i0] + f * (SX[i0 + 1] - SX[i0]), sy = SY[i0] + f * (SY[i0 + 1] - SY[i0]);
                double r = RR[i0] + f * (RR[i0 + 1] - RR[i0]);
                if (!isfinite(r))
                    r = 0.0;
                if (r > 0.99)
                    r = 0.99;
                if (r < -0.99)
                    r = -0.99;
                if (sx < 0.0)
                    sx = 0.0;
                if (sy < 0.0)
                    sy = 0.0;
                o->sxx[i] = sx * sx;
                o->syy[i] = sy * sy;
                o->sxy[i] = r * sx * sy;
            }
            (void)h0;
        }
    }
    DkCtx C = {cfg, g, eg, ob, E, K, top, mode_obj, M, rects, n, rects + n};
    for (int m = 0; m < M; m++)                  /* the others: their most probable modes, with the margins */
        for (int i = 0; i < n; i++) {
            DkRect *r = &C.oth[(size_t)m * n + i];
            if (top[m] < 0) {
                r->hl = r->hw = 0.0;
                r->cx = r->cy = 1e9;
                r->c = 1.0;
                r->s = 0.0;
                continue;
            }
            dk_obj_rect(cfg, &ob[top[m]], i, 0.0, 0.0, 0.0, 0.0, r);
            r->hl += cfg[DC_MLONG];
            r->hw += cfg[DC_MLAT];
        }
    double along = cfg[DC_ALONG], alat = cfg[DC_ALAT], amax = cfg[DC_AFAC] * (along > alat ? along : alat);
    int ndirs = (int)cfg[DC_NDIRS];
    double tmax = (nh - 1) * dt;
    for (int m = 0; m < M; m++) {
        const double *hd = OH + (size_t)m * ohw;
        int k0 = (int)hd[0], nk = (int)hd[1], set = (int)hd[8];
        if (set < 0 || set >= nsets)
            set = 0;
        double *oo = OO + (size_t)m * DK_OF;
        for (int f = 0; f < DK_OF; f++)
            oo[f] = 0.0;
        oo[DO_BEST] = -1.0;
        oo[DO_TTC] = NAN;
        oo[DO_THW] = NAN;
        oo[DO_SEP] = INFINITY;
        double best_score = -1.0;
        double mass_c[4] = {0, 0, 0, 0};
        for (int k = k0; k < k0 + nk && k < K; k++) {
            const double *row = MD + (size_t)k * (2 + 5 * N);
            for (int j = 0; j < E; j++) {
                double *pp = PO + ((size_t)k * E + j) * DK_PF;
                double pe = EG[(size_t)j * (1 + 3 * N)];
                double w0 = row[0] * pe, wc = row[1] * pe;
                for (int f = 0; f < DK_PF; f++)
                    pp[f] = NAN;
                pp[DP_W0] = w0;
                pp[DP_WC] = wc;
                pp[DP_AVL] = along;
                pp[DP_AVT] = alat;
                pp[DP_LE] = pp[DP_LX] = 0.0;
                pp[DP_COURSE] = 0.0;
                if (!(w0 >= cfg[DC_PMIN] || wc >= cfg[DC_PMIN])) {
                    pp[DP_PC] = 0.0;
                    continue;
                }
                double tcon;
                dk_pair_prep(&C, j, k);
                pp[DP_PC] = dk_pc(&C, j, k, &tcon);
                pp[DP_TCON] = tcon;
                double sep;
                pp[DP_TTC] = dk_first_contact(&C, j, k, 0, &sep);
                pp[DP_SEP] = sep;
                pp[DP_TTCI] = dk_first_contact(&C, j, k, 1, NULL);
                C.i_course = isnan(pp[DP_TTCI]) ? n : (int)floor(pp[DP_TTCI] / dt + 0.5);
                pp[DP_THW] = dk_thw(&C, j, k);
                if (isnan(pp[DP_TTCI])) {           /* no collision course: nothing is required */
                    pp[DP_AB] = pp[DP_AL] = pp[DP_AR] = pp[DP_AC] = 0.0;
                    pp[DP_DIR] = 0.0;
                    pp[DP_BTN] = pp[DP_STN] = pp[DP_TN] = 0.0;
                } else {
                    pp[DP_COURSE] = 1.0;
                    double ab = dk_required(&C, j, k, 0.0, amax);
                    double al = dk_required(&C, j, k, M_PI / 2, amax), ar = dk_required(&C, j, k, -M_PI / 2, amax);
                    /* the share of the friction ellipse a demand a in direction phi uses */
#define DK_UTIL(a, ph) ((a) * sqrt((cos(ph) / along) * (cos(ph) / along) + (sin(ph) / alat) * (sin(ph) / alat)))
                    double ac = ab, dir = 0.0, best = ab / along;
                    if (al / alat < best) {
                        best = al / alat;
                        ac = al;
                        dir = M_PI / 2;
                    }
                    if (ar / alat < best) {
                        best = ar / alat;
                        ac = ar;
                        dir = -M_PI / 2;
                    }
                    /* combinations on the friction circle: braking and steering together */
                    for (int q = 1; q <= ndirs; q++) {
                        double ph = (M_PI / 2) * q / (ndirs + 1);
                        for (int sg = -1; sg <= 1; sg += 2) {
                            /* only a demand that uses less of the ellipse than the best so far can win: try that */
                            double a_eq = isfinite(best) ? best / DK_UTIL(1.0, sg * ph) : amax;
                            if (a_eq > amax)
                                a_eq = amax;
                            if (!dk_clear(&C, j, k, 0.0, a_eq * cos(ph), a_eq * sg * sin(ph), NULL))
                                continue;
                            double a = dk_required_hi(&C, j, k, sg * ph, a_eq);
                            double u = DK_UTIL(a, sg * ph);
                            if (u < best) {
                                best = u;
                                ac = a;
                                dir = sg * ph;
                            }
                        }
                    }
#undef DK_UTIL
                    pp[DP_AB] = ab;
                    pp[DP_AL] = al;
                    pp[DP_AR] = ar;
                    pp[DP_AC] = ac;
                    pp[DP_DIR] = dir;
                    pp[DP_BTN] = ab / along;
                    pp[DP_STN] = (al < ar ? al : ar) / alat;
                    double tn = pp[DP_BTN];
                    if (pp[DP_STN] < tn)
                        tn = pp[DP_STN];
                    if (best < tn)
                        tn = best;
                    pp[DP_TN] = tn;
                    /* the time margin: the time left to start the last action that avoids the collision with its
                     * margin (the inflated footprint, as the threat numbers). It needs a collision to have a time:
                     * where the mean futures miss (cfg DC_TMHIT; only the margin is breached) there is none, and
                     * the pair's level is its threat number's (and its probability's mass) */
                    if (cfg[DC_TMHIT] > 0.5 && isnan(pp[DP_TTC])) {
                        pp[DP_TTB] = pp[DP_TTS] = pp[DP_TM] = NAN;
                    } else {
                        double tb = dk_latest(&C, j, k, along, 0.0, tmax);
                        double tl = dk_latest(&C, j, k, 0.0, alat, tmax), tr = dk_latest(&C, j, k, 0.0, -alat, tmax);
                        double ts = tl > tr ? tl : tr;
                        pp[DP_TTB] = tb < 0.0 ? 0.0 : tb;
                        pp[DP_TTS] = ts < 0.0 ? 0.0 : ts;
                        double latest = tb > ts ? tb : ts;
                        pp[DP_TM] = (latest < 0.0 ? 0.0 : latest) - cfg[DC_TREACT];
                    }
                }
                int le = dk_level(thr, set, 0, pp[DP_COURSE], pp[DP_TN], isnan(pp[DP_TM]) ? INFINITY : pp[DP_TM]);
                int lx = dk_level(thr, set, 1, pp[DP_COURSE], pp[DP_TN], isnan(pp[DP_TM]) ? INFINITY : pp[DP_TM]);
                pp[DP_LE] = le;
                pp[DP_LX] = lx;
                oo[DO_PC] += w0 * pp[DP_PC];
                oo[DO_PCC] += wc * pp[DP_PC];
                for (int l = 0; l < 4; l++) {
                    if (le > l)
                        oo[DO_ME + l] += w0 * pp[DP_PC];
                    if (lx > l)
                        oo[DO_MX + l] += w0 * pp[DP_PC];
                    if (le > l)
                        mass_c[l] += wc * pp[DP_PC];
                }
                double score = le * 10.0 + w0 * pp[DP_PC];
                if (pp[DP_COURSE] > 0.5 && score > best_score) {
                    best_score = score;
                    oo[DO_BEST] = (double)((size_t)(k - k0) * E + j);
                }
                if (!isnan(pp[DP_TTC]) && w0 >= cfg[DC_PMIN] && (isnan(oo[DO_TTC]) || pp[DP_TTC] < oo[DO_TTC]))
                    oo[DO_TTC] = pp[DP_TTC];
                if (k == top[m] && j == 0)
                    oo[DO_THW] = pp[DP_THW];
                if (!isnan(pp[DP_SEP]) && w0 >= cfg[DC_PMIN] && pp[DP_SEP] < oo[DO_SEP])
                    oo[DO_SEP] = pp[DP_SEP];
            }
        }
        /* the object's level: the highest whose pairs' probability mass reaches the level's minimum */
        const double *be = thr + ((set * 2 + 0) * 4) * 3, *bx = thr + ((set * 2 + 1) * 4) * 3;
        int LE = 0, LX = 0, LC = 0;
        for (int l = 0; l < 4; l++) {
            if (oo[DO_ME + l] >= be[3 * l + 2] && oo[DO_ME + l] > 0.0)
                LE = l + 1;
            if (oo[DO_MX + l] >= bx[3 * l + 2] && oo[DO_MX + l] > 0.0)
                LX = l + 1;
            if (mass_c[l] >= be[3 * l + 2] && mass_c[l] > 0.0)
                LC = l + 1;
        }
        oo[DO_LE] = LE;
        oo[DO_LX] = LX;
        oo[DO_LC] = LC;
        /* why: 1 the threat number, 2 the time margin, 3 the collision course alone */
        if (oo[DO_BEST] >= 0.0) {
            const double *pp = PO + ((size_t)k0 * E + (size_t)oo[DO_BEST]) * DK_PF;
            const double *b = thr + ((set * 2) * 4) * 3;
            int Lp = (int)pp[DP_LE];
            if (Lp > 0) {
                oo[DO_WHY] = pp[DP_TN] >= b[3 * (Lp - 1)] ? 1.0 : (pp[DP_TM] <= b[3 * (Lp - 1) + 1] ? 2.0 : 3.0);
            } else
                oo[DO_WHY] = 3.0;
        }
        if (oo[DO_SEP] == INFINITY)
            oo[DO_SEP] = NAN;
    }
    *nh_out = nh;
    *n_out = n;
    rc = 0;
done:
    PyMem_Free(buf);
    PyMem_Free(ibuf);
    PyMem_Free(eg);
    PyMem_Free(ob);
    PyMem_Free(rects);
    return rc;
}

static PyObject *n_decide_eval(PyObject *self, PyObject *args)
{
    PyObject *ot, *oe, *oh, *om, *oc, *othr, *opair, *oobj;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &ot, &oe, &oh, &om, &oc, &othr, &opair, &oobj))
        return NULL;
    View vt, ve, vh, vm, vc, vthr, vp, vo;
    vt.held = ve.held = vh.held = vm.held = vc.held = vthr.held = vp.held = vo.held = 0;
    PyObject *res = NULL;
    if (view_get(ot, &vt, 0, 'd', 1, "decide_eval t") < 0 || view_get(oe, &ve, 0, 'd', 2, "decide_eval ego") < 0 ||
        view_get(oh, &vh, 0, 'd', 2, "decide_eval ohdr") < 0 || view_get(om, &vm, 0, 'd', 2, "decide_eval modes") < 0 ||
        view_get(oc, &vc, 0, 'd', 1, "decide_eval cfg") < 0 || view_get(othr, &vthr, 0, 'd', 1, "decide_eval thr") < 0 ||
        view_get(opair, &vp, 1, 'd', 2, "decide_eval out_pair") < 0 || view_get(oobj, &vo, 1, 'd', 2, "decide_eval out_obj") < 0)
        goto done;
    int N = (int)vt.b.shape[0], E = (int)ve.b.shape[0], M = (int)vh.b.shape[0], K = (int)vm.b.shape[0];
    if (N < 2 || N > DK_MAXN || E < 1 || vc.b.shape[0] < DC_NC || ve.b.shape[1] != 1 + 3 * N ||
        (vh.b.shape[1] != DK_OH && vh.b.shape[1] != DK_OHC) || vm.b.shape[1] != 2 + 5 * N || vp.b.shape[0] != (Py_ssize_t)K * E ||
        vp.b.shape[1] != DK_PF || vo.b.shape[0] != M || vo.b.shape[1] != DK_OF) {
        PyErr_SetString(PyExc_ValueError, "decide_eval: arrays of the documented shapes are needed");
        goto done;
    }
    int nh, n;
    if (dk_eval(DATA(vt), N, DATA(ve), E, DATA(vh), (int)vh.b.shape[1], M, DATA(vm), K, DATA(vc), DATA(vthr),
                (int)(vthr.b.shape[0] / 24), DATA(vp), DATA(vo), &nh, &n) == 0)
        res = Py_BuildValue("(ii)", nh, n);
done:
    view_drop(&vt);
    view_drop(&ve);
    view_drop(&vh);
    view_drop(&vm);
    view_drop(&vc);
    view_drop(&vthr);
    view_drop(&vp);
    view_drop(&vo);
    return res;
}

/* decide_inst_prob(ego_rect[6], obj_rect[6], mu[2], cov[3]): the exact instantaneous probability (tests, tools):
 * rects as (cx, cy, heading, half length, half width, unused); the object's centre ~ N(mu, cov) */
static PyObject *n_decide_inst_prob(PyObject *self, PyObject *args)
{
    double e[5], o[5], mx, my, sxx, sxy, syy;
    if (!PyArg_ParseTuple(args, "(ddddd)(ddddd)(dd)(ddd)", &e[0], &e[1], &e[2], &e[3], &e[4], &o[0], &o[1], &o[2],
                          &o[3], &o[4], &mx, &my, &sxx, &sxy, &syy))
        return NULL;
    DkRect er = {e[0], e[1], cos(e[2]), sin(e[2]), e[3], e[4]}, orr = {o[0], o[1], cos(o[2]), sin(o[2]), o[3], o[4]};
    return PyFloat_FromDouble(dk_inst_prob(&er, &orr, mx, my, sxx, sxy, syy));
}

/* decide_draws(n): the first n fixed standard-normal draws (n x 5), as a list (tests) */
static PyObject *n_decide_draws(PyObject *self, PyObject *arg)
{
    long n = PyLong_AsLong(arg);
    if (n == -1 && PyErr_Occurred())
        return NULL;
    if (n < 0 || n > DK_MAXS) {
        PyErr_SetString(PyExc_ValueError, "decide_draws: 0..4096");
        return NULL;
    }
    dk_draws((int)n);
    PyObject *out = PyList_New(n);
    if (!out)
        return NULL;
    for (long i = 0; i < n; i++) {
        PyObject *r = Py_BuildValue("(ddddd)", DK_Z[i][0], DK_Z[i][1], DK_Z[i][2], DK_Z[i][3], DK_Z[i][4]);
        if (!r) {
            Py_DECREF(out);
            return NULL;
        }
        PyList_SetItem(out, i, r);
    }
    return out;
}

/* ====================================================================================================
 * The frame: a radar message's objects and the futures of its time in, the decision's records out
 * ====================================================================================================
 * decide_frame is what the plug-in calls once per radar message (chris_integrator/decide/frame.py): it reads the
 * radar's objects (the message's dicts), takes the contract's futures as arrays (decide/futures.py `Futures`) or
 * builds the degraded ones (an object without futures: constant velocity, along the path where it drives along it;
 * the car without its own: its motion now and the plan), sets each footprint (extent, heading, centre, the lanes'
 * weight of a standing mode), evaluates (dk_eval) and writes the records of the objects worth listing.
 *
 * The arithmetic is the Python's it replaced (2026-11-24), in its order: numpy's elementwise steps, numpy.interp
 * (df_interp), Python's float `%` and `**` (the C library's fmod and pow), its sum() of floats (compensated:
 * df_sum), math.hypot (CPython's own: df_hypot), numpy.hypot and atan2 / sin / cos (the C library's). tests/test_decide_port.py holds it to the messages recorded before. */

#define DF_STRS(X)                                                                                                     \
    X(state) X(id) X(x_m) X(y_m) X(vx_mps) X(vy_mps) X(std) X(p_stationary) X(flags) X(motion) X(class) X(label)       \
    X(len) X(wid) X(hdg) X(len_std) X(std_len) X(wid_std) X(std_wid) X(hdg_std) X(std_hdg) X(ext_std) X(heading_rad)   \
    X(speed_mps) X(ctr) X(value) X(v) X(accel_mps2) X(yaw_rate_rps) X(parked) X(in_path) X(threat) X(ego) X(cross)     \
    X(left) X(right) X(onc) X(park) X(off) X(level) X(level_name) X(level_exit) X(level_cond) X(p_coll)                \
    X(p_coll_cond) X(ttc_s) X(thw_s) X(sep_m) X(kind) X(size) X(size_src) X(fut) X(mass) X(btn) X(stn) X(tn) X(a_req)  \
    X(brake) X(comb) X(comb_dir_deg) X(ttb_s) X(tts_s) X(tm_s) X(why) X(mode) X(p0) X(p) X(ego_mode) X(measure) X(pc)  \
    X(t_contact_s) X(lane) X(shown) X(pairs) X(w0) X(w) X(course) X(radar) X(cv) X(contract) X(vehicle) X(vru)         \
    X(unknown) X(stand) X(stand_path) X(cv_path) X(path) X(turn) X(none) X(information) X(caution) X(warning)          \
    X(imminent) X(threat_number) X(time_margin) X(collision_course) X(confirmed) X(coast) X(stationary) X(moving)
#define X(n) S_##n,
enum { DF_STRS(X) S_N };
#undef X
#define X(n) #n,
static const char *const DF_STR[S_N] = {DF_STRS(X)};
#undef X
static PyObject *df_s[S_N];

static int df_strings(void)
{
    if (df_s[S_N - 1] != NULL)
        return 0;
    for (int i = 0; i < S_N; i++)
        if (df_s[i] == NULL && (df_s[i] = PyUnicode_InternFromString(DF_STR[i])) == NULL)
            return -1;
    return 0;
}

/* the futures' cfg (decide/futures.py DEFAULT_CFG, as frame.py packs it) */
enum {
    FC_SPEED_CAP, FC_VSTD_CAP, FC_STILL_VSTD, FC_ALONG_PATH, FC_MODES_RADAR, FC_STAND_MIN, FC_ALONG_HDG_DEG,
    FC_ALONG_MIN, FC_EGO_PATH_P, FC_EGO_A_LO, FC_EGO_A_HI, FC_N
};
/* the threat model's cfg beyond the evaluation's (threat.py DEFAULT_CFG) */
enum {
    DX_BRAKE_SHARE = DC_NC, DX_STEER_SHARE, DX_SPEED_FADE, DX_TURN_R, DX_HDG_RES, DX_AXIS_STANDING, DX_AXIS_SNAP,
    DX_AXIS_STD, DX_LANE_AXIS_P, DX_LANE_WEIGHT, DX_LANE_FLOOR, DX_STAND_MOVE, DX_LIST_MIN_P, DX_NC
};
#define DF_G 9.81

/* classes: prior footprint (length, width, 1 sigma of each) and the threshold set (0 vehicle, 1 vru, 2 unknown) */
enum { DF_CAR, DF_VAN, DF_TRUCK, DF_BUS, DF_TRAM, DF_HUMAN, DF_BIKE, DF_MOTORBIKE, DF_OTHER };
static const double DF_SIZE[9][4] = {{4.5, 1.8, 0.4, 0.1},  {5.2, 2.0, 0.5, 0.15},  {9.0, 2.5, 3.0, 0.15},
                                     {12.0, 2.55, 1.5, 0.1}, {25.0, 2.65, 5.0, 0.1}, {0.6, 0.6, 0.15, 0.15},
                                     {1.8, 0.7, 0.2, 0.15},  {2.1, 0.85, 0.2, 0.15}, {4.2, 1.8, 1.0, 0.3}};
static const int DF_SET[9] = {0, 0, 0, 0, 0, 1, 1, 1, 2};
static const struct {
    const char *name;
    int cls;
} DF_LABELS[] = {{"car", DF_CAR},         {"van", DF_VAN},           {"truck", DF_TRUCK},     {"bus", DF_BUS},
                 {"tram", DF_TRAM},       {"human", DF_HUMAN},       {"bike", DF_BIKE},       {"motorbike", DF_MOTORBIKE},
                 {"person", DF_HUMAN},    {"pedestrian", DF_HUMAN},  {"bicycle", DF_BIKE},    {"cyclist", DF_BIKE},
                 {"rider", DF_BIKE},      {"motorcycle", DF_MOTORBIKE}, {"motor", DF_MOTORBIKE}, {"lorry", DF_TRUCK},
                 {"coach", DF_BUS},       {"train", DF_TRAM},        {NULL, 0}};

/* Python's and numpy's small steps, as they compute them */
static volatile double df_two = 2.0;
#define DF_POW2(x) pow((x), df_two)            /* float ** 2: the C library's pow (the compiler must not fold it) */
#define DF_MAX(a, b) ((b) > (a) ? (b) : (a))   /* max(a, b) */
#define DF_MIN(a, b) ((b) < (a) ? (b) : (a))   /* min(a, b) */

/* math.hypot(a, b) as CPython computes it (3.11 on: mathmodule.c vector_norm, the squares summed without error and
 * the root corrected once): not always the C library's hypot in the last bit */
static double df_hypot(double a, double b)
{
    double v[2] = {fabs(a), fabs(b)}, max = v[0] > v[1] ? v[0] : v[1];
    if (isinf(v[0]) || isinf(v[1]))
        return INFINITY;
    if (isnan(v[0]) || isnan(v[1]))
        return NAN;
    if (max == 0.0)
        return max;
    int max_e;
    frexp(max, &max_e);
    if (max_e < -1023)
        return 2.2250738585072014e-308 * df_hypot(a / 2.2250738585072014e-308, b / 2.2250738585072014e-308);
    double scale = ldexp(1.0, -max_e), csum = 1.0, frac1 = 0.0, frac2 = 0.0;
    for (int i = 0; i < 2; i++) {
        double x = v[i] * scale, hi = x * x, lo = fma(x, x, -hi);
        double sm = csum + hi, sl = (csum - sm) + hi;
        csum = sm;
        frac1 += lo;
        frac2 += sl;
    }
    double h = sqrt(csum - 1.0 + (frac1 + frac2));
    double hi = -h * h, lo = fma(-h, h, -hi), sm = csum + hi, sl = (csum - sm) + hi;
    csum = sm;
    frac1 += lo;
    frac2 += sl;
    double x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    return h / scale;
}

static double df_mod(double a, double b)       /* float a % b */
{
    double m = fmod(a, b);
    if (m != 0.0) {
        if ((b < 0.0) != (m < 0.0))
            m += b;
    } else
        m = copysign(0.0, b);
    return m;
}

/* Python's sum() of floats: the first as it is, the further ones compensated (Neumaier; CPython from 3.12), an int
 * added as it is */
typedef struct {
    double hi, lo;
    int started;
} DfSum;

static void df_sum_add(DfSum *s, double x, int is_int)
{
    if (is_int || !s->started) {
        s->hi += x;
        if (!is_int)
            s->started = 1;
        return;
    }
    double t = s->hi + x;
    if (fabs(s->hi) >= fabs(x))
        s->lo += (s->hi - t) + x;
    else
        s->lo += (x - t) + s->hi;
    s->hi = t;
}

static double df_sum(const DfSum *s)
{
    return (s->lo != 0.0 && isfinite(s->lo)) ? s->hi + s->lo : s->hi;
}

static PyObject *df_get(PyObject *d, int key)   /* d.get(key): borrowed, NULL where absent */
{
    PyObject *v = PyDict_GetItemWithError(d, df_s[key]);
    if (v == NULL)
        PyErr_Clear();
    return v;
}

/* _fin(v): an int or a float (no bool), finite */
static int df_fin(PyObject *v, double *out)
{
    if (v == NULL)
        return 0;
    if (PyFloat_Check(v)) {
        double d = PyFloat_AsDouble(v);
        if (!isfinite(d))
            return 0;
        *out = d;
        return 1;
    }
    if (PyLong_Check(v) && !PyBool_Check(v)) {
        double d = PyLong_AsDouble(v);
        if (d == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return 0;
        }
        *out = d;
        return 1;
    }
    return 0;
}

/* float(v or 0.0) */
static int df_num0(PyObject *v, double *out)
{
    *out = 0.0;
    if (v == NULL || v == Py_None)
        return 0;
    double d = PyFloat_AsDouble(v);
    if (d == -1.0 && PyErr_Occurred())
        return -1;
    *out = d == 0.0 ? 0.0 : d;
    return 0;
}

/* float(d.get(key, 0.0)) */
static int df_numd(PyObject *d, int key, double *out)
{
    PyObject *v = df_get(d, key);
    *out = 0.0;
    if (v == NULL)
        return 0;
    double x = PyFloat_AsDouble(v);
    if (x == -1.0 && PyErr_Occurred())
        return -1;
    *out = x;
    return 0;
}

static int df_is(PyObject *v, int key)          /* v == "<key>" */
{
    return v != NULL && (v == df_s[key] || (PyUnicode_Check(v) && PyUnicode_CompareWithASCIIString(v, DF_STR[key]) == 0));
}

/* the first `max` items of a list / tuple: values and whether each is _fin; its length (0: not such a sequence) */
static Py_ssize_t df_seq(PyObject *v, double *out, int *fin, int max)
{
    if (v == NULL || !(PyList_Check(v) || PyTuple_Check(v)))
        return 0;
    Py_ssize_t n = PySequence_Size(v);
    for (int i = 0; i < max && i < n; i++) {
        PyObject *it = PyList_Check(v) ? PyList_GetItem(v, i) : PyTuple_GetItem(v, i);
        fin[i] = df_fin(it, &out[i]);
    }
    return n;
}

/* futures._vs: a value with its 1 sigma: a number + `<k>_std`, a [value, std] pair, or a {"value" / "v", "std"} */
static void df_vs(PyObject *o, int key, int k_std, int std_k, double *val, int *has, double *sd, int *has_sd)
{
    *has = *has_sd = 0;
    PyObject *v = df_get(o, key);
    if (v == NULL)
        return;
    if (PyList_Check(v) || PyTuple_Check(v)) {
        double q[2];
        int f[2] = {0, 0};
        if (df_seq(v, q, f, 2) == 2 && f[0]) {
            *val = q[0];
            *has = 1;
            if (f[1]) {
                *sd = q[1];
                *has_sd = 1;
            }
        }
        return;
    }
    if (PyDict_Check(v)) {
        PyObject *x = df_get(v, S_value);
        if (x == NULL)
            x = df_get(v, S_v);
        *has = df_fin(x, val);
        *has_sd = df_fin(df_get(v, S_std), sd);
        return;
    }
    if (df_fin(v, val)) {
        *has = 1;
        PyObject *x = df_get(o, k_std);
        if (x == NULL)
            x = df_get(o, std_k);
        *has_sd = df_fin(x, sd);
    }
}

static int df_class(PyObject *cls)              /* classvote.norm_label of the class's label, as an index */
{
    if (cls == NULL || !PyDict_Check(cls))
        return DF_OTHER;
    PyObject *lab = df_get(cls, S_label);
    if (lab == NULL || !PyUnicode_Check(lab))
        return DF_OTHER;
    Py_ssize_t n;
    const char *s = PyUnicode_AsUTF8AndSize(lab, &n);
    if (s == NULL) {
        PyErr_Clear();
        return DF_OTHER;
    }
    while (n > 0 && (*s == ' ' || (*s >= '\t' && *s <= '\r'))) {
        s++;
        n--;
    }
    while (n > 0 && (s[n - 1] == ' ' || (s[n - 1] >= '\t' && s[n - 1] <= '\r')))
        n--;
    char b[16];
    if (n <= 0 || n >= (Py_ssize_t)sizeof(b))
        return DF_OTHER;
    for (Py_ssize_t i = 0; i < n; i++)
        b[i] = (char)(s[i] >= 'A' && s[i] <= 'Z' ? s[i] + 32 : s[i]);
    b[n] = 0;
    for (int i = 0; DF_LABELS[i].name; i++)
        if (strcmp(b, DF_LABELS[i].name) == 0)
            return DF_LABELS[i].cls;
    return DF_OTHER;
}

/* a buffer of float64: an array (any shape) or bytes; its numbers */
static int df_buf(PyObject *o, View *v, const double **p, Py_ssize_t *n, const char *what)
{
    v->held = 0;
    if (PyObject_GetBuffer(o, &v->b, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return -1;
    v->held = 1;
    const char *f = v->b.format ? v->b.format : "B";
    if (*f == '@' || *f == '=' || *f == '<')
        f++;
    int ok = (f[0] == 'd' && f[1] == 0 && v->b.itemsize == 8) ||
             ((f[0] == 'B' || f[0] == 'c' || f[0] == 'b') && f[1] == 0 && v->b.len % 8 == 0);
    if (!ok) {
        view_drop(v);
        PyErr_Format(PyExc_TypeError, "decide: %s: float64 numbers (an array or bytes) are needed", what);
        return -1;
    }
    *p = (const double *)v->b.buf;
    *n = v->b.len / 8;
    return 0;
}

/* ---- the path: the road model's plan as a polyline ---- */
typedef struct {
    int n;
    const double *px, *py, *ps;
    double *hd, *mid;            /* per segment: its direction; the arc length of its middle */
    double c0, s0, c1, s1;
} DfPath;

/* numpy.interp(x, xp, fp) for one x (xp ascending, n >= 2) */
static double df_interp(double x, const double *xp, const double *fp, int n)
{
    if (isnan(x))
        return x;
    if (x > xp[n - 1])
        return fp[n - 1];
    if (x < xp[0])
        return fp[0];
    int lo = 0, hi = n;
    while (lo < hi) {
        int m = (lo + hi) >> 1;
        if (x >= xp[m])
            lo = m + 1;
        else
            hi = m;
    }
    int j = lo - 1;
    if (j == n - 1 || xp[j] == x)
        return fp[j];
    double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    double r = slope * (x - xp[j]) + fp[j];
    if (isnan(r)) {
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if (isnan(r) && fp[j] == fp[j + 1])
            r = fp[j];
    }
    return r;
}

/* futures._along: the point of the polyline at arc length s (beyond its ends: along the end segments), the
 * direction there */
static void df_along(const DfPath *P, double s, double *x, double *y, double *h)
{
    int n = P->n;
    *x = df_interp(s, P->ps, P->px, n);
    *y = df_interp(s, P->ps, P->py, n);
    *h = n > 2 ? df_interp(s, P->mid, P->hd, n - 1) : P->hd[0];
    if (s < P->ps[0]) {
        *x = P->px[0] + (s - P->ps[0]) * P->c0;
        *y = P->py[0] + (s - P->ps[0]) * P->s0;
    }
    if (s > P->ps[n - 1]) {
        *x = P->px[n - 1] + (s - P->ps[n - 1]) * P->c1;
        *y = P->py[n - 1] + (s - P->ps[n - 1]) * P->s1;
    }
}

/* futures._project: (arc length, signed offset left) of a point on the polyline: the nearest segment */
static void df_project(const DfPath *P, double x, double y, double *s, double *off)
{
    int n = P->n, have = 0;
    double bd = 0.0, bs = 0.0, bo = 0.0;
    for (int i = 0; i < n - 1; i++) {
        double ax = P->px[i], ay = P->py[i], ex = P->px[i + 1] - ax, ey = P->py[i + 1] - ay;
        double L2 = ex * ex + ey * ey;
        if (L2 <= 1e-12)
            continue;
        double f = ((x - ax) * ex + (y - ay) * ey) / L2;
        double lo = i == 0 ? -INFINITY : 0.0, hi = i == n - 2 ? INFINITY : 1.0;
        double m = DF_MAX(lo, f), fc = DF_MIN(hi, m);
        double qx = ax + fc * ex, qy = ay + fc * ey;
        double d2 = DF_POW2(x - qx) + DF_POW2(y - qy);
        if (!have || d2 < bd) {
            double L = sqrt(L2);
            have = 1;
            bd = d2;
            bs = P->ps[i] + fc * L;
            bo = ((x - ax) * (-ey) + (y - ay) * ex) / L;
        }
    }
    *s = have ? bs : 0.0;
    *off = have ? bo : 0.0;
}

/* decide_path(x, y): the plan's line (x forward, y RIGHT as the road model gives it) as (px, py, ps) in the car
 * frame (y left), each bytes of float64, from the car (a first point at x = 0 where the line starts ahead); None
 * where it is not a line (fewer than two points, a number that is none, x going back, shorter than 5 m) */
static PyObject *n_decide_path(PyObject *self, PyObject *args)
{
    PyObject *ox, *oy, *res = NULL, *bx = NULL, *by = NULL, *bs = NULL;
    if (!PyArg_ParseTuple(args, "OO", &ox, &oy))
        return NULL;
    if (!(PyList_Check(ox) || PyTuple_Check(ox)) || !(PyList_Check(oy) || PyTuple_Check(oy)))
        Py_RETURN_NONE;
    Py_ssize_t n = PySequence_Size(ox);
    if (n < 2 || n != PySequence_Size(oy))
        Py_RETURN_NONE;
    bx = PyBytes_FromStringAndSize(NULL, (n + 1) * 8);
    by = PyBytes_FromStringAndSize(NULL, (n + 1) * 8);
    bs = PyBytes_FromStringAndSize(NULL, (n + 1) * 8);
    if (!bx || !by || !bs)
        goto done;
    double *px = (double *)PyBytes_AsString(bx) + 1, *py = (double *)PyBytes_AsString(by) + 1;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *a = PyList_Check(ox) ? PyList_GetItem(ox, i) : PyTuple_GetItem(ox, i);
        PyObject *b = PyList_Check(oy) ? PyList_GetItem(oy, i) : PyTuple_GetItem(oy, i);
        double u = PyFloat_AsDouble(a), w = (u == -1.0 && PyErr_Occurred()) ? 0.0 : PyFloat_AsDouble(b);
        if (PyErr_Occurred()) {
            PyErr_Clear();
            goto none;
        }
        px[i] = u;
        py[i] = -w;
        if (!isfinite(u) || !isfinite(w) || (i > 0 && !(px[i] - px[i - 1] >= 0.0)))
            goto none;
    }
    if (px[n - 1] < 5.0)
        goto none;
    Py_ssize_t m = n;
    if (px[0] > 0.0) {                           /* from the car */
        px--;
        py--;
        px[0] = 0.0;
        py[0] = py[1];
        m = n + 1;
    }
    double *ps = (double *)PyBytes_AsString(bs), acc = 0.0;
    ps[0] = 0.0;
    for (Py_ssize_t i = 1; i < m; i++) {
        acc += hypot(px[i] - px[i - 1], py[i] - py[i - 1]);
        ps[i] = acc;
    }
    {
        PyObject *cx = PyBytes_FromStringAndSize((const char *)px, m * 8), *cy = PyBytes_FromStringAndSize((const char *)py, m * 8);
        PyObject *cs = PyBytes_FromStringAndSize((const char *)ps, m * 8);
        if (cx && cy && cs)
            res = PyTuple_Pack(3, cx, cy, cs);
        Py_XDECREF(cx);
        Py_XDECREF(cy);
        Py_XDECREF(cs);
    }
    goto done;
none:
    res = Py_None;
    Py_INCREF(res);
done:
    Py_XDECREF(bx);
    Py_XDECREF(by);
    Py_XDECREF(bs);
    return res;
}

/* ---- a radar object as the decision reads it ---- */
typedef struct {
    PyObject *o, *id;            /* borrowed */
    double x, y;                 /* float(x_m), float(y_m): 0 where absent */
    double sd[4];                /* `std`, as far as it reaches */
    int sdfin[4];
    Py_ssize_t nsd;
    int motion;                  /* 1 stationary, -1 moving, 0 neither */
    int parked, in_path, threat, ps_fin;
    double ps, vx, vy;           /* vx, vy: the finite value or 0 */
    double L, W, sL, sW, h, sh;  /* futures.extent: h, sh NaN where not known */
    int cls, src_radar;
    int has_ctr;
    double cdx, cdy;             /* futures.centre_offset */
    int stands, hfm;             /* futures.stands, heading_from_motion */
    double trav_h, trav_sh;      /* futures.travel_heading */
    /* its futures */
    int first, nm, cv, along, row;
} DfObj;

static int df_parse(PyObject *o, DfObj *q)
{
    q->o = o;
    q->id = df_get(o, S_id);
    if (q->id == NULL)
        q->id = Py_None;
    if (df_numd(o, S_x_m, &q->x) < 0 || df_numd(o, S_y_m, &q->y) < 0)
        return -1;
    PyObject *sd = df_get(o, S_std);
    q->sdfin[0] = q->sdfin[1] = q->sdfin[2] = q->sdfin[3] = 0;
    q->nsd = df_seq(sd, q->sd, q->sdfin, 4);
    PyObject *mo = df_get(o, S_motion);
    q->motion = df_is(mo, S_stationary) ? 1 : df_is(mo, S_moving) ? -1 : 0;
    PyObject *fl = df_get(o, S_flags);
    q->parked = q->in_path = q->threat = 0;
    if (fl != NULL && fl != Py_None && (PyList_Check(fl) || PyTuple_Check(fl))) {
        if ((q->parked = PySequence_Contains(fl, df_s[S_parked])) < 0 || (q->in_path = PySequence_Contains(fl, df_s[S_in_path])) < 0 ||
            (q->threat = PySequence_Contains(fl, df_s[S_threat])) < 0)
            return -1;
    }
    q->ps = 0.0;
    q->ps_fin = df_fin(df_get(o, S_p_stationary), &q->ps);
    PyObject *ovx = df_get(o, S_vx_mps), *ovy = df_get(o, S_vy_mps);
    q->vx = q->vy = 0.0;
    df_fin(ovx, &q->vx);
    df_fin(ovy, &q->vy);
    /* futures.extent */
    q->cls = df_class(df_get(o, S_class));
    const double *sz = DF_SIZE[q->cls];
    double L, W, sL, sW, h, sh;
    int hL, hW, hsL, hsW, hh, hsh;
    df_vs(o, S_len, S_len_std, S_std_len, &L, &hL, &sL, &hsL);
    df_vs(o, S_wid, S_wid_std, S_std_wid, &W, &hW, &sW, &hsW);
    df_vs(o, S_hdg, S_hdg_std, S_std_hdg, &h, &hh, &sh, &hsh);
    if (!hsh && hh) {                             /* the radar's 1 sigma of len, wid, hdg: the heading's */
        double es[3];
        int ef[3] = {0, 0, 0};
        if (df_seq(df_get(o, S_ext_std), es, ef, 3) >= 3 && ef[2]) {
            sh = es[2];
            hsh = 1;
        }
    }
    q->src_radar = hL && hW;
    q->L = hL && 0.1 <= L && L <= 40.0 ? L : sz[0];
    q->W = hW && 0.1 <= W && W <= 5.0 ? W : sz[1];
    q->sL = hsL && sL >= 0.0 ? sL : sz[2];
    q->sW = hsW && sW >= 0.0 ? sW : sz[3];
    /* the radar's direction of travel: where the speed stands clear of its spread */
    double hr = 0.0, sp = 0.0;
    int hr_fin = df_fin(df_get(o, S_heading_rad), &hr);
    int sv_ok = q->nsd >= 4 && q->sdfin[2] && q->sdfin[3];
    double sv = sv_ok ? DF_MAX(q->sd[2], q->sd[3]) : 0.0;
    if (!df_fin(df_get(o, S_speed_mps), &sp)) {
        double a, b;
        if (df_num0(ovx, &a) < 0 || df_num0(ovy, &b) < 0)
            return -1;
        sp = df_hypot(a, b);
    }
    double lim = sv_ok ? 2.0 * sv : INFINITY;
    q->hfm = hr_fin && sp >= DF_MAX(2.0, lim);
    q->trav_h = hr;
    q->trav_sh = atan2(sv, DF_MAX(sp, 1e-6));
    if (!hh && q->hfm) {
        h = hr;
        hh = 1;
    }
    q->h = hh ? h : NAN;
    q->sh = hsh ? sh : NAN;
    /* futures.centre_offset */
    q->has_ctr = 0;
    double c[2], xm, ym;
    int cf[2] = {0, 0};
    if (df_seq(df_get(o, S_ctr), c, cf, 2) == 2 && cf[0] && cf[1] && df_fin(df_get(o, S_x_m), &xm) && df_fin(df_get(o, S_y_m), &ym)) {
        q->cdx = c[0] - xm;
        q->cdy = c[1] - ym;
        q->has_ctr = df_hypot(q->cdx, q->cdy) <= 25.0;
    }
    q->stands = q->motion == -1 ? 0 : (q->motion == 1 || q->parked || (q->ps_fin && q->ps >= 0.5));
    return 0;
}

/* futures.cv_future: one constant-velocity mode into a row (p0, p, x, y, sx, sy, rxy); 1 where it follows the path */
static int df_cv(const DfObj *q, int stationary, const double *t, int N, const double *fc, const DfPath *P, double *row)
{
    double sd[4], vx, vy, svx, svy, cap = fc[FC_VSTD_CAP];
    for (int i = 0; i < 4; i++)
        sd[i] = i < q->nsd && q->sdfin[i] ? q->sd[i] : 1.0;
    if (stationary) {
        vx = vy = 0.0;
        svx = svy = fc[FC_STILL_VSTD];
    } else {
        vx = q->vx;
        vy = q->vy;
        double sp = df_hypot(vx, vy);
        if (sp > fc[FC_SPEED_CAP]) {
            vx = vx * fc[FC_SPEED_CAP] / sp;
            vy = vy * fc[FC_SPEED_CAP] / sp;
        }
        svx = DF_MIN(sd[2], cap);
        svy = DF_MIN(sd[3], cap);
    }
    double *X = row + 2, *Y = X + N, *SX = Y + N, *SY = SX + N, *R = SY + N;
    double a0 = DF_POW2(sd[0]), a1 = DF_POW2(sd[1]);
    row[0] = row[1] = 1.0;
    for (int i = 0; i < N; i++) {
        double u = svx * t[i], w = svy * t[i];
        SX[i] = sqrt(a0 + u * u);
        SY[i] = sqrt(a1 + w * w);
        X[i] = q->x + vx * t[i];
        Y[i] = q->y + vy * t[i];
        R[i] = 0.0;
    }
    if (P != NULL && fc[FC_ALONG_PATH] > 0.5 && df_hypot(vx, vy) >= fc[FC_ALONG_MIN]) {
        double s0, off, hx, hy, h0;
        df_project(P, q->x, q->y, &s0, &off);
        df_along(P, s0, &hx, &hy, &h0);
        double dv = atan2(vy, vx) - h0;
        dv = df_mod(dv + M_PI, 2.0 * M_PI) - M_PI;
        double lim = fc[FC_ALONG_HDG_DEG] * (M_PI / 180.0);
        if (fabs(dv) <= lim || fabs(dv) >= M_PI - lim) {
            double va = df_hypot(vx, vy) * (fabs(dv) <= lim ? 1.0 : -1.0);
            for (int i = 0; i < N; i++) {
                double cx, cy, ch;
                df_along(P, s0 + va * t[i], &cx, &cy, &ch);
                X[i] = cx - off * sin(ch);
                Y[i] = cy + off * cos(ch);
            }
            return 1;
        }
    }
    return 0;
}

static PyObject *df_round(double v, int n)       /* round(v, n) */
{
    int ok;
    double r = round_fast(v, n, &ok);
    if (ok)
        return PyFloat_FromDouble(r);
    PyObject *a = Py_BuildValue("(di)", v, n), *res = NULL;
    if (a != NULL) {
        res = PyObject_CallObject(builtin_round, a);
        Py_DECREF(a);
    }
    return res;
}

static PyObject *df_r(double v, int n)           /* threat._r: None where it is no number */
{
    if (!isfinite(v))
        Py_RETURN_NONE;
    return df_round(v, n);
}

static int df_set(PyObject *d, int key, PyObject *v)      /* d[key] = v; takes v */
{
    if (v == NULL)
        return -1;
    int rc = PyDict_SetItem(d, df_s[key], v);
    Py_DECREF(v);
    return rc;
}

static int df_setb(PyObject *d, int key, PyObject *v)     /* d[key] = v; v borrowed */
{
    return v == NULL ? -1 : PyDict_SetItem(d, df_s[key], v);
}

static const int DF_LEVEL_NAME[5] = {S_none, S_information, S_caution, S_warning, S_imminent};
static const int DF_LANE_KEY[7] = {S_ego, S_left, S_right, S_onc, S_cross, S_park, S_off};

/* what the prediction's lane probabilities say of one object: the mass, P(ego), P(cross), P(a lane along the road) */
typedef struct {
    int has;
    double sum, ego, cross, along;
    PyObject *dict;              /* the message's own dict (borrowed), or NULL: from the array */
    const double *arr;
} DfLane;

static void df_lane(PyObject *lanes, const double *larr, Py_ssize_t nl, int row, DfLane *ln)
{
    DfSum all = {0.0, 0.0, 0}, al = {0.0, 0.0, 0};
    ln->has = 0;
    ln->dict = NULL;
    ln->arr = NULL;
    ln->sum = ln->ego = ln->cross = ln->along = 0.0;
    if (row < 0)
        return;
    if (larr != NULL) {
        if (row >= nl)
            return;
        const double *a = larr + 7 * row;
        ln->has = 1;
        ln->arr = a;
        for (int i = 0; i < 7; i++)
            if (isfinite(a[i])) {
                df_sum_add(&all, a[i], 0);
                if (i < 4)
                    df_sum_add(&al, a[i], 0);
            }
        ln->ego = isfinite(a[0]) ? a[0] : 0.0;
        ln->cross = isfinite(a[4]) ? a[4] : 0.0;
        ln->sum = df_sum(&all);
        ln->along = df_sum(&al);
        return;
    }
    if (lanes == NULL || !PyList_Check(lanes) || row >= PyList_Size(lanes))
        return;
    PyObject *d = PyList_GetItem(lanes, row);
    if (d == NULL || !PyDict_Check(d))
        return;
    ln->has = 1;
    ln->dict = d;
    Py_ssize_t pos = 0;
    PyObject *k, *v;
    double x;
    while (PyDict_Next(d, &pos, &k, &v))
        if (df_fin(v, &x))
            df_sum_add(&all, x, PyLong_Check(v));
    df_fin(df_get(d, S_ego), &ln->ego);
    df_fin(df_get(d, S_cross), &ln->cross);
    static const int AL[4] = {S_ego, S_left, S_right, S_onc};
    for (int i = 0; i < 4; i++) {
        v = df_get(d, AL[i]);
        if (df_fin(v, &x))
            df_sum_add(&al, x, PyLong_Check(v));
    }
    ln->sum = df_sum(&all);
    ln->along = df_sum(&al);
}

/* futures.road_axis: a heading taken to the road's axis (either direction) where it is within snap of it */
static void df_road_axis(double *h, double *sh, double road, double snap, double sd)
{
    if (!isfinite(*h)) {
        *h = road;
        *sh = sd;
        return;
    }
    double d = df_mod(*h - road + M_PI / 2.0, M_PI) - M_PI / 2.0;
    if (fabs(d) > snap)
        return;
    *h = *h - d;
    *sh = isfinite(*sh) ? DF_MIN(*sh, sd) : sd;
}

/* decide_frame(objects, ego, futures, path, t_own, fcfg, tcfg, thr, mu, shown, opt)
 *   objects   the radar message's `objects` (a list of dicts) or None
 *   ego       the radar message's `ego` (a dict) or None
 *   futures   None, or (t, ego, ego_names, ego_given, modes, names, rows, lanes, sel) of decide/futures.py Futures
 *   path      None, or (px, py, ps) of decide_path
 *   t_own     the grid of the degraded futures (no grid in `futures`)
 *   fcfg, tcfg, thr   the futures' constants (FC_*), the threat model's (DC_*, DX_*; V0, A0 and the available
 *             accelerations are set here), the thresholds
 *   mu        the road's friction
 *   shown     {object id: the level shown}: the frame before
 *   opt       1: every judged pair in the listed records (`pairs`); 2: every judged object gets a record, and the
 *             arrays are returned (tests, tools)
 * -> (records, listed, objects by id, shown, judged, cv, invalid, ego_own, steps, a_long, a_lat, ego_modes,
 *     horizon, detail)
 *   records   the dicts of decide.objects[] of the objects that matter to a consumer or an alert (a level or an
 *             exit level, a collision probability from list_min_p, a headway); listed: those the message lists,
 *             in its order; objects by id: {id: radar object} of the records
 *   steps     of the horizon (its simulation's), -1 where nothing was judged
 *   detail    None, or {"t", "ego", "ohdr", "modes", "pair", "obj": bytes of float64; "names", "ego_names",
 *             "src": lists} */
static PyObject *n_decide_frame(PyObject *self, PyObject *args)
{
    PyObject *objects, *ego, *fut, *path, *ot, *ofc, *otc, *othr, *shown;
    double mu;
    int opt;
    if (!PyArg_ParseTuple(args, "OOOOOOOOdO!i", &objects, &ego, &fut, &path, &ot, &ofc, &otc, &othr, &mu, &PyDict_Type,
                          &shown, &opt))
        return NULL;
    if (df_strings() < 0)
        return NULL;
    View vt, vfc, vtc, vthr, vg, veg, vmd, vrw, vln, vpx, vpy, vps;
    vt.held = vfc.held = vtc.held = vthr.held = vg.held = veg.held = vmd.held = vrw.held = vln.held = vpx.held = vpy.held = vps.held = 0;
    PyObject *res = NULL, *recs = NULL, *listed = NULL, *byid = NULL, *shown_new = NULL, *ego_modes = NULL, *detail = NULL;
    double *mem = NULL;
    DfObj *Q = NULL;
    PyObject **mname = NULL, **ename = NULL, **lrec = NULL;
    const double *t_own, *fc, *tc, *thr, *grid = NULL, *hego = NULL, *hmodes = NULL, *hrows = NULL, *hlanes = NULL;
    Py_ssize_t n_own, nfc, ntc, nthr, ng = 0, nheg = 0, nhm = 0, nhr = 0, nhl = 0;
    PyObject *h_enames = NULL, *h_names = NULL, *h_lanes = NULL, *h_sel = NULL;
    int ego_given = 0;
    if (df_buf(ot, &vt, &t_own, &n_own, "t_own") < 0 || df_buf(ofc, &vfc, &fc, &nfc, "fcfg") < 0 ||
        df_buf(otc, &vtc, &tc, &ntc, "tcfg") < 0 || df_buf(othr, &vthr, &thr, &nthr, "thr") < 0)
        goto done;
    if (nfc < FC_N || ntc < DX_NC || nthr < 24 || nthr % 24) {
        PyErr_SetString(PyExc_ValueError, "decide_frame: the constants' arrays are too short");
        goto done;
    }
    if (fut != Py_None) {
        if (!PyTuple_Check(fut) || PyTuple_Size(fut) != 9) {
            PyErr_SetString(PyExc_TypeError, "decide_frame: futures: the nine fields of Futures");
            goto done;
        }
        PyObject *f_t = PyTuple_GetItem(fut, 0), *f_ego = PyTuple_GetItem(fut, 1), *f_lanes = PyTuple_GetItem(fut, 7);
        h_enames = PyTuple_GetItem(fut, 2);
        ego_given = PyObject_IsTrue(PyTuple_GetItem(fut, 3));
        h_names = PyTuple_GetItem(fut, 5);
        h_sel = PyTuple_GetItem(fut, 8);
        if (f_t != Py_None && df_buf(f_t, &vg, &grid, &ng, "futures t") < 0)
            goto done;
        if (f_ego != Py_None && df_buf(f_ego, &veg, &hego, &nheg, "futures ego") < 0)
            goto done;
        if (df_buf(PyTuple_GetItem(fut, 4), &vmd, &hmodes, &nhm, "futures modes") < 0 ||
            df_buf(PyTuple_GetItem(fut, 6), &vrw, &hrows, &nhr, "futures rows") < 0)
            goto done;
        if (f_lanes != Py_None) {
            if (PyList_Check(f_lanes))
                h_lanes = f_lanes;
            else if (df_buf(f_lanes, &vln, &hlanes, &nhl, "futures lanes") < 0)
                goto done;
        }
        if (grid != NULL && ng < 1)
            grid = NULL;
        if (!PyList_Check(h_names) || !PyList_Check(h_enames) || !(PyDict_Check(h_sel) || PyList_Check(h_sel)) ||
            (grid != NULL && (nhm % (2 + 5 * ng) || nheg % (1 + 3 * ng) || PyList_Size(h_names) < nhm / (2 + 5 * ng) ||
                              PyList_Size(h_enames) < nheg / (1 + 3 * ng))) || nhr % 3 || nhl % 7) {
            PyErr_SetString(PyExc_ValueError, "decide_frame: futures: array sizes do not fit together");
            goto done;
        }
    }
    DfPath P;
    P.n = 0;
    P.hd = NULL;
    if (path != Py_None) {
        Py_ssize_t a, b, c;
        if (!PyTuple_Check(path) || PyTuple_Size(path) != 3 || df_buf(PyTuple_GetItem(path, 0), &vpx, &P.px, &a, "path x") < 0 ||
            df_buf(PyTuple_GetItem(path, 1), &vpy, &P.py, &b, "path y") < 0 || df_buf(PyTuple_GetItem(path, 2), &vps, &P.ps, &c, "path s") < 0)
            goto done;
        if (a < 2 || a != b || a != c) {
            PyErr_SetString(PyExc_ValueError, "decide_frame: path: three arrays of one length, two points at least");
            goto done;
        }
        P.n = (int)a;
    }
    /* the grid: the contract's (now put before it where it starts later), or the degraded futures' own */
    const int prepend = grid != NULL && grid[0] > 1e-9;
    const int Nin = (int)ng, N = grid != NULL ? Nin + prepend : (int)n_own;
    if (N < 1 || N > DK_MAXN) {
        PyErr_SetString(PyExc_ValueError, "decide_frame: a grid of 1 to 512 times is needed");
        goto done;
    }
    /* the objects judged: confirmed and coast */
    Py_ssize_t no = objects != Py_None && PyList_Check(objects) ? PyList_Size(objects) : 0;
    int M = 0;
    for (Py_ssize_t i = 0; i < no; i++) {
        PyObject *o = PyList_GetItem(objects, i);
        if (PyDict_Check(o)) {
            PyObject *st = df_get(o, S_state);
            M += df_is(st, S_confirmed) || df_is(st, S_coast);
        }
    }
    PyObject *eg_d = ego != Py_None && PyDict_Check(ego) && PyDict_Size(ego) > 0 ? ego : NULL;
    double e_speed = 0.0, e_yaw = 0.0, e_acc = 0.0;
    if (eg_d != NULL && (df_num0(df_get(eg_d, S_speed_mps), &e_speed) < 0 || df_num0(df_get(eg_d, S_yaw_rate_rps), &e_yaw) < 0 ||
                         df_num0(df_get(eg_d, S_accel_mps2), &e_acc) < 0))
        goto done;
    /* threat.available: the braking and the lateral acceleration a driver can get */
    const double v0 = DF_MAX(0.0, e_speed);
    double fade = DF_MAX(0.5, 1.0 - tc[DX_SPEED_FADE] * DF_MAX(0.0, v0));
    double a_long = mu * DF_G * tc[DX_BRAKE_SHARE] * fade, a_lat = mu * DF_G * tc[DX_STEER_SHARE] * fade;
    {
        double vr = DF_MAX(v0, 0.5), cap = DF_POW2(vr) / tc[DX_TURN_R];
        a_lat = DF_MIN(a_lat, cap);
        a_lat = DF_MAX(a_lat, 0.05);
    }
    recs = PyList_New(0);
    listed = PyList_New(0);
    byid = PyDict_New();
    shown_new = PyDict_New();
    ego_modes = PyList_New(0);
    if (!recs || !listed || !byid || !shown_new || !ego_modes)
        goto done;
    int n_cv = 0, invalid = 0, ego_own = 0, steps = -1;
    double horizon = grid != NULL ? grid[Nin - 1] : t_own[N - 1];
    if (M == 0) {                                 /* nothing to judge: the car's futures are not made */
        ego_own = !(fut != Py_None && ego_given);
        goto out;
    }
    if (N < 2) {
        PyErr_SetString(PyExc_ValueError, "decide_eval: arrays of the documented shapes are needed");
        goto done;
    }
    /* storage: the grid, the path's segments, the car's modes, the objects' modes, the results */
    int Emax = hego != NULL && grid != NULL ? (int)(nheg / (1 + 3 * Nin)) : 0;
    if (Emax < 2)
        Emax = 2;
    Py_ssize_t Kmax = 0;
    Q = (DfObj *)PyMem_Calloc((size_t)M, sizeof(DfObj));
    if (Q == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    {
        int m = 0;
        for (Py_ssize_t i = 0; i < no; i++) {
            PyObject *o = PyList_GetItem(objects, i);
            if (!PyDict_Check(o))
                continue;
            PyObject *st = df_get(o, S_state);
            if (!(df_is(st, S_confirmed) || df_is(st, S_coast)))
                continue;
            DfObj *q = &Q[m++];
            if (df_parse(o, q) < 0)
                goto done;
            q->row = -1;
            if (fut != Py_None) {
                if (PyDict_Check(h_sel)) {
                    PyObject *r = PyDict_GetItemWithError(h_sel, q->id);
                    if (r == NULL && PyErr_Occurred())
                        goto done;
                    if (r != NULL)
                        q->row = (int)PyLong_AsLong(r);
                } else if (i < PyList_Size(h_sel))
                    q->row = (int)PyLong_AsLong(PyList_GetItem(h_sel, i));
                if (q->row == -1 && PyErr_Occurred())
                    goto done;
                if (q->row >= nhr / 3)
                    q->row = -1;
            }
            int cnt = q->row >= 0 ? (int)hrows[3 * q->row + 1] : 0;
            Kmax += cnt > 2 ? cnt : 2;
        }
    }
    const int wE = 1 + 3 * N, wM = 2 + 5 * N;
    size_t need = (size_t)N + 2 * (size_t)(P.n > 0 ? P.n : 1) + (size_t)Emax * wE + (size_t)Kmax * (wM + 2) +
                  (size_t)M * DK_OHC + (size_t)Kmax * Emax * DK_PF + (size_t)M * DK_OF + DX_NC;
    mem = (double *)PyMem_Malloc(need * sizeof(double));
    mname = (PyObject **)PyMem_Malloc((size_t)(Kmax + 1) * sizeof(PyObject *));
    ename = (PyObject **)PyMem_Malloc((size_t)Emax * sizeof(PyObject *));
    if (!mem || !mname || !ename) {
        PyErr_NoMemory();
        goto done;
    }
    double *w = mem;
    double *t = w;
    w += N;
    if (grid != NULL) {
        if (prepend)
            t[0] = 0.0;
        for (int i = 0; i < Nin; i++)
            t[i + prepend] = grid[i];
    } else
        for (int i = 0; i < N; i++)
            t[i] = t_own[i];
    const DfPath *PP = NULL;
    if (P.n > 0) {
        P.hd = w;
        P.mid = w + P.n;
        w += 2 * P.n;
        for (int i = 0; i < P.n - 1; i++) {
            P.hd[i] = atan2(P.py[i + 1] - P.py[i], P.px[i + 1] - P.px[i]);
            P.mid[i] = 0.5 * (P.ps[i + 1] + P.ps[i]);
        }
        P.c0 = cos(P.hd[0]);
        P.s0 = sin(P.hd[0]);
        P.c1 = cos(P.hd[P.n - 2]);
        P.s1 = sin(P.hd[P.n - 2]);
        PP = &P;
    }
    /* ---- the car's futures: the contract's, else its own from its motion now and the plan ---- */
    double *EG = w;
    w += (size_t)Emax * wE;
    int E = 0;
    if (grid != NULL && hego != NULL) {
        E = (int)(nheg / (1 + 3 * Nin));
        DfSum sum = {0.0, 0.0, 0};
        for (int j = 0; j < E; j++) {
            const double *src = hego + (size_t)j * (1 + 3 * Nin);
            double *row = EG + (size_t)j * wE, p = isfinite(src[0]) ? DF_MIN(1.0, DF_MAX(0.0, src[0])) : 1.0 / E;
            row[0] = p;
            df_sum_add(&sum, p, 0);
            for (int q = 0; q < 3; q++) {
                double *d = row + 1 + q * N;
                if (prepend)
                    d[0] = q == 2 ? e_speed : 0.0;
                for (int i = 0; i < Nin; i++) {
                    double v = src[1 + q * Nin + i];
                    d[i + prepend] = q == 2 ? ((v >= 0.0 || isnan(v)) ? v : 0.0) : v;
                }
            }
            ename[j] = PyList_GetItem(h_enames, j);
        }
        double sp = df_sum(&sum);
        if (E > 0 && sp > 0.0)
            for (int j = 0; j < E; j++)
                EG[(size_t)j * wE] = EG[(size_t)j * wE] / sp;
        else
            E = 0;
    }
    if (E == 0) {                                 /* futures.ego_futures */
        ego_own = 1;
        double w0 = e_yaw, lo = fc[FC_EGO_A_LO], hi = fc[FC_EGO_A_HI];
        double am = DF_MAX(lo, e_acc), a0 = DF_MIN(hi, am);
        double t_stop = a0 < 0.0 ? v0 / -a0 : INFINITY, half = 0.5 * a0;
        double s_stop = a0 < 0.0 ? v0 * t_stop + half * t_stop * t_stop : 0.0;
        double k = v0 > 0.5 ? w0 / v0 : 0.0;
        E = PP != NULL ? 2 : 1;
        double *turn = EG + (size_t)(E - 1) * wE, *pth = EG;
        double s0 = 0.0, off0 = 0.0;
        if (PP != NULL) {
            df_project(PP, 0.0, 0.0, &s0, &off0);
            pth[0] = fc[FC_EGO_PATH_P];
            turn[0] = 1.0 - fc[FC_EGO_PATH_P];
            ename[0] = df_s[S_path];
            ename[1] = df_s[S_turn];
        } else {
            turn[0] = 1.0;
            ename[0] = df_s[S_turn];
        }
        for (int i = 0; i < N; i++) {
            double ti = t[i], vv = v0 + a0 * ti, v = (0.0 >= vv) ? 0.0 : vv;
            double s = (a0 < 0.0 && !(ti <= t_stop)) ? s_stop : v0 * ti + half * ti * ti;
            double psi = k * s;
            if (fabs(k) > 1e-6) {
                turn[1 + i] = sin(psi) / k;
                turn[1 + N + i] = (1.0 - cos(psi)) / k;
            } else {
                turn[1 + i] = s;
                turn[1 + N + i] = 0.0;
            }
            turn[1 + 2 * N + i] = v;
            if (PP != NULL) {
                /* the plan's line from the car's own place: its offset at the car taken out over the first 2 s */
                double cx, cy, ch, f = 1.0 - ti / 2.0;
                df_along(PP, s0 + s, &cx, &cy, &ch);
                f = f < 0.0 ? 0.0 : (f > 1.0 ? 1.0 : f);
                pth[1 + i] = cx - (-off0) * f * sin(ch);
                pth[1 + N + i] = cy + (-off0) * f * cos(ch);
                pth[1 + 2 * N + i] = v;
            }
        }
    }
    for (int j = 0; j < E; j++) {                 /* the message's ego_modes: [name, round(p, 3)] */
        PyObject *pr = df_round(EG[(size_t)j * wE], 3), *pair = pr ? PyList_New(2) : NULL;
        if (pair == NULL) {
            Py_XDECREF(pr);
            goto done;
        }
        Py_INCREF(ename[j]);
        PyList_SetItem(pair, 0, ename[j]);
        PyList_SetItem(pair, 1, pr);
        int rc = PyList_Append(ego_modes, pair);
        Py_DECREF(pair);
        if (rc < 0)
            goto done;
    }
    /* ---- the objects' futures ---- */
    double *MD = w;
    w += (size_t)Kmax * wM;
    double *pn = w;                               /* per mode: p0, p as normalised (before the lanes' weight) */
    w += (size_t)Kmax * 2;
    int K = 0;
    for (int m = 0; m < M; m++) {
        DfObj *q = &Q[m];
        int state = q->row >= 0 ? (int)hrows[3 * q->row + 2] : 0;
        q->first = K;
        q->nm = 0;
        q->cv = 1;
        q->along = 0;
        if (grid != NULL && state == 1) {         /* futures.contract_modes */
            int first = (int)hrows[3 * q->row], cnt = (int)hrows[3 * q->row + 1];
            int two = q->nsd >= 2 && q->sdfin[0] && q->sdfin[1];
            double sx0 = two ? q->sd[0] : 1.0, sy0 = two ? q->sd[1] : 1.0;
            DfSum sum0 = {0.0, 0.0, 0}, sumc = {0.0, 0.0, 0};
            if (first < 0 || cnt < 0 || (Py_ssize_t)(first + cnt) * (2 + 5 * Nin) > nhm)
                cnt = 0;
            for (int i = 0; i < cnt; i++) {
                const double *src = hmodes + (size_t)(first + i) * (2 + 5 * Nin);
                double *row = MD + (size_t)(K + i) * wM;
                for (int c = 0; c < 5; c++) {
                    const double *a = src + 2 + c * Nin;
                    double *d = row + 2 + c * N;
                    int given = !isnan(a[0]);
                    if (prepend)
                        d[0] = c == 0 ? q->x : c == 1 ? q->y : c == 2 ? sx0 : c == 3 ? sy0 : 0.0;
                    for (int j = 0; j < Nin; j++) {
                        double v = given ? a[j] : (c == 2 ? sx0 : c == 3 ? sy0 : 0.0);
                        if (c == 2 || c == 3)
                            v = fabs(v);
                        else if (c == 4)
                            v = v < -0.99 ? -0.99 : (v > 0.99 ? 0.99 : v);
                        d[j + prepend] = v;
                    }
                }
                double p = isfinite(src[1]) ? DF_MIN(1.0, DF_MAX(0.0, src[1])) : 1.0 / cnt;
                double p0 = isfinite(src[0]) ? DF_MIN(1.0, DF_MAX(0.0, src[0])) : p;
                pn[2 * (K + i)] = p0;
                pn[2 * (K + i) + 1] = p;
                df_sum_add(&sum0, p0, 0);
                df_sum_add(&sumc, p, 0);
                mname[K + i] = PyList_GetItem(h_names, first + i);
            }
            double s0 = df_sum(&sum0), sc = df_sum(&sumc);
            if (cnt > 0 && s0 > 0.0 && sc > 0.0) {
                for (int i = 0; i < cnt; i++) {
                    pn[2 * (K + i)] = pn[2 * (K + i)] / s0;
                    pn[2 * (K + i) + 1] = pn[2 * (K + i) + 1] / sc;
                }
                q->nm = cnt;
                q->cv = 0;
            }
        }
        if (q->cv) {                              /* futures.cv_futures: the degraded adapter */
            if (state != 0)
                invalid++;
            n_cv++;
            double ps = q->ps;
            int one = fc[FC_MODES_RADAR] < 0.5;
            if (!one) {
                if (q->motion == 1 || q->parked)
                    ps = 1.0;
                else if (q->motion == -1)
                    ps = 0.0;
                else if (!q->ps_fin)
                    one = 1;
            }
            double *r0 = MD + (size_t)K * wM, *r1 = r0 + wM;
            if (one) {
                int st = q->motion == 1;
                int ap = df_cv(q, st, t, N, fc, PP, r0);
                mname[K] = df_s[st ? (ap ? S_stand_path : S_stand) : (ap ? S_cv_path : S_cv)];
                q->along = ap;
                q->nm = 1;
            } else {
                ps = DF_MIN(1.0, DF_MAX(0.0, ps));
                if (ps >= 1.0 - fc[FC_STAND_MIN] || ps <= fc[FC_STAND_MIN]) {
                    int st = ps >= 1.0 - fc[FC_STAND_MIN];
                    int ap = df_cv(q, st, t, N, fc, PP, r0);
                    mname[K] = df_s[st ? (ap ? S_stand_path : S_stand) : (ap ? S_cv_path : S_cv)];
                    q->along = ap;
                    q->nm = 1;
                } else {
                    int a = df_cv(q, 1, t, N, fc, PP, r0), b = df_cv(q, 0, t, N, fc, PP, r1);
                    mname[K] = df_s[a ? S_stand_path : S_stand];
                    mname[K + 1] = df_s[b ? S_cv_path : S_cv];
                    r0[0] = r0[1] = ps;
                    r1[0] = r1[1] = 1.0 - ps;
                    q->along = a || b;
                    q->nm = 2;
                }
            }
            for (int i = 0; i < q->nm; i++) {
                pn[2 * (K + i)] = MD[(size_t)(K + i) * wM];
                pn[2 * (K + i) + 1] = MD[(size_t)(K + i) * wM + 1];
            }
        }
        K += q->nm;
    }
    /* ---- threat.evaluate: each footprint, the lanes' weight of the standing modes ---- */
    double *OH = w;
    w += (size_t)M * DK_OHC;
    for (int m = 0; m < M; m++) {
        DfObj *q = &Q[m];
        DfLane ln;
        df_lane(h_lanes, hlanes, nhl / 7, q->row, &ln);
        double h = q->h, sh = q->sh, rh = 0.0;
        if (PP != NULL) {                         /* the path's direction at its place */
            double s, off, x, y;
            df_project(PP, q->x, q->y, &s, &off);
            df_along(PP, s, &x, &y, &rh);
        }
        if (isfinite(h) && (isfinite(sh) && sh > tc[DX_HDG_RES])) {       /* a box heading the radar did not resolve */
            if (q->hfm) {
                h = q->trav_h;
                sh = q->trav_sh;
            } else if (PP != NULL) {
                sh = NAN;
                df_road_axis(&h, &sh, rh, M_PI, tc[DX_AXIS_STD]);
            }
        } else if (!isfinite(h) && PP != NULL)
            h = rh;
        else if (PP != NULL && DF_SET[q->cls] == 0 && (ln.along >= tc[DX_LANE_AXIS_P] || q->along))
            df_road_axis(&h, &sh, rh, M_PI, tc[DX_AXIS_STD]);
        else if (PP != NULL && tc[DX_AXIS_STANDING] > 0.5 && q->stands && !q->hfm)
            df_road_axis(&h, &sh, rh, tc[DX_AXIS_SNAP], tc[DX_AXIS_STD]);
        double w_stand = 1.0;
        if (tc[DX_LANE_WEIGHT] > 0.5 && !(q->in_path || q->threat || !ln.has) && !(ln.sum < 0.5)) {
            double a = ln.ego + ln.cross, b = DF_MAX(tc[DX_LANE_FLOOR], a);
            w_stand = DF_MIN(1.0, b);
        }
        double *hd = OH + (size_t)m * DK_OHC;
        hd[0] = q->first;
        hd[1] = q->nm;
        hd[2] = q->L;
        hd[3] = q->W;
        hd[4] = q->sL;
        hd[5] = q->sW;
        hd[6] = h;
        hd[7] = sh;
        hd[8] = DF_SET[q->cls];
        hd[9] = q->has_ctr ? 1.0 : 0.0;
        hd[10] = q->has_ctr ? q->cdx : 0.0;
        hd[11] = q->has_ctr ? q->cdy : 0.0;
        for (int i = 0; i < q->nm; i++) {
            double *row = MD + (size_t)(q->first + i) * wM;
            const double *X = row + 2, *Y = X + N;
            double f = 1.0;
            if (w_stand < 1.0) {                  /* futures.mode_stands */
                double far = hypot(X[0] - X[0], Y[0] - Y[0]);
                for (int j = 1; j < N; j++) {
                    double d = hypot(X[j] - X[0], Y[j] - Y[0]);
                    if (d > far || isnan(d))
                        far = d;
                }
                if (far < tc[DX_STAND_MOVE])
                    f = w_stand;
            }
            row[0] = pn[2 * (q->first + i)] * f;
            row[1] = pn[2 * (q->first + i) + 1] * f;
        }
    }
    double *cfg = w;
    w += DX_NC;
    for (int i = 0; i < DC_NC; i++)
        cfg[i] = tc[i];
    cfg[DC_V0] = v0;
    cfg[DC_A0] = e_acc;
    cfg[DC_ALONG] = a_long;
    cfg[DC_ALAT] = a_lat;
    double *PO = w, *OO = w + (size_t)K * E * DK_PF;
    {
        int nsim;
        if (dk_eval(t, N, EG, E, OH, DK_OHC, M, MD, K, cfg, thr, (int)(nthr / 24), PO, OO, &steps, &nsim) < 0)
            goto done;
    }
    /* ---- the records (threat.object_record) of the objects that matter, the level shown, the list ---- */
    {
        const int all = (opt & 2) != 0, pairs = (opt & 1) != 0;
        int nlist = 0;
        lrec = (PyObject **)PyMem_Calloc((size_t)M + 1, sizeof(PyObject *));
        double *lkey = (double *)PyMem_Malloc((size_t)(2 * M + 1) * sizeof(double));
        if (lrec == NULL || lkey == NULL) {
            PyMem_Free(lkey);
            PyErr_NoMemory();
            goto done;
        }
        int bad = 0;
        for (int m = 0; m < M && !bad; m++) {
            const DfObj *q = &Q[m];
            const double *oo = OO + (size_t)m * DK_OF;
            int lv = (int)oo[DO_LE], lx = (int)oo[DO_LX];
            PyObject *pc = df_r(oo[DO_PC], 3);
            if (pc == NULL) {
                bad = 1;
                break;
            }
            double pr = pc == Py_None ? 0.0 : PyFloat_AsDouble(pc);
            int thw = isfinite(oo[DO_THW]);
            if (!(all || lv > 0 || lx > 0 || pr >= tc[DX_LIST_MIN_P] || thw)) {
                Py_DECREF(pc);
                continue;
            }
            PyObject *r = PyDict_New(), *x;
            if (r == NULL) {
                Py_DECREF(pc);
                bad = 1;
                break;
            }
            const double *ms = pn + 2 * q->first;
            bad |= df_setb(r, S_id, q->id) < 0 || df_set(r, S_level, PyLong_FromLong(lv)) < 0 ||
                   df_setb(r, S_level_name, df_s[DF_LEVEL_NAME[lv < 0 || lv > 4 ? 0 : lv]]) < 0 ||
                   df_set(r, S_level_exit, PyLong_FromLong(lx)) < 0 || df_set(r, S_level_cond, PyLong_FromLong((int)oo[DO_LC])) < 0;
            bad |= df_set(r, S_p_coll, pc) < 0;
            bad |= df_set(r, S_p_coll_cond, df_r(oo[DO_PCC], 3)) < 0 || df_set(r, S_ttc_s, df_r(oo[DO_TTC], 2)) < 0 ||
                   df_set(r, S_thw_s, df_r(oo[DO_THW], 2)) < 0 || df_set(r, S_sep_m, df_r(oo[DO_SEP], 2)) < 0 ||
                   df_setb(r, S_kind, df_s[DF_SET[q->cls] == 0 ? S_vehicle : DF_SET[q->cls] == 1 ? S_vru : S_unknown]) < 0;
            x = PyList_New(2);
            if (x != NULL) {
                PyObject *a = df_r(q->L, 1), *b = df_r(q->W, 1);
                if (a == NULL || b == NULL) {
                    Py_XDECREF(a);
                    Py_XDECREF(b);
                    Py_CLEAR(x);
                } else {
                    PyList_SetItem(x, 0, a);
                    PyList_SetItem(x, 1, b);
                }
            }
            bad |= df_set(r, S_size, x) < 0 || df_setb(r, S_size_src, df_s[q->src_radar ? S_radar : S_class]) < 0 ||
                   df_setb(r, S_fut, df_s[q->cv ? S_cv : S_contract]) < 0;
            x = PyList_New(4);
            for (int l = 0; x != NULL && l < 4; l++) {
                PyObject *a = df_r(oo[DO_ME + l], 3);
                if (a == NULL)
                    Py_CLEAR(x);
                else
                    PyList_SetItem(x, l, a);
            }
            bad |= df_set(r, S_mass, x) < 0;
            int b = (int)oo[DO_BEST];
            if (b >= 0) {
                int km = b / E, j = b % E;
                const double *p = PO + ((size_t)(q->first + km) * E + j) * DK_PF;
                PyObject *ar = PyDict_New(), *why = PyDict_New();
                if (ar == NULL || why == NULL) {
                    Py_XDECREF(ar);
                    Py_XDECREF(why);
                    bad = 1;
                } else {
                    bad |= df_set(r, S_btn, df_r(p[DP_BTN], 2)) < 0 || df_set(r, S_stn, df_r(p[DP_STN], 2)) < 0 ||
                           df_set(r, S_tn, df_r(p[DP_TN], 2)) < 0;
                    bad |= df_set(ar, S_brake, df_r(p[DP_AB], 2)) < 0 || df_set(ar, S_left, df_r(p[DP_AL], 2)) < 0 ||
                           df_set(ar, S_right, df_r(p[DP_AR], 2)) < 0 || df_set(ar, S_comb, df_r(p[DP_AC], 2)) < 0 ||
                           df_set(ar, S_comb_dir_deg, df_r(p[DP_DIR] * (180.0 / M_PI), 0)) < 0;
                    bad |= df_set(r, S_a_req, ar) < 0;
                    bad |= df_set(r, S_ttb_s, df_r(p[DP_TTB], 2)) < 0 || df_set(r, S_tts_s, df_r(p[DP_TTS], 2)) < 0 ||
                           df_set(r, S_tm_s, df_r(p[DP_TM], 2)) < 0;
                    int me = (int)oo[DO_WHY];
                    bad |= df_setb(why, S_mode, mname[q->first + km]) < 0 || df_set(why, S_p0, df_r(ms[2 * km], 3)) < 0 ||
                           df_set(why, S_p, df_r(ms[2 * km + 1], 3)) < 0 || df_setb(why, S_ego_mode, ename[j]) < 0 ||
                           df_setb(why, S_measure, df_s[me == 1 ? S_threat_number : me == 2 ? S_time_margin : S_collision_course]) < 0 ||
                           df_set(why, S_pc, df_r(p[DP_PC], 3)) < 0 || df_set(why, S_t_contact_s, df_r(p[DP_TCON], 2)) < 0;
                    bad |= df_set(r, S_why, why) < 0;
                }
            } else {
                bad |= df_set(r, S_btn, PyFloat_FromDouble(0.0)) < 0 || df_set(r, S_stn, PyFloat_FromDouble(0.0)) < 0 ||
                       df_set(r, S_tn, PyFloat_FromDouble(0.0)) < 0 || df_setb(r, S_a_req, Py_None) < 0 ||
                       df_setb(r, S_ttb_s, Py_None) < 0 || df_setb(r, S_tts_s, Py_None) < 0 || df_setb(r, S_tm_s, Py_None) < 0 ||
                       df_setb(r, S_why, Py_None) < 0;
            }
            DfLane ln;
            df_lane(h_lanes, hlanes, nhl / 7, q->row, &ln);
            if (ln.has && ln.dict != NULL)
                bad |= df_setb(r, S_lane, ln.dict) < 0;
            else if (ln.has) {
                PyObject *d = PyDict_New();
                for (int i = 0; d != NULL && i < 7; i++)
                    if (df_set(d, DF_LANE_KEY[i], PyFloat_FromDouble(ln.arr[i])) < 0)
                        Py_CLEAR(d);
                bad |= df_set(r, S_lane, d) < 0;
            }
            /* the level shown: up at the entry level, down below the exit level (hysteresis) */
            long sh = 0;
            PyObject *was = PyDict_GetItemWithError(shown, q->id);
            if (was == NULL && PyErr_Occurred())
                bad = 1;
            else if (was != NULL)
                sh = PyLong_AsLong(was);
            sh = lv > sh ? lv : (lx < sh ? lx : sh);
            x = PyLong_FromLong(sh);
            bad |= x == NULL || PyDict_SetItem(shown_new, q->id, x) < 0;
            bad |= df_set(r, S_shown, x) < 0 || df_setb(r, S_in_path, q->in_path ? Py_True : Py_False) < 0;
            int list = lv || sh || pr >= tc[DX_LIST_MIN_P] || thw;
            if (list && pairs && !bad) {          /* threat.pairs_record */
                PyObject *pl = PyList_New(0);
                for (int km = 0; pl != NULL && km < q->nm; km++)
                    for (int j = 0; pl != NULL && j < E; j++) {
                        const double *p = PO + ((size_t)(q->first + km) * E + j) * DK_PF;
                        if (!(p[DP_W0] > 0.0 || p[DP_WC] > 0.0))
                            continue;
                        PyObject *d = PyDict_New();
                        int e = d == NULL || df_setb(d, S_mode, mname[q->first + km]) < 0 || df_setb(d, S_ego_mode, ename[j]) < 0 ||
                                df_set(d, S_w0, df_r(p[DP_W0], 3)) < 0 || df_set(d, S_w, df_r(p[DP_WC], 3)) < 0 ||
                                df_set(d, S_pc, df_r(p[DP_PC], 3)) < 0 || df_setb(d, S_course, p[DP_COURSE] > 0.5 ? Py_True : Py_False) < 0 ||
                                df_set(d, S_tn, df_r(p[DP_TN], 2)) < 0 || df_set(d, S_btn, df_r(p[DP_BTN], 2)) < 0 ||
                                df_set(d, S_stn, df_r(p[DP_STN], 2)) < 0 || df_set(d, S_tm_s, df_r(p[DP_TM], 2)) < 0 ||
                                df_set(d, S_ttc_s, df_r(p[DP_TTC], 2)) < 0 ||
                                df_set(d, S_level, PyLong_FromLong(isfinite(p[DP_LE]) ? (long)p[DP_LE] : 0)) < 0 ||
                                PyList_Append(pl, d) < 0;
                        Py_XDECREF(d);
                        if (e)
                            Py_CLEAR(pl);
                    }
                bad |= df_set(r, S_pairs, pl) < 0;
            }
            bad |= PyList_Append(recs, r) < 0 || PyDict_SetItem(byid, q->id, q->o) < 0;
            if (list && !bad) {                   /* the message's order: by level, then by collision probability */
                int at = nlist++;
                while (at > 0 && (lkey[2 * (at - 1)] < lv || (lkey[2 * (at - 1)] == lv && lkey[2 * (at - 1) + 1] < pr))) {
                    lrec[at] = lrec[at - 1];
                    lkey[2 * at] = lkey[2 * (at - 1)];
                    lkey[2 * at + 1] = lkey[2 * (at - 1) + 1];
                    at--;
                }
                lrec[at] = r;
                lkey[2 * at] = lv;
                lkey[2 * at + 1] = pr;
            }
            Py_DECREF(r);
        }
        PyMem_Free(lkey);
        for (int i = 0; i < nlist && !bad; i++)
            bad |= PyList_Append(listed, lrec[i]) < 0;
        if (bad) {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_RuntimeError, "decide_frame: a record could not be made");
            goto done;
        }
        if (all) {
            PyObject *names = PyList_New(K), *enames = PyList_New(E), *src = PyList_New(M);
            if (names && enames && src) {
                for (int k = 0; k < K; k++) {
                    Py_INCREF(mname[k]);
                    PyList_SetItem(names, k, mname[k]);
                }
                for (int j = 0; j < E; j++) {
                    Py_INCREF(ename[j]);
                    PyList_SetItem(enames, j, ename[j]);
                }
                for (int m = 0; m < M; m++) {
                    PyObject *s = df_s[Q[m].cv ? S_cv : S_contract];
                    Py_INCREF(s);
                    PyList_SetItem(src, m, s);
                }
                detail = Py_BuildValue("{s:y#,s:y#,s:y#,s:y#,s:y#,s:y#,s:y#,s:O,s:O,s:O}", "t", (const char *)t, (Py_ssize_t)N * 8,
                                       "ego", (const char *)EG, (Py_ssize_t)E * wE * 8, "ohdr", (const char *)OH,
                                       (Py_ssize_t)M * DK_OHC * 8, "modes", (const char *)MD, (Py_ssize_t)K * wM * 8, "pair",
                                       (const char *)PO, (Py_ssize_t)K * E * DK_PF * 8, "obj", (const char *)OO,
                                       (Py_ssize_t)M * DK_OF * 8, "p", (const char *)pn, (Py_ssize_t)K * 2 * 8, "names", names,
                                       "ego_names", enames, "src", src);
            }
            Py_XDECREF(names);
            Py_XDECREF(enames);
            Py_XDECREF(src);
            if (detail == NULL)
                goto done;
        }
    }
out:
    res = Py_BuildValue("(OOOOiiiiiddOdO)", recs, listed, byid, shown_new, M, n_cv, invalid, ego_own, steps, a_long, a_lat,
                        ego_modes, horizon, detail ? detail : Py_None);
done:
    PyMem_Free(mem);
    PyMem_Free(Q);
    PyMem_Free(mname);
    PyMem_Free(ename);
    PyMem_Free(lrec);
    Py_XDECREF(recs);
    Py_XDECREF(listed);
    Py_XDECREF(byid);
    Py_XDECREF(shown_new);
    Py_XDECREF(ego_modes);
    Py_XDECREF(detail);
    view_drop(&vt);
    view_drop(&vfc);
    view_drop(&vtc);
    view_drop(&vthr);
    view_drop(&vg);
    view_drop(&veg);
    view_drop(&vmd);
    view_drop(&vrw);
    view_drop(&vln);
    view_drop(&vpx);
    view_drop(&vpy);
    view_drop(&vps);
    return res;
}

#define DECIDE_METHODS                                                                                                 \
    {"decide_eval", n_decide_eval, METH_VARARGS, "the decision layer's threat measures per object, mode and ego mode (decide.c)"},   \
    {"decide_inst_prob", n_decide_inst_prob, METH_VARARGS, "the exact probability that two footprints overlap at one time (decide.c)"}, \
    {"decide_draws", n_decide_draws, METH_O, "the decision layer's fixed standard-normal draws (decide.c)"},          \
    {"decide_path", n_decide_path, METH_VARARGS, "the road model's plan as the decision layer's path polyline (decide.c)"}, \
    {"decide_frame", n_decide_frame, METH_VARARGS, "one radar message's decision: futures, footprints, threat, records (decide.c)"},
