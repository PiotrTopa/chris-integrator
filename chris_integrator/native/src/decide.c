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
 * It is included by chris_native.c (one translation unit) and uses its buffer helpers. Frame: the car frame of the
 * output time held fixed on the ground (x forward, y left, m); the car starts at the origin, heading 0.
 *
 * decide_eval(t[N], ego[E, 1+3N], ohdr[M, 10], modes[K, 2+5N], cfg[NC], thr[S*2*4*3], out_pair[K*E, PF], out_obj[M, OF])
 *   ego rows      p, x[N], y[N], v[N]
 *   ohdr rows     first mode, number of modes, len, wid, sd_len, sd_wid, hdg (NaN: unknown), sd_hdg, threshold set, 0
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
#define DK_OH 10
#define DK_PF 24
#define DK_OF 18

enum {
    DC_FRONT, DC_REAR, DC_HALFW, DC_V0, DC_A0, DC_ALONG, DC_ALAT, DC_TREACT, DC_TAUB, DC_TAUS, DC_DT, DC_NMAX,
    DC_KSIG, DC_MLONG, DC_MLAT, DC_METHOD, DC_NSAMP, DC_PRUNE, DC_SDEL, DC_SDET, DC_NDIRS, DC_AFAC, DC_TOLA,
    DC_TOLT, DC_REF, DC_OTHERS, DC_PMIN, DC_HMAX, DC_HEXT, DC_ECAPL, DC_ECAPT, DC_ICAPL, DC_ICAPT, DC_NC
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

/* the object's rectangle at step i, inflated by k sigma along its axes (+ margins), with the car's covariance */
static void dk_obj_rect(const double *cfg, const DkObj *o, int i, double esxx, double esxy, double esyy, int infl,
                        DkRect *r)
{
    r->cx = o->cx[i];
    r->cy = o->cy[i];
    r->c = o->c[i];
    r->s = o->s[i];
    r->hl = o->hl;
    r->hw = o->hw;
    if (infl) {
        double sxx = o->sxx[i] + esxx, sxy = o->sxy[i] + esxy, syy = o->syy[i] + esyy;
        double c = r->c, s = r->s;
        double va = c * c * sxx + 2.0 * c * s * sxy + s * s * syy;
        double vc = s * s * sxx - 2.0 * c * s * sxy + c * c * syy;
        double k = cfg[DC_KSIG];
        double ia = k * sqrt(va > 0.0 ? va + 0.25 * o->sdl * o->sdl : 0.25 * o->sdl * o->sdl);
        double ic = k * sqrt(vc > 0.0 ? vc + 0.25 * o->sdw * o->sdw : 0.25 * o->sdw * o->sdw);
        /* bounded: a poorly ranged object's spread is the collision probability's matter, not a wall */
        r->hl += (ia < cfg[DC_ICAPL] ? ia : cfg[DC_ICAPL]) + cfg[DC_MLONG];
        r->hw += (ic < cfg[DC_ICAPT] ? ic : cfg[DC_ICAPT]) + cfg[DC_MLAT];
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
 * collision course: what the car must keep clear of; the car's own spread enters the collision probability only) */
static void dk_pair_prep(DkCtx *C, int j, int k)
{
    (void)j;                                   /* the car executes the demand: its own spread is not added here */
    for (int i = 0; i < C->g.n; i++)
        dk_obj_rect(C->cfg, &C->ob[k], i, 0.0, 0.0, 0.0, 1, &C->tgt[i]);
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
            dk_obj_rect(cfg, o, i, 0.0, 0.0, 0.0, 0, &orr);
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
            dk_obj_rect(C->cfg, &C->ob[k], i, 0.0, 0.0, 0.0, 0, &orr);
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

static PyObject *n_decide_eval(PyObject *self, PyObject *args)
{
    PyObject *ot, *oe, *oh, *om, *oc, *othr, *opair, *oobj;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &ot, &oe, &oh, &om, &oc, &othr, &opair, &oobj))
        return NULL;
    View vt, ve, vh, vm, vc, vthr, vp, vo;
    vt.held = ve.held = vh.held = vm.held = vc.held = vthr.held = vp.held = vo.held = 0;
    PyObject *res = NULL;
    double *buf = NULL;
    int *ibuf = NULL;
    DkEgo *eg = NULL;
    DkObj *ob = NULL;
    DkRect *rects = NULL;
    if (view_get(ot, &vt, 0, 'd', 1, "decide_eval t") < 0 || view_get(oe, &ve, 0, 'd', 2, "decide_eval ego") < 0 ||
        view_get(oh, &vh, 0, 'd', 2, "decide_eval ohdr") < 0 || view_get(om, &vm, 0, 'd', 2, "decide_eval modes") < 0 ||
        view_get(oc, &vc, 0, 'd', 1, "decide_eval cfg") < 0 || view_get(othr, &vthr, 0, 'd', 1, "decide_eval thr") < 0 ||
        view_get(opair, &vp, 1, 'd', 2, "decide_eval out_pair") < 0 || view_get(oobj, &vo, 1, 'd', 2, "decide_eval out_obj") < 0)
        goto done;
    int N = (int)vt.b.shape[0], E = (int)ve.b.shape[0], M = (int)vh.b.shape[0], K = (int)vm.b.shape[0];
    const double *t = DATA(vt), *EG = DATA(ve), *OH = DATA(vh), *MD = DATA(vm), *cfg = DATA(vc), *thr = DATA(vthr);
    double *PO = DATA(vp), *OO = DATA(vo);
    if (N < 2 || N > DK_MAXN || E < 1 || vc.b.shape[0] < DC_NC || ve.b.shape[1] != 1 + 3 * N ||
        vh.b.shape[1] != DK_OH || vm.b.shape[1] != 2 + 5 * N || vp.b.shape[0] != (Py_ssize_t)K * E ||
        vp.b.shape[1] != DK_PF || vo.b.shape[0] != M || vo.b.shape[1] != DK_OF) {
        PyErr_SetString(PyExc_ValueError, "decide_eval: arrays of the documented shapes are needed");
        goto done;
    }
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
    int nsets = (int)(vthr.b.shape[0] / 24);
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
        const double *hd = OH + (size_t)m * DK_OH;
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
            /* the heading per grid interval: of the motion where it moves, else the stated heading */
            double offb_x = 0.0, offb_y = 0.0;           /* the reference point to the centre, in the body frame */
            for (int i = 0; i < n; i++) {
                double tau = t[0] + i * dt, f;
                int i0;
                dk_interp_idx(&g, tau, &i0, &f);
                double x = X[i0] + f * (X[i0 + 1] - X[i0]), y = Y[i0] + f * (Y[i0 + 1] - Y[i0]);
                double dx = X[i0 + 1] - X[i0], dy = Y[i0 + 1] - Y[i0];
                double h = (dx * dx + dy * dy > 0.09 * (t[i0 + 1] - t[i0]) * (t[i0 + 1] - t[i0]) * 4.0) ? atan2(dy, dx)
                           : (isfinite(hd[6]) ? hd[6] : hprev);
                hprev = h;
                double c = cos(h), s = sin(h);
                if (i == 0 && cfg[DC_REF] > 0.5) {
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
            dk_obj_rect(cfg, &ob[top[m]], i, 0.0, 0.0, 0.0, 0, r);
            r->hl += cfg[DC_MLONG];
            r->hw += cfg[DC_MLAT];
        }
    double along = cfg[DC_ALONG], alat = cfg[DC_ALAT], amax = cfg[DC_AFAC] * (along > alat ? along : alat);
    int ndirs = (int)cfg[DC_NDIRS];
    double tmax = (nh - 1) * dt;
    for (int m = 0; m < M; m++) {
        const double *hd = OH + (size_t)m * DK_OH;
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
                    double tb = dk_latest(&C, j, k, along, 0.0, tmax);
                    double tl = dk_latest(&C, j, k, 0.0, alat, tmax), tr = dk_latest(&C, j, k, 0.0, -alat, tmax);
                    double ts = tl > tr ? tl : tr;
                    pp[DP_TTB] = tb < 0.0 ? 0.0 : tb;
                    pp[DP_TTS] = ts < 0.0 ? 0.0 : ts;
                    double latest = tb > ts ? tb : ts;
                    pp[DP_TM] = (latest < 0.0 ? 0.0 : latest) - cfg[DC_TREACT];
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
    res = Py_BuildValue("(ii)", nh, n);
done:
    PyMem_Free(buf);
    PyMem_Free(ibuf);
    PyMem_Free(eg);
    PyMem_Free(ob);
    PyMem_Free(rects);
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
