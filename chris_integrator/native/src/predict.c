/* The `predict` part of chris-integrator's native core: where every object around the car will be, as several
 * futures with probabilities (chris_integrator/predict; the design is in that package's __init__.py).
 *
 * Included by chris_native.c (one translation unit, one module); it uses that file's View / dview_get /
 * view_drop / wrap_pi and nothing of the radar part. Everything numerical of the predictor is here and only
 * here: the road's curvilinear frame, the lane assignment, the manoeuvre likelihoods and the recursive intent
 * filter, the trajectory generators with their covariance, car following, the joint reweighting of interacting
 * objects, the ego car's futures, and the scoring of a past prediction against what happened.
 *
 *   predict_cfg()                      {name: default} of every constant (PCFG below): the one list of them
 *   predict_names()                    (manoeuvre names by code, lane names, intent slot names)
 *   predict_frame(...)                 one frame: all objects, into given arrays
 *   predict_score(...)                 one matured prediction against the realised positions
 *   predict_futs(...)                  the futures as the message carries them (lists of dicts, rounded)
 *
 * Frame: the car frame of the frame's time, held fixed on the ground: x forward, y LEFT, metres.
 *
 * SPDX-License-Identifier: MIT
 */
#include <stdlib.h>

/* ====================================================================================================
 * Constants: one list (name, default). Python reads it (predict_cfg) and hands back an array in this order.
 * ==================================================================================================== */
#define PCFG(X)                                                                                                     \
    X(step_s, 0.25)             /* internal integration step of a trajectory (s) */                                 \
    X(modes_max, 6.0)           /* futures per object at most (<= 8) */                                             \
    X(mode_p_min, 0.02)         /* a manoeuvre below this prior probability gets no trajectory */                   \
    X(interact, 1.0)            /* 0: no interaction at all (no car following, no joint reweighting) */             \
    X(joint, 1.0)               /* 0: no joint reweighting (car following stays) */                                 \
    X(clique_max, 8.0)          /* agents of one exact joint search at most; larger groups: belief propagation */   \
    X(joint_nodes, 20000.0)     /* search nodes of one group at most; beyond: belief propagation */                 \
    X(joint_prune, 1e-4)        /* a partial combination below this share of the best complete one is cut */        \
    X(conflict_gamma, 0.97)     /* how surely two road users avoid a combination in which they collide */           \
    X(conflict_headway_s, 0.7)  /* the gap a road user keeps to the one ahead of it, as a time */                   \
    X(row_wrong, 0.3)           /* weight of "the one with priority gives way" against "the other one does" */      \
    X(path_back_m, 80.0)        /* the road's centre line is extended this far behind the car */                    \
    X(path_ahead_m, 320.0)      /* ... and this far ahead */                                                        \
    X(evid_tau_s, 0.4)          /* a track's frames are one observation per this time (their errors are shared) */  \
    X(lane_switch_hz, 0.25)     /* lane filter: rate of an unseen change of lane */                                 \
    X(lane_floor, 0.01)         /* lane filter: what a lane kind keeps whatever the evidence */                     \
    X(motion_tau_s, 0.15)       /* the radar's standing / moving verdict takes hold of the manoeuvres in this time */ \
    X(motion_v_sd, 2.5)         /* a speed known no better than this says nothing about standing or moving ... */   \
    X(motion_ps_band, 0.15)     /* ... while the radar's p_stationary is within this of one half */                 \
    X(cross_deg, 55.0)          /* heading against the road from which a road user crosses it */                    \
    X(keep_sd_m, 0.35)          /* lane keeping: 1 sigma of the offset from the lane's centre */                    \
    X(keep_rate_sd, 0.25)       /* lane keeping: 1 sigma of the lateral speed (m/s) */                              \
    X(keep_return_s, 3.0)       /* lane keeping: an offset is driven out in this time */                            \
    X(lc_rate, 1.0)             /* lane change: typical lateral speed (m/s) */                                      \
    X(lc_rate_sd, 0.55)                                                                                             \
    X(lc_off_m, 1.5)            /* lane change: offset towards the target while it lasts (anywhere on the way) */         \
    X(lc_off_sd, 1.3)                                                                                               \
    X(lc_dur_s, 4.5)            /* lane change: its duration over a full lane width */                              \
    X(lc_onset_sd_s, 1.0)       /* execution: when it really starts */                                              \
    X(drift_rate, 0.3)          /* oncoming drift: lateral speed towards the car's lane (m/s) */                    \
    X(drift_rate_sd, 0.2)                                                                                           \
    X(acc_sd, 0.7)              /* longitudinal acceleration: 1 sigma around what a manoeuvre asks for */           \
    X(acc_meas_sd, 1.0)         /* ... of the tracker's acceleration itself */                                      \
    X(yaw_sd, 0.08)             /* yaw rate against the lane's (rad/s): 1 sigma while keeping the lane */           \
    X(turn_lat_acc, 2.2)        /* a turn is driven at this lateral acceleration (m/s2) */                          \
    X(turn_r_right_m, 7.0)      /* corner radii of a turn at a junction */                                          \
    X(turn_r_left_m, 11.0)                                                                                          \
    X(turn_reach_m, 120.0)      /* a junction counts from this distance */                                          \
    X(uturn_r_m, 5.5)                                                                                               \
    X(uturn_v_max, 8.0)                                                                                             \
    X(idm_t_s, 1.4)             /* car following: time gap, standstill gap, acceleration, comfortable braking */    \
    X(idm_s0_m, 2.0)                                                                                                \
    X(idm_a, 1.5)                                                                                                   \
    X(idm_b, 2.0)                                                                                                   \
    X(brake_max, 8.0)                                                                                               \
    X(lead_reach_m, 150.0)                                                                                          \
    X(acc_tau_s, 2.0)           /* an observed acceleration persists this long */                                   \
    X(acc_noise, 0.16)          /* how much a road user's acceleration wanders: share of what its kind uses */      \
    X(follow_v_tau_s, 4.0)      /* following: the speed's uncertainty relaxes to the leader's in this time */       \
    X(lat_wn, 1.05)             /* lateral control of a path follower: natural frequency (rad/s), damping */        \
    X(lat_zeta, 0.7)                                                                                                \
    X(pull_onset_s, 1.5)        /* pulling out: mean time until it moves, and its spread (execution) */             \
    X(pull_onset_sd_s, 1.5)                                                                                         \
    X(pull_acc, 1.5)                                                                                                \
    X(pull_v, 8.0)                                                                                                  \
    X(ped_v, 1.4)               /* walking speed */                                                                 \
    X(ped_v_sd, 0.5)                                                                                                \
    X(ped_cross_onset_s, 1.0)   /* about to cross: mean time until the first step, and its spread */                \
    X(ped_cross_onset_sd_s, 1.2)                                                                                    \
    X(ped_kerb_m, 1.5)          /* "at the kerb": within this of the road's edge */                                 \
    X(ped_lat_sd, 0.25)         /* pedestrians wander: lateral speed noise (m/s per sqrt s) */                      \
    X(bike_swerve_m, 1.2)                                                                                           \
    X(bike_swerve_s, 2.0)                                                                                           \
    X(bike_obstacle_m, 30.0)    /* a standing thing this near ahead of a cyclist makes it swerve */                 \
    X(free_floor, 0.03)         /* what the free (kinematic) future always keeps as a prior */                      \
    X(free_noctx, 0.35)         /* ... and keeps where the road is not known */                                     \
    X(free_loglik, -4.5)        /* the free future explains anything this well (3 sigma) */                       \
    X(stand_drift, 0.05)        /* a standing object's place may wander (m/s) */                                    \
    X(stop_sd_m, 1.5)           /* where exactly a stop ends */                                                     \
    X(path_kappa_sd, 0.003)     /* beyond the known centre line the road may bend: 1 sigma of its curvature (1/m) */ \
    X(go_onset_s, 2.5)          /* a vehicle standing in a lane moves off: mean time until then, and its spread */  \
    X(go_onset_sd_s, 2.5)                                                                                           \
    X(ego_len, 4.5)                                                                                                 \
    X(ego_wid, 1.8)                                                                                                 \
    X(ego_centre_x, -0.6)       /* the car's centre in its own frame (the origin is the camera) */                  \
    X(ego_path_p, 0.8)          /* the car follows the road's path with this prior when one is known */             \
    X(right_hand, 1.0)          /* traffic keeps right (-1: left) */

enum {
#define X(n, d) PC_##n,
    PCFG(X)
#undef X
    PC_N
};
static const char *const PCFG_NAME[] = {
#define X(n, d) #n,
    PCFG(X)
#undef X
};
static const double PCFG_DEF[] = {
#define X(n, d) d,
    PCFG(X)
#undef X
};

/* manoeuvre slots of the intent filter */
enum { PM_KEEP, PM_FOLLOW, PM_LCL, PM_LCR, PM_TURNL, PM_TURNR, PM_STOP, PM_PULL, PM_STAND, PM_UTURN, PM_DRIFT, PM_CROSS, PM_FREE,
       PM_STARTX, PM_SWERVE, P_NM };
/* names a future is published under (a slot's name depends on who does it and where) */
enum { PN_KEEP, PN_FOLLOW, PN_LCL, PN_LCR, PN_TURNL, PN_TURNR, PN_STOP, PN_PULL, PN_PARKED, PN_UTURN, PN_ONC_DRIFT, PN_CROSS,
       PN_FREE, PN_STARTX, PN_SWERVE, PN_ONC_KEEP, PN_CUT_IN, PN_CUT_OUT, PN_WALK, PN_WAIT, PN_RIDE, PN_STOPPED, PN_STAND,
       PN_CROSSING, PN_OVERTAKE, PN_EGO_PATH, PN_EGO_FOLLOW, PN_EGO_CTRA, PN_MOVE_OFF, P_NNAME };
static const char *const PNAME[P_NNAME] = {
    "keep", "follow", "lc_left", "lc_right", "turn_left", "turn_right", "stop", "pull_out", "parked", "u_turn",
    "onc_drift", "cross", "free", "start_cross", "swerve", "onc_keep", "cut_in", "cut_out", "walk", "wait", "ride",
    "stopped", "stand", "crossing", "onc_overtake", "path", "follow", "ctra", "move_off"};
static const char *const PSLOT[P_NM] = {"keep", "follow", "lc_left", "lc_right", "turn_left", "turn_right", "stop",
                                        "pull_out", "stand", "u_turn", "drift", "cross", "free", "start_cross",
                                        "swerve"};
enum { PL_EGO, PL_LEFT, PL_RIGHT, PL_ONC, PL_CROSS, PL_PARK, PL_OFF, P_NLANE };
static const char *const PLANE[P_NLANE] = {"ego", "left", "right", "onc", "cross", "park", "off"};

/* object row */
enum { PO_X, PO_Y, PO_VX, PO_VY, PO_AX, PO_AY, PO_SX, PO_SY, PO_SVX, PO_SVY, PO_KIND, PO_PSTAT, PO_LEN, PO_WID, PO_HDG, PO_HDGSD,
       PO_EXIST, PO_SEEN, PO_IND, PO_BRAKE, PO_PARKED, PO_ID, PO_VIS, PO_MOTION, PO_WITH, P_NO };
/* an object's kept state */
enum { PS_INIT, PS_T, PS_LANE, PS_INT = PS_LANE + P_NLANE, PS_D = PS_INT + P_NM, PS_DR, PS_PDD, PS_PDR, PS_PRR, PS_V, PS_A, PS_PVV,
       PS_PVA, PS_PAA, PS_TSTAND, PS_TMOVE, PS_LMAP, PS_TKERB, PS_TOP, PS_DIR, PS_TLANE, PS_GSTAND, PS_PSL, P_NS = 48 };
/* per object diagnostics (`feat`) */
enum { PF_S, PF_D, PF_SD, PF_DC, PF_DIR, PF_LANE, PF_LEAD, PF_GAP, PF_E, PF_R, PF_A, PF_PSI, PF_STOP, PF_JDIST, PF_KERB, PF_OM, P_NF };
/* the road */
enum { PR_W, PR_Q, PR_KERBL, PR_KERBR, PR_VLIM, PR_SPARE1, PR_SPARE2, PR_SPARE3, P_NR };
enum { PB_LO, PB_HI, PB_TYPE, PB_P, PB_DIR, PB_SPARE, P_NB };
enum { PA_I0, PA_N, PA_SJ, PA_SIDE, PA_PRIO, PA_ENTRY, PA_EXIT, PA_CTRL, P_NA };
enum { PE_V, PE_W, PE_A, PE_SIG, PE_PATH, PE_SPARE1, PE_SPARE2, PE_SPARE3, P_NE };
enum { PK_ALON, PK_ALAT, PK_VMAX, PK_VSTILL, PK_WID, PK_LEN, P_NK };

#define P_KMAX 8
#define P_EMODES 3
#define P_NT_MAX 64
#define P_STEPS_MAX 128
#define P_PTS 360
#define P_AGENTS_MAX 64
#define P_BIG 1e30

static inline double p_sq(double v) { return v * v; }
static inline double p_clip(double v, double lo, double hi) { return v < lo ? lo : v > hi ? hi : v; }
static inline double p_phi(double z) { return 0.5 * erfc(-z * 0.7071067811865476); }
/* a Gaussian feature: its density against a reference spread (1 at the mean when sd == ref) */
static inline double p_logn(double z, double mu, double sd, double ref)
{
    double u = (z - mu) / sd;
    return log(ref / sd) - 0.5 * u * u;
}

/* ====================================================================================================
 * Paths: polylines with arc length and a tangent angle at every vertex
 * ==================================================================================================== */
typedef struct {
    int n;
    double x[P_PTS], y[P_PTS], s[P_PTS], th[P_PTS];
} PPath;

static void pp_add(PPath *p, double x, double y)
{
    if (p->n >= P_PTS)
        return;
    if (p->n > 0 && hypot(x - p->x[p->n - 1], y - p->y[p->n - 1]) < 0.3)
        return;
    p->x[p->n] = x;
    p->y[p->n] = y;
    p->n++;
}

/* arc lengths (s0 at the first vertex) and vertex tangents (the mean of the two segments', unwrapped) */
static void pp_finish(PPath *p, double s0)
{
    int n = p->n;
    if (n < 2) {
        if (n == 0) {
            p->x[0] = p->y[0] = 0.0;
        }
        p->x[1] = p->x[0] + 1.0;
        p->y[1] = p->y[0];
        p->n = n = 2;
    }
    p->s[0] = s0;
    double prev = 0.0;
    for (int i = 0; i + 1 < n; i++) {
        double dx = p->x[i + 1] - p->x[i], dy = p->y[i + 1] - p->y[i];
        p->s[i + 1] = p->s[i] + hypot(dx, dy);
        double h = atan2(dy, dx);
        if (i == 0) {
            p->th[0] = h;
        } else {
            h = prev + wrap_pi(h - prev);
            p->th[i] = 0.5 * (prev + h);
        }
        prev = h;
    }
    p->th[n - 1] = prev;
}

static int pp_seg(const PPath *p, double s)
{
    int lo = 0, hi = p->n - 2;
    if (s <= p->s[0])
        return 0;
    if (s >= p->s[p->n - 1])
        return hi;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (p->s[mid] <= s)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

/* the point at arc length s (beyond the ends: straight on), its tangent angle and curvature */
static void pp_eval(const PPath *p, double s, double *x, double *y, double *th, double *kap)
{
    int n = p->n;
    if (s <= p->s[0]) {
        double c = cos(p->th[0]), sn = sin(p->th[0]), u = s - p->s[0];
        *x = p->x[0] + u * c;
        *y = p->y[0] + u * sn;
        *th = p->th[0];
        *kap = 0.0;
        return;
    }
    if (s >= p->s[n - 1]) {
        double c = cos(p->th[n - 1]), sn = sin(p->th[n - 1]), u = s - p->s[n - 1];
        *x = p->x[n - 1] + u * c;
        *y = p->y[n - 1] + u * sn;
        *th = p->th[n - 1];
        *kap = 0.0;
        return;
    }
    int i = pp_seg(p, s);
    double ds = p->s[i + 1] - p->s[i], u = (s - p->s[i]) / ds;
    *x = p->x[i] + u * (p->x[i + 1] - p->x[i]);
    *y = p->y[i] + u * (p->y[i + 1] - p->y[i]);
    *th = p->th[i] + u * (p->th[i + 1] - p->th[i]);
    *kap = (p->th[i + 1] - p->th[i]) / ds;
}

/* the nearest point of the path to (x, y): its arc length and the offset to the LEFT of the path */
static void pp_project(const PPath *p, double x, double y, double *s, double *d)
{
    double best = P_BIG;
    int n = p->n;
    *s = p->s[0];
    *d = 0.0;
    for (int i = 0; i + 1 < n; i++) {
        double dx = p->x[i + 1] - p->x[i], dy = p->y[i + 1] - p->y[i];
        double L2 = dx * dx + dy * dy, rx = x - p->x[i], ry = y - p->y[i];
        double t = (rx * dx + ry * dy) / L2;
        if (t < 0.0 && i > 0)
            t = 0.0;
        if (t > 1.0 && i + 2 < n)
            t = 1.0;
        double ex = rx - t * dx, ey = ry - t * dy, e2 = ex * ex + ey * ey;
        if (e2 < best) {
            double L = sqrt(L2);
            best = e2;
            *s = p->s[i] + t * L;
            /* the offset is measured across the path's own direction at that point: by the vertex tangents */
            double u = t < 0.0 ? 0.0 : t > 1.0 ? 1.0 : t;
            double th = p->th[i] + u * (p->th[i + 1] - p->th[i]);
            double px = p->x[i] + t * dx, py = p->y[i] + t * dy;
            *d = -(x - px) * sin(th) + (y - py) * cos(th);
            if (t >= 0.0 && t <= 1.0)
                *d = (dx * ry - dy * rx) / L;
        }
    }
}

/* the point at arc length s, offset d to the left */
static void pp_point(const PPath *p, double s, double d, double *x, double *y, double *th, double *kap)
{
    pp_eval(p, s, x, y, th, kap);
    *x -= d * sin(*th);
    *y += d * cos(*th);
}

/* the road's centre line: the given points, extended behind the car along the arc it drove and ahead with the
 * end's curvature dying away; without points, the arc of the car's own speed and yaw rate. s = 0 beside the car */
static void pp_base(PPath *b, const double *c, int nc, double v, double w, double back, double ahead)
{
    PPath t;
    t.n = 0;
    double k0 = v > 1.5 ? p_clip(w / v, -0.08, 0.08) : 0.0;
    double x0 = 0.0, y0 = 0.0, h0 = 0.0;
    if (nc >= 2) {
        x0 = c[0];
        y0 = c[1];
        h0 = atan2(c[3] - c[1], c[2] - c[0]);
    }
    /* behind: back from the first point along the arc the car came on (the list is filled in reverse) */
    int nb = (int)(back / 5.0);
    if (nb > 40)
        nb = 40;
    double bx[40], by[40];
    double x = x0, y = y0, h = h0;
    for (int i = 0; i < nb; i++) {
        h -= k0 * 5.0 * exp(-(i * 5.0) / 40.0);
        x -= 5.0 * cos(h);
        y -= 5.0 * sin(h);
        bx[i] = x;
        by[i] = y;
    }
    for (int i = nb - 1; i >= 0; i--)
        pp_add(&t, bx[i], by[i]);
    double kend = k0;
    if (nc >= 2) {
        for (int i = 0; i < nc && t.n < P_PTS - 80; i++)
            pp_add(&t, c[2 * i], c[2 * i + 1]);
        /* the end's direction and curvature from its last stretch */
        int n = t.n;
        x = t.x[n - 1];
        y = t.y[n - 1];
        h = atan2(t.y[n - 1] - t.y[n - 2], t.x[n - 1] - t.x[n - 2]);
        kend = 0.0;
        int j = n - 2;
        double len = hypot(t.x[n - 1] - t.x[j], t.y[n - 1] - t.y[j]);
        while (j > nb && len < 15.0) {
            j--;
            len += hypot(t.x[j + 1] - t.x[j], t.y[j + 1] - t.y[j]);
        }
        if (j < n - 2 && len > 3.0) {
            double hj = atan2(t.y[j + 1] - t.y[j], t.x[j + 1] - t.x[j]);
            kend = p_clip(wrap_pi(h - hj) / len, -0.05, 0.05);
        }
    } else {
        pp_add(&t, 0.0, 0.0);
        x = y = 0.0;
        h = 0.0;
    }
    double tau = nc >= 2 ? 40.0 : (v > 3.0 ? v : 3.0) * 4.0, run = 0.0;
    double far = hypot(x, y);
    while (far + run < ahead && t.n < P_PTS) {
        h += kend * 5.0 * exp(-run / tau);
        x += 5.0 * cos(h);
        y += 5.0 * sin(h);
        run += 5.0;
        pp_add(&t, x, y);
    }
    pp_finish(&t, 0.0);
    double s0, d0;
    pp_project(&t, 0.0, 0.0, &s0, &d0);
    *b = t;
    for (int i = 0; i < b->n; i++)
        b->s[i] -= s0;
}

/* a quadratic corner between two directed points (the control point where their tangents meet) */
static void pp_corner(PPath *p, double ax, double ay, double ah, double bx, double by, double bh)
{
    double ca = cos(ah), sa = sin(ah), cb = cos(bh), sb = sin(bh);
    double det = ca * sb - sa * cb, cx, cy;
    if (fabs(det) > 0.2) {
        double t = ((bx - ax) * sb - (by - ay) * cb) / det;
        double chord = hypot(bx - ax, by - ay);
        t = p_clip(t, 0.1 * chord, 2.0 * chord);
        cx = ax + t * ca;
        cy = ay + t * sa;
    } else {
        cx = 0.5 * (ax + bx);
        cy = 0.5 * (ay + by);
    }
    for (int k = 1; k <= 7; k++) {
        double u = k / 8.0, v = 1.0 - u;
        pp_add(p, v * v * ax + 2.0 * u * v * cx + u * u * bx, v * v * ay + 2.0 * u * v * cy + u * u * by);
    }
}

/* a stretch of a path walked from s0 to s1 (either way) at offset d (left of the path's own direction) */
static void pp_walk(PPath *o, const PPath *p, double s0, double s1, double d, double step)
{
    double len = fabs(s1 - s0);
    int k = (int)(len / step) + 1;
    for (int i = 0; i <= k; i++) {
        double x, y, th, kap;
        pp_point(p, s0 + (s1 - s0) * i / k, d, &x, &y, &th, &kap);
        pp_add(o, x, y);
    }
}

/* ====================================================================================================
 * One trajectory along a path
 * ==================================================================================================== */
typedef struct {
    const PPath *path;
    int dir;                      /* +1: along the path's arc length, -1: against it */
    double s0, d0, dr0;           /* start: arc length, offset and offset rate (left of the PATH's direction) */
    double d1, t_on, t_dur;       /* the offset is taken to d1, from t_on on, within t_dur (min-jerk) */
    double drift;                 /* or: it keeps changing at this rate (t_dur <= 0) */
    double v0, a0;                /* speed and acceleration now */
    int go;                       /* 1: accelerates to v_des (a start); 0: its acceleration dies away */
    double v_des, a_go;
    double t_start;               /* it stands until then */
    double s_stop;                /* path arc length at which it stops (NaN: none) */
    const double *lead_s, *lead_v;/* the one it follows: front-to-rear position along the path and speed per step */
    double lead_len, len;
    double a_lat;                 /* lateral acceleration it corners with */
    double v_max;
    /* covariance */
    double pss, pdd, psd, pvv, paa, pdr, prr;
    double sig_a, tau_a, k_v;     /* acceleration noise and persistence; pull of the speed to its reference */
    double q_lat, kp, kd;         /* lateral control (kp = 0: none, a free offset) */
    double onset_sd;              /* execution: spread of the time the lateral move really begins */
    double onset_lon_sd;          /* ... and of the time it starts to move at all */
    double stop_sd;
    double extra_lat;             /* execution: what is not known of the path itself (a turn's line), m */
    double known_lo, known_hi;    /* the path is known between these arc lengths; beyond, it is a guess */
    double kappa_sd;
} PRoll;

static void roll_defaults(PRoll *r, const double *cfg, const double *kin)
{
    memset(r, 0, sizeof *r);
    r->dir = 1;
    r->s_stop = NAN;
    r->known_lo = -P_BIG;
    r->known_hi = P_BIG;
    r->kappa_sd = cfg[PC_path_kappa_sd];
    r->a_lat = cfg[PC_turn_lat_acc] < kin[PK_ALAT] * 1.3 ? cfg[PC_turn_lat_acc] : kin[PK_ALAT] * 1.3;
    r->v_max = kin[PK_VMAX];
    r->sig_a = cfg[PC_acc_noise] * kin[PK_ALON];
    r->tau_a = cfg[PC_acc_tau_s];
    double wn = cfg[PC_lat_wn], z = cfg[PC_lat_zeta];
    r->kp = wn * wn;
    r->kd = 2.0 * z * wn;
    r->q_lat = 2.0 * r->kd * r->kp * p_sq(cfg[PC_keep_sd_m]);
    r->stop_sd = cfg[PC_stop_sd_m];
    r->a_go = cfg[PC_idm_a];
}

/* min-jerk offset from (d0, dr0) to (d1, 0) over T: value and rate at u in [0, T] */
static void quintic(double d0, double dr0, double d1, double T, double u, double *d, double *dr)
{
    if (u <= 0.0) {
        *d = d0;
        *dr = dr0;
        return;
    }
    if (u >= T) {
        *d = d1;
        *dr = 0.0;
        return;
    }
    double D = d1 - d0, T2 = T * T, T3 = T2 * T;
    double a3 = (10.0 * D - 6.0 * dr0 * T) / T3, a4 = (-15.0 * D + 8.0 * dr0 * T) / (T3 * T),
           a5 = (6.0 * D - 3.0 * dr0 * T) / (T3 * T2);
    double u2 = u * u, u3 = u2 * u;
    *d = d0 + dr0 * u + a3 * u3 + a4 * u3 * u + a5 * u3 * u2;
    *dr = dr0 + 3.0 * a3 * u2 + 4.0 * a4 * u3 + 5.0 * a5 * u2 * u2;
}

/* The trajectory at the output times: out[k] = (x, y, sx, sy, rxy, v). sprof / vprof (may be NULL): arc length and
 * speed at every internal step (nstep + 1 values). The mean is integrated with the step h; the covariance of
 * (s, v, a) along and of (d, d rate) across the path is propagated with it and turned into the frame at each
 * output time, with what is not known of the execution added. */
static void roll_out(const PRoll *r, const double *cfg, const double *times, int nt, double h, int nstep, double *out,
                     double *sprof, double *vprof)
{
    const PPath *p = r->path;
    double s = r->s0, v = r->v0 > 0.0 ? r->v0 : 0.0, a = r->a0, t = 0.0;
    double Ps[6] = {r->pss, 0.0, 0.0, r->pvv, 0.0, r->paa};       /* ss sv sa vv va aa */
    double Pl[3] = {r->pdd, r->pdr, r->prr};                      /* dd dr rr */
    double psd = r->psd;
    double rho = exp(-h / r->tau_a), qa = p_sq(r->sig_a) * (1.0 - rho * rho);
    double fv = exp(-r->k_v * h);
    int ko = 0;
    double px = 0.0, py = 0.0, pth = 0.0, pk = 0.0, d = r->d0, dr = r->dr0;
    double prev[6] = {0, 0, 0, 0, 0, 0};
    double tprev = 0.0;
    double b = cfg[PC_idm_b], T = cfg[PC_idm_t_s], s0g = cfg[PC_idm_s0_m], amax = r->a_go, bmax = cfg[PC_brake_max];
    for (int k = 0; k <= nstep; k++) {
        /* the lateral reference */
        if (r->t_dur > 0.0) {
            quintic(r->d0, r->dr0, r->d1, r->t_dur, t - r->t_on, &d, &dr);
            if (t < r->t_on) {
                d = r->d0;
                dr = 0.0;
            }
        } else {
            d = r->d0 + r->drift * t;
            dr = r->drift;
        }
        pp_point(p, s, d, &px, &py, &pth, &pk);
        if (sprof != NULL) {
            sprof[k] = s;
            vprof[k] = v;
        }
        /* this step's covariance in the frame */
        double c = cos(pth), sn = sin(pth);
        double vss = Ps[0] + p_sq(v * r->onset_lon_sd);            /* a start that comes sooner or later */
        double vdd = Pl[0] + p_sq(dr * r->onset_sd) + p_sq(r->extra_lat) * (1.0 - exp(-t / 2.0));
        double beyond = s > r->known_hi ? s - r->known_hi : s < r->known_lo ? r->known_lo - s : 0.0;
        if (beyond > 0.0)
            vdd += p_sq(0.5 * r->kappa_sd * beyond * beyond);
        double cur[6];
        cur[0] = px;
        cur[1] = py;
        double cxx = c * c * vss - 2.0 * c * sn * psd + sn * sn * vdd;
        double cyy = sn * sn * vss + 2.0 * c * sn * psd + c * c * vdd;
        double cxy = c * sn * (vss - vdd) + (c * c - sn * sn) * psd;
        cur[2] = sqrt(cxx > 1e-6 ? cxx : 1e-6);
        cur[3] = sqrt(cyy > 1e-6 ? cyy : 1e-6);
        cur[4] = p_clip(cxy / (cur[2] * cur[3]), -0.99, 0.99);
        cur[5] = v;
        while (ko < nt && times[ko] <= t + 1e-9) {
            double u = k == 0 || t - tprev < 1e-9 ? 1.0 : (times[ko] - tprev) / (t - tprev);
            if (u < 0.0)
                u = 0.0;
            for (int q = 0; q < 6; q++)
                out[6 * ko + q] = prev[q] + u * (cur[q] - prev[q]);
            if (k == 0)
                for (int q = 0; q < 6; q++)
                    out[6 * ko + q] = cur[q];
            ko++;
        }
        memcpy(prev, cur, sizeof prev);
        tprev = t;
        if (k == nstep)
            break;
        /* the acceleration of this step */
        double acc;
        if (t < r->t_start) {
            acc = 0.0;
            v = 0.0;
        } else if (r->go) {
            double q = v / (r->v_des > 0.5 ? r->v_des : 0.5);
            acc = amax * (1.0 - q * q * q * q);
        } else {
            acc = a;
        }
        /* a bend ahead is not entered faster than it can be driven */
        if (v > 1.0) {
            for (int j = 0; j < 4; j++) {
                double la = (j == 0 ? 0.0 : j == 1 ? 0.6 : j == 2 ? 1.5 : 3.0) * v + (j == 0 ? 0.0 : 2.0);
                double qx, qy, qth, qk;
                pp_eval(p, s + r->dir * la, &qx, &qy, &qth, &qk);
                double ak = fabs(qk);
                if (ak > 1e-3) {
                    double vc = sqrt(r->a_lat / ak);
                    if (vc < v) {
                        double need = la > 1.0 ? (vc * vc - v * v) / (2.0 * la) : -b;
                        if (need < -0.3 && need < acc)
                            acc = need;
                    }
                }
            }
        }
        /* the one ahead, and the place to stop: the car-following law's interaction term against each */
        for (int which = 0; which < 2; which++) {
            double gap, dv;
            if (which == 0) {
                if (r->lead_s == NULL)
                    continue;
                gap = r->dir * (r->lead_s[k] - s) - 0.5 * (r->lead_len + r->len);
                dv = v - r->dir * r->lead_v[k];
            } else {
                if (!(r->s_stop == r->s_stop))
                    continue;
                gap = r->dir * (r->s_stop - s);
                dv = v;
            }
            double ai;
            if (which == 0) {
                if (gap < 0.3)
                    gap = 0.3;
                double ss = s0g + v * T + v * dv / (2.0 * sqrt(amax * b));
                if (ss < s0g)
                    ss = s0g;
                ai = amax * (1.0 - p_sq(ss / gap));
            } else {
                /* a stop is braked for from where a comfortable braking still reaches it, with what it needs */
                double need = -v * v / (2.0 * (gap > 0.5 ? gap : 0.5));
                if (gap <= 0.2)
                    ai = v > 0.0 ? -0.5 * bmax : 0.0;
                else if (need <= -0.6 * b)
                    ai = need;
                else if (v < 1.5 && gap < 4.0)
                    ai = need - 0.2;                             /* creeping up to the line */
                else
                    continue;
            }
            if (ai < acc)
                acc = ai;
        }
        if (acc < -bmax)
            acc = -bmax;
        if (acc > 1.2 * amax + 2.0)
            acc = 1.2 * amax + 2.0;
        double vn = v + acc * h;
        if (vn < 0.0)
            vn = 0.0;
        if (vn > r->v_max)
            vn = r->v_max;
        double dist = 0.5 * (v + vn) * h;
        double sc = 1.0 - pk * d;                                  /* an offset path is longer or shorter in a bend */
        if (sc < 0.3)
            sc = 0.3;
        double along = v > 0.5 && fabs(dr) < v ? sqrt(1.0 - p_sq(dr / v)) : 1.0;
        s += r->dir * dist * along / sc;
        v = vn;
        if (!r->go)
            a *= rho;
        /* covariance: (s, v, a) */
        {
            double ss = Ps[0], sv = Ps[1], sa = Ps[2], vv = Ps[3], va = Ps[4], aa = Ps[5];
            double fvv = fv, stop = (v < 0.3 && (r->s_stop == r->s_stop || t >= r->t_start)) ? 1.0 : 0.0;
            if (t < r->t_start)
                fvv = 1.0, stop = 0.0;
            double h2 = 0.5 * h * h;
            double nss = ss + 2.0 * h * sv + 2.0 * h2 * sa + h * h * vv + 2.0 * h * h2 * va + h2 * h2 * aa;
            double nsv = fvv * (sv + h * vv + h2 * va) + h * (sa + h * va + h2 * aa);
            double nsa = rho * (sa + h * va + h2 * aa);
            double nvv = fvv * fvv * vv + 2.0 * fvv * h * va + h * h * aa;
            double nva = rho * (fvv * va + h * aa);
            double naa = rho * rho * aa + qa;
            if (stop > 0.0) {                                    /* it stands: what was not known of its speed is gone */
                nsv *= 0.5;
                nvv *= 0.25;
                nva *= 0.5;
                nsa *= 0.5;
                naa *= 0.5;
                if (r->s_stop == r->s_stop && nss > p_sq(r->stop_sd) + r->pss)
                    nss = 0.7 * nss + 0.3 * (p_sq(r->stop_sd) + r->pss);
            }
            Ps[0] = nss, Ps[1] = nsv, Ps[2] = nsa, Ps[3] = nvv, Ps[4] = nva, Ps[5] = naa;
        }
        /* (d, d rate): a controlled offset is pulled back, a free one is not */
        {
            double dd = Pl[0], dq = Pl[1], rr = Pl[2];
            double f10 = -r->kp * h, f11 = 1.0 - r->kd * h;
            double ndd = dd + 2.0 * h * dq + h * h * rr;
            double ndq = f10 * dd + (f11 + f10 * h) * dq + f11 * h * rr;
            double nrr = f10 * f10 * dd + 2.0 * f10 * f11 * dq + f11 * f11 * rr + r->q_lat * h;
            Pl[0] = ndd, Pl[1] = ndq, Pl[2] = nrr;
        }
        psd *= exp(-h / 2.0);
        t += h;
    }
    for (; ko < nt; ko++)                                        /* (times beyond the last step: held) */
        memcpy(out + 6 * ko, prev, sizeof prev);
}

/* A free future: the tracker's own motion carried on (velocity, an acceleration that dies away), no road. The
 * covariance along and across the heading with the kind's acceleration noise. */
static void roll_free(const double *o, const double *kin, const double *cfg, const double *times, int nt, double *out,
                      int stand)
{
    double vx = o[PO_VX], vy = o[PO_VY], ax = o[PO_AX], ay = o[PO_AY];
    double sp = hypot(vx, vy), tau = cfg[PC_acc_tau_s];
    double hx = sp > 0.3 ? vx / sp : 1.0, hy = sp > 0.3 ? vy / sp : 0.0;
    double qlon = 0.6 * kin[PK_ALON], qlat = 0.6 * kin[PK_ALAT];
    if (stand)
        vx = vy = ax = ay = sp = 0.0;
    for (int k = 0; k < nt; k++) {
        double t = times[k];
        double g = tau * t - tau * tau * (1.0 - exp(-t / tau));      /* integral of the dying acceleration, twice */
        double x = o[PO_X] + vx * t + ax * g, y = o[PO_Y] + vy * t + ay * g;
        double vxt = vx + ax * tau * (1.0 - exp(-t / tau)), vyt = vy + ay * tau * (1.0 - exp(-t / tau));
        double v = hypot(vxt, vyt);
        if (v > kin[PK_VMAX]) {                                      /* not faster than its kind drives */
            double f = kin[PK_VMAX] / v;
            x = o[PO_X] + (x - o[PO_X]) * f;
            y = o[PO_Y] + (y - o[PO_Y]) * f;
            v = kin[PK_VMAX];
        }
        double cxx, cyy, cxy;
        if (stand) {
            double w = p_sq(cfg[PC_stand_drift] * t);
            cxx = p_sq(o[PO_SX]) + w;
            cyy = p_sq(o[PO_SY]) + w;
            cxy = 0.0;
        } else {
            /* an acceleration that wanders with spread q and persistence tau: position variance q^2 g(t)^2-like */
            double gl = p_sq(qlon * g), gc = p_sq(qlat * g);
            cxx = p_sq(o[PO_SX]) + p_sq(o[PO_SVX] * t) + hx * hx * gl + hy * hy * gc;
            cyy = p_sq(o[PO_SY]) + p_sq(o[PO_SVY] * t) + hy * hy * gl + hx * hx * gc;
            cxy = hx * hy * (gl - gc);
        }
        out[6 * k] = x;
        out[6 * k + 1] = y;
        out[6 * k + 2] = sqrt(cxx);
        out[6 * k + 3] = sqrt(cyy);
        out[6 * k + 4] = p_clip(cxy / sqrt(cxx * cyy), -0.99, 0.99);
        out[6 * k + 5] = v;
    }
}

/* ====================================================================================================
 * One frame's working data
 * ==================================================================================================== */
typedef struct {
    const double *o;
    double *st;
    const double *kin;
    int grp;                        /* 0 vehicle, 1 cyclist, 2 pedestrian */
    double s, d, sd_d, sd_s, th, kap, vt, vn, sp, psi, spsi;
    double len, wid, pstat, minfo, exist;
    int lane, band, dir;
    double dc, bw;                  /* its lane's centre and width (base frame) */
    int nb[2];                      /* the band on ITS left / right (-1: none) */
    double e, r, se, sr, v, a, sa, om, som;   /* offset and rate to ITS left, speed, acceleration, yaw rate */
    int lead;                       /* the one ahead in its lane: index, -1 none, -2 the car itself */
    double gap, lead_v, a_idm;
    double stop_s, stop_p;          /* where it would stop (base arc length; on its arm for a crossing one), prior */
    int stop_why;                   /* 0 none, 1 junction, 2 queue, 3 pedestrian crossing */
    int jarm[2];                    /* the arm ahead on ITS left / right (-1: none) */
    double jdist;
    int arm, arm_in;                /* the side road it is on (-1: none); 1: heading for the junction */
    double arm_s, arm_d, arm_hd;
    double kerb;                    /* metres inside the carriageway (negative: outside) */
    int side;                       /* +1: on the left half of the road */
    double xing;                    /* distance along the road to the nearest marked crossing (inf: none) */
    int obstacle;                   /* a standing thing ahead in a cyclist's way */
    double q[P_NM], ll[P_NM];
    int nm, slot[P_KMAX], name[P_KMAX], yield[P_KMAX];
    double pri[P_KMAX], rank[P_KMAX];
    double *traj;                   /* [P_KMAX][nt][6] */
    double sprof[P_STEPS_MAX + 1], vprof[P_STEPS_MAX + 1];
    double bb[P_KMAX][4];           /* each future's bounding box, spread included */
} PObj;

typedef struct {
    const double *cfg, *road, *bands, *ameta, *xing, *times, *ego, *kin;
    int nb, na, nx, nt, nstep, n;
    double h, w, rh, known_lo, known_hi;
    PPath base;
    PPath *arm;
    double asj[32];
    PObj *obj;
    /* the car */
    int enm, ename[P_EMODES], eyield[P_EMODES], elead;
    double epri[P_EMODES], etraj[P_EMODES][P_NT_MAX][6], esprof[P_STEPS_MAX + 1], evprof[P_STEPS_MAX + 1];
    double ebb[P_EMODES][4], egap, ed0;
    double stat[8];
} PCtx;

/* Does it stand: the radar's word. Its p_stationary; its verdicts, which it holds with a hysteresis of their own
 * and which therefore say more than the number of one frame (`motion` stationary, the flag `parked`: half the
 * doubt that it stands is taken away; `motion` moving, the flag `with_traffic`: half the doubt that it moves).
 * Where the radar itself does not know (no verdict, p_stationary near one half: its prior, not evidence) and the
 * speed is known no better than a few m/s, the answer is "not known": drawn towards one half by *info, the
 * weight of the radar's word (0.1 to 1). */
static double motion_state(const double *cfg, const double *o, double *info)
{
    double ps = p_clip(o[PO_PSTAT], 0.0, 1.0);
    int verdict = 0;
    if (o[PO_PARKED] > 0.5 || o[PO_MOTION] > 0.5) {
        ps = 1.0 - 0.5 * (1.0 - ps);
        verdict = 1;
    }
    if (o[PO_MOTION] < -0.5 || o[PO_WITH] > 0.5) {
        ps = 0.5 * ps;
        verdict = 1;
    }
    double u = 1.0;
    if (!verdict) {
        double band = cfg[PC_motion_ps_band], sv = o[PO_SVX] > o[PO_SVY] ? o[PO_SVX] : o[PO_SVY];
        double u_ps = p_clip((fabs(ps - 0.5) - band) / band, 0.0, 1.0);
        double u_v = p_clip(cfg[PC_motion_v_sd] - sv, 0.0, 1.0);
        u = u_ps > u_v ? u_ps : u_v;
        if (u < 0.1)
            u = 0.1;
    }
    *info = u;
    return 0.5 + u * (ps - 0.5);
}

/* ---- lanes -------------------------------------------------------------------------------------------- */
static double band_occ(const double *b, double d, double sd)
{
    return p_phi((b[PB_HI] - d) / sd) - p_phi((b[PB_LO] - d) / sd);
}

/* the driving band (ego / left / right / oncoming) that holds offset d best; -1: none */
static int band_at(const PCtx *c, double d, int need_dir)
{
    int best = -1;
    double bp = 0.02;
    for (int b = 0; b < c->nb; b++) {
        const double *B = c->bands + P_NB * b;
        int ty = (int)B[PB_TYPE];
        if (ty > PL_ONC || (need_dir != 0 && (int)B[PB_DIR] != need_dir))
            continue;
        if (d < B[PB_LO] || d > B[PB_HI])
            continue;
        if (B[PB_P] > bp) {
            bp = B[PB_P];
            best = b;
        }
    }
    return best;
}

/* Where the object is in the road's frame, with what its position uncertainty does to that on a bend (three
 * points along the long axis of its uncertainty, each projected); what each kind of lane says to that place
 * and to the way it moves; one step of the lane filter. */
static void lane_step(PCtx *c, PObj *q, double dt, double *lane_out)
{
    const double *o = q->o, *cfg = c->cfg;
    double geo[P_NLANE] = {0, 0, 0, 0, 0, 0, 0};
    double sx = o[PO_SX], sy = o[PO_SY];
    int along_x = sx >= sy;
    double big = along_x ? sx : sy, small = along_x ? sy : sx;
    static const double W3[3] = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};
    double dm = 0.0, d2 = 0.0, res2 = 0.0, sm = 0.0, s2 = 0.0, dk[3], sk[3];
    for (int k = 0; k < 3; k++) {
        double off = (k - 1) * 1.7320508075688772 * big;
        double x = o[PO_X] + (along_x ? off : 0.0), y = o[PO_Y] + (along_x ? 0.0 : off);
        double s, d, px, py, th, kap;
        pp_project(&c->base, x, y, &s, &d);
        pp_eval(&c->base, s, &px, &py, &th, &kap);
        double comp = along_x ? cos(th) : -sin(th);            /* the short axis across the path */
        double rs = sqrt(p_sq(comp * small) + 0.0225);
        dk[k] = d;
        sk[k] = s;
        dm += W3[k] * d;
        sm += W3[k] * s;
        res2 += W3[k] * rs * rs;
        double tot = 0.0;
        for (int b = 0; b < c->nb; b++) {
            const double *B = c->bands + P_NB * b;
            double oc = B[PB_P] * band_occ(B, d, rs);
            /* the likelihood of this place for an object spread evenly over the band: occupancy per metre */
            geo[(int)B[PB_TYPE]] += W3[k] * oc / (B[PB_HI] - B[PB_LO]);
            tot += oc;
        }
        geo[PL_OFF] += W3[k] * (tot < 0.98 ? 1.0 - tot : 0.02) / 30.0;   /* "off the road": 30 m of it */
    }
    for (int k = 0; k < 3; k++) {
        d2 += W3[k] * p_sq(dk[k] - dm);
        s2 += W3[k] * p_sq(sk[k] - sm);
    }
    double px, py;
    pp_project(&c->base, o[PO_X], o[PO_Y], &q->s, &q->d);
    pp_eval(&c->base, q->s, &px, &py, &q->th, &q->kap);
    q->sd_d = sqrt(d2 + res2);
    q->sd_s = sqrt(s2 + p_sq(along_x ? sin(q->th) * small : cos(q->th) * small) + 0.01);
    double ct = cos(q->th), sn = sin(q->th);
    q->vt = o[PO_VX] * ct + o[PO_VY] * sn;
    q->vn = -o[PO_VX] * sn + o[PO_VY] * ct;
    q->sp = hypot(o[PO_VX], o[PO_VY]);
    /* its direction against the road's: from its velocity; a slow vehicle's from its body when that is known */
    double svn = sqrt(p_sq(sn * o[PO_SVX]) + p_sq(ct * o[PO_SVY]));
    int have_dir = 0, body = 0;
    if (q->sp > q->kin[PK_VSTILL] + 0.3) {
        q->psi = atan2(q->vn, q->vt);
        q->spsi = atan2(svn, q->sp);
        have_dir = 1;
    } else if (o[PO_HDG] == o[PO_HDG] && q->grp == 0) {
        q->psi = wrap_pi(o[PO_HDG] - q->th);
        q->spsi = o[PO_HDGSD] == o[PO_HDGSD] ? o[PO_HDGSD] : 0.3;
        have_dir = body = 1;                                    /* a body's axis: either way round */
    } else {
        q->psi = 0.0;
        q->spsi = 3.0;
    }
    /* the side roads: is it on one */
    q->arm = -1;
    double occ_arm = 0.0, occ_box = 0.0;
    int arm_box = -1;
    double box_s = 0.0, box_d = 0.0;
    for (int a = 0; a < c->na; a++) {
        double as, ad;
        pp_project(&c->arm[a], o[PO_X], o[PO_Y], &as, &ad);
        if (as < -c->w)
            continue;
        double sdx = sqrt(p_sq(sx * sin(c->arm[a].th[0])) + p_sq(sy * cos(c->arm[a].th[0])) + 0.04);
        double oc = p_phi((c->w - ad) / sdx) - p_phi((-c->w - ad) / sdx);
        if (as < 0.5 * c->w + 1.0) {                            /* in the junction itself: only a crossing one's arm */
            if (oc > occ_box) {
                occ_box = oc;
                arm_box = a;
                box_s = as;
                box_d = ad;
            }
            continue;
        }
        if (oc > occ_arm) {
            occ_arm = oc;
            q->arm = a;
            q->arm_s = as;
            q->arm_d = ad;
        }
    }
    if (q->arm < 0 && arm_box >= 0 && occ_box > 0.3) {
        q->arm = arm_box;
        q->arm_s = box_s;
        q->arm_d = box_d;
    }
    {
        /* no side road known: a crossing road user is on or near the road */
        double out = q->d > c->road[PR_KERBL] ? q->d - c->road[PR_KERBL] : q->d < c->road[PR_KERBR] ? c->road[PR_KERBR] - q->d : 0.0;
        geo[PL_CROSS] = (c->na > 0 ? (occ_arm > 0.05 ? occ_arm : 0.05) : 0.5 * exp(-(out > 3.0 ? out - 3.0 : 0.0) / 4.0)) / (2.0 * c->w);
    }
    double em[P_NLANE];
    double st = q->pstat, mv = 1.0 - st;
    double sdir = hypot(0.44, q->spsi), scr = hypot(0.6, q->spsi);
    double m_with = 1.0, m_onc = 1.0, m_cross = 1.0;
    if (have_dir) {
        m_with = exp(-0.5 * p_sq(wrap_pi(q->psi) / sdir));
        m_onc = exp(-0.5 * p_sq(wrap_pi(q->psi - PI) / sdir));
        double a1 = wrap_pi(q->psi - 0.5 * PI), a2 = wrap_pi(q->psi + 0.5 * PI);
        m_cross = exp(-0.5 * p_sq((fabs(a1) < fabs(a2) ? a1 : a2) / scr));
        if (body)
            m_with = m_onc = (m_with > m_onc ? m_with : m_onc);
        /* a road user that crosses is not "in" the lanes it passes over */
        if (q->arm >= 0 && !body) {
            double ah = c->arm[q->arm].th[pp_seg(&c->arm[q->arm], q->arm_s)];
            double rel = wrap_pi(atan2(o[PO_VY], o[PO_VX]) - ah);
            double al = exp(-0.5 * p_sq((fabs(rel) < 0.5 * PI ? rel : wrap_pi(rel - PI)) / sdir));
            if (al > m_cross)
                m_cross = al;
        }
    }
    if (q->grp == 2) {                                          /* a pedestrian is where it is, however it walks */
        for (int l = 0; l < P_NLANE; l++)
            em[l] = geo[l];
    } else {
        double slow = q->sp < 2.0 ? 0.3 : 0.05;
        /* standing: any lane may hold a standing vehicle; where its body's direction is known, it points along
         * the lane it stands in (across the road: it waits on a side road) */
        double s_lane = body ? 0.05 + 0.5 * m_with : 0.35, s_cross = body ? 0.05 + 0.95 * m_cross : 0.25;
        double s_park = body ? 0.2 + 0.8 * m_with : 1.0;
        em[PL_EGO] = geo[PL_EGO] * (st * s_lane + mv * m_with);
        em[PL_LEFT] = geo[PL_LEFT] * (st * s_lane + mv * m_with);
        em[PL_RIGHT] = geo[PL_RIGHT] * (st * s_lane + mv * m_with);
        em[PL_ONC] = geo[PL_ONC] * (st * s_lane + mv * m_onc);
        em[PL_CROSS] = geo[PL_CROSS] * (st * s_cross + mv * m_cross);
        em[PL_PARK] = geo[PL_PARK] * (st * s_park + mv * slow);
        em[PL_OFF] = geo[PL_OFF] * (st * 1.0 + mv * 0.3);
    }
    static const double PRIOR[P_NLANE] = {0.2, 0.12, 0.12, 0.15, 0.1, 0.16, 0.15};
    double *lp = q->st + PS_LANE, post[P_NLANE], sum = 0.0;
    int fresh = !(q->st[PS_INIT] > 0.5);
    if (o[PO_PARKED] > 0.5 && q->grp != 2) {                   /* the radar: it stands beside the path */
        em[PL_PARK] *= 1.0 + 3.0 * st;
        em[PL_OFF] *= 1.0 + 3.0 * st;
    }
    if (!fresh) {
        /* which lane it is in was concluded from how it moved; when the radar changes its mind about that (it
         * stands after all, it drives after all), that much of the conclusion is void. Not where the change is
         * one of motion itself (it came to a stop, it moves off: slow before and after): the lane holds */
        double vk = q->st[PS_V] > q->sp ? q->st[PS_V] : q->sp;
        double ch = fabs(st - q->st[PS_PSL]) * p_clip((vk - 2.0) / 3.0, 0.0, 1.0);
        if (ch > 1.0)
            ch = 1.0;
        for (int l = 0; l < P_NLANE; l++)
            lp[l] = (1.0 - ch) * lp[l] + ch * PRIOR[l];
    }
    q->st[PS_PSL] = st;
    double lam = fresh ? 1.0 : 1.0 - exp(-cfg[PC_lane_switch_hz] * dt);
    double wgt = fresh ? 1.0 : dt / cfg[PC_evid_tau_s];
    if (wgt > 1.0)
        wgt = 1.0;
    for (int l = 0; l < P_NLANE; l++) {
        double pred = (1.0 - lam) * lp[l] + lam * PRIOR[l];
        post[l] = pred * pow(em[l] > 1e-7 ? em[l] : 1e-7, wgt);
        sum += post[l];
    }
    double fl = cfg[PC_lane_floor], sum2 = 0.0;
    for (int l = 0; l < P_NLANE; l++) {
        post[l] = post[l] / sum;
        if (post[l] < fl)
            post[l] = fl;
        sum2 += post[l];
    }
    int best = 0;
    for (int l = 0; l < P_NLANE; l++) {
        lp[l] = lane_out[l] = post[l] / sum2;
        if (lp[l] > lp[best])
            best = l;
    }
    /* the lane it was last said to be in holds unless another is clearly ahead */
    int was = (int)q->st[PS_LMAP];
    if (!fresh && was >= 0 && was < P_NLANE && was != best && lp[was] > 0.7 * lp[best])
        best = was;
    q->lane = best;
    /* its band of that kind, and the bands beside it */
    q->band = -1;
    q->dir = 0;
    q->dc = q->d;
    q->bw = c->w;
    q->nb[0] = q->nb[1] = -1;
    if (best <= PL_ONC || best == PL_PARK) {
        double bv = 0.0;
        for (int b = 0; b < c->nb; b++) {
            const double *B = c->bands + P_NB * b;
            if ((int)B[PB_TYPE] != best)
                continue;
            double v = B[PB_P] * band_occ(B, dm, q->sd_d);
            if (v > bv) {
                bv = v;
                q->band = b;
            }
        }
    }
    if (q->band >= 0 && best <= PL_ONC) {
        const double *B = c->bands + P_NB * q->band;
        q->dc = 0.5 * (B[PB_LO] + B[PB_HI]);
        q->bw = B[PB_HI] - B[PB_LO];
        q->dir = (int)B[PB_DIR];
        if (q->dir != 0) {
            q->nb[0] = band_at(c, q->dc + q->dir * q->bw, 0);
            q->nb[1] = band_at(c, q->dc - q->dir * q->bw, 0);
        }
    }
    if (q->dir == 0 && q->grp == 1 && q->sp > 1.0 && fabs(cos(q->psi)) > 0.75)
        q->dir = q->vt > 0.0 ? 1 : -1;                          /* a cyclist rides along wherever it is */
    q->arm_hd = 0.0;
    if (q->arm >= 0 && best == PL_CROSS) {
        double ah = c->arm[q->arm].th[pp_seg(&c->arm[q->arm], q->arm_s)];
        double vout = o[PO_VX] * cos(ah) + o[PO_VY] * sin(ah);
        q->arm_in = vout < -0.3 || (fabs(vout) <= 0.3 && q->arm_s < 40.0);
        if (q->sp > 1.0)                                         /* its direction against the arm's, towards the junction */
            q->arm_hd = wrap_pi(atan2(o[PO_VY], o[PO_VX]) - (ah + PI));
    } else if (best != PL_CROSS) {
        q->arm = -1;
    }
}

/* ---- what is observed of it: the filters on its offset and its speed ------------------------------------ */
static void feat_step(PCtx *c, PObj *q, double dt)
{
    const double *o = q->o, *cfg = c->cfg;
    double *st = q->st;
    double infl = cfg[PC_evid_tau_s] / (dt > 0.02 ? dt : 0.02);
    if (infl < 1.0)
        infl = 1.0;
    double sn = sin(q->th), ct = cos(q->th);
    double rvn = (p_sq(sn * o[PO_SVX]) + p_sq(ct * o[PO_SVY]) + 0.01) * infl, rd = p_sq(q->sd_d) * infl;
    double a_m = q->sp > 0.5 ? (o[PO_AX] * o[PO_VX] + o[PO_AY] * o[PO_VY]) / q->sp : 0.0;
    double rsp = ((q->sp > 0.5 ? (p_sq(o[PO_VX] * o[PO_SVX]) + p_sq(o[PO_VY] * o[PO_SVY])) / p_sq(q->sp)
                               : 0.5 * (p_sq(o[PO_SVX]) + p_sq(o[PO_SVY]))) + 0.01) * infl;
    double ra = p_sq(cfg[PC_acc_meas_sd]) * infl;
    if (!(st[PS_INIT] > 0.5) || fabs(q->d - st[PS_D]) > 3.0 * c->w) {
        st[PS_D] = q->d;
        st[PS_DR] = q->vn;
        st[PS_PDD] = p_sq(q->sd_d);
        st[PS_PDR] = 0.0;
        st[PS_PRR] = rvn / infl;
        st[PS_V] = q->sp;
        st[PS_A] = a_m;
        st[PS_PVV] = rsp / infl;
        st[PS_PVA] = 0.0;
        st[PS_PAA] = p_sq(cfg[PC_acc_meas_sd]);
        st[PS_TSTAND] = st[PS_TMOVE] = st[PS_TKERB] = 0.0;
        st[PS_TLANE] = 10.0;
        st[PS_LMAP] = -1.0;
        st[PS_TOP] = -1.0;
    } else {
        /* the offset and its rate: steady rate with a wandering acceleration; measured: the place and the velocity */
        double d = st[PS_D] + st[PS_DR] * dt, r = st[PS_DR];
        double qd = (q->grp == 2 ? 0.6 : 0.8), qr = qd * qd * dt;
        double pdd = st[PS_PDD] + 2.0 * dt * st[PS_PDR] + dt * dt * st[PS_PRR] + qr * dt * dt / 3.0;
        double pdr = st[PS_PDR] + dt * st[PS_PRR] + qr * dt / 2.0, prr = st[PS_PRR] + qr;
        double k0 = pdd / (pdd + rd), k1 = pdr / (pdd + rd), inn = q->d - d;
        d += k0 * inn;
        r += k1 * inn;
        prr -= k1 * pdr;
        pdr -= k1 * pdd;
        pdd -= k0 * pdd;
        k0 = pdr / (prr + rvn), k1 = prr / (prr + rvn), inn = q->vn - r;
        d += k0 * inn;
        r += k1 * inn;
        pdd -= k0 * pdr;
        pdr -= k0 * prr;
        prr -= k1 * prr;
        st[PS_D] = d, st[PS_DR] = r, st[PS_PDD] = pdd, st[PS_PDR] = pdr, st[PS_PRR] = prr;
        /* the speed and its change */
        double rho = exp(-dt / cfg[PC_acc_tau_s]);
        double v = st[PS_V] + st[PS_A] * dt, a = st[PS_A] * rho;
        double qa = p_sq(2.0 * cfg[PC_acc_noise] * q->kin[PK_ALON]) * (1.0 - rho * rho);
        double pvv = st[PS_PVV] + 2.0 * dt * st[PS_PVA] + dt * dt * st[PS_PAA];
        double pva = rho * (st[PS_PVA] + dt * st[PS_PAA]), paa = rho * rho * st[PS_PAA] + qa;
        k0 = pvv / (pvv + rsp), k1 = pva / (pvv + rsp), inn = q->sp - v;
        v += k0 * inn;
        a += k1 * inn;
        paa -= k1 * pva;
        pva -= k1 * pvv;
        pvv -= k0 * pvv;
        k0 = pva / (paa + ra), k1 = paa / (paa + ra), inn = a_m - a;
        v += k0 * inn;
        a += k1 * inn;
        pvv -= k0 * pva;
        pva -= k0 * paa;
        paa -= k1 * paa;
        st[PS_V] = v > 0.0 ? v : 0.0, st[PS_A] = a, st[PS_PVV] = pvv, st[PS_PVA] = pva, st[PS_PAA] = paa;
    }
    if (q->pstat > 0.5) {
        st[PS_TSTAND] += dt;
        st[PS_TMOVE] = 0.0;
    } else {
        st[PS_TMOVE] += dt;
        if (st[PS_TMOVE] > 3.0)
            st[PS_TSTAND] = 0.0;
    }
    int dir = q->dir != 0 ? q->dir : 1;
    q->e = dir * (st[PS_D] - q->dc);
    q->r = dir * st[PS_DR];
    q->se = sqrt(st[PS_PDD] > 1e-4 ? st[PS_PDD] : 1e-4);
    q->sr = sqrt(st[PS_PRR] > 1e-4 ? st[PS_PRR] : 1e-4);
    q->v = st[PS_V] * (1.0 - q->pstat);
    q->a = p_clip(st[PS_A], -cfg[PC_brake_max], 1.5 * q->kin[PK_ALON]) * (1.0 - q->pstat);
    q->sa = sqrt(st[PS_PAA] > 0.01 ? st[PS_PAA] : 0.01);
    double sp2 = q->sp > 1.5 ? q->sp * q->sp : 2.25;
    q->om = (o[PO_VX] * o[PO_AY] - o[PO_VY] * o[PO_AX]) / sp2 - q->dir * q->kap * q->sp;
    q->som = 1.4 * cfg[PC_acc_meas_sd] / sqrt(sp2);
    /* the road's edges */
    double kl = c->road[PR_KERBL], kr = c->road[PR_KERBR];
    q->side = q->d >= 0.5 * (kl + kr) ? 1 : -1;
    q->kerb = q->side > 0 ? kl - st[PS_D] : st[PS_D] - kr;
    if (q->grp == 2) {
        if (fabs(q->kerb) < cfg[PC_ped_kerb_m] && q->sp < 0.8)
            st[PS_TKERB] += dt;
        else if (st[PS_TKERB] > 0.0)
            st[PS_TKERB] = st[PS_TKERB] > dt * 2.0 ? st[PS_TKERB] - dt * 2.0 : 0.0;
    }
    q->xing = P_BIG;
    for (int i = 0; i < c->nx; i++) {
        double ds = fabs(q->s - c->xing[2 * i]) - c->xing[2 * i + 1];
        if (ds < q->xing)
            q->xing = ds > 0.0 ? ds : 0.0;
    }
}

/* ---- the one ahead, the junction ahead, the place to stop ------------------------------------------------- */
static void context_step(PCtx *c, int i)
{
    PObj *q = c->obj + i;
    const double *cfg = c->cfg;
    q->lead = -1;
    q->gap = P_BIG;
    q->lead_v = 0.0;
    q->a_idm = 0.0;
    q->stop_s = NAN;
    q->stop_p = 0.0;
    q->stop_why = 0;
    q->jarm[0] = q->jarm[1] = -1;
    q->jdist = P_BIG;
    q->obstacle = 0;
    double reach = cfg[PC_lead_reach_m];
    if (q->dir != 0) {
        double best = reach;
        for (int j = -1; j < c->n; j++) {
            double sj, dj, lj, vj;
            if (j < 0) {                                          /* the car itself */
                sj = cfg[PC_ego_centre_x];
                dj = c->ed0;                                      /* its own offset from the centre line */
                lj = cfg[PC_ego_len];
                vj = c->ego[PE_V];
            } else {
                const PObj *u = c->obj + j;
                if (j == i || (u->dir == -q->dir && u->pstat < 0.5) || (u->lane == PL_CROSS && u->sp > 2.0))
                    continue;
                sj = u->s;
                dj = u->st[PS_D];
                lj = u->len;
                vj = u->vt * (1.0 - u->pstat);
            }
            double ahead = q->dir * (sj - q->s);
            if (ahead < 0.5 || ahead >= best)
                continue;
            double half = q->grp == 1 ? 0.9 : 0.5 * q->bw;        /* a cyclist: its own corridor */
            if (fabs(dj - (q->grp == 1 ? q->st[PS_D] : q->dc)) > half)
                continue;
            if (q->grp == 1 && j >= 0 && c->obj[j].pstat > 0.5 && ahead < cfg[PC_bike_obstacle_m])
                q->obstacle = 1;
            best = ahead;
            q->lead = j < 0 ? -2 : j;
            q->gap = ahead - 0.5 * (lj + q->len);
            q->lead_v = q->dir * vj;
        }
        if (q->lead != -1) {
            double g = q->gap > 0.5 ? q->gap : 0.5, a = cfg[PC_idm_a], b = cfg[PC_idm_b];
            double ss = cfg[PC_idm_s0_m] + q->v * cfg[PC_idm_t_s] + q->v * (q->v - q->lead_v) / (2.0 * sqrt(a * b));
            if (ss < cfg[PC_idm_s0_m])
                ss = cfg[PC_idm_s0_m];
            q->a_idm = p_clip(a * (1.0 - p_sq(ss / g)), -cfg[PC_brake_max], 0.3);
            if (q->lead_v < 1.0 && q->gap < reach) {             /* a queue's tail: the stop is behind it */
                int j = q->lead;
                double sj = j == -2 ? cfg[PC_ego_centre_x] : c->obj[j].s, lj = j == -2 ? cfg[PC_ego_len] : c->obj[j].len;
                q->stop_s = sj - q->dir * (0.5 * (lj + q->len) + cfg[PC_idm_s0_m]);
                q->stop_p = 0.9;
                q->stop_why = 2;
            }
        }
        /* the junction ahead: its arms on this object's left and right, and whether it has to give way there */
        double jbest = cfg[PC_turn_reach_m], pri = -2.0, ctrl = 0.0, jd[2] = {P_BIG, P_BIG};
        for (int a = 0; a < c->na; a++) {
            const double *A = c->ameta + P_NA * a;
            double ahead = q->dir * (c->asj[a] - q->s);
            if (ahead < 1.0 || ahead > cfg[PC_turn_reach_m])
                continue;
            int side = (int)A[PA_SIDE] * q->dir > 0 ? 0 : 1;
            if (A[PA_EXIT] > 0.5 && ahead < jd[side]) {
                jd[side] = ahead;
                q->jarm[side] = a;
            }
            if (ahead < jbest - 12.0) {
                jbest = ahead;
                pri = A[PA_PRIO];
                ctrl = A[PA_CTRL];
            } else if (ahead < jbest + 12.0) {
                if (ahead < jbest)
                    jbest = ahead;
                if (A[PA_PRIO] > pri)
                    pri = A[PA_PRIO];
                if (A[PA_CTRL] > ctrl)
                    ctrl = A[PA_CTRL];
            }
        }
        if (pri > -2.0) {
            q->jdist = jbest;
            double line = c->w + 1.5, sp = ctrl >= 1.5 ? 0.9 : ctrl >= 0.5 ? 0.3 : pri > 0.5 ? 0.5 : pri > -0.5 ? 0.12 : 0.02;
            double ss = q->s + q->dir * (jbest - line);
            if (jbest > line + 1.0 && (q->stop_why == 0 || q->dir * (q->stop_s - ss) > 0.0)) {
                q->stop_s = ss;
                q->stop_p = sp;
                q->stop_why = 1;
            }
        }
        /* a marked crossing with someone at it */
        for (int x = 0; x < c->nx; x++) {
            double sx = c->xing[2 * x], hl = c->xing[2 * x + 1], ahead = q->dir * (sx - q->s);
            if (ahead < hl + 3.0 || ahead > reach || q->grp == 2)
                continue;
            int someone = 0;
            for (int j = 0; j < c->n && !someone; j++) {
                const PObj *u = c->obj + j;
                someone = u->grp == 2 && fabs(u->s - sx) < hl + 3.0 && u->kerb > -2.5;
            }
            double ss = sx - q->dir * (hl + 2.0);
            if (someone && (q->stop_why == 0 || q->dir * (q->stop_s - ss) > 0.0)) {
                q->stop_s = ss;
                q->stop_p = 0.5;
                q->stop_why = 3;
            }
        }
    } else if (q->grp == 2) {
        for (int a = 0; a < c->na; a++)
            if (fabs(c->asj[a] - q->s) < q->jdist)
                q->jdist = fabs(c->asj[a] - q->s);
    } else if (q->arm >= 0 && q->arm_in) {
        /* on a side road, heading for the junction: it stops at the edge of the road it meets */
        const double *A = c->ameta + P_NA * q->arm;
        double edge = ((int)A[PA_SIDE] > 0 ? c->road[PR_KERBL] : -c->road[PR_KERBR]) + 1.5;
        if (edge < 0.5 * c->w + 1.5)
            edge = 0.5 * c->w + 1.5;
        if (q->arm_s > edge + 1.0) {
            q->stop_s = edge;                                    /* on the arm, from the junction outwards */
            q->stop_p = A[PA_CTRL] >= 1.5 ? 0.9 : A[PA_PRIO] < -0.5 ? 0.6 : A[PA_PRIO] < 0.5 ? 0.25 : 0.03;
            q->stop_why = 1;
            q->jdist = q->arm_s;
        }
    }
}

/* ---- the manoeuvres: what is possible here (prior weights) and how well each explains what is seen --------- */
static const double P_EXIT_HZ[P_NM] = {0.15, 0.2, 0.25, 0.25, 0.2, 0.2, 0.2, 0.3, 0.05, 0.2, 0.5, 0.1, 0.5, 0.4, 0.5};

static double a_for_turn(const PCtx *c, const PObj *q, double dist, double R)
{
    double vt = sqrt(c->cfg[PC_turn_lat_acc] * R), d = dist - R > 3.0 ? dist - R : 3.0;
    if (q->v <= vt)
        return 0.0;
    double need = (vt * vt - q->v * q->v) / (2.0 * d);
    return need <= -0.6 * c->cfg[PC_idm_b] ? need : 0.0;
}

static double a_for_stop(const PCtx *c, const PObj *q, double dist)
{
    double need = -q->v * q->v / (2.0 * (dist > 0.5 ? dist : 0.5));
    return need <= -0.6 * c->cfg[PC_idm_b] ? (need < -c->cfg[PC_brake_max] ? -c->cfg[PC_brake_max] : need) : 0.0;
}

static void intent_step(PCtx *c, PObj *q, double dt, double *intent_out)
{
    const double *cfg = c->cfg, *o = q->o;
    double *Q = q->q, *L = q->ll;
    for (int m = 0; m < P_NM; m++) {
        Q[m] = 0.0;
        L[m] = 0.0;
    }
    /* standing or moving is not weighed here: the manoeuvres are of two groups (it stands: stand, and "it will
     * start" where that begins from standstill; it moves: all others), the evidence below sorts the manoeuvres
     * within each, and the groups' shares are the radar's word (below, after the filter's step) */
    double st = q->pstat, lmove = 0.0, lstand = 0.0;
    int pull_moving = 0;
    double quality = c->road[PR_Q], rh = c->rh;
    double ksd = cfg[PC_keep_sd_m], krs = cfg[PC_keep_rate_sd], asd = cfg[PC_acc_sd], ysd = cfg[PC_yaw_sd];
    double se = hypot(ksd, q->se), sr = hypot(krs, q->sr), sa = hypot(asd, q->sa), so = hypot(ysd, q->som);
    /* just arrived in this lane: it is still settling, its lateral speed is that of the move that brought it */
    double settle = exp(-q->st[PS_TLANE] / 1.0);
    se = hypot(se, 0.8 * settle);
    sr = hypot(sr, 1.2 * settle);
    double l_e = p_logn(q->e, 0.0, se, se), l_r = p_logn(q->r, -q->e / cfg[PC_keep_return_s], sr, sr);
    double l_a0 = p_logn(q->a, 0.0, sa, sa), l_o0 = p_logn(q->om, 0.0, so, so);
    double ind = o[PO_IND] == o[PO_IND] ? o[PO_IND] : 0.0;          /* +1: its left indicator, -1: its right */
    int ind_known = o[PO_IND] == o[PO_IND];
    double brake = o[PO_BRAKE] == o[PO_BRAKE] ? (o[PO_BRAKE] > 0.5 ? log(2.5) : log(0.6)) : 0.0;
    Q[PM_FREE] = cfg[PC_free_floor] + (1.0 - quality) * cfg[PC_free_noctx];
    L[PM_FREE] = cfg[PC_free_loglik] + lmove;
    Q[PM_STAND] = st > 0.05 || q->sp < 1.0 ? 1.0 : 0.0;
    L[PM_STAND] = lstand;
    if (q->grp == 2) {
        /* a pedestrian: walking along, standing (waiting at the kerb), about to cross, crossing */
        double vin = -q->side * q->st[PS_DR], svin = hypot(0.35, q->sr);
        double near = fabs(q->kerb) < cfg[PC_ped_kerb_m] + 1.0;
        Q[PM_KEEP] = 1.0;
        L[PM_KEEP] = lmove + p_logn(vin, 0.0, svin, svin) + (q->kerb > 0.5 ? -(q->kerb - 0.5) : 0.0);
        Q[PM_STAND] = 0.6;
        double boost = (q->xing < 2.0 ? 8.0 : 1.0) * (q->jdist < 15.0 ? 3.0 : 1.0);
        double waited = q->st[PS_TKERB] / 3.0;
        if (near && q->kerb < 0.7) {
            Q[PM_STARTX] = 0.03 * boost * (1.0 + (waited < 2.0 ? waited : 2.0));
            L[PM_STARTX] = st > 0.5 ? lstand
                                   : lmove + p_logn(vin, 0.5, hypot(0.5, q->sr), hypot(0.35, q->sr)) + p_logn(q->sp, 0.6, 0.6, 0.6);
        }
        if (q->kerb > -1.5) {
            Q[PM_CROSS] = q->kerb > 0.3 ? 0.6 : 0.1;
            L[PM_CROSS] = lmove + p_logn(fabs(vin), cfg[PC_ped_v], hypot(cfg[PC_ped_v_sd], q->sr), hypot(0.35, q->sr));
        }
        Q[PM_FREE] += 0.03;
    } else if (q->dir != 0 && (q->lane <= PL_ONC || q->grp == 1)) {
        /* in a lane, travelling along it */
        int bike = q->grp == 1;
        if (bike) {                                             /* a cyclist keeps its own line, not the lane's centre */
            l_e = 0.0;
            l_r = p_logn(q->r, 0.0, hypot(0.3, q->sr), hypot(krs, q->sr));
        }
        Q[PM_KEEP] = 1.0;
        L[PM_KEEP] = lmove + l_e + l_r + l_a0 + l_o0;
        if (q->lead != -1 && cfg[PC_interact] > 0.5 && q->stop_why != 2) {
            double near = q->gap < 3.0 * q->v + 15.0 ? 1.0 : 0.3;
            Q[PM_FOLLOW] = near;
            /* it may not have begun to brake for the one ahead yet: braking less than the law asks is no surprise */
            double late = q->a > q->a_idm ? hypot(sa, 0.5 * (q->a - q->a_idm) + 0.5) : sa;
            L[PM_FOLLOW] = lmove + l_e + l_r + p_logn(q->a, q->a_idm, late, late) + l_o0 + brake * (q->a_idm < -0.5);
        }
        if (!bike) {
            for (int sd = 0; sd < 2; sd++) {                    /* a change to ITS left (0) / right (1) */
                int nb = q->nb[sd], m = sd == 0 ? PM_LCL : PM_LCR;
                double sg = sd == 0 ? 1.0 : -1.0;
                if (nb < 0)
                    continue;
                const double *B = c->bands + P_NB * nb;
                int opp = (int)B[PB_DIR] != q->dir;
                Q[m] = 0.04 * B[PB_P] * (opp ? 0.25 : 1.0);
                L[m] = lmove + p_logn(sg * q->e, cfg[PC_lc_off_m], hypot(cfg[PC_lc_off_sd], q->se), hypot(ksd, q->se))
                       + p_logn(sg * q->r, cfg[PC_lc_rate], hypot(cfg[PC_lc_rate_sd], q->sr), hypot(krs, q->sr)) + l_a0 + l_o0;
                if (ind_known)
                    L[m] += ind * sg > 0.5 ? log(4.0) : ind * sg < -0.5 ? log(0.2) : log(0.7);
            }
            if (q->lane == PL_ONC) {                             /* an oncoming one wandering towards the car's lane */
                double sg = rh;                                 /* its left, where traffic keeps right */
                Q[PM_DRIFT] = 0.01;
                L[PM_DRIFT] = lmove + p_logn(sg * q->e, 0.5, hypot(0.6, q->se), hypot(ksd, q->se))
                             + p_logn(sg * q->r, cfg[PC_drift_rate], hypot(cfg[PC_drift_rate_sd], q->sr), hypot(krs, q->sr)) + l_a0 + l_o0;
            }
        } else {
            double sg = rh;
            Q[PM_SWERVE] = q->obstacle ? 0.25 : 0.02;
            /* something stands in its way: that it has not begun to move out yet is no surprise */
            double late = q->obstacle && sg * q->r < 0.7 ? 0.5 * (0.7 - sg * q->r) + 0.3 : 0.0;
            L[PM_SWERVE] = lmove + p_logn(sg * q->r, 0.7, hypot(hypot(0.4, late), q->sr), hypot(hypot(krs, late), q->sr)) + l_a0;
        }
        for (int sd = 0; sd < 2; sd++) {                        /* a turn to ITS left / right */
            int m = sd == 0 ? PM_TURNL : PM_TURNR;
            double sg = sd == 0 ? 1.0 : -1.0;
            int a = q->jarm[sd];
            double R = (sd == 0) == (rh > 0.0) ? cfg[PC_turn_r_left_m] : cfg[PC_turn_r_right_m];
            if (a >= 0) {
                double dist = q->dir * (c->asj[a] - q->s);
                Q[m] = 0.15 * (dist < 60.0 ? 1.0 : 0.5);
                double l_pos = dist < 50.0 ? p_logn(sg * q->e, 0.3, hypot(0.6, q->se), hypot(ksd, q->se)) : l_e;
                double l_yaw = l_o0, l_rate = l_r;
                if (dist < R + 8.0) {                            /* in the corner: it leaves its lane to that side */
                    l_pos = p_logn(sg * q->e, 1.0, hypot(1.5, q->se), hypot(ksd, q->se));
                    l_rate = p_logn(sg * q->r, 1.5, hypot(1.5, q->sr), hypot(krs, q->sr));
                    l_yaw = p_logn(sg * q->om, q->v / R, hypot(0.25, q->som), hypot(ysd, q->som));
                }
                L[m] = lmove + l_pos + l_rate + p_logn(q->a, a_for_turn(c, q, dist, R), sa, sa) + l_yaw;
            } else if (q->v > 2.0) {                            /* no junction known: a turn shows as a yaw rate */
                Q[m] = 0.01 + 0.03 * (1.0 - quality);
                L[m] = lmove + p_logn(q->e, 0.0, hypot(1.5, q->se), hypot(ksd, q->se)) + p_logn(sg * q->r, 1.0, hypot(1.0, q->sr), hypot(krs, q->sr))
                       + l_a0 + p_logn(sg * q->om, 0.25, hypot(0.15, q->som), hypot(ysd, q->som));
            } else {
                continue;
            }
            if (ind_known)
                L[m] += ind * sg > 0.5 ? log(4.0) : ind * sg < -0.5 ? log(0.2) : log(0.7);
        }
        if (q->stop_why != 0 && q->v > 0.5) {
            double dist = q->dir * (q->stop_s - q->s);
            if (dist > 0.0) {
                Q[PM_STOP] = q->stop_p;
                L[PM_STOP] = lmove + l_e + l_r + p_logn(q->a, a_for_stop(c, q, dist), sa, sa) + l_o0 + brake;
            }
        }
        if (!bike && q->v < cfg[PC_uturn_v_max] && q->v > 1.0) {
            /* a U-turn where there is room to its left for it */
            double room = rh * q->dir > 0.0 ? c->road[PR_KERBL] - q->st[PS_D] : q->st[PS_D] - c->road[PR_KERBR];
            if (room > 1.6 * cfg[PC_uturn_r_m] || q->jdist < 25.0) {
                Q[PM_UTURN] = 0.003;
                L[PM_UTURN] = lmove + p_logn(q->e, 0.0, hypot(1.5, q->se), hypot(ksd, q->se)) + p_logn(rh * q->r, 1.0, hypot(1.0, q->sr), hypot(krs, q->sr))
                               + p_logn(rh * q->om, q->v / cfg[PC_uturn_r_m], hypot(0.25, q->som), hypot(ysd, q->som));
            }
        }
        if (!bike && q->lane <= PL_ONC && st > 0.3) {
            /* it stands in a traffic lane (a queue, a junction): it will move off; sooner with nothing ahead */
            Q[PM_PULL] = q->lead != -1 && q->lead_v < 1.0 && q->gap < 15.0 ? 0.25 : 0.6;
            L[PM_PULL] = lstand;
        }
        if (q->lane > PL_ONC)
            Q[PM_FREE] += 0.1;
    } else if (q->lane == PL_CROSS || (q->grp == 1 && q->sp > 1.0)) {
        /* crossing the road, or on a side road */
        double shd = hypot(0.12, q->spsi), l_hd0 = q->arm >= 0 && q->arm_in ? p_logn(q->arm_hd, 0.0, shd, shd) : 0.0;
        Q[PM_CROSS] = 1.0;
        L[PM_CROSS] = lmove + l_a0 + l_o0 + l_hd0;
        if (q->grp == 0 && st > 0.3) {                           /* it waits there: it will move off */
            Q[PM_PULL] = 0.5;
            L[PM_PULL] = lstand;
        }
        if (q->stop_why == 1 && q->v > 0.5) {
            Q[PM_STOP] = q->stop_p;
            L[PM_STOP] = lmove + p_logn(q->a, a_for_stop(c, q, q->arm_s - q->stop_s), sa, sa) + brake + l_o0 + l_hd0;
        }
        if (q->arm >= 0 && q->arm_in && c->ameta[P_NA * q->arm + PA_ENTRY] > 0.5 && q->grp == 0) {
            double R = cfg[PC_turn_r_right_m];
            double la = p_logn(q->a, a_for_turn(c, q, q->arm_s, R), sa, sa);
            for (int sd = 0; sd < 2; sd++) {
                int m = sd == 0 ? PM_TURNL : PM_TURNR;
                double sg = sd == 0 ? 1.0 : -1.0;
                Q[m] = 0.3;
                L[m] = lmove + la + l_o0 + (q->arm_s < R + 8.0 ? p_logn(sg * q->arm_hd, 0.6, hypot(0.5, q->spsi), hypot(0.12, q->spsi)) : l_hd0);
                if (ind_known)
                    L[m] += ind * sg > 0.5 ? log(4.0) : ind * sg < -0.5 ? log(0.2) : log(0.7);
            }
        }
        Q[PM_FREE] += 0.05;
    } else if (q->lane == PL_PARK || (q->grp == 0 && st > 0.5 && q->lane == PL_OFF && fabs(q->kerb) < 3.0)) {
        /* parked, or beside the road: it stays, or it pulls out */
        int tb = band_at(c, q->st[PS_D] + (q->side > 0 ? -1.0 : 1.0) * (fabs(q->kerb) + 0.5 * c->w), 0);
        if (tb >= 0 && q->grp == 0) {
            double toward = -q->side * q->st[PS_DR];
            /* its lights are part of the situation while it stands: brake lights on, the indicator towards the road */
            Q[PM_PULL] = 0.005 * (o[PO_BRAKE] > 0.5 ? 5.0 : 1.0) * (!ind_known ? 1.0 : ind * q->side < -0.5 ? 20.0 : 0.7);
            if (st > 0.5)
                L[PM_PULL] = lstand;
            else
            {
                /* it moves, in the strip: pulling out shows as a speed towards the road. Where the tracker's
                 * lateral speed says nothing (a new or poorly seen object), or it runs against the lane it would
                 * join, this explains it no better than its own motion carried on (free) */
                double known = p_clip((0.8 - q->sr) / 0.4, 0.0, 1.0);
                int against = (int)c->bands[P_NB * tb + PB_DIR] * q->vt < -1.0;
                pull_moving = 1;
                L[PM_PULL] = lmove + p_logn(toward, 0.5, hypot(0.5, q->sr), hypot(krs, q->sr)) + p_logn(q->sp, 1.5, 1.5, 1.5)
                             + known * 2.0 + (1.0 - known) * cfg[PC_free_loglik] - (against ? 3.0 : 0.0);
            }
        }
        Q[PM_FREE] += 0.05;
    } else {
        Q[PM_FREE] += 0.5;                                       /* off the road, no lane: its own motion is all there is */
    }
    /* the filter: a manoeuvre lasts (its own exit rate); when it ends the next one is drawn from what is
     * possible here; the evidence of this frame, as one observation per evid_tau */
    double *ip = q->st + PS_INT, qs = 0.0, qls = 0.0, post[P_NM], sum = 0.0;
    int fresh = !(q->st[PS_INIT] > 0.5);
    for (int m = 0; m < P_NM; m++) {
        qs += Q[m];
        qls += Q[m] * P_EXIT_HZ[m];
    }
    int was = (int)q->st[PS_LMAP];
    if (!fresh && was != q->lane) {
        /* it is in another lane now: a change that got it there is over, what belonged to the old place is void */
        double moved = ip[PM_LCL] + ip[PM_LCR] + ip[PM_DRIFT] + ip[PM_PULL] + ip[PM_SWERVE] + ip[PM_STARTX];
        ip[PM_KEEP] += 0.8 * moved;
        ip[PM_LCL] = ip[PM_LCR] = ip[PM_DRIFT] = ip[PM_PULL] = ip[PM_SWERVE] = ip[PM_STARTX] = 0.0;
        for (int m = 0; m < P_NM; m++)
            ip[m] = 0.5 * ip[m] / (1.0 - 0.2 * moved > 0.1 ? 1.0 - 0.2 * moved : 0.1) + 0.5 * Q[m] / qs;
    }
    double wgt = fresh ? 1.0 : dt / cfg[PC_evid_tau_s], gone = 0.0;
    if (wgt > 1.0)
        wgt = 1.0;
    if (!fresh)
        for (int m = 0; m < P_NM; m++)
            gone += ip[m] * (Q[m] > 0.0 ? 1.0 - exp(-P_EXIT_HZ[m] * dt) : 1.0);
    else
        gone = 1.0;
    double lmax = -P_BIG;
    for (int m = 0; m < P_NM; m++)
        if (Q[m] > 0.0 && L[m] > lmax)
            lmax = L[m];
    for (int m = 0; m < P_NM; m++) {
        if (!(Q[m] > 0.0)) {
            post[m] = 0.0;
            continue;
        }
        /* (the next manoeuvre is drawn in proportion to prior weight x exit rate: left alone, the filter then
         * settles on the prior weights themselves, not on the long-lasting manoeuvres) */
        double pred = fresh ? Q[m] / qs : ip[m] * exp(-P_EXIT_HZ[m] * dt) + gone * Q[m] * P_EXIT_HZ[m] / qls;
        post[m] = pred * exp(wgt * (L[m] - lmax > -40.0 ? L[m] - lmax : -40.0));
        sum += post[m];
    }
    double sum2 = 0.0;
    for (int m = 0; m < P_NM; m++) {
        if (Q[m] > 0.0) {
            post[m] /= sum;
            double fl = 1e-4 < 0.2 * Q[m] / qs ? 1e-4 : 0.2 * Q[m] / qs;
            /* the future in which it gives way to the one ahead is always kept in play: whether it is taken is
             * for the joint weights to say */
            if (((m == PM_FOLLOW && Q[m] >= 1.0) || (m == PM_SWERVE && q->obstacle)) && cfg[PC_mode_p_min] * 1.5 > fl)
                fl = cfg[PC_mode_p_min] * 1.5;
            if (post[m] < fl)
                post[m] = fl;
        }
        sum2 += post[m];
    }
    for (int m = 0; m < P_NM; m++)
        post[m] /= sum2;
    {
        /* the share of "it stands": the radar's (as far as it knows: motion_state), taking hold of the kept
         * share within motion_tau_s */
        double gs = 0.0, g = st;
        int stand[P_NM];
        for (int m = 0; m < P_NM; m++) {
            stand[m] = m == PM_STAND || (m == PM_PULL && !pull_moving) || (m == PM_STARTX && st > 0.5);
            if (stand[m])
                gs += post[m];
        }
        if (!fresh && cfg[PC_motion_tau_s] > 1e-3)
            g = q->st[PS_GSTAND] + (1.0 - exp(-dt / cfg[PC_motion_tau_s])) * (st - q->st[PS_GSTAND]);
        q->st[PS_GSTAND] = g;
        if (gs > 1e-9 && gs < 1.0 - 1e-9) {
            for (int m = 0; m < P_NM; m++)
                post[m] *= stand[m] ? g / gs : (1.0 - g) / (1.0 - gs);
        }
    }
    if (st > 0.5) {
        /* it stands: "it stays" and "it will start" (move off, pull out, step into the road) look the same now;
         * how the standing share splits among them is the situation's word alone */
        double tot = post[PM_STAND] + post[PM_PULL] + post[PM_STARTX], qq = Q[PM_STAND] + Q[PM_PULL] + Q[PM_STARTX];
        if (qq > 0.0) {
            post[PM_STAND] = tot * Q[PM_STAND] / qq;
            post[PM_PULL] = tot * Q[PM_PULL] / qq;
            post[PM_STARTX] = tot * Q[PM_STARTX] / qq;
        }
    }
    for (int m = 0; m < P_NM; m++)
        ip[m] = intent_out[m] = post[m];
    q->st[PS_TLANE] = !fresh && was != q->lane ? 0.0 : q->st[PS_TLANE] + dt;
    q->st[PS_LMAP] = q->lane;
    q->st[PS_DIR] = q->dir;
}

/* the name a slot is published under for this object */
static int slot_name(const PCtx *c, const PObj *q, int m)
{
    switch (m) {
    case PM_KEEP:
        return q->grp == 2 ? PN_WALK : q->grp == 1 ? PN_RIDE : q->lane == PL_ONC ? PN_ONC_KEEP : PN_KEEP;
    case PM_FOLLOW:
        return PN_FOLLOW;
    case PM_LCL:
    case PM_LCR: {
        int nb = q->nb[m == PM_LCL ? 0 : 1];
        int tt = nb >= 0 ? (int)c->bands[P_NB * nb + PB_TYPE] : -1;
        if (q->lane == PL_ONC && tt != PL_ONC)
            return PN_OVERTAKE;
        if (tt == PL_EGO && q->s > 0.0)
            return PN_CUT_IN;
        if (q->lane == PL_EGO)
            return PN_CUT_OUT;
        return m == PM_LCL ? PN_LCL : PN_LCR;
    }
    case PM_TURNL:
        return PN_TURNL;
    case PM_TURNR:
        return PN_TURNR;
    case PM_STOP:
        return PN_STOP;
    case PM_PULL:
        return (q->dir != 0 && q->lane <= PL_ONC) || q->lane == PL_CROSS ? PN_MOVE_OFF : PN_PULL;
    case PM_STAND:
        if (q->grp == 2)
            return fabs(q->kerb) < c->cfg[PC_ped_kerb_m] ? PN_WAIT : PN_STAND;
        return q->grp == 0 && (q->lane == PL_PARK || q->lane == PL_OFF) ? PN_PARKED : PN_STOPPED;
    case PM_UTURN:
        return PN_UTURN;
    case PM_DRIFT:
        return PN_ONC_DRIFT;
    case PM_CROSS:
        return q->grp == 2 ? PN_CROSSING : PN_CROSS;
    case PM_STARTX:
        return PN_STARTX;
    case PM_SWERVE:
        return PN_SWERVE;
    default:
        return PN_FREE;
    }
}

/* ---- one future ------------------------------------------------------------------------------------------- */
static void roll_on_base(const PCtx *c, const PObj *q, PRoll *r)
{
    const double *st = q->st;
    double ct = cos(q->th), sn = sin(q->th), sx2 = p_sq(q->o[PO_SX]), sy2 = p_sq(q->o[PO_SY]);
    r->path = &c->base;
    r->known_lo = c->known_lo;
    r->known_hi = c->known_hi;
    r->dir = q->dir != 0 ? q->dir : 1;
    r->s0 = q->s;
    r->d0 = st[PS_D];
    r->dr0 = st[PS_DR];
    r->pss = ct * ct * sx2 + sn * sn * sy2;
    r->psd = ct * sn * (sy2 - sx2);
    r->pdd = st[PS_PDD];
    r->pdr = st[PS_PDR];
    r->prr = st[PS_PRR];
    r->pvv = st[PS_PVV];
    r->paa = st[PS_PAA];
    r->v0 = q->v;
    r->a0 = q->a;
    r->len = q->len;
}

/* the object put on another path: where it is along and beside it */
static void roll_on_path(const PObj *q, PRoll *r, const PPath *p, int dir)
{
    const double *o = q->o, *st = q->st;
    double x, y, th, kap;
    r->path = p;
    r->dir = dir;
    pp_project(p, o[PO_X], o[PO_Y], &r->s0, &r->d0);
    pp_eval(p, r->s0, &x, &y, &th, &kap);
    double ct = cos(th), sn = sin(th), sx2 = p_sq(o[PO_SX]), sy2 = p_sq(o[PO_SY]);
    r->dr0 = (-o[PO_VX] * sn + o[PO_VY] * ct) * (1.0 - q->pstat);
    r->pss = ct * ct * sx2 + sn * sn * sy2;
    r->pdd = sn * sn * sx2 + ct * ct * sy2;
    r->psd = ct * sn * (sy2 - sx2);
    r->pdr = 0.0;
    r->prr = p_sq(sn * o[PO_SVX]) + p_sq(ct * o[PO_SVY]);
    r->pvv = st[PS_PVV];
    r->paa = st[PS_PAA];
    r->v0 = q->v;
    r->a0 = q->a;
    r->len = q->len;
}

static void path_line(PPath *p, double x, double y, double h, double len)
{
    p->n = 0;
    pp_add(p, x, y);
    pp_add(p, x + len * cos(h), y + len * sin(h));
    pp_finish(p, 0.0);
}

/* from (x, y) heading h: an arc of curvature k over the angle ang, then straight on */
static void path_arc(PPath *p, double x, double y, double h, double k, double ang, double lead_in)
{
    p->n = 0;
    pp_add(p, x, y);
    if (lead_in > 0.5) {
        x += lead_in * cos(h);
        y += lead_in * sin(h);
        pp_add(p, x, y);
    }
    int n = 12;
    double ds = ang / fabs(k) / n;
    for (int i = 0; i < n; i++) {
        h += k * ds;
        x += ds * cos(h - 0.5 * k * ds);
        y += ds * sin(h - 0.5 * k * ds);
        pp_add(p, x, y);
    }
    pp_add(p, x + 250.0 * cos(h), y + 250.0 * sin(h));
    pp_finish(p, 0.0);
}

/* the one ahead of object i in the band centred at dc (width bw), driving `dir`: index, -2 the car, -1 none */
static int leader_in(const PCtx *c, int i, double dc, double bw, int dir)
{
    const PObj *q = c->obj + i;
    const double *cfg = c->cfg;
    int lead = -1;
    double best = cfg[PC_lead_reach_m];
    for (int j = -1; j < c->n; j++) {
        double sj, dj;
        if (j < 0) {
            sj = cfg[PC_ego_centre_x];
            dj = c->ed0;
        } else {
            const PObj *u = c->obj + j;
            if (j == i || (u->dir == -dir && u->pstat < 0.5) || (u->lane == PL_CROSS && u->sp > 2.0))
                continue;
            sj = u->s;
            dj = u->st[PS_D];
        }
        double ahead = dir * (sj - q->s);
        if (ahead < 0.5 || ahead >= best || fabs(dj - dc) > 0.5 * bw)
            continue;
        best = ahead;
        lead = j < 0 ? -2 : j;
    }
    return lead;
}

static void roll_lead(const PCtx *c, PRoll *r, int lead)
{
    if (lead == -1 || !(c->cfg[PC_interact] > 0.5))
        return;
    r->lead_s = lead == -2 ? c->esprof : c->obj[lead].sprof;
    r->lead_v = lead == -2 ? c->evprof : c->obj[lead].vprof;
    r->lead_len = lead == -2 ? c->cfg[PC_ego_len] : c->obj[lead].len;
}

/* The future number k of object i: its trajectory into traj, its name, its standing in a conflict. Returns 0
 * when this manoeuvre has no trajectory here after all. */
static int mode_build(PCtx *c, int i, int k, int m, PPath *tmp)
{
    PObj *q = c->obj + i;
    const double *cfg = c->cfg, *o = q->o;
    double *out = q->traj + (size_t)k * c->nt * 6;
    PRoll r;
    double rh = c->rh;
    int top = k == 0;
    roll_defaults(&r, cfg, q->kin);
    q->name[k] = slot_name(c, q, m);
    q->yield[k] = 0;
    q->rank[k] = 3.0;
    double *sp = NULL, *vp = NULL;
    if (m == PM_FREE || m == PM_STAND) {
        roll_free(o, q->kin, cfg, c->times, c->nt, out, m == PM_STAND);
        q->rank[k] = m == PM_STAND ? 5.0 : 3.0;
        return 1;
    }
    if (q->grp == 2) {
        /* pedestrians */
        r.sig_a = 0.3;
        r.kp = r.kd = 0.0;
        r.q_lat = p_sq(cfg[PC_ped_lat_sd]);
        r.a_lat = 5.0;
        if (m == PM_KEEP) {
            roll_on_base(c, q, &r);
            r.dir = q->vt >= 0.0 ? 1 : -1;
            r.v0 = fabs(q->vt) * (1.0 - q->pstat);
            r.a0 = 0.0;
            r.d1 = r.d0;
            r.dr0 = 0.0;
            r.t_dur = 1.0;
        } else {                                                /* across the road, to its far side */
            double hn = q->th + (q->side > 0 ? -0.5 * PI : 0.5 * PI);
            double hv = atan2(o[PO_VY], o[PO_VX]);
            int moving = q->sp > 0.6 && q->pstat < 0.5;
            if (moving && fabs(wrap_pi(hv - hn)) < 1.05)
                hn = hv;
            else if (moving && fabs(wrap_pi(hv - hn - PI)) < 1.05 && m == PM_CROSS)
                hn = hv;                                         /* it walks off the road again */
            path_line(tmp, o[PO_X], o[PO_Y], hn, 60.0);
            roll_on_path(q, &r, tmp, 1);
            r.d0 = r.d1 = r.dr0 = 0.0;
            r.t_dur = 1.0;
            double width = c->road[PR_KERBL] - c->road[PR_KERBR], cosang = fabs(cos(wrap_pi(hn - q->th - 0.5 * PI)));
            double to_far = (q->kerb < 0.0 ? width - q->kerb : width - q->kerb) + 1.2;
            if (fabs(wrap_pi(hn - (q->th + (q->side > 0 ? -0.5 * PI : 0.5 * PI)))) > 0.5 * PI)
                to_far = (q->kerb > 0.0 ? q->kerb : 0.0) + 1.2;  /* back to its own side */
            r.s_stop = to_far / (cosang > 0.5 ? cosang : 0.5);
            r.stop_sd = 1.5;
            r.go = 1;
            r.a_go = 1.0;
            r.v_des = q->sp > cfg[PC_ped_v] && moving ? q->sp : cfg[PC_ped_v];
            r.v0 = moving ? q->sp : 0.0;
            r.pss = r.pdd = 0.5 * (p_sq(o[PO_SX]) + p_sq(o[PO_SY]));
            r.psd = 0.0;
            if (m == PM_STARTX) {
                r.t_start = moving ? 0.0 : cfg[PC_ped_cross_onset_s];
                r.onset_lon_sd = moving ? 0.4 : cfg[PC_ped_cross_onset_sd_s];
                q->rank[k] = q->xing < 2.0 ? 4.0 : 1.0;
            } else {
                q->rank[k] = 3.0;
            }
        }
        roll_out(&r, cfg, c->times, c->nt, c->h, c->nstep, out, NULL, NULL);
        return 1;
    }
    int on_base = q->dir != 0;
    if (q->grp == 1) {
        r.sig_a = 1.5 * cfg[PC_acc_noise] * q->kin[PK_ALON];
    }
    double ts[P_STEPS_MAX + 1], tv[P_STEPS_MAX + 1];
    /* what those behind it follow: its run along the road in its first future, and in any other future of weight
     * in which it gets less far (it brakes for the one ahead of it, it stops) */
    int must = (m == PM_FOLLOW && q->a_idm < -1.0) || (m == PM_STOP && q->stop_why == 2);    /* or it ends in the one ahead */
    int prof = on_base && (top || q->pri[k] >= 0.15 || must) && (m == PM_KEEP || m == PM_FOLLOW || m == PM_STOP || m == PM_LCL
                                                          || m == PM_LCR || m == PM_SWERVE || m == PM_DRIFT || m == PM_PULL);
    if (prof) {
        sp = ts;
        vp = tv;
    }
    switch (m) {
    case PM_KEEP:
    case PM_FOLLOW:
    case PM_STOP:
        if (!on_base) {
            /* a stop on a side road, before the road it meets */
            if (m != PM_STOP || q->arm < 0)
                return 0;
            tmp->n = 0;
            pp_walk(tmp, &c->arm[q->arm], q->arm_s + 30.0, -20.0, q->arm_d, 3.0);
            pp_finish(tmp, 0.0);
            roll_on_path(q, &r, tmp, 1);
            r.d1 = 0.0;
            r.t_dur = cfg[PC_keep_return_s];
            r.s_stop = r.s0 + (q->arm_s - q->stop_s);
            q->yield[k] = 1;
            break;
        }
        roll_on_base(c, q, &r);
        r.d1 = q->grp == 1 ? r.d0 : q->dc + 0.3 * (r.d0 - q->dc);
        r.t_dur = cfg[PC_keep_return_s];
        if (m == PM_FOLLOW && q->lead != -1) {
            roll_lead(c, &r, q->lead);
            r.k_v = 1.0 / cfg[PC_follow_v_tau_s];
            q->yield[k] = 1;
        }
        if (m == PM_STOP) {
            r.s_stop = q->stop_s;
            q->yield[k] = 1;
        }
        break;
    case PM_LCL:
    case PM_LCR:
    case PM_DRIFT:
    case PM_SWERVE: {
        if (!on_base)
            return 0;
        roll_on_base(c, q, &r);
        double sg = m == PM_LCL ? 1.0 : m == PM_LCR ? -1.0 : rh, rate = cfg[PC_lc_rate];
        if (m == PM_SWERVE) {
            r.d1 = r.d0 + q->dir * sg * cfg[PC_bike_swerve_m];
            r.t_dur = cfg[PC_bike_swerve_s];
        } else {
            int nb = q->nb[sg > 0.0 ? 0 : 1];
            if (nb < 0)
                return 0;
            r.d1 = 0.5 * (c->bands[P_NB * nb + PB_LO] + c->bands[P_NB * nb + PB_HI]);
            /* in the lane it moves into it drives behind the one that is ahead there */
            if ((int)c->bands[P_NB * nb + PB_DIR] == q->dir)
                roll_lead(c, &r, leader_in(c, i, r.d1, c->bands[P_NB * nb + PB_HI] - c->bands[P_NB * nb + PB_LO], q->dir));
            double span = fabs(r.d1 - r.d0);
            if (m == PM_DRIFT) {
                rate = fabs(q->r) > cfg[PC_drift_rate] ? fabs(q->r) : cfg[PC_drift_rate];
                r.t_dur = p_clip(1.875 * span / rate, 3.0, 12.0);
                r.kp = r.kd = 0.0;                              /* nobody steers it: the offset's doubt grows */
                r.q_lat = 0.09;
            } else {
                r.t_dur = p_clip(cfg[PC_lc_dur_s] * sqrt(span / c->w), 2.5, 7.0);
            }
        }
        int under_way = sg * q->r > 0.25;
        r.t_on = under_way || m == PM_DRIFT ? 0.0 : 0.5;
        r.onset_sd = under_way ? 0.3 : cfg[PC_lc_onset_sd_s];
        if (!under_way && m != PM_DRIFT)
            r.dr0 = 0.0;
        q->rank[k] = 2.0;
        break;
    }
    case PM_TURNL:
    case PM_TURNR: {
        int sd = m == PM_TURNL ? 0 : 1;
        double sg = sd == 0 ? 1.0 : -1.0;
        double R = (sd == 0) == (rh > 0.0) ? cfg[PC_turn_r_left_m] : cfg[PC_turn_r_right_m];
        double x, y, th, kap, x2, y2, th2;
        if (on_base && q->jarm[sd] >= 0) {
            /* along its lane to the junction, round the corner, out along the arm (on the arm's own right) */
            int a = q->jarm[sd];
            const PPath *A = &c->arm[a];
            double sj = c->asj[a], aoff = -rh * 0.25 * c->w;
            tmp->n = 0;
            double s_in = sj - q->dir * R;
            if (q->dir * (s_in - q->s) > 1.0)
                pp_walk(tmp, &c->base, q->s - q->dir * 2.0, s_in, q->dc, 3.0);
            pp_point(&c->base, q->dir * (s_in - q->s) > 1.0 ? s_in : q->s, q->dc, &x, &y, &th, &kap);
            pp_point(A, R, aoff, &x2, &y2, &th2, &kap);
            pp_add(tmp, x, y);
            pp_corner(tmp, x, y, th + (q->dir < 0 ? PI : 0.0), x2, y2, th2);
            pp_walk(tmp, A, R, R + 260.0, aoff, 6.0);
            pp_finish(tmp, 0.0);
            roll_on_path(q, &r, tmp, 1);
            r.pdd = q->st[PS_PDD];
            r.extra_lat = 0.5;
            q->rank[k] = (sd == 0) == (rh > 0.0) ? 2.0 : 2.5;
        } else if (!on_base && q->arm >= 0 && q->arm_in) {
            /* from the side road into the road: towards the car (the oncoming side) or away from it */
            int a = q->arm;
            const PPath *A = &c->arm[a];
            double sj = c->asj[a];
            pp_point(A, R + 2.0, q->arm_d, &x, &y, &th, &kap);
            double hin = th + PI;
            /* which way round is this: to ITS left the road runs along hin + pi/2 */
            double bx, by, bth, bk;
            pp_eval(&c->base, sj, &bx, &by, &bth, &bk);
            int fwd = cos(wrap_pi(bth - (hin + sg * 0.5 * PI))) > 0.0;     /* this turn ends along the base's direction */
            int tb = -1;
            for (int b = 0; b < c->nb; b++) {
                const double *B = c->bands + P_NB * b;
                if ((int)B[PB_TYPE] <= PL_ONC && (int)B[PB_DIR] == (fwd ? 1 : -1) && (tb < 0 || B[PB_P] > c->bands[P_NB * tb + PB_P]))
                    tb = b;
            }
            double td = tb >= 0 ? 0.5 * (c->bands[P_NB * tb + PB_LO] + c->bands[P_NB * tb + PB_HI]) : (fwd ? 0.0 : rh * c->w);
            tmp->n = 0;
            if (q->arm_s > R + 3.0)
                pp_walk(tmp, A, q->arm_s + 2.0, R + 2.0, q->arm_d, 3.0);
            pp_add(tmp, x, y);
            double so = sj + (fwd ? 1.0 : -1.0) * (R + 0.5 * c->w);
            pp_point(&c->base, so, td, &x2, &y2, &th2, &kap);
            pp_corner(tmp, x, y, hin, x2, y2, th2 + (fwd ? 0.0 : PI));
            pp_walk(tmp, &c->base, so, so + (fwd ? 260.0 : -260.0), td, 6.0);
            pp_finish(tmp, 0.0);
            roll_on_path(q, &r, tmp, 1);
            r.extra_lat = 0.5;
            q->rank[k] = 2.0;
        } else if (on_base) {
            /* no junction known: the turn it shows, carried on over a quarter circle */
            double hv = q->sp > 0.5 ? atan2(o[PO_VY], o[PO_VX]) : q->th + (q->dir < 0 ? PI : 0.0);
            double kk = sg * q->om > 0.05 && q->v > 1.0 ? fabs(q->om + q->dir * q->kap * q->sp) / q->v : 0.1;
            kk = p_clip(kk, 1.0 / 40.0, 1.0 / 6.0);
            path_arc(tmp, o[PO_X], o[PO_Y], hv, sg * kk, 0.5 * PI, 0.0);
            roll_on_path(q, &r, tmp, 1);
            r.d0 = r.dr0 = 0.0;
            r.extra_lat = 1.2;
            q->rank[k] = 2.0;
        } else {
            return 0;
        }
        r.d1 = 0.0;
        r.t_dur = 2.5;
        break;
    }
    case PM_UTURN: {
        if (!on_base)
            return 0;
        double hv = q->sp > 0.5 ? atan2(o[PO_VY], o[PO_VX]) : q->th + (q->dir < 0 ? PI : 0.0);
        path_arc(tmp, o[PO_X], o[PO_Y], hv, rh / cfg[PC_uturn_r_m], PI, 3.0);
        roll_on_path(q, &r, tmp, 1);
        r.d0 = r.d1 = r.dr0 = 0.0;
        r.t_dur = 1.0;
        r.extra_lat = 1.2;
        q->rank[k] = 0.5;
        break;
    }
    case PM_PULL: {
        if (on_base && q->lane <= PL_ONC) {
            /* standing in its lane, it moves off: after a while, behind the one ahead */
            roll_on_base(c, q, &r);
            r.d1 = r.d0;
            r.dr0 = 0.0;
            r.t_dur = 1.0;
            roll_lead(c, &r, q->lead);
            r.t_start = cfg[PC_go_onset_s];
            r.onset_lon_sd = cfg[PC_go_onset_sd_s];
            r.go = 1;
            r.v0 = 0.0;
            r.v_des = c->road[PR_VLIM] == c->road[PR_VLIM] ? c->road[PR_VLIM] : cfg[PC_pull_v] + 3.0;
            break;
        }
        prof = 0;
        sp = vp = NULL;
        if (!on_base && q->lane == PL_CROSS) {
            /* waiting on a side road (or across the road): on over the junction, after a while */
            if (q->arm >= 0) {
                tmp->n = 0;
                pp_walk(tmp, &c->arm[q->arm], q->arm_s + 2.0, -60.0, q->arm_d, 4.0);
                pp_finish(tmp, 0.0);
            } else {
                path_line(tmp, o[PO_X], o[PO_Y], o[PO_HDG] == o[PO_HDG] ? o[PO_HDG] : q->th + 0.5 * PI, 200.0);
            }
            roll_on_path(q, &r, tmp, 1);
            r.d1 = r.d0;
            r.dr0 = 0.0;
            r.t_dur = 1.0;
            r.t_start = cfg[PC_go_onset_s];
            r.onset_lon_sd = cfg[PC_go_onset_sd_s];
            r.go = 1;
            r.v0 = 0.0;
            r.v_des = cfg[PC_pull_v];
            r.extra_lat = 1.5;                                  /* straight on or a turn: not known */
            q->rank[k] = q->arm >= 0 ? 3.0 + 0.5 * c->ameta[P_NA * q->arm + PA_PRIO] : 3.0;
            break;
        }
        /* out of the parking strip into the lane beside it */
        int tb = band_at(c, q->st[PS_D] + (q->side > 0 ? -1.0 : 1.0) * (fabs(q->kerb) + 0.5 * c->w), 0);
        if (tb < 0)
            return 0;
        const double *B = c->bands + P_NB * tb;
        roll_on_base(c, q, &r);
        r.dir = (int)B[PB_DIR] != 0 ? (int)B[PB_DIR] : 1;
        r.d1 = 0.5 * (B[PB_LO] + B[PB_HI]);
        roll_lead(c, &r, leader_in(c, i, r.d1, B[PB_HI] - B[PB_LO], r.dir));
        int moving = q->pstat < 0.5 && q->sp > 0.5;
        r.t_start = moving ? 0.0 : cfg[PC_pull_onset_s];
        r.onset_lon_sd = moving ? 0.3 : cfg[PC_pull_onset_sd_s];
        r.t_on = r.t_start;
        r.t_dur = 3.5;
        r.onset_sd = moving ? 0.3 : cfg[PC_pull_onset_sd_s];
        r.dr0 = moving ? r.dr0 : 0.0;
        r.go = 1;
        r.a_go = cfg[PC_pull_acc];
        r.v_des = cfg[PC_pull_v];
        r.v0 = moving ? q->sp : 0.0;
        q->rank[k] = 1.0;
        break;
    }
    case PM_CROSS: {
        if (q->arm >= 0) {
            /* along the side road, over the junction, on along the arm opposite (or straight on) */
            int a = q->arm, opp = -1;
            const PPath *A = &c->arm[a];
            for (int b = 0; b < c->na; b++)
                if (b != a && fabs(c->asj[b] - c->asj[a]) < 15.0 && c->ameta[P_NA * b + PA_SIDE] * c->ameta[P_NA * a + PA_SIDE] < 0.0)
                    opp = b;
            tmp->n = 0;
            if (q->arm_in) {
                pp_walk(tmp, A, q->arm_s + 2.0, 0.0, q->arm_d, 3.0);
                if (opp >= 0)
                    pp_walk(tmp, &c->arm[opp], 2.0, 260.0, -q->arm_d, 6.0);
            } else {
                pp_walk(tmp, A, q->arm_s - 2.0, q->arm_s + 260.0, q->arm_d, 6.0);
            }
            pp_finish(tmp, 0.0);
            roll_on_path(q, &r, tmp, 1);
            r.d1 = 0.0;
            r.t_dur = cfg[PC_keep_return_s];
            q->rank[k] = 3.0 + 0.5 * c->ameta[P_NA * a + PA_PRIO];
        } else {
            double hv = q->sp > 0.5 ? atan2(o[PO_VY], o[PO_VX]) : q->th + 0.5 * PI;
            path_line(tmp, o[PO_X], o[PO_Y], hv, 300.0);
            roll_on_path(q, &r, tmp, 1);
            r.d0 = r.d1 = r.dr0 = 0.0;
            r.t_dur = 1.0;
            r.extra_lat = 0.8;
        }
        break;
    }
    default:
        return 0;
    }
    roll_out(&r, cfg, c->times, c->nt, c->h, c->nstep, out, sp, vp);
    if (prof && r.path == &c->base) {
        for (int j = 0; j <= c->nstep; j++)
            if (top || r.dir * ts[j] < r.dir * q->sprof[j]) {
                q->sprof[j] = ts[j];
                q->vprof[j] = r.dir * tv[j];
            }
    }
    return 1;
}

/* the bounding box of a future, with three sigma of its spread */
static void traj_bb(const double *t, int nt, double *bb)
{
    bb[0] = bb[1] = P_BIG;
    bb[2] = bb[3] = -P_BIG;
    for (int k = 0; k < nt; k++) {
        const double *p = t + 6 * k;
        double ex = 3.0 * p[2], ey = 3.0 * p[3];
        if (p[0] - ex < bb[0])
            bb[0] = p[0] - ex;
        if (p[1] - ey < bb[1])
            bb[1] = p[1] - ey;
        if (p[0] + ex > bb[2])
            bb[2] = p[0] + ex;
        if (p[1] + ey > bb[3])
            bb[3] = p[1] + ey;
    }
}

/* ---- the futures of one object: which manoeuvres, their trajectories ---------------------------------------- */
static void object_modes(PCtx *c, int i, PPath *tmp)
{
    PObj *q = c->obj + i;
    const double *cfg = c->cfg, *ip = q->st + PS_INT;
    int kmax = (int)cfg[PC_modes_max];
    if (kmax > P_KMAX)
        kmax = P_KMAX;
    if (kmax < 1)
        kmax = 1;
    int order[P_NM], used = 0;
    for (int m = 0; m < P_NM; m++)
        order[m] = m;
    for (int a = 1; a < P_NM; a++) {
        int v = order[a], b = a - 1;
        while (b >= 0 && ip[order[b]] < ip[v]) {
            order[b + 1] = order[b];
            b--;
        }
        order[b + 1] = v;
    }
    double sum = 0.0;
    q->nm = 0;
    for (int a = 0; a < P_NM && q->nm < kmax; a++) {
        int m = order[a];
        if (!(ip[m] > 0.0) || (q->nm > 0 && ip[m] < cfg[PC_mode_p_min]))
            break;
        q->pri[q->nm] = ip[m];
        if (!mode_build(c, i, q->nm, m, tmp))
            continue;
        q->slot[q->nm] = m;
        sum += ip[m];
        traj_bb(q->traj + (size_t)q->nm * c->nt * 6, c->nt, q->bb[q->nm]);
        q->nm++;
        used = 1;
    }
    if (!used) {                                                /* nothing could be built: its own motion */
        mode_build(c, i, 0, PM_FREE, tmp);
        q->slot[0] = PM_FREE;
        q->pri[0] = sum = 1.0;
        traj_bb(q->traj, c->nt, q->bb[0]);
        q->nm = 1;
    }
    for (int k = 0; k < q->nm; k++)
        q->pri[k] /= sum;
}

/* ---- the car's own futures ------------------------------------------------------------------------------------ */
static void ego_modes(PCtx *c, PPath *tmp)
{
    const double *cfg = c->cfg, *e = c->ego;
    static const double CAR[P_NK] = {2.5, 2.0, 60.0, 0.7, 1.8, 4.5};
    double d0 = c->ed0;
    /* the one ahead in the car's lane */
    c->elead = -1;
    c->egap = P_BIG;
    double best = cfg[PC_lead_reach_m], lv = 0.0;
    for (int j = 0; j < c->n; j++) {
        const PObj *u = c->obj + j;
        if ((u->dir < 0 && u->pstat < 0.5) || (u->lane == PL_CROSS && u->sp > 2.0))
            continue;
        double ahead = u->s - cfg[PC_ego_centre_x];
        if (ahead < 0.5 || ahead >= best || fabs(u->st[PS_D] - d0) > 0.5 * c->w)
            continue;
        best = ahead;
        c->elead = j;
        c->egap = ahead - 0.5 * (u->len + cfg[PC_ego_len]);
        lv = u->vt * (1.0 - u->pstat);
    }
    int n = 0, have_path = e[PE_PATH] > 0.5;
    double a_idm = 0.0;
    if (c->elead >= 0) {
        double g = c->egap > 0.5 ? c->egap : 0.5, a = cfg[PC_idm_a], b = cfg[PC_idm_b];
        double ss = cfg[PC_idm_s0_m] + e[PE_V] * cfg[PC_idm_t_s] + e[PE_V] * (e[PE_V] - lv) / (2.0 * sqrt(a * b));
        if (ss < cfg[PC_idm_s0_m])
            ss = cfg[PC_idm_s0_m];
        a_idm = p_clip(a * (1.0 - p_sq(ss / g)), -cfg[PC_brake_max], 0.3);
    }
    double x, y, th, kap;
    pp_eval(&c->base, 0.0, &x, &y, &th, &kap);
    double l_yaw = exp(p_logn(e[PE_W] - kap * e[PE_V], 0.0, 0.05, 0.05));
    double p_path = have_path ? cfg[PC_ego_path_p] * l_yaw / (cfg[PC_ego_path_p] * l_yaw + 1.0 - cfg[PC_ego_path_p]) : 0.0;
    if (p_path < 0.05 && have_path)
        p_path = 0.05;
    double lf = exp(p_logn(e[PE_A], a_idm, cfg[PC_acc_sd], cfg[PC_acc_sd])), l0 = exp(p_logn(e[PE_A], 0.0, cfg[PC_acc_sd], cfg[PC_acc_sd]));
    double p_follow = c->elead >= 0 && cfg[PC_interact] > 0.5 ? lf / (lf + l0) : 0.0;
    for (int which = 0; which < 3; which++) {                   /* along the road, the same behind the one ahead, by its own motion */
        PRoll r;
        roll_defaults(&r, cfg, CAR);
        double p;
        if (which == 0) {
            p = (have_path ? p_path : 1.0) * (1.0 - p_follow);
        } else if (which == 1) {
            p = (have_path ? p_path : 1.0) * p_follow;
        } else {
            p = have_path ? 1.0 - p_path : 0.0;
        }
        if (!(p > 0.0))
            continue;
        if (which < 2) {
            r.path = &c->base;
            r.known_lo = c->known_lo;
            r.known_hi = c->known_hi;
            r.s0 = 0.0;
            r.d0 = d0;
            r.d1 = have_path ? 0.3 * d0 : d0;
            r.t_dur = cfg[PC_keep_return_s];
            if (which == 1) {
                r.lead_s = c->obj[c->elead].sprof;
                r.lead_v = c->obj[c->elead].vprof;
                r.lead_len = c->obj[c->elead].len;
                r.k_v = 1.0 / cfg[PC_follow_v_tau_s];
            }
            c->ename[n] = !have_path ? (which == 1 ? PN_EGO_FOLLOW : PN_EGO_CTRA) : which == 1 ? PN_EGO_FOLLOW : PN_EGO_PATH;
        } else {
            pp_base(tmp, NULL, 0, e[PE_V], e[PE_W], 10.0, cfg[PC_path_ahead_m]);
            r.path = tmp;
            r.t_dur = 1.0;
            c->ename[n] = PN_EGO_CTRA;
        }
        r.v0 = e[PE_V] > 0.0 ? e[PE_V] : 0.0;
        r.a0 = e[PE_A];
        r.len = cfg[PC_ego_len];
        r.s0 += 0.0;
        r.pss = r.pdd = 0.01;
        r.pvv = 0.04;
        r.paa = 0.25;
        r.prr = 0.01;
        /* what those behind the car follow: its run in its first future and in a braking one of weight; the
         * car's reference point for the others is its centre */
        double ts[P_STEPS_MAX + 1], tv[P_STEPS_MAX + 1];
        int prof = which < 2 && (n == 0 || p >= 0.15);
        roll_out(&r, cfg, c->times, c->nt, c->h, c->nstep, &c->etraj[n][0][0], prof ? ts : NULL, prof ? tv : NULL);
        if (prof)
            for (int k = 0; k <= c->nstep; k++)
                if (n == 0 || ts[k] + cfg[PC_ego_centre_x] < c->esprof[k]) {
                    c->esprof[k] = ts[k] + cfg[PC_ego_centre_x];
                    c->evprof[k] = tv[k];
                }
        c->epri[n] = p;
        c->eyield[n] = which == 1;
        traj_bb(&c->etraj[n][0][0], c->nt, c->ebb[n]);
        n++;
    }
    c->enm = n;
}

/* ====================================================================================================
 * Interaction: which futures of two road users cannot both happen, who gives way, and the joint weights
 * ==================================================================================================== */
typedef struct {
    int nm;
    const double *pri, *rank;
    const int *yield;
    const double *traj;             /* [nm][nt][6], stride given */
    size_t stride;
    const double (*bb)[4];
    double len, wid, exist;
} PAgent;

static void agent_of(const PCtx *c, int a, PAgent *g)
{
    static const double ERANK[P_EMODES] = {3.0, 3.0, 3.0};
    if (a == c->n) {
        g->nm = c->enm;
        g->pri = c->epri;
        g->rank = ERANK;
        g->yield = c->eyield;
        g->traj = &c->etraj[0][0][0];
        g->stride = (size_t)P_NT_MAX * 6;
        g->bb = c->ebb;
        g->len = c->cfg[PC_ego_len];
        g->wid = c->cfg[PC_ego_wid];
        g->exist = 1.0;
    } else {
        const PObj *q = c->obj + a;
        g->nm = q->nm;
        g->pri = q->pri;
        g->rank = q->rank;
        g->yield = q->yield;
        g->traj = q->traj;
        g->stride = (size_t)c->nt * 6;
        g->bb = q->bb;
        g->len = q->len;
        g->wid = q->wid;
        g->exist = q->exist;
    }
}

/* the direction of a future at sample k (0: it does not move there) */
static int traj_dir(const double *t, int nt, int k, double *cx, double *sy)
{
    int a = k > 0 ? k - 1 : 0, b = k + 1 < nt ? k + 1 : nt - 1;
    double dx = t[6 * b] - t[6 * a], dy = t[6 * b + 1] - t[6 * a + 1], n = hypot(dx, dy);
    if (n < 0.15) {
        *cx = 1.0;
        *sy = 0.0;
        return 0;
    }
    *cx = dx / n;
    *sy = dy / n;
    return 1;
}

/* The probability that two futures meet: at each sample the two places are Gaussian; the pair is in conflict when
 * the other's centre lies inside the rectangle the two bodies cover together, lengthened behind a road user that
 * drives the same way by the gap the follower keeps. The largest over the horizon. *now: the plain overlap of the
 * bodies at the first sample (two tracks that already lie on each other say nothing about the future). */
static double pair_conflict(const PCtx *c, const PAgent *A, int ma, const PAgent *B, int mb, double *now)
{
    const double *ta = A->traj + ma * A->stride, *tb = B->traj + mb * B->stride;
    double hw = c->cfg[PC_conflict_headway_s], best = 0.0;
    int nt = c->nt;
    *now = 0.0;
    for (int k = 0; k < nt; k++) {
        const double *pa = ta + 6 * k, *pb = tb + 6 * k;
        double dx = pb[0] - pa[0], dy = pb[1] - pa[1];
        double sxx = pa[2] * pa[2] + pb[2] * pb[2], syy = pa[3] * pa[3] + pb[3] * pb[3];
        double vmax = pa[5] > pb[5] ? pa[5] : pb[5];
        double reach = 0.5 * (A->len + B->len) + hw * vmax + 1.0;
        if (k > 0 && (fabs(dx) > reach + 3.5 * sqrt(sxx) || fabs(dy) > reach + 3.5 * sqrt(syy)))
            continue;
        double sxy = pa[4] * pa[2] * pa[3] + pb[4] * pb[2] * pb[3];
        double ca, sa, cb, sb;
        int mva = traj_dir(ta, nt, k, &ca, &sa), mvb = traj_dir(tb, nt, k, &cb, &sb);
        double lb = B->len, wb = B->wid, la = A->len, wa = A->wid;
        if (!mva && mvb) {                                      /* the frame of the one that moves */
            ca = cb, sa = sb;
            dx = -dx, dy = -dy;
            double t = la;
            la = lb, lb = t;
            t = wa, wa = wb, wb = t;
        }
        double cr = mva && mvb ? ca * cb + sa * sb : 1.0, sr = mva && mvb ? fabs(ca * sb - sa * cb) : 0.0;
        if (!mva && !mvb) {
            lb = wb = 0.5 * (B->len + B->wid);
            la = wa = 0.5 * (A->len + A->wid);
        } else if (!(mva && mvb)) {
            lb = wb = 0.5 * (lb + wb);                          /* a standing body: which way it points is not known */
        }
        double acr = fabs(cr);
        double Lh = 0.5 * la + 0.5 * (acr * lb + sr * wb), W = 0.5 * wa + 0.5 * (sr * lb + acr * wb) + 0.2;
        double dl = dx * ca + dy * sa, dw = -dx * sa + dy * ca;
        double sl = sqrt(ca * ca * sxx + 2.0 * ca * sa * sxy + sa * sa * syy + 1e-4);
        double sw = sqrt(sa * sa * sxx - 2.0 * ca * sa * sxy + ca * ca * syy + 1e-4);
        double Lp = Lh, Lm = Lh;
        if (mva && mvb && cr > 0.7) {
            double va = !mva && mvb ? pb[5] : pa[5], vb = !mva && mvb ? pa[5] : pb[5];
            Lp += hw * va;                                      /* the other ahead: this one's gap */
            Lm += hw * vb;
        }
        if (k == 0) {
            *now = (p_phi((Lh - dl) / sl) - p_phi((-Lh - dl) / sl)) * (p_phi((W - dw) / sw) - p_phi((-W - dw) / sw));
            continue;
        }
        if (fabs(dl) > (Lp > Lm ? Lp : Lm) + 4.0 * sl || fabs(dw) > W + 4.0 * sw)
            continue;
        double P = (p_phi((Lp - dl) / sl) - p_phi((-Lm - dl) / sl)) * (p_phi((W - dw) / sw) - p_phi((-W - dw) / sw));
        if (P > best)
            best = P;
    }
    return best;
}

/* who has priority when future ma of A and future mb of B meet: +1 A, -1 B, 0 nobody in particular */
static int pair_priority(const PCtx *c, const PAgent *A, int ma, const PAgent *B, int mb)
{
    double ra = A->rank[ma], rb = B->rank[mb];
    if (ra > rb + 0.25)
        return 1;
    if (rb > ra + 0.25)
        return -1;
    const double *ta = A->traj + ma * A->stride, *tb = B->traj + mb * B->stride;
    double ca, sa, cb, sb;
    int mva = traj_dir(ta, c->nt, 1, &ca, &sa), mvb = traj_dir(tb, c->nt, 1, &cb, &sb);
    if (!mva || !mvb)
        return 0;
    double dx = tb[0] - ta[0], dy = tb[1] - ta[1];
    if (ca * cb + sa * sb > 0.5)                                /* the same way: the one in front */
        return dx * ca + dy * sa > 0.0 ? -1 : 1;
    if (ca * cb + sa * sb < -0.5)
        return 0;
    double side = (ca * dy - sa * dx) * c->rh;                  /* > 0: B is on A's left: A comes from B's right */
    return side > 0.0 ? 1 : -1;
}

/* the table of a pair: psi[ma * P_KMAX + mb]; returns 0 when the two do not touch each other's futures */
static int pair_table(const PCtx *c, int a, int b, double *psi)
{
    PAgent A, B;
    agent_of(c, a, &A);
    agent_of(c, b, &B);
    double conf[P_KMAX][P_KMAX];
    int any = 0;
    for (int i = 0; i < A.nm; i++)
        for (int j = 0; j < B.nm; j++) {
            conf[i][j] = 0.0;
            const double *ba = A.bb[i], *bq = B.bb[j];
            /* (the boxes hold three sigma of the places; what is added: the bodies and the gap the faster one keeps) */
            const double *fa = A.traj + i * A.stride, *fb = B.traj + j * B.stride;
            int last = 6 * (c->nt - 1) + 5;
            double vm = fa[5] > fa[last] ? fa[5] : fa[last];
            if (fb[5] > vm)
                vm = fb[5];
            if (fb[last] > vm)
                vm = fb[last];
            double m = 0.5 * (A.len + B.len) + vm * c->cfg[PC_conflict_headway_s] + 1.0;
            if (ba[0] > bq[2] + m || bq[0] > ba[2] + m || ba[1] > bq[3] + m || bq[1] > ba[3] + m)
                continue;
            double now;
            double v = pair_conflict(c, &A, i, &B, j, &now);
            if (now > 0.3)
                return 0;                                       /* they lie on each other now: no statement */
            conf[i][j] = v;
            if (v > 0.02)
                any = 1;
        }
    if (!any)
        return 0;
    double g = c->cfg[PC_conflict_gamma] * A.exist * B.exist, row = c->cfg[PC_row_wrong];
    for (int i = 0; i < A.nm; i++)
        for (int j = 0; j < B.nm; j++) {
            double f = 1.0 - g * conf[i][j];
            if (A.yield[i] && !B.yield[j]) {                    /* A gives way to this future of B: does it have to? */
                double cg = 0.0;
                int gi = -1;
                for (int k = 0; k < A.nm; k++)
                    if (!A.yield[k] && conf[k][j] > cg) {
                        cg = conf[k][j];
                        gi = k;
                    }
                if (gi >= 0 && cg > 0.2 && pair_priority(c, &A, gi, &B, j) > 0)
                    f *= 1.0 - (1.0 - row) * cg;
            }
            if (B.yield[j] && !A.yield[i]) {
                double cg = 0.0;
                int gj = -1;
                for (int k = 0; k < B.nm; k++)
                    if (!B.yield[k] && conf[i][k] > cg) {
                        cg = conf[i][k];
                        gj = k;
                    }
                if (gj >= 0 && cg > 0.2 && pair_priority(c, &A, i, &B, gj) < 0)
                    f *= 1.0 - (1.0 - row) * cg;
            }
            psi[i * P_KMAX + j] = f < 1e-6 ? 1e-6 : f;
        }
    return 1;
}

#define P_CL 16
typedef struct {
    int m, K[P_CL], ag[P_CL], sel[P_CL];
    const double *pri[P_CL];
    const double *tab[P_CL][P_CL];      /* tab[d][e] for e < d; NULL: no factor */
    int flip[P_CL][P_CL];               /* the table's first index is e's */
    double marg[P_CL][P_KMAX], best, prune;
    long nodes, budget;
} PJoint;

static void joint_dfs(PJoint *J, int d, double w)
{
    if (J->nodes > J->budget)
        return;
    J->nodes++;
    if (d == J->m) {
        for (int e = 0; e < J->m; e++)
            J->marg[e][J->sel[e]] += w;
        if (w > J->best)
            J->best = w;
        return;
    }
    for (int a = 0; a < J->K[d]; a++) {
        double ww = w * J->pri[d][a];
        for (int e = 0; e < d; e++) {
            const double *t = J->tab[d][e];
            if (t != NULL)
                ww *= J->flip[d][e] ? t[J->sel[e] * P_KMAX + a] : t[a * P_KMAX + J->sel[e]];
        }
        if (ww < J->prune * J->best)
            continue;
        J->sel[d] = a;
        joint_dfs(J, d + 1, ww);
    }
}

/* Joint weights of a group of interacting road users: P(m_1..m_k) ~ prod prior_i(m_i) prod psi_ij(m_i, m_j); each
 * one's marginal into out[agent][mode]. Exactly, by a depth-first search over the combinations with what cannot
 * matter cut off, while the group and the search stay small; otherwise by belief propagation over the pairs. */
static void joint_solve(PCtx *c, const int *ag, int m, double *const *tab, int nag, double (*out)[P_KMAX])
{
    const double *cfg = c->cfg;
    PAgent G[P_AGENTS_MAX];
    for (int d = 0; d < m; d++)
        agent_of(c, ag[d], &G[d]);
    int done = 0;
    if (m <= (int)cfg[PC_clique_max] && m <= P_CL) {
        PJoint J;
        J.m = m;
        J.nodes = 0;
        J.budget = (long)cfg[PC_joint_nodes];
        J.prune = cfg[PC_joint_prune];
        double w0 = 1.0;
        for (int d = 0; d < m; d++) {
            J.ag[d] = ag[d];
            J.K[d] = G[d].nm;
            J.pri[d] = G[d].pri;
            J.sel[d] = 0;
            w0 *= G[d].pri[0];
            for (int k = 0; k < P_KMAX; k++)
                J.marg[d][k] = 0.0;
            for (int e = 0; e < d; e++) {
                int lo = ag[d] < ag[e] ? ag[d] : ag[e], hi = ag[d] < ag[e] ? ag[e] : ag[d];
                J.tab[d][e] = tab[lo * nag + hi];
                J.flip[d][e] = ag[e] == lo;
                if (J.tab[d][e] != NULL)
                    w0 *= J.tab[d][e][0];
            }
        }
        J.best = w0;                                            /* every road user's first future together */
        joint_dfs(&J, 0, 1.0);
        c->stat[3] += (double)J.nodes;
        if (J.nodes <= J.budget) {
            for (int d = 0; d < m; d++) {
                double s = 0.0;
                for (int k = 0; k < J.K[d]; k++)
                    s += J.marg[d][k];
                for (int k = 0; k < J.K[d]; k++)
                    out[d][k] = s > 0.0 ? J.marg[d][k] / s : G[d].pri[k];
            }
            done = 1;
        }
    }
    if (done)
        return;
    /* belief propagation (sum-product, damped) on the pairs of the group */
    c->stat[4] += 1.0;
    int ne = 0;
    for (int d = 0; d < m; d++)
        for (int e = 0; e < d; e++) {
            int lo = ag[d] < ag[e] ? ag[d] : ag[e], hi = ag[d] < ag[e] ? ag[e] : ag[d];
            if (tab[lo * nag + hi] != NULL)
                ne++;
        }
    double *msg = (double *)PyMem_Malloc((size_t)(ne > 0 ? ne : 1) * 2 * P_KMAX * sizeof(double));
    int *ed = (int *)PyMem_Malloc((size_t)(ne > 0 ? ne : 1) * 2 * sizeof(int));
    if (msg == NULL || ed == NULL) {
        PyMem_Free(msg);
        PyMem_Free(ed);
        for (int d = 0; d < m; d++)
            for (int k = 0; k < G[d].nm; k++)
                out[d][k] = G[d].pri[k];
        return;
    }
    ne = 0;
    for (int d = 0; d < m; d++)
        for (int e = 0; e < d; e++) {
            int lo = ag[d] < ag[e] ? ag[d] : ag[e], hi = ag[d] < ag[e] ? ag[e] : ag[d];
            if (tab[lo * nag + hi] == NULL)
                continue;
            ed[2 * ne] = ag[d] == lo ? d : e;                   /* the table's first index */
            ed[2 * ne + 1] = ag[d] == lo ? e : d;
            ne++;
        }
    for (int i = 0; i < ne * 2 * P_KMAX; i++)
        msg[i] = 1.0;
    double bel[P_AGENTS_MAX][P_KMAX];
    for (int it = 0; it < 15; it++) {
        for (int d = 0; d < m; d++)
            for (int k = 0; k < G[d].nm; k++)
                bel[d][k] = G[d].pri[k];
        for (int x = 0; x < ne; x++) {
            int u = ed[2 * x], v = ed[2 * x + 1];
            for (int k = 0; k < G[v].nm; k++)
                bel[v][k] *= msg[(2 * x) * P_KMAX + k];         /* u -> v */
            for (int k = 0; k < G[u].nm; k++)
                bel[u][k] *= msg[(2 * x + 1) * P_KMAX + k];     /* v -> u */
        }
        if (it == 14)
            break;
        for (int x = 0; x < ne; x++) {
            int u = ed[2 * x], v = ed[2 * x + 1];
            const double *t = tab[ag[u] * nag + ag[v]];
            double nu[P_KMAX], nv[P_KMAX], su = 0.0, sv = 0.0;
            for (int k = 0; k < G[v].nm; k++) {                 /* u -> v */
                double acc = 0.0;
                for (int a = 0; a < G[u].nm; a++)
                    acc += bel[u][a] / msg[(2 * x + 1) * P_KMAX + a] * t[a * P_KMAX + k];
                nv[k] = acc;
                sv += acc;
            }
            for (int a = 0; a < G[u].nm; a++) {                 /* v -> u */
                double acc = 0.0;
                for (int k = 0; k < G[v].nm; k++)
                    acc += bel[v][k] / msg[(2 * x) * P_KMAX + k] * t[a * P_KMAX + k];
                nu[a] = acc;
                su += acc;
            }
            for (int k = 0; k < G[v].nm; k++) {
                double z = sv > 0.0 ? nv[k] / sv * G[v].nm : 1.0;
                msg[(2 * x) * P_KMAX + k] = 0.5 * msg[(2 * x) * P_KMAX + k] + 0.5 * (z > 1e-9 ? z : 1e-9);
            }
            for (int a = 0; a < G[u].nm; a++) {
                double z = su > 0.0 ? nu[a] / su * G[u].nm : 1.0;
                msg[(2 * x + 1) * P_KMAX + a] = 0.5 * msg[(2 * x + 1) * P_KMAX + a] + 0.5 * (z > 1e-9 ? z : 1e-9);
            }
        }
    }
    for (int d = 0; d < m; d++) {
        double s = 0.0;
        for (int k = 0; k < G[d].nm; k++)
            s += bel[d][k];
        for (int k = 0; k < G[d].nm; k++)
            out[d][k] = s > 0.0 ? bel[d][k] / s : G[d].pri[k];
    }
    PyMem_Free(msg);
    PyMem_Free(ed);
}

/* p (with the car's own futures among the others) and p0 (without the car: nobody is taken to give way to it) */
static int interact(PCtx *c, double *mp, double *mp0)
{
    int n = c->n, nag = n + 1;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < P_KMAX; k++)
            mp[i * P_KMAX + k] = mp0[i * P_KMAX + k] = k < c->obj[i].nm ? c->obj[i].pri[k] : 0.0;
    if (!(c->cfg[PC_interact] > 0.5 && c->cfg[PC_joint] > 0.5) || n == 0)
        return 0;
    double **tab = (double **)PyMem_Calloc((size_t)nag * nag, sizeof(double *));
    int *comp = (int *)PyMem_Malloc((size_t)nag * 3 * sizeof(int));
    if (tab == NULL || comp == NULL) {
        PyMem_Free(tab);
        PyMem_Free(comp);
        PyErr_NoMemory();
        return -1;
    }
    int *root = comp + nag, *list = comp + 2 * nag, rc = 0;
    for (int a = 0; a < nag; a++)
        comp[a] = a;
    double psi[P_KMAX * P_KMAX];
    for (int a = 0; a < nag && rc == 0; a++)
        for (int b = a + 1; b < nag; b++) {
            if (b < n && c->obj[a].nm == 1 && c->obj[b].nm == 1)
                continue;                                       /* neither has a choice */
            if (!pair_table(c, a, b, psi))
                continue;
            double *t = (double *)PyMem_Malloc(sizeof psi);
            if (t == NULL) {
                PyErr_NoMemory();
                rc = -1;
                break;
            }
            memcpy(t, psi, sizeof psi);
            tab[a * nag + b] = t;
            c->stat[0] += 1.0;
            int ra = a, rb = b;
            while (comp[ra] != ra)
                ra = comp[ra];
            while (comp[rb] != rb)
                rb = comp[rb];
            if (ra != rb)
                comp[rb] = ra;
        }
    for (int a = 0; a < nag; a++) {
        int r = a;
        while (comp[r] != r)
            r = comp[r];
        root[a] = r;
    }
    double out[P_AGENTS_MAX][P_KMAX];
    for (int r = 0; r < nag && rc == 0; r++) {
        int m = 0, has_ego = 0;
        for (int a = 0; a < nag; a++)
            if (root[a] == r) {
                list[m++] = a;
                has_ego |= a == n;
            }
        if (m < 2)
            continue;
        c->stat[1] += 1.0;
        if (m > c->stat[2])
            c->stat[2] = m;
        joint_solve(c, list, m, tab, nag, out);
        for (int d = 0; d < m; d++)
            if (list[d] < n)
                for (int k = 0; k < c->obj[list[d]].nm; k++)
                    mp[list[d] * P_KMAX + k] = out[d][k];
        if (has_ego) {
            if (m > 2)
                joint_solve(c, list, m - 1, tab, nag, out);     /* the car is the last of the list */
            for (int d = 0; d < m - 1 && m > 2; d++)
                for (int k = 0; k < c->obj[list[d]].nm; k++)
                    mp0[list[d] * P_KMAX + k] = out[d][k];
        } else {
            for (int d = 0; d < m; d++)
                for (int k = 0; k < c->obj[list[d]].nm; k++)
                    mp0[list[d] * P_KMAX + k] = out[d][k];
        }
    }
    for (int i = 0; i < nag * nag; i++)
        PyMem_Free(tab[i]);
    PyMem_Free(tab);
    PyMem_Free(comp);
    return rc;
}

/* ====================================================================================================
 * The module's functions
 * ==================================================================================================== */
/* a C-contiguous float64 buffer of rows of `width` numbers (any shape); the number of rows into *rows */
static int pview(PyObject *o, View *v, int writable, int width, Py_ssize_t *rows, const char *what)
{
    v->held = 0;
    if (PyObject_GetBuffer(o, &v->b, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0)) < 0)
        return -1;
    v->held = 1;
    const char *f = v->b.format ? v->b.format : "B";
    if (*f == '@' || *f == '=' || *f == '<')
        f++;
    Py_ssize_t nd = v->b.len / (Py_ssize_t)sizeof(double);
    if (!(f[0] == 'd' && f[1] == 0 && v->b.itemsize == 8) || nd % width != 0) {
        view_drop(v);
        PyErr_Format(PyExc_TypeError, "predict: %s: a C-contiguous float64 array of rows of %d is needed", what, width);
        return -1;
    }
    *rows = nd / width;
    return 0;
}

static PyObject *n_predict_cfg(PyObject *self, PyObject *noargs)
{
    PyObject *d = PyDict_New();
    if (d == NULL)
        return NULL;
    for (int i = 0; i < PC_N; i++) {
        PyObject *v = PyFloat_FromDouble(PCFG_DEF[i]);
        if (v == NULL || PyDict_SetItemString(d, PCFG_NAME[i], v) < 0) {
            Py_XDECREF(v);
            Py_DECREF(d);
            return NULL;
        }
        Py_DECREF(v);
    }
    return d;
}

static PyObject *names_tuple(const char *const *names, int n)
{
    PyObject *t = PyTuple_New(n);
    if (t == NULL)
        return NULL;
    for (int i = 0; i < n; i++) {
        PyObject *s = PyUnicode_FromString(names[i]);
        if (s == NULL) {
            Py_DECREF(t);
            return NULL;
        }
        PyTuple_SetItem(t, i, s);
    }
    return t;
}

static PyObject *n_predict_names(PyObject *self, PyObject *noargs)
{
    PyObject *a = names_tuple(PNAME, P_NNAME), *b = names_tuple(PLANE, P_NLANE), *s = names_tuple(PSLOT, P_NM);
    if (a == NULL || b == NULL || s == NULL) {
        Py_XDECREF(a);
        Py_XDECREF(b);
        Py_XDECREF(s);
        return NULL;
    }
    return Py_BuildValue("(NNN(iiiiiiiiii))", a, b, s, P_NO, P_NS, P_NF, P_NR, P_NB, P_NA, P_NE, P_NK, P_KMAX, P_EMODES);
}

static int cmp_desc(const void *a, const void *b)
{
    double x = ((const double *)a)[0], y = ((const double *)b)[0];
    return x < y ? 1 : x > y ? -1 : 0;
}

/* predict_frame(cfg, times, ego, road, bands, arm_meta, arm_pts, xing, centre, kin, objs, state, T,
 *               lane, intent, feat, modes, traj, ego_out, ego_traj, stats) -> number of ego futures
 *   objs (n, P_NO), state (n, P_NS) read and written, lane (n, 7), intent (n, P_NM), feat (n, P_NF),
 *   modes (n, 5, P_KMAX): rows count (first value) / slot / name / p / p0, traj (n, P_KMAX, nt, 6),
 *   ego_out (P_EMODES, 2): p, name; ego_traj (P_EMODES, nt, 3): x, y, v; stats (8) */
static PyObject *n_predict_frame(PyObject *self, PyObject *args)
{
    PyObject *o[20];
    double T;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOdOOOOOOOO", &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &o[7], &o[8],
                          &o[9], &o[10], &o[11], &T, &o[12], &o[13], &o[14], &o[15], &o[16], &o[17], &o[18], &o[19]))
        return NULL;
    static const int WIDTH[20] = {PC_N, 1, P_NE, P_NR, P_NB, P_NA, 2, 2, 2, P_NK, P_NO, P_NS, P_NLANE, P_NM, P_NF,
                                  5 * P_KMAX, 6, 2, 3, 8};
    static const int WR[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    static const char *const WHAT[20] = {"cfg", "times", "ego", "road", "bands", "arm_meta", "arm_pts", "xing", "centre",
                                         "kin", "objs", "state", "lane", "intent", "feat", "modes", "traj", "ego_out",
                                         "ego_traj", "stats"};
    View v[20];
    Py_ssize_t rows[20];
    PyObject *res = NULL;
    PCtx *c = NULL;
    PPath *tmp = NULL;
    double *key = NULL;
    for (int i = 0; i < 20; i++)
        v[i].held = 0;
    for (int i = 0; i < 20; i++)
        if (pview(o[i], &v[i], WR[i], WIDTH[i], &rows[i], WHAT[i]) < 0)
            goto done;
    Py_ssize_t n = rows[10], nt = rows[1], na = rows[5];
    if (rows[0] != 1 || rows[2] != 1 || rows[3] != 1 || rows[9] < 9 || rows[19] != 1 || nt < 1 || nt > P_NT_MAX
        || n > P_AGENTS_MAX - 1 || na > 32 || rows[11] != n || rows[12] != n || rows[13] != n || rows[14] != n
        || rows[15] != n || rows[16] != n * P_KMAX * nt || rows[17] != P_EMODES || rows[18] != P_EMODES * nt) {
        PyErr_SetString(PyExc_ValueError, "predict_frame: array sizes do not fit together");
        goto done;
    }
    c = (PCtx *)PyMem_Calloc(1, sizeof(PCtx));
    tmp = (PPath *)PyMem_Malloc(sizeof(PPath));
    if (c == NULL || tmp == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    c->cfg = (const double *)v[0].b.buf;
    c->times = (const double *)v[1].b.buf;
    c->ego = (const double *)v[2].b.buf;
    c->road = (const double *)v[3].b.buf;
    c->bands = (const double *)v[4].b.buf;
    c->ameta = (const double *)v[5].b.buf;
    c->xing = (const double *)v[7].b.buf;
    c->kin = (const double *)v[9].b.buf;
    c->nb = (int)rows[4];
    c->na = (int)na;
    c->nx = (int)rows[7];
    c->nt = (int)nt;
    c->n = (int)n;
    c->w = c->road[PR_W] > 2.0 ? c->road[PR_W] : 3.5;
    c->rh = c->cfg[PC_right_hand] >= 0.0 ? 1.0 : -1.0;
    c->h = c->cfg[PC_step_s] > 0.02 ? c->cfg[PC_step_s] : 0.25;
    for (Py_ssize_t k = 1; k < nt; k++)
        if (!(c->times[k] > c->times[k - 1]) || !(c->times[0] >= 0.0)) {
            PyErr_SetString(PyExc_ValueError, "predict_frame: times must rise");
            goto done;
        }
    c->nstep = (int)ceil(c->times[nt - 1] / c->h - 1e-9);
    if (c->nstep > P_STEPS_MAX) {
        c->nstep = P_STEPS_MAX;
        c->h = c->times[nt - 1] / P_STEPS_MAX;
    }
    if (c->nstep < 1)
        c->nstep = 1;
    pp_base(&c->base, (const double *)v[8].b.buf, (int)rows[8], c->ego[PE_V], c->ego[PE_W], c->cfg[PC_path_back_m],
            c->cfg[PC_path_ahead_m]);
    {
        double s0, d0;
        const double *ce = (const double *)v[8].b.buf;
        pp_project(&c->base, 0.0, 0.0, &s0, &c->ed0);
        if (rows[8] >= 2) {                                      /* the stretch the centre line was given for */
            pp_project(&c->base, ce[0], ce[1], &c->known_lo, &d0);
            pp_project(&c->base, ce[2 * (rows[8] - 1)], ce[2 * (rows[8] - 1) + 1], &c->known_hi, &d0);
        } else {                                                /* the car's own arc: good for the next seconds */
            c->known_lo = -5.0;
            c->known_hi = 10.0 + 2.0 * c->ego[PE_V];
        }
    }
    if (na > 0) {
        c->arm = (PPath *)PyMem_Malloc((size_t)na * sizeof(PPath));
        if (c->arm == NULL) {
            PyErr_NoMemory();
            goto done;
        }
        const double *pts = (const double *)v[6].b.buf;
        for (int a = 0; a < na; a++) {
            const double *A = c->ameta + P_NA * a;
            Py_ssize_t i0 = (Py_ssize_t)A[PA_I0], np = (Py_ssize_t)A[PA_N];
            if (i0 < 0 || np < 2 || i0 + np > rows[6]) {
                PyErr_SetString(PyExc_ValueError, "predict_frame: an arm's points are not in arm_pts");
                goto done;
            }
            c->arm[a].n = 0;
            for (Py_ssize_t k = 0; k < np; k++)
                pp_add(&c->arm[a], pts[2 * (i0 + k)], pts[2 * (i0 + k) + 1]);
            pp_finish(&c->arm[a], 0.0);
            double d;
            if (A[PA_SJ] == A[PA_SJ])
                c->asj[a] = A[PA_SJ];
            else
                pp_project(&c->base, c->arm[a].x[0], c->arm[a].y[0], &c->asj[a], &d);
        }
    }
    double *objs = (double *)v[10].b.buf, *state = (double *)v[11].b.buf, *lane = (double *)v[12].b.buf;
    double *intent = (double *)v[13].b.buf, *feat = (double *)v[14].b.buf, *modes = (double *)v[15].b.buf;
    double *traj = (double *)v[16].b.buf, *eout = (double *)v[17].b.buf, *etraj = (double *)v[18].b.buf;
    double *stats = (double *)v[19].b.buf;
    if (n > 0) {
        c->obj = (PObj *)PyMem_Calloc((size_t)n, sizeof(PObj));
        key = (double *)PyMem_Malloc((size_t)n * 2 * sizeof(double));
        if (c->obj == NULL || key == NULL) {
            PyErr_NoMemory();
            goto done;
        }
    }
    for (int i = 0; i < n; i++) {
        PObj *q = c->obj + i;
        q->o = objs + P_NO * i;
        q->st = state + P_NS * i;
        int kind = (int)q->o[PO_KIND];
        if (kind < 0 || kind > 8)
            kind = 8;
        q->kin = c->kin + P_NK * kind;
        q->grp = kind == 7 ? 2 : kind == 6 ? 1 : 0;
        q->len = q->o[PO_LEN] == q->o[PO_LEN] && q->o[PO_LEN] > 0.2 ? q->o[PO_LEN] : q->kin[PK_LEN];
        q->wid = q->o[PO_WID] == q->o[PO_WID] && q->o[PO_WID] > 0.2 ? q->o[PO_WID] : q->kin[PK_WID];
        q->pstat = motion_state(c->cfg, q->o, &q->minfo);
        q->exist = p_clip(q->o[PO_EXIST], 0.0, 1.0);
        q->traj = traj + (size_t)i * P_KMAX * nt * 6;
        double dt = T - q->st[PS_T];
        if (!(q->st[PS_INIT] > 0.5) || !(dt >= 0.0) || dt > 2.0) {
            q->st[PS_INIT] = 0.0;
            dt = 0.0;
        }
        lane_step(c, q, dt, lane + P_NLANE * i);
        feat_step(c, q, dt);
        for (int k = 0; k <= c->nstep; k++) {
            q->sprof[k] = q->s + q->vt * (1.0 - q->pstat) * k * c->h;
            q->vprof[k] = q->vt * (1.0 - q->pstat);
        }
    }
    for (int i = 0; i < n; i++)
        context_step(c, i);
    for (int i = 0; i < n; i++) {
        PObj *q = c->obj + i;
        double dt = q->st[PS_INIT] > 0.5 ? T - q->st[PS_T] : 0.0;
        intent_step(c, q, dt, intent + P_NM * i);
    }
    /* the futures, leaders before followers: along the road from the far end to the car, the car, those behind
     * it; then the oncoming ones from the near end */
    for (int i = 0; i < n; i++) {
        const PObj *q = c->obj + i;
        key[2 * i] = q->dir < 0 ? -1e6 - q->s : q->s;          /* the oncoming ones after all others, nearest the start first */
        key[2 * i + 1] = i;
    }
    if (n > 0)
        qsort(key, (size_t)n, 2 * sizeof(double), cmp_desc);
    int ego_done = 0;
    for (int a = 0; a < n; a++) {
        int i = (int)key[2 * a + 1];
        if (!ego_done && (c->obj[i].dir < 0 || c->obj[i].s <= c->cfg[PC_ego_centre_x])) {
            ego_modes(c, tmp);
            ego_done = 1;
        }
        object_modes(c, i, tmp);
    }
    if (!ego_done)
        ego_modes(c, tmp);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 5 * P_KMAX; k++)
            modes[(size_t)i * 5 * P_KMAX + k] = 0.0;
    {
        double *mp = (double *)PyMem_Malloc((size_t)(n > 0 ? n : 1) * 2 * P_KMAX * sizeof(double));
        if (mp == NULL) {
            PyErr_NoMemory();
            goto done;
        }
        int rc = interact(c, mp, mp + (size_t)n * P_KMAX);
        for (int i = 0; i < n && rc == 0; i++) {
            const PObj *q = c->obj + i;
            double *M = modes + (size_t)i * 5 * P_KMAX;
            M[0] = q->nm;
            for (int k = 0; k < q->nm; k++) {
                M[P_KMAX + k] = q->slot[k];
                M[2 * P_KMAX + k] = q->name[k];
                M[3 * P_KMAX + k] = mp[i * P_KMAX + k];
                M[4 * P_KMAX + k] = mp[(n + i) * P_KMAX + k];
            }
            c->stat[6] += q->nm;
        }
        PyMem_Free(mp);
        if (rc < 0)
            goto done;
    }
    for (int i = 0; i < n; i++) {
        PObj *q = c->obj + i;
        double *F = feat + P_NF * i;
        F[PF_S] = q->s, F[PF_D] = q->st[PS_D], F[PF_SD] = q->sd_d, F[PF_DC] = q->dc, F[PF_DIR] = q->dir, F[PF_LANE] = q->lane;
        F[PF_LEAD] = q->lead, F[PF_GAP] = q->gap < P_BIG ? q->gap : NAN, F[PF_E] = q->e, F[PF_R] = q->r, F[PF_A] = q->a;
        F[PF_PSI] = q->psi, F[PF_STOP] = q->stop_why, F[PF_JDIST] = q->jdist < P_BIG ? q->jdist : NAN, F[PF_KERB] = q->kerb;
        F[PF_OM] = q->om;
        q->st[PS_INIT] = 1.0;
        q->st[PS_T] = T;
        q->st[PS_TOP] = q->slot[0];
    }
    for (int m = 0; m < P_EMODES; m++) {
        eout[2 * m] = m < c->enm ? c->epri[m] : 0.0;
        eout[2 * m + 1] = m < c->enm ? c->ename[m] : -1.0;
        for (int k = 0; k < nt; k++) {
            etraj[((size_t)m * nt + k) * 3] = m < c->enm ? c->etraj[m][k][0] : NAN;
            etraj[((size_t)m * nt + k) * 3 + 1] = m < c->enm ? c->etraj[m][k][1] : NAN;
            etraj[((size_t)m * nt + k) * 3 + 2] = m < c->enm ? c->etraj[m][k][5] : NAN;
        }
    }
    c->stat[5] = (double)n;
    c->stat[7] = c->elead;
    memcpy(stats, c->stat, sizeof c->stat);
    res = PyLong_FromLong(c->enm);
done:
    if (c != NULL) {
        PyMem_Free(c->arm);
        PyMem_Free(c->obj);
    }
    PyMem_Free(c);
    PyMem_Free(tmp);
    PyMem_Free(key);
    for (int i = 0; i < 20; i++)
        view_drop(&v[i]);
    return res;
}

/* predict_score(traj (K, nt, 6), p (K), actual (nt, 2), valid (nt), out (nt, 6)) -> (winner, p of the winner)
 * A prediction against what the object did. Per sample (NaN where the object was not seen): the best future's
 * mean error up to it (minADE) and its error there (minFDE), the same of the most probable future, the negative
 * log-likelihood of the place under the futures' mixture, and the squared Mahalanobis distance to the future
 * that was nearest in the end (2 on average when the stated spread is right). The winner is that future. With two
 * more rows in `out`, they take every future's mean error (the first 12 futures). */
static PyObject *n_predict_score(PyObject *self, PyObject *args)
{
    PyObject *ot, *op, *oa, *ov, *oo;
    View vt, vp, va, vv, vo;
    Py_ssize_t rt, K, ra, nt, ro;
    PyObject *res = NULL;
    vt.held = vp.held = va.held = vv.held = vo.held = 0;
    if (!PyArg_ParseTuple(args, "OOOOO", &ot, &op, &oa, &ov, &oo))
        return NULL;
    if (pview(ot, &vt, 0, 6, &rt, "traj") < 0 || pview(op, &vp, 0, 1, &K, "p") < 0 || pview(oa, &va, 0, 2, &ra, "actual") < 0
        || pview(ov, &vv, 0, 1, &nt, "valid") < 0 || pview(oo, &vo, 1, 6, &ro, "out") < 0)
        goto done;
    if (K < 1 || K > 64 || nt < 1 || rt != K * nt || ra != nt || (ro != nt && ro != nt + 2)) {
        PyErr_SetString(PyExc_ValueError, "predict_score: array sizes do not fit together");
        goto done;
    }
    const double *t = (const double *)vt.b.buf, *p = (const double *)vp.b.buf, *act = (const double *)va.b.buf;
    const double *val = (const double *)vv.b.buf;
    double *out = (double *)vo.b.buf;
    double cum[64], last[64];
    int top = 0, cnt = 0, win = 0;
    for (int m = 0; m < K; m++) {
        cum[m] = 0.0;
        last[m] = NAN;
        if (p[m] > p[top])
            top = m;
    }
    for (Py_ssize_t k = 0; k < nt; k++) {
        double *O = out + 6 * k;
        if (!(val[k] > 0.5)) {
            for (int j = 0; j < 6; j++)
                O[j] = NAN;
            continue;
        }
        cnt++;
        double mix = 0.0, best_a = P_BIG, best_f = P_BIG;
        for (int m = 0; m < K; m++) {
            const double *s = t + ((size_t)m * nt + k) * 6;
            double dx = act[2 * k] - s[0], dy = act[2 * k + 1] - s[1], e = hypot(dx, dy);
            cum[m] += e;
            last[m] = e;
            if (cum[m] / cnt < best_a)
                best_a = cum[m] / cnt;
            if (e < best_f)
                best_f = e;
            double r = s[4], det = 1.0 - r * r;
            double zx = dx / s[2], zy = dy / s[3];
            double q = (zx * zx - 2.0 * r * zx * zy + zy * zy) / det;
            mix += p[m] * exp(-0.5 * (q < 1400.0 ? q : 1400.0)) / (2.0 * PI * s[2] * s[3] * sqrt(det));
        }
        O[0] = best_a;
        O[1] = best_f;
        O[2] = cum[top] / cnt;
        O[3] = last[top];
        O[4] = -log(mix > 1e-300 ? mix : 1e-300);
        O[5] = NAN;
    }
    if (cnt > 0) {
        for (int m = 1; m < K; m++)
            if (cum[m] < cum[win])
                win = m;
        for (Py_ssize_t k = 0; k < nt; k++) {
            if (!(val[k] > 0.5))
                continue;
            const double *s = t + ((size_t)win * nt + k) * 6;
            double dx = act[2 * k] - s[0], dy = act[2 * k + 1] - s[1], r = s[4];
            double zx = dx / s[2], zy = dy / s[3];
            out[6 * k + 5] = (zx * zx - 2.0 * r * zx * zy + zy * zy) / (1.0 - r * r);
        }
    }
    if (ro > nt)                                                 /* a further row: every future's mean error */
        for (int m = 0; m < 12; m++)
            out[6 * nt + m] = cnt > 0 && m < K ? cum[m] / cnt : NAN;
    res = Py_BuildValue("(id)", cnt > 0 ? win : -1, cnt > 0 ? p[win] : 0.0);
done:
    view_drop(&vt);
    view_drop(&vp);
    view_drop(&va);
    view_drop(&vv);
    view_drop(&vo);
    return res;
}

/* predict_futs(traj (n, P_KMAX, nt, 6), modes (n, 5, P_KMAX), names, decimals) -> [[future, ...] per object]
 * The message form of the futures: {"p", "p0", "m", "x", "y", "sx", "sy", "rxy", "v"}, numbers rounded as
 * round_lists rounds them (rint(v * 10^d) / 10^d; p and p0 to three decimals). */
static PyObject *n_predict_futs(PyObject *self, PyObject *args)
{
    PyObject *ot, *om, *names, *res = NULL;
    int dec;
    View vt, vm;
    Py_ssize_t rt, n;
    static const char *const KEYS[6] = {"x", "y", "sx", "sy", "rxy", "v"};
    vt.held = vm.held = 0;
    if (!PyArg_ParseTuple(args, "OOO!i", &ot, &om, &PyTuple_Type, &names, &dec))
        return NULL;
    if (pview(ot, &vt, 0, 6, &rt, "traj") < 0 || pview(om, &vm, 0, 5 * P_KMAX, &n, "modes") < 0)
        goto done;
    if (dec < 0 || dec > 9 || (n > 0 && rt % (n * P_KMAX) != 0) || (n > 0 && rt / (n * P_KMAX) < 1)) {
        PyErr_SetString(PyExc_ValueError, "predict_futs: array sizes do not fit together");
        goto done;
    }
    Py_ssize_t nt = n > 0 ? rt / (n * P_KMAX) : 0, nn = PyTuple_Size(names);
    const double *t = (const double *)vt.b.buf, *M = (const double *)vm.b.buf;
    double sc = POW10[dec];
    res = PyList_New(n);
    if (res == NULL)
        goto done;
    for (Py_ssize_t i = 0; i < n; i++) {
        const double *m = M + i * 5 * P_KMAX;
        int nm = (int)m[0];
        if (nm < 0 || nm > P_KMAX)
            nm = 0;
        PyObject *row = PyList_New(nm);
        if (row == NULL)
            goto fail;
        PyList_SetItem(res, i, row);
        for (int k = 0; k < nm; k++) {
            PyObject *d = PyDict_New();
            if (d == NULL)
                goto fail;
            PyList_SetItem(row, k, d);
            int code = (int)m[2 * P_KMAX + k];
            PyObject *v = PyFloat_FromDouble(rint(m[3 * P_KMAX + k] * 1000.0) / 1000.0);
            if (v == NULL || PyDict_SetItemString(d, "p", v) < 0) {
                Py_XDECREF(v);
                goto fail;
            }
            Py_DECREF(v);
            v = PyFloat_FromDouble(rint(m[4 * P_KMAX + k] * 1000.0) / 1000.0);
            if (v == NULL || PyDict_SetItemString(d, "p0", v) < 0) {
                Py_XDECREF(v);
                goto fail;
            }
            Py_DECREF(v);
            if (code < 0 || code >= nn || PyDict_SetItemString(d, "m", PyTuple_GetItem(names, code)) < 0) {
                if (!PyErr_Occurred())
                    PyErr_SetString(PyExc_ValueError, "predict_futs: a name code without a name");
                goto fail;
            }
            const double *tr = t + ((size_t)i * P_KMAX + k) * nt * 6;
            for (int q = 0; q < 6; q++) {
                PyObject *l = PyList_New(nt);
                if (l == NULL)
                    goto fail;
                for (Py_ssize_t j = 0; j < nt; j++) {
                    PyObject *f = PyFloat_FromDouble(rint(tr[6 * j + q] * sc) / sc);
                    if (f == NULL) {
                        Py_DECREF(l);
                        goto fail;
                    }
                    PyList_SetItem(l, j, f);
                }
                int rc = PyDict_SetItemString(d, KEYS[q], l);
                Py_DECREF(l);
                if (rc < 0)
                    goto fail;
            }
        }
    }
    goto done;
fail:
    Py_CLEAR(res);
done:
    view_drop(&vt);
    view_drop(&vm);
    return res;
}

#define PREDICT_METHODS                                                                                                \
    {"predict_cfg", n_predict_cfg, METH_NOARGS, "the predictor's constants: {name: default}, in the array's order"},  \
    {"predict_names", n_predict_names, METH_NOARGS, "(manoeuvre names, lane names, intent slots, array widths)"},     \
    {"predict_frame", n_predict_frame, METH_VARARGS, "one frame of the predictor: lanes, intents, futures"},          \
    {"predict_futs", n_predict_futs, METH_VARARGS, "the futures' message form: per object a list of dicts"},      \
    {"predict_score", n_predict_score, METH_VARARGS, "a past prediction against what the object did"},
