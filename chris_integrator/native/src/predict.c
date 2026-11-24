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
    X(ego_path_p, 0.95)         /* the car follows the road's path with this prior when one is known (0.8 until      \
                                   2026-11-24: a fifth of every frame's weight ran straight on through the bends) */             \
    X(right_hand, 1.0)          /* traffic keeps right (-1: left) */                                                \
    X(lane_los, 1.0)            /* 1: an object's place is spread along the line of sight (a camera's range), read   \
                                   at up to P_LOS ranges; 0: three points along the frame's longer axis */             \
    X(lane_mem, 0.0)            /* 1: the lane filter multiplies every frame's place into what it held (a range error \
                                   that lasts is counted again each evid_tau); 0: the track's place now is all the    \
                                   place evidence there is (the track is the sum of its sightings) */                 \
    X(lane_hyp, 1.0)            /* 1: a vehicle whose range leaves open whether it is in the car's lane or beside it  \
                                   gets futures of both places, each at the range that place means (needs lane_los) */ \
    X(lane_hyp_min, 0.02)       /* ... from this probability of the less likely place */                           \
    X(lane_hyp_stand, 0.3)      /* ... and of standing (a moving vehicle's futures are its lane's) */                 \
    X(ego_kap_back, 0.0)        /* the car's own futures: where the stretch starts over which the path's curvature is \
                                   compared with the yaw rate (ego_modes) */                                          \
    X(motion_scale, 1.0)        /* 1: "it moves" counts no further than its velocity shows once the range's scale is  \
                                   allowed for (motion_scale_bound) */                                                \
    X(motion_scale_rel, 0.15)   /* ... the range's relative error at least (what the tracker's ranges showed) */      \
    X(motion_scale_v, 8.0)      /* ... and the speeds of things that do move: 1 sigma (m/s) */                        \
    X(ped_gap, 1.0)             /* a pedestrian standing between or beside parked vehicles, in their line at the     \
                                   road's edge, is that many times likelier about to step out than one at an open    \
                                   kerb (1: no difference) */                                                         \
    X(ped_gap_s_m, 6.0)         /* ... a standing vehicle's end within this of it along the road */                   \
    X(ped_onset, 0.0)           /* 1: a pedestrian's first step towards the road is read from its place (how far it   \
                                   has come from where it stood, against what its place is known to), before the      \
                                   tracker's velocity shows it */                                                     \
    X(ped_onset_m, 0.3)         /* ... from this far (m) */                                                           \
    X(ped_onset_span_m, 1.5)    /* ... a step out begun within the last second has come up to this far */            \
    X(ped_onset_range, 1.0)     /* ... the share of the range's spread that a correction may move within that time */ \
    X(ped_onset_tau_s, 0.7)     /* ... the place it stood at follows a slow change of the offset in this time (a      \
                                   track's place settling as the car closes: 0.3 m/s is followed, 1.4 m/s is not) */  \
    X(ped_onset_v_min, 0.6)     /* ... half a second on, it must have come on at this pace at least (m/s) */          \
    X(ped_onset_v_max, 4.0)     /* ... a change of the place faster than this is the track's correction (m/s) */      \
    X(ped_carry, 0.0)           /* 1: a pedestrian's manoeuvres are carried over the edge of a lane (the step out it  \
                                   began goes on as the crossing it is); 0: as a vehicle's: in another lane "the     \
                                   change that got it there is over", four fifths of "about to step out" become       \
                                   "walks along" and half of the rest the priors, at the moment it enters the car's   \
                                   lane */                                                                            \
    X(ped_cov, 0.0)             /* 1: a pedestrian's crossing futures start with its place's own spread (a camera's:  \
                                   metres along the line of sight, centimetres across it: how far into the road it   \
                                   is, is known); 0: with a round one of the same size (a pedestrian 2.9 m from the   \
                                   lane's middle, range known to 2.8 m, stood "in the lane" with a fifth of it) */    \
    X(std_rho, 0.0)             /* 1: where the radar gives the correlation of a place's x and y errors (`std_rho`),  \
                                   the place's covariance is that one: the needle is read along its own long axis     \
                                   (lane_step) and the futures start with it; 0: as if it were not given (the needle   \
                                   rebuilt from `std` and the bearing, the futures' spread upright) */

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
       PO_EXIST, PO_SEEN, PO_IND, PO_BRAKE, PO_PARKED, PO_ID, PO_VIS, PO_MOTION, PO_WITH, PO_CX, PO_CY, PO_RHO, P_NO };
/* PO_X, PO_Y: the radar's reference point (the near face: where the futures start, what the decision layer's
 * footprint is drawn from); PO_CX, PO_CY: the object's centre (NaN: not given), its place on the road; PO_RHO: the
 * correlation of the place's x and y errors (the radar's `std_rho`; NaN: not given) */

/* an object's kept state */
enum { PS_INIT, PS_T, PS_LANE, PS_INT = PS_LANE + P_NLANE, PS_D = PS_INT + P_NM, PS_DR, PS_PDD, PS_PDR, PS_PRR, PS_V, PS_A, PS_PVV,
       PS_PVA, PS_PAA, PS_TSTAND, PS_TMOVE, PS_LMAP, PS_TKERB, PS_TOP, PS_DIR, PS_TLANE, PS_GSTAND, PS_PSL, PS_MOVED, PS_DANCH,
       PS_TON, PS_DPREV, PS_TCH, P_NS = 48 };
/* PS_MOVED: it has been seen to move (half a second of "it moves" on end): its lane is then also what its way of
 * moving has shown over time. PS_DANCH, PS_TON, PS_DPREV, PS_TCH (a pedestrian, cfg ped_onset): the offset at which
 * it stood, for how long it has been away from it towards the road, its offset when that last changed and how long
 * ago that was (ped_step) */
/* per object diagnostics (`feat`) */
enum { PF_S, PF_D, PF_SD, PF_DC, PF_DIR, PF_LANE, PF_LEAD, PF_GAP, PF_E, PF_R, PF_A, PF_PSI, PF_STOP, PF_JDIST, PF_KERB, PF_OM, P_NF };
/* the road */
enum { PR_W, PR_Q, PR_KERBL, PR_KERBR, PR_VLIM, PR_ROWL, PR_ROWR, PR_SPARE3, P_NR };
/* PR_ROWL, PR_ROWR: how many times likelier a standing vehicle on that side is one of those that stand beside the road
 * there than one standing in a lane (0 or 1: nothing known; predictor.py `row_prior`) */
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
#define P_LOS 9                     /* ranges along the line of sight at most (lane_step) */

static inline double p_sq(double v) { return v * v; }
static inline double p_clip(double v, double lo, double hi) { return v < lo ? lo : v > hi ? hi : v; }
static inline double p_phi(double z) { return 0.5 * erfc(-z * 0.7071067811865476); }
/* the covariance of an object's place between x and y: the radar's where it gives the correlation (cfg std_rho) */
static inline double p_cxy(const double *cfg, const double *o)
{
    return cfg[PC_std_rho] > 0.5 && o[PO_RHO] == o[PO_RHO] ? o[PO_RHO] * o[PO_SX] * o[PO_SY] : 0.0;
}
/* ... as a VEHICLE's futures start with it. A pedestrian's and a cyclist's keep the spread they had: a standing
 * vehicle whose needle crosses the car's way gets two places and the lanes' floor for the one beside it (lane_hyp);
 * a pedestrian has no such two places yet, and its one standing future as a needle across a bend was said (two of
 * 24 runs of a pedestrian standing at a parked row). Its lanes are read along the needle all the same. */
static inline double p_cxy_fut(const double *cfg, const double *o)
{
    const int kind = (int)o[PO_KIND];
    return kind == 6 || kind == 7 ? 0.0 : p_cxy(cfg, o);
}
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
    double best = P_BIG, bt = 0.0;
    int n = p->n, bi = -1;
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
            best = e2;
            bi = i;
            bt = t;
        }
    }
    if (bi >= 0) {                                /* the nearest segment's numbers, once it is known */
        const int i = bi;
        const double t = bt;
        double dx = p->x[i + 1] - p->x[i], dy = p->y[i + 1] - p->y[i];
        double L2 = dx * dx + dy * dy, rx = x - p->x[i], ry = y - p->y[i];
        double L = sqrt(L2);
        *s = p->s[i] + t * L;
        if (t >= 0.0 && t <= 1.0)
            *d = (dx * ry - dy * rx) / L;
        else {
            /* the offset is measured across the path's own direction at that point: by the vertex tangents */
            double u = t < 0.0 ? 0.0 : t > 1.0 ? 1.0 : t;
            double th = p->th[i] + u * (p->th[i + 1] - p->th[i]);
            double px = p->x[i] + t * dx, py = p->y[i] + t * dy;
            *d = -(x - px) * sin(th) + (y - py) * cos(th);
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
    /* what does not change from step to step is taken once: the decay over a step, the car-following law's root;
     * and what depends on the step's time alone (exp(-t / 2), t summed up step by step as below) is the same for
     * every future rolled out with this step: kept from one roll-out to the next */
    const double eh2 = exp(-h / 2.0), sab = sqrt(amax * b);
    static double et_h = -1.0, et_t = 0.0, ET[P_STEPS_MAX + 1];
    static int et_n = 0;
    if (h != et_h) {
        et_h = h;
        et_t = 0.0;
        et_n = 0;
    }
    const int tabled = nstep <= P_STEPS_MAX;
    while (tabled && et_n <= nstep) {
        ET[et_n++] = exp(-et_t / 2.0);
        et_t += h;
    }
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
        /* the point at arc length s, offset d to the left (pp_point), its tangent's cos and sin kept */
        pp_eval(p, s, &px, &py, &pth, &pk);
        const double c = cos(pth), sn = sin(pth);
        px -= d * sn;
        py += d * c;
        if (sprof != NULL) {
            sprof[k] = s;
            vprof[k] = v;
        }
        /* this step's covariance in the frame */
        double vss = Ps[0] + p_sq(v * r->onset_lon_sd);            /* a start that comes sooner or later */
        double vdd = Pl[0] + p_sq(dr * r->onset_sd) + p_sq(r->extra_lat) * (1.0 - (tabled ? ET[k] : exp(-t / 2.0)));
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
                double qx, qy, qth, qk = pk;              /* (at the place itself: the curvature just read) */
                if (!(la == 0.0))
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
                double ss = s0g + v * T + v * dv / (2.0 * sab);
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
        psd *= eh2;
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
            cxy = p_cxy_fut(cfg, o);
        } else {
            /* an acceleration that wanders with spread q and persistence tau: position variance q^2 g(t)^2-like */
            double gl = p_sq(qlon * g), gc = p_sq(qlat * g);
            cxx = p_sq(o[PO_SX]) + p_sq(o[PO_SVX] * t) + hx * hx * gl + hy * hy * gc;
            cyy = p_sq(o[PO_SY]) + p_sq(o[PO_SVY] * t) + hy * hy * gl + hx * hx * gc;
            cxy = hx * hy * (gl - gc);
            if (p_cxy_fut(cfg, o) != 0.0)
                cxy += p_cxy_fut(cfg, o);
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
    double sref, dref;              /* the reference point (near face) less the centre, along and beside the base path */
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
    int between;                    /* a pedestrian in the line of the parked vehicles, one of them next to it (ped_step) */
    double onset;                   /* ... the log-likelihood ratio of "it has begun to step out" from how far its place
                                     * has come towards the road (0: nothing) */
    int side;                       /* +1: on the left half of the road */
    double xing;                    /* distance along the road to the nearest marked crossing (inf: none) */
    int obstacle;                   /* a standing thing ahead in a cyclist's way */
    double q[P_NM], ll[P_NM];
    int nm, slot[P_KMAX], name[P_KMAX], yield[P_KMAX];
    double pri[P_KMAX], rank[P_KMAX];
    double *traj;                   /* [P_KMAX][nt][6] */
    double sprof[P_STEPS_MAX + 1], vprof[P_STEPS_MAX + 1];
    double bb[P_KMAX][4];           /* each future's bounding box, spread included */
    /* its place along the line of sight (lane_step, cfg lane_los; los_n 0: not read so) and the two places a
     * vehicle may be in (cfg lane_hyp; hyp 0: one place): 0 the car's lane or across it, 1 elsewhere */
    int los_n, hyp, side_h;         /* hyp 1: two places; 2: surely in the car's way (one place); side_h: the place its futures are of */
    double los_r0, los_sr, los_sc;  /* the range of its centre, its 1 sigma along and across the line of sight */
    int needle;                     /* the needle is the radar's own (`std_rho`): its long axis (nx, ny), a unit
                                     * vector away from the car; a place at "range" r is the track's place moved by
                                     * r - los_r0 along it. 0: the line of sight itself (the place scaled by r / los_r0) */
    double nx, ny;
    double cxy;                     /* the covariance of its place between x and y as its futures start with it (p_cxy_fut) */
    double hyp_p[2], hyp_r[2], hyp_sd[2];   /* per place: its probability, the range's mean and 1 sigma given it */
    double cs, cd, cvar;            /* the place of the futures being built (object_modes): arc length and offset
                                     * beyond the track's own, its variance as a share of the track's */
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

/* What a camera's velocity can say of "it moves" (cfg motion_scale). A camera measures bearing and size; the range
 * is known to a share rho of itself, and while the car closes at v a range read long by the share b makes a standing
 * thing move the car's way at b v (short: towards the car). The tracker's parked cars showed exactly that: called
 * moving in a fifth of their frames, their velocity along the car's own within a point or two of it, 0.15-0.35 of
 * the car's speed. So the velocity along the car's travel is, for a standing thing, N(0, (rho v)^2 + 1), and across
 * it N(0, stated); for a moving one N(0, motion_scale_v^2 + stated) along, N(0, 3^2 + stated)
 * across. With even odds before, that ratio is the most "it stands" has to be: the radar's p_stationary is raised to
 * it, never lowered, and only for a vehicle beside the car's way (predict_frame). A lead driving at the car's speed,
 * an oncoming car, anything crossing are far outside it. */
static double motion_scale_bound(const double *cfg, const double *o, double v_ego)
{
    double r = hypot(o[PO_X], o[PO_Y]), sr = hypot(o[PO_SX], o[PO_SY]);
    double rho = r > 1.0 ? sr / r : 0.0;
    if (rho < cfg[PC_motion_scale_rel])
        rho = cfg[PC_motion_scale_rel];
    /* (a range known no better than that still does not make a car that keeps the car's own speed a standing one:
     * real drive 20261122-202233 at 327 s, a car 14 m ahead at the car's speed, its range known to 9 m) */
    if (rho > 0.3)
        rho = 0.3;
    double svx = o[PO_SVX] == o[PO_SVX] ? o[PO_SVX] : 1.0, svy = o[PO_SVY] == o[PO_SVY] ? o[PO_SVY] : 1.0;
    /* (the scale's share alone, with a metre a second of the filter's own: a velocity that is merely poorly known
     * (a lead just born, 18 +- 9 m/s) is not thereby one a standing thing would show) */
    double sx_s = p_sq(rho * v_ego) + 1.0, sy_s = p_sq(rho * v_ego * (r > 1.0 ? o[PO_Y] / r : 0.0)) + p_sq(svy) + 0.01;
    double sx_m = p_sq(cfg[PC_motion_scale_v]) + p_sq(svx), sy_m = 9.0 + p_sq(svy);
    /* (across, only what a velocity there takes away from standing: things that move drive along the road too) */
    if (p_sq(o[PO_VY]) > 4.0 * sy_s)                 /* it moves across the car's travel: no scale explains that */
        return 0.0;
    double ll = 0.5 * log(sx_m / sx_s) - 0.5 * p_sq(o[PO_VX]) * (1.0 / sx_s - 1.0 / sx_m)
                - 0.5 * p_sq(o[PO_VY]) * (1.0 / sy_s - 1.0 / sy_m);
    return 1.0 / (1.0 + exp(-p_clip(ll, -30.0, 30.0)));
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
    /* its place on the road is its centre where the radar gives it (the near face of a car beside the lane is up
     * to half its width nearer the lane than the car); the futures start at the reference point (roll_on_base) */
    int has_c = o[PO_CX] == o[PO_CX] && o[PO_CY] == o[PO_CY];
    double ox = has_c ? o[PO_CX] : o[PO_X], oy = has_c ? o[PO_CY] : o[PO_Y];
    double geo[P_NLANE] = {0, 0, 0, 0, 0, 0, 0};
    double sx = o[PO_SX], sy = o[PO_SY];
    int along_x = sx >= sy;
    double big = along_x ? sx : sy, small = along_x ? sy : sx;
    static const double W3[3] = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};
    double dm = 0.0, d2 = 0.0, res2 = 0.0, sm = 0.0, s2 = 0.0, dk[P_LOS], sk[P_LOS];
    /* Along the line of sight (cfg lane_los). A camera knows the bearing to a tenth of a degree and the range to a
     * tenth or a fifth of itself: the place's spread is a needle along the line of sight. The message states its
     * extent along the frame's axes only (`std`); for a needle along the line of sight at bearing b these are
     * sx^2 = vr cos^2 b + vc sin^2 b, sy^2 = vr sin^2 b + vc cos^2 b, which give vr (along) and vc (across) back.
     * Where the two do not fit such a needle (vc well below zero, the bearing near 45 degrees) the frame's axes
     * stand as before. On a bend the needle crosses the lanes: its ranges are read one by one (three where the
     * needle stays within 0.6 m across the road, P_LOS else), each with what the range between it and the next
     * adds across the road. */
    int los = 0;
    double gk[P_LOS][P_NLANE], wk[P_LOS], rk[P_LOS], vr = 0.0, vc = 0.0;
    const double r0 = hypot(ox, oy);
    q->los_n = q->hyp = q->needle = 0;
    q->nx = r0 > 1e-6 ? ox / r0 : 1.0;
    q->ny = r0 > 1e-6 ? oy / r0 : 0.0;
    if (cfg[PC_lane_los] > 0.5 && r0 > 3.0 && cfg[PC_std_rho] > 0.5 && o[PO_RHO] == o[PO_RHO]) {
        /* the radar states the place's whole covariance (`std_rho`): the needle is its long axis as it is, wherever
         * it points (a camera's: within a degree of the line of sight; a microwave radar's: across it) */
        const double cxy = p_cxy(cfg, o), tot = sx * sx + sy * sy, dif = sx * sx - sy * sy;
        const double disc = sqrt(dif * dif + 4.0 * cxy * cxy), ang = 0.5 * atan2(2.0 * cxy, dif);
        double ex = cos(ang), ey = sin(ang);
        if (ex * ox + ey * oy < 0.0) {
            ex = -ex;
            ey = -ey;
        }
        vr = 0.5 * (tot + disc);
        vc = 0.5 * (tot - disc);
        if (vc < 0.0025)
            vc = 0.0025;
        q->nx = ex;
        q->ny = ey;
        q->needle = 1;
        los = 3;
    } else if (cfg[PC_lane_los] > 0.5 && r0 > 3.0) {
        double c2 = (ox * ox - oy * oy) / (r0 * r0);
        if (fabs(c2) > 0.25) {
            double tot = sx * sx + sy * sy, dif = (sx * sx - sy * sy) / c2;
            vr = 0.5 * (tot + dif);
            vc = 0.5 * (tot - dif);
            if (vr > 0.0 && vr >= vc && vc > -0.05 * vr) {
                if (vc < 0.0025)
                    vc = 0.0025;
                los = 3;
            }
        }
    }
    if (los) {
        const double sr = sqrt(vr);
        for (int k = 0; k < 3; k++) {
            rk[k] = r0 + (k - 1) * 1.7320508075688772 * sr;
            if (rk[k] < 0.1 * r0)
                rk[k] = 0.1 * r0;
            wk[k] = W3[k];
            if (q->needle)
                pp_project(&c->base, ox + (rk[k] - r0) * q->nx, oy + (rk[k] - r0) * q->ny, &sk[k], &dk[k]);
            else
                pp_project(&c->base, ox * rk[k] / r0, oy * rk[k] / r0, &sk[k], &dk[k]);
        }
        /* (the radar's own needle of a thing behind the car: three places do; whether it is in the car's way is
         * no question any more. The rebuilt needle left most of those out by its fit) */
        if ((fabs(dk[2] - dk[0]) > 0.6 || fabs(dk[1] - 0.5 * (dk[0] + dk[2])) > 0.2) && !(q->needle && ox <= 0.0)) {
            double ws = 0.0;
            los = P_LOS;
            for (int k = 0; k < P_LOS; k++) {
                double z = (k - (P_LOS - 1) / 2) * (6.0 / (P_LOS - 1));
                rk[k] = r0 + z * sr;
                if (rk[k] < 0.1 * r0)
                    rk[k] = 0.1 * r0;
                wk[k] = exp(-0.5 * z * z);
                ws += wk[k];
                if (q->needle)
                    pp_project(&c->base, ox + (rk[k] - r0) * q->nx, oy + (rk[k] - r0) * q->ny, &sk[k], &dk[k]);
                else
                    pp_project(&c->base, ox * rk[k] / r0, oy * rk[k] / r0, &sk[k], &dk[k]);
            }
            for (int k = 0; k < P_LOS; k++)
                wk[k] /= ws;
        }
        for (int k = 0; k < los; k++) {
            double step = k == 0 ? dk[1] - dk[0] : k == los - 1 ? dk[k] - dk[k - 1] : 0.5 * (dk[k + 1] - dk[k - 1]);
            double rs = sqrt(vc + 0.0225 + step * step / 12.0), tot = 0.0;
            for (int l = 0; l < P_NLANE; l++)
                gk[k][l] = 0.0;
            for (int b = 0; b < c->nb; b++) {
                const double *B = c->bands + P_NB * b;
                double oc = B[PB_P] * band_occ(B, dk[k], rs);
                gk[k][(int)B[PB_TYPE]] += wk[k] * oc / (B[PB_HI] - B[PB_LO]);
                tot += oc;
            }
            gk[k][PL_OFF] += wk[k] * (tot < 0.98 ? 1.0 - tot : 0.02) / 30.0;
            for (int l = 0; l < P_NLANE; l++)
                geo[l] += gk[k][l];
            dm += wk[k] * dk[k];
            sm += wk[k] * sk[k];
        }
        res2 = vc + 0.0225;
        for (int k = 0; k < los; k++) {
            d2 += wk[k] * p_sq(dk[k] - dm);
            s2 += wk[k] * p_sq(sk[k] - sm);
        }
        q->los_n = los;
        q->los_r0 = r0;
        q->los_sr = sr;
        q->los_sc = sqrt(vc);
    } else {
    for (int k = 0; k < 3; k++) {
        double off = (k - 1) * 1.7320508075688772 * big;
        double x = ox + (along_x ? off : 0.0), y = oy + (along_x ? 0.0 : off);
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
    }
    double px, py;
    pp_project(&c->base, ox, oy, &q->s, &q->d);
    pp_eval(&c->base, q->s, &px, &py, &q->th, &q->kap);
    q->sref = q->dref = 0.0;
    if (has_c) {
        double sr, dr;
        pp_project(&c->base, o[PO_X], o[PO_Y], &sr, &dr);
        q->sref = sr - q->s;
        q->dref = dr - q->d;
    }
    q->sd_d = sqrt(d2 + res2);
    q->sd_s = sqrt(s2 + (los ? vc : p_sq(along_x ? sin(q->th) * small : cos(q->th) * small)) + 0.01);
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
        pp_project(&c->arm[a], ox, oy, &as, &ad);
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
        /* where vehicles have been seen to stand beside the road on its side (and how many against those standing
         * in a lane), one standing in a lane is that many times less likely than one of them (the lanes' standing
         * share is divided: what stands across the road or on a side road is not part of that count) */
        double row = c->road[dm >= 0.0 ? PR_ROWL : PR_ROWR];
        if (row > 1.0 && q->grp == 0) {
            em[PL_EGO] = geo[PL_EGO] * (st * s_lane / row + mv * m_with);
            em[PL_LEFT] = geo[PL_LEFT] * (st * s_lane / row + mv * m_with);
            em[PL_RIGHT] = geo[PL_RIGHT] * (st * s_lane / row + mv * m_with);
            em[PL_ONC] = geo[PL_ONC] * (st * s_lane / row + mv * m_onc);
        }
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
    if (!(cfg[PC_lane_mem] > 0.5) && !(q->st[PS_MOVED] >= 1.0) && !fresh && occ_arm < 0.3 && occ_box < 0.3) {
        /* A thing never seen to move. The track's place now IS the sum of what was seen of it: its lane is the
         * prior times what that place says, not that multiplied into what the same track said a moment ago.
         * Multiplied in frame after frame, a range error that put a roadside car's mean into the lane for a second
         * became certainty (P(ego) above 0.5 for 22 % of the parked cars' frames at 40-50 m, twice the share whose
         * mean was in the lane at all). What has moved keeps the filter: how it moved (along a lane, across the
         * road, into the next lane) is evidence that does add up, and a car that drove up and stopped keeps the
         * lane it drove in. So does what stands on a side road (a car waiting at the junction): which of the
         * roads it is on is told apart by its body's direction, frame by frame. */
        lam = 1.0;
        wgt = 1.0;
    }
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
    /* The two places of a vehicle seen along a line of sight that crosses the car's lane (cfg lane_hyp): in the
     * car's lane (or across the road), or elsewhere (another lane, the parking strip, off the road). Each range read
     * above says how much it belongs to either (the lanes' priors, what each lane says to the way it moves, the
     * bands there); the probabilities are the lane filter's own; given a place, the range is the mean and spread of
     * the ranges that belong to it. */
    if (los && q->grp == 0 && cfg[PC_lane_hyp] > 0.5) {
        double a[P_LOS], b[P_LOS], sa = 0.0, sb = 0.0, ma = 0.0, mb = 0.0, va = 0.0, vb = 0.0;
        double fac[P_NLANE];
        for (int l = 0; l < P_NLANE; l++)
            fac[l] = geo[l] > 0.0 ? PRIOR[l] * em[l] / geo[l] : 0.0;
        const double cross = PRIOR[PL_CROSS] * em[PL_CROSS];
        /* "in the car's way": its body reaches into the strip the car's body sweeps along its lane (a car parked
         * half in the lane is in the way though its lane is the parking strip's) */
        /* (the bodies themselves, no margin: a parked car 0.1 m clear of the strip is not "surely in the way"; real
         * drive 20261122-202233 at 331 s) */
        const double reach = 0.5 * (cfg[PC_ego_wid] + q->wid), mid = 0.3 * c->ed0;
        for (int k = 0; k < los; k++) {
            double step = k == 0 ? dk[1] - dk[0] : k == los - 1 ? dk[k] - dk[k - 1] : 0.5 * (dk[k + 1] - dk[k - 1]);
            double rs = sqrt(vc + 0.0225 + step * step / 12.0);
            double in = p_phi((reach - (dk[k] - mid)) / rs) - p_phi((-reach - (dk[k] - mid)) / rs), all = 0.0;
            for (int l = 0; l < P_NLANE; l++)
                if (l != PL_CROSS)
                    all += fac[l] * gk[k][l];
            all += cross * wk[k];                     /* ("across the road" is no place of its own: wherever it is) */
            a[k] = all * in;
            b[k] = all * (1.0 - in);
            sa += a[k];
            sb += b[k];
            ma += a[k] * rk[k];
            mb += b[k] * rk[k];
        }
        const double hA = sa + sb > 0.0 ? sa / (sa + sb) : 0.0, pmin = cfg[PC_lane_hyp_min];
        /* Two places only for what (probably) stands, and only where they are apart in range: a moving vehicle's
         * futures are its lane's, and a place open across the line of sight alone (a car at the lane's edge, a car
         * parked half in the lane) is one place whose spread says it: those stay as they were, weighed by the lanes
         * in the decision unless the radar's own corridor test puts them in the path. (Weighing them by "its body
         * reaches the strip" instead said a parked car on a real drive, 20261122-202233 at 331 s, whose track stood
         * a metre nearer the plan's line than the car was passed.) */
        const int stands = q->pstat >= cfg[PC_lane_hyp_stand];
        if (sa > 0.0 && sb > 0.0) {
            ma /= sa;
            mb /= sb;
        }
        /* apart in range: the ranges that belong to the one place are mostly not those of the other (a car in the lane
         * on a bend has the other place on both sides of it: the means coincide, the stretches do not) */
        double ov = 0.0;
        for (int k = 0; k < los && sa > 0.0 && sb > 0.0; k++)
            ov += a[k] / sa < b[k] / sb ? a[k] / sa : b[k] / sb;
        const int apart = sa > 0.0 && sb > 0.0 && ov < 0.5 && los == P_LOS;
        if (stands && apart && hA >= pmin) {
            for (int k = 0; k < los; k++) {
                va += a[k] * p_sq(rk[k] - ma);
                vb += b[k] * p_sq(rk[k] - mb);
            }
            const double cell = p_sq(rk[1] - rk[0]) / 12.0;      /* the ranges between two read */
            q->hyp = 1;
            q->hyp_p[0] = hA;
            q->hyp_p[1] = 1.0 - hA;
            q->hyp_r[0] = ma;
            q->hyp_r[1] = mb;
            q->hyp_sd[0] = sqrt(va / sa + cell);
            q->hyp_sd[1] = sqrt(vb / sb + cell);
            /* A future is a normal spread, a place is a stretch of the line of sight: each place's spread is kept
             * within its stretch (two sigma to where the other place takes over between the two means), or the
             * roadside future's tail lay in the lane again (pc 0.2-0.4 against the car's path for "parked") and
             * the lane's mass was counted twice. */
            double lo = ma < mb ? ma : mb, hi = ma < mb ? mb : ma, edge = 0.5 * (ma + mb);
            for (int k = 0; k + 1 < los; k++) {
                if (rk[k + 1] <= lo || rk[k] >= hi)
                    continue;
                double f0 = a[k] - b[k], f1 = a[k + 1] - b[k + 1];
                if ((f0 > 0.0) != (f1 > 0.0)) {
                    edge = rk[k] + (rk[k + 1] - rk[k]) * f0 / (f0 - f1);
                    break;
                }
            }
            if (edge > lo && edge < hi) {
                if (q->hyp_sd[0] > 0.5 * fabs(ma - edge))
                    q->hyp_sd[0] = 0.5 * fabs(ma - edge);
                if (q->hyp_sd[1] > 0.5 * fabs(mb - edge))
                    q->hyp_sd[1] = 0.5 * fabs(mb - edge);
            }
            for (int h = 0; h < 2; h++)
                if (q->hyp_sd[h] < 0.3)
                    q->hyp_sd[h] = 0.3;
        }
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
        st[PS_TSTAND] = st[PS_TMOVE] = st[PS_TKERB] = st[PS_MOVED] = 0.0;
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
    if (st[PS_MOVED] < 1.0) {                                   /* seen to move: half a second of a clear "it moves" on end */
        st[PS_MOVED] = q->pstat < cfg[PC_lane_hyp_stand] ? st[PS_MOVED] + dt : 0.0;
        if (st[PS_MOVED] >= 0.5)
            st[PS_MOVED] = 1.0;
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

/* ---- a pedestrian at parked vehicles ------------------------------------------------------------------------- */
/* Two things a pedestrian's track says before its velocity does (cfg ped_gap, ped_onset; after every object's
 * lane_step and feat_step).
 *
 * Where it stands: in the line of the vehicles parked beside the road (between two of them, or at the end of one),
 * not behind them on the pavement. Someone standing there is about to cross or to get into a car; from behind a
 * parked vehicle it is seen late and is in the lane after two steps. `between`: a standing vehicle on its side of
 * the road ends within ped_gap_s_m of it along the road, and its offset lies within that vehicle's body's (0.3 m
 * beyond its far face at most).
 *
 * That it has begun to move: the tracker's velocity of a pedestrian follows a step by a box or two (0.15-1.1 s on
 * the simulator's streams); its place is the first thing a box moves. The offset at which it stood is kept
 * (PS_DANCH: it follows the offset within ped_onset_tau_s while that stays within ped_onset_m of it, so a track's
 * place settling at 0.3 m/s as the car closes is followed and a walk at 1.4 m/s is not); once the offset has come
 * d > ped_onset_m towards the road, the likelihood ratio of "a step out begun within the last second" (d anywhere
 * up to ped_onset_span_m) against "it stands" (d normal around 0 with the spread a change of the offset has) is
 * `onset`: it lowers the share of "it stands" (predict_frame) and, among the futures in which it moves, weighs the
 * crossing ones against walking along (intent_step). Not a step: a change faster than ped_onset_v_max between two
 * changes of the place (the track's correction: the place it stood at goes with it), and one that half a second on
 * has not come on at ped_onset_v_min. After 4 s away the place it has reached is where it stands. */
static void ped_step(PCtx *c, PObj *q, double dt)
{
    const double *cfg = c->cfg;
    double *st = q->st;
    q->between = 0;
    q->onset = 0.0;
    if (q->grp != 2)
        return;
    if (cfg[PC_ped_gap] != 1.0 && q->kerb < 0.7 && q->kerb > -(cfg[PC_ped_kerb_m] + 1.0)) {
        for (int j = 0; j < c->n && !q->between; j++) {
            const PObj *u = c->obj + j;
            if (u == q || u->grp != 0 || u->pstat < 0.5 || (u->d >= 0.0) != (q->d >= 0.0))
                continue;
            if (fabs(u->s - q->s) > 0.5 * u->len + cfg[PC_ped_gap_s_m])
                continue;
            if (fabs(q->d) <= fabs(u->d) + 0.5 * u->wid + 0.3 && fabs(q->d) >= fabs(u->d) - 0.5 * u->wid - 1.0)
                q->between = 1;
        }
    }
    if (!(cfg[PC_ped_onset] > 0.5))
        return;
    if (!(st[PS_INIT] > 0.5) || dt <= 0.0) {
        if (!(st[PS_INIT] > 0.5)) {
            st[PS_DANCH] = st[PS_DPREV] = q->d;
            st[PS_TON] = st[PS_TCH] = 0.0;
        }
        return;
    }
    /* a change of the place faster than anyone runs is the track's correction, not a step: the place it stood at
     * goes with it (the change over the time since the place last changed: between two boxes a track is carried) */
    st[PS_TCH] += dt;
    const double dd = q->d - st[PS_DPREV];
    if (fabs(dd) > 0.02) {
        if (fabs(dd) > 0.15 && fabs(dd) > cfg[PC_ped_onset_v_max] * (st[PS_TCH] > dt ? st[PS_TCH] : dt))
            st[PS_DANCH] += dd;
        st[PS_DPREV] = q->d;
        st[PS_TCH] = 0.0;
    }
    const double d = q->side * (st[PS_DANCH] - q->d);            /* towards the road's middle: positive */
    if (d > cfg[PC_ped_onset_m] && st[PS_TON] < 4.0) {
        /* how well the CHANGE of its offset is known: across the line of sight (the bearing: centimetres), twice;
         * and what a correction of its range would move it across the road (the needle's share of the offset's
         * spread; ped_onset_range of it). Where the place was not read along a needle: the offset's whole spread */
        double sd = q->sd_d;
        if (q->los_n) {
            const double along = p_sq(q->sd_d) - p_sq(q->los_sc) - 0.0225;
            sd = sqrt(2.0 * p_sq(q->los_sc) + p_sq(cfg[PC_ped_onset_range]) * (along > 0.0 ? along : 0.0));
        }
        sd = hypot(sd, 0.1);
        const double z = d / sd;
        const double ll = -log(cfg[PC_ped_onset_span_m]) + 0.5 * z * z + log(sd * 2.5066282746310002);
        st[PS_TON] += dt;
        if (st[PS_TON] > 0.5 && d < cfg[PC_ped_onset_m] + cfg[PC_ped_onset_v_min] * (st[PS_TON] - 0.5)) {
            st[PS_DANCH] = q->d;                                /* slower than anyone walks out: the track's place settling */
            st[PS_TON] = 0.0;
        } else if (d <= cfg[PC_ped_onset_span_m] + 2.0 * sd)
            q->onset = p_clip(ll, 0.0, 8.0);
        else
            q->onset = 8.0;                                     /* further than a second's steps: it walks */
    } else if (d > cfg[PC_ped_onset_m] || d < -cfg[PC_ped_onset_m]) {
        st[PS_DANCH] = q->d;                                    /* it stands (or walks) somewhere else now */
        st[PS_TON] = 0.0;
    } else {
        st[PS_DANCH] += (q->d - st[PS_DANCH]) * (1.0 - exp(-dt / cfg[PC_ped_onset_tau_s]));
        st[PS_TON] = 0.0;
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
        if (q->onset > 0.0)                                      /* its place has come towards the road: not "along" */
            L[PM_KEEP] -= q->onset;
        if (near && q->kerb < 0.7) {
            Q[PM_STARTX] = 0.03 * boost * (1.0 + (waited < 2.0 ? waited : 2.0)) * (q->between ? cfg[PC_ped_gap] : 1.0);
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
    const int ped_carry = q->grp == 2 && cfg[PC_ped_carry] > 0.5;
    if (ped_carry && !fresh && !(Q[PM_STARTX] > 0.0) && Q[PM_CROSS] > 0.0) {
        /* a pedestrian that was about to step out and is in the road now: that IS the crossing it began */
        ip[PM_CROSS] += ip[PM_STARTX];
        ip[PM_STARTX] = 0.0;
    }
    if (!fresh && was != q->lane && !ped_carry) {
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
    r->s0 = q->s + q->sref;                 /* the futures carry the reference point */
    r->d0 = st[PS_D] + q->dref;
    r->dr0 = st[PS_DR];
    r->pss = ct * ct * sx2 + sn * sn * sy2;
    r->psd = ct * sn * (sy2 - sx2);
    {
        const double cxy = q->cxy;
        if (cxy != 0.0) {                   /* the place's covariance as the radar states it (cfg std_rho) */
            r->pss += 2.0 * ct * sn * cxy;
            r->psd += (ct * ct - sn * sn) * cxy;
            if (r->pss < 0.0025)
                r->pss = 0.0025;
        }
    }
    r->pdd = st[PS_PDD];
    r->pdr = st[PS_PDR];
    r->prr = st[PS_PRR];
    r->pvv = st[PS_PVV];
    r->paa = st[PS_PAA];
    r->v0 = q->v;
    r->a0 = q->a;
    r->len = q->len;
    if (q->hyp == 1) {                      /* one of its two places (object_modes): there, as surely as that place is known */
        r->s0 += q->cs;
        r->d0 += q->cd;
        r->pss *= q->cvar;
        r->psd *= q->cvar;
        r->pdd = r->pdd * q->cvar + p_sq(q->los_sc);
    }
    if (q->cxy != 0.0) {                    /* (a covariance: the lateral filter's own spread may be the smaller) */
        const double lim = 0.99 * sqrt(r->pss * (r->pdd > 0.0 ? r->pdd : 0.0));
        r->psd = p_clip(r->psd, -lim, lim);
    }
}

/* A standing future at one of a vehicle's two places (lane_step, cfg lane_hyp): the reference point at that place's
 * range along the line of sight, the spread a needle along it (the range's 1 sigma given the place, the bearing's
 * across). */
static void roll_stand_at(const PObj *q, int side, const double *cfg, const double *times, int nt, double *out)
{
    const double *o = q->o;
    double f = q->hyp_r[side] / q->los_r0, rr = hypot(o[PO_X], o[PO_Y]);
    double ux = rr > 1e-6 ? o[PO_X] / rr : 1.0, uy = rr > 1e-6 ? o[PO_Y] / rr : 0.0;
    double vr = p_sq(q->hyp_sd[side]), vc = p_sq(q->los_sc);
    double x0 = o[PO_X] * f, y0 = o[PO_Y] * f;
    if (q->needle) {                        /* along the radar's own needle (lane_step) */
        ux = q->nx;
        uy = q->ny;
        x0 = o[PO_X] + (q->hyp_r[side] - q->los_r0) * ux;
        y0 = o[PO_Y] + (q->hyp_r[side] - q->los_r0) * uy;
    }
    for (int k = 0; k < nt; k++) {
        double w = p_sq(cfg[PC_stand_drift] * times[k]);
        double cxx = vr * ux * ux + vc * uy * uy + w, cyy = vr * uy * uy + vc * ux * ux + w, cxy = (vr - vc) * ux * uy;
        out[6 * k] = x0;
        out[6 * k + 1] = y0;
        out[6 * k + 2] = sqrt(cxx);
        out[6 * k + 3] = sqrt(cyy);
        out[6 * k + 4] = p_clip(cxy / sqrt(cxx * cyy), -0.99, 0.99);
        out[6 * k + 5] = 0.0;
    }
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
    if (q->cxy != 0.0) {                    /* the place's covariance as the radar states it (cfg std_rho) */
        r->pss += 2.0 * ct * sn * q->cxy;
        r->pdd -= 2.0 * ct * sn * q->cxy;
        r->psd += (ct * ct - sn * sn) * q->cxy;
        if (r->pss < 0.0025)
            r->pss = 0.0025;
        if (r->pdd < 0.0025)
            r->pdd = 0.0025;
        const double lim = 0.99 * sqrt(r->pss * r->pdd);
        r->psd = p_clip(r->psd, -lim, lim);
    }
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
    if (m == PM_STAND && q->hyp == 1) {
        roll_stand_at(q, q->side_h, cfg, c->times, c->nt, out);
        q->rank[k] = 5.0;
        return 1;
    }
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
            /* (its place has come towards the road, its velocity not yet: it walks, at a walking pace; ped_step) */
            const int stepped = q->onset >= 2.0 && !(q->sp > 0.6);
            int moving = (q->sp > 0.6 && q->pstat < 0.5) || stepped;
            if (stepped)
                ;
            else if (moving && fabs(wrap_pi(hv - hn)) < 1.05)
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
            r.v0 = stepped ? cfg[PC_ped_v] : moving ? q->sp : 0.0;
            if (!(cfg[PC_ped_cov] > 0.5)) {
                r.pss = r.pdd = 0.5 * (p_sq(o[PO_SX]) + p_sq(o[PO_SY]));
                r.psd = 0.0;
            }
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
    const int side = q->lane == PL_EGO || q->lane == PL_CROSS || q->hyp_p[0] >= 0.5 ? 0 : 1;
    q->side_h = side;
    if (q->hyp == 1) {
        /* its futures are those of the place its lane is of (side), at that place's range; the other place gets a
         * standing future of its own below */
        double f = q->hyp_r[side] / q->los_r0, s1, d1, s0, d0;
        if (q->needle)
            pp_project(&c->base, q->o[PO_X] + (q->hyp_r[side] - q->los_r0) * q->nx,
                       q->o[PO_Y] + (q->hyp_r[side] - q->los_r0) * q->ny, &s1, &d1);
        else
            pp_project(&c->base, q->o[PO_X] * f, q->o[PO_Y] * f, &s1, &d1);
        pp_project(&c->base, q->o[PO_X], q->o[PO_Y], &s0, &d0);
        q->cs = s1 - s0;
        q->cd = d1 - d0;
        q->cvar = p_clip(p_sq(q->hyp_sd[side] / q->los_sr), 0.01, 1.0);
    }
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
    if (q->hyp == 1) {
        /* The futures built are of one place and weigh what that place weighs. The other place gets one future: it
         * stands there ("stopped" in the car's lane, "parked" beside it). Whether it moves is part of what is not
         * known of it; a vehicle that may move, in the car's lane, is judged as one standing there, and one beside
         * the road as parked. */
        const int alt = 1 - side;
        const double p_st = q->hyp_p[alt];
        for (int k = 0; k < q->nm; k++) {
            q->pri[k] *= q->hyp_p[side];
            if (q->slot[k] == PM_STAND && (q->name[k] == PN_PARKED || q->name[k] == PN_STOPPED))
                q->name[k] = side == 0 ? PN_STOPPED : PN_PARKED;      /* named by the place, as the other one is */
        }
        if (q->nm < P_KMAX) {
            int k = q->nm++;
            roll_stand_at(q, alt, cfg, c->times, c->nt, q->traj + (size_t)k * c->nt * 6);
            q->slot[k] = PM_STAND;
            q->name[k] = alt == 0 ? PN_STOPPED : PN_PARKED;
            q->yield[k] = 0;
            q->rank[k] = 5.0;
            q->pri[k] = p_st;
            traj_bb(q->traj + (size_t)k * c->nt * 6, c->nt, q->bb[k]);
        } else
            q->pri[0] += p_st;
    }
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
    /* the yaw rate the road asks for here: the path's mean curvature over the next max(5 m, v x 1 s), from the
     * turn of its chords (positions at 0, half, all of it). Not the curvature at s = 0: the path there joins the
     * arc behind the car (5 m steps) to the given points (decimetres apart), and the half heading step at the join
     * read over a short segment made 3-4 times the road's curvature, so every steady bend took the car off its path
     * (p_path at its 0.05 floor, curve_parked R 110 m, 2026-11-24) */
    double kap;
    {
        /* cfg ego_kap_back: the stretch starts this share of itself behind the car (0: all of it ahead; 0.5: the car
         * in its middle). Ahead only, the road's turn is asked of the car a half second before it is there: at every
         * bend's start and end the yaw rate "disagreed" with the path and the straight-on future took up to half
         * the weight */
        double s1 = e[PE_V] > 5.0 ? e[PE_V] : 5.0, xa, ya, xb, yb, xc, yc, th, k, sb = -p_clip(cfg[PC_ego_kap_back], 0.0, 1.0) * s1;
        pp_eval(&c->base, sb, &xa, &ya, &th, &k);
        pp_eval(&c->base, sb + 0.5 * s1, &xb, &yb, &th, &k);
        pp_eval(&c->base, sb + s1, &xc, &yc, &th, &k);
        kap = wrap_pi(atan2(yc - yb, xc - xb) - atan2(yb - ya, xb - xa)) / (0.5 * s1);
    }
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
 * bodies at the first sample (two tracks that already lie on each other say nothing about the future).
 *
 * now == NULL: only whether the largest exceeds P_CONF_ANY (1.0) or not (0.0), the question pair_table asks of
 * every pair of futures before it needs a number. A sample further than 2.2 sigma outside the rectangle along or
 * across has a share there below 1 - Phi(2.2) = 0.014, so a product below P_CONF_ANY: it is left out without its
 * four erfc; the other samples are computed as they are. */
#define P_CONF_ANY 0.02
static double pair_conflict(const PCtx *c, const PAgent *A, int ma, const PAgent *B, int mb, double *now)
{
    const int screen = now == NULL;
    double now_;
    if (screen)
        now = &now_;
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
            if (!screen)
                *now = (p_phi((Lh - dl) / sl) - p_phi((-Lh - dl) / sl)) * (p_phi((W - dw) / sw) - p_phi((-W - dw) / sw));
            continue;
        }
        if (fabs(dl) > (Lp > Lm ? Lp : Lm) + 4.0 * sl || fabs(dw) > W + 4.0 * sw)
            continue;
        if (screen) {
            if (fabs(dw) > W + 2.2 * sw || dl > Lp + 2.2 * sl || dl < -Lm - 2.2 * sl)
                continue;
            if ((p_phi((Lp - dl) / sl) - p_phi((-Lm - dl) / sl)) * (p_phi((W - dw) / sw) - p_phi((-W - dw) / sw)) > P_CONF_ANY)
                return 1.0;
            continue;
        }
        /* the largest over the horizon: a sample whose share across alone (a probability: at most one times the
         * share along) does not exceed the largest so far cannot raise it, and its share along is not computed */
        double Pw = p_phi((W - dw) / sw) - p_phi((-W - dw) / sw);
        if (Pw >= 0.0 && Pw <= best)
            continue;
        double Pl = p_phi((Lp - dl) / sl) - p_phi((-Lm - dl) / sl);
        if (Pl >= 0.0 && Pl <= best)
            continue;
        double P = Pl * Pw;
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
    /* most pairs of road users never meet: whether any pair of their futures does (a conflict above P_CONF_ANY)
     * is asked first, without the numbers; only a pair that does is worked out (below, as it always was) */
    for (int i = 0; i < A.nm && !any; i++)
        for (int j = 0; j < B.nm && !any; j++) {
            const double *ba = A.bb[i], *bq = B.bb[j];
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
            any = pair_conflict(c, &A, i, &B, j, NULL) > 0.5;
        }
    if (!any)
        return 0;
    any = 0;
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
            if (v > P_CONF_ANY)
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
        q->cxy = p_cxy_fut(c->cfg, q->o);
        q->between = 0;
        q->onset = 0.0;
        q->pstat = motion_state(c->cfg, q->o, &q->minfo);
        if (c->cfg[PC_motion_scale] > 0.5 && q->grp == 0 && c->ego[PE_V] > 2.0) {
            /* only beside the car's way (its place as the track has it): there "it may stand after all" takes a
             * threat away. In the car's way the radar's word stands: a lead whose speed the tracker has not found yet
             * (26 m/s read as 17) must not become a car standing in the lane */
            double lo = motion_scale_bound(c->cfg, q->o, c->ego[PE_V]);
            if (lo > q->pstat) {
                int has_c = q->o[PO_CX] == q->o[PO_CX] && q->o[PO_CY] == q->o[PO_CY];
                double s0, d0;
                pp_project(&c->base, has_c ? q->o[PO_CX] : q->o[PO_X], has_c ? q->o[PO_CY] : q->o[PO_Y], &s0, &d0);
                if (fabs(d0 - 0.3 * c->ed0) > 0.5 * (c->cfg[PC_ego_wid] + q->wid) + 0.15)
                    q->pstat = lo;
            }
        }
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
    if (c->cfg[PC_ped_gap] != 1.0 || c->cfg[PC_ped_onset] > 0.5)
        for (int i = 0; i < n; i++) {
            PObj *q = c->obj + i;
            if (q->grp != 2)
                continue;
            double dt = q->st[PS_INIT] > 0.5 ? T - q->st[PS_T] : 0.0;
            ped_step(c, q, dt);
            if (q->onset > 0.0) {                               /* "it stands" against how far it has come */
                const double a = q->pstat * exp(-q->onset);
                q->pstat = a / (a + 1.0 - q->pstat);
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
            M[1] = q->hyp ? (q->hyp_p[0] > 1e-6 ? q->hyp_p[0] : 1e-6) : 0.0;   /* its futures carry both places: P(in the car's way) */
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
/* the scoring of one prediction (predict_score's rows into out; more: the further rows too): the winner (-1: never
 * seen), its probability into *pw */
static int score_one(const double *t, int K, Py_ssize_t nt, const double *p, const double *act, const double *val, double *out,
                     int more, double *pw)
{
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
    if (more)                                                    /* a further row: every future's mean error */
        for (int m = 0; m < 12; m++)
            out[6 * nt + m] = cnt > 0 && m < K ? cum[m] / cnt : NAN;
    *pw = cnt > 0 ? p[win] : 0.0;
    return cnt > 0 ? win : -1;
}

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
    double pw;
    int win = score_one((const double *)vt.b.buf, (int)K, nt, (const double *)vp.b.buf, (const double *)va.b.buf,
                        (const double *)vv.b.buf, (double *)vo.b.buf, ro > nt, &pw);
    res = Py_BuildValue("(id)", win, pw);
done:
    view_drop(&vt);
    view_drop(&vp);
    view_drop(&va);
    view_drop(&vv);
    view_drop(&vo);
    return res;
}

/* ====================================================================================================
 * The radar message in, the `radar_fut` message out, the self-check's sums: the predictor's glue
 * ==================================================================================================== */
#define PK_STRS(X)                                                                                                     \
    X(x_m) X(y_m) X(vx_mps) X(vy_mps) X(ax_mps2) X(ay_mps2) X(std) X(class) X(label) X(p_stationary) X(len) X(wid)     \
    X(hdg) X(hdg_std) X(exist) X(state) X(indicator) X(brake_lights) X(vis) X(flags) X(motion) X(ctr) X(id)            \
    X(tentative) X(memory) X(left) X(right) X(parked) X(occluded) X(hidden) X(stationary) X(moving) X(with_traffic)    \
    X(oncoming) X(objects) X(type) X(radar_fut) X(fut) X(t) X(ego_fut) X(modes) X(fut_ctx) X(lane) X(intent) X(p)      \
    X(p0) X(m) X(x) X(y) X(sx) X(sy) X(rxy) X(v) X(lane_res) X(std_rho)
#define X(n) K_##n,
enum { PK_STRS(X) K_N };
#undef X
#define X(n) #n,
static const char *const PK_STR[K_N] = {PK_STRS(X)};
#undef X
static PyObject *pk_s[K_N];

static int pk_strings(void)
{
    if (pk_s[K_N - 1] != NULL)
        return 0;
    for (int i = 0; i < K_N; i++)
        if (pk_s[i] == NULL && (pk_s[i] = PyUnicode_InternFromString(PK_STR[i])) == NULL)
            return -1;
    return 0;
}

static PyObject *pk_get(PyObject *d, int key)   /* d.get(key): borrowed, NULL where absent */
{
    PyObject *v = PyDict_GetItemWithError(d, pk_s[key]);
    if (v == NULL)
        PyErr_Clear();
    return v;
}

/* Python's `_fin(v)`: an int or a float (no bool), finite; its value */
static int py_fin(PyObject *v, double *out)
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

static int pk_is(PyObject *v, int key)          /* v == "<key>" */
{
    return v != NULL && (v == pk_s[key] || (PyUnicode_Check(v) && PyUnicode_CompareWithASCIIString(v, PK_STR[key]) == 0));
}

/* float(v) of a present value, `absent` where it is None or not there; -1 with an exception where it is no number */
static int pk_num(PyObject *v, double absent, double *out)
{
    if (v == NULL || v == Py_None) {
        *out = absent;
        return 0;
    }
    *out = PyFloat_AsDouble(v);
    return *out == -1.0 && PyErr_Occurred() ? -1 : 0;
}

static int pk_or0(PyObject *v, double *out)      /* float(v or 0.0) */
{
    *out = 0.0;
    if (v == NULL || v == Py_None)
        return 0;
    int t = PyObject_IsTrue(v);
    if (t <= 0)
        return t;
    *out = PyFloat_AsDouble(v);
    return *out == -1.0 && PyErr_Occurred() ? -1 : 0;
}

/* predict_pack(objects, tentative, objects_max) -> (rows, ids, objects taken, {id: row})
 * A radar message's objects as predict_frame reads them (P_NO numbers each: PO_*): those with a place (x_m, y_m
 * finite numbers) that are not tentative (or all, `tentative`), the nearest `objects_max` where there are more (in
 * order of distance). rows: bytes of float64 (n, P_NO). A row with a number that is none among its first twelve is
 * made harmless (no motion, wide spreads, p_stationary one half); spreads are 0.05 at least. */
static const char *const PK_KIND[8] = {"car", "van", "truck", "bus", "tram", "motorbike", "bike", "human"};

static PyObject *n_predict_pack(PyObject *self, PyObject *args)
{
    PyObject *objects, *res = NULL, *rows = NULL, *ids = NULL, *taken = NULL, *sel = NULL;
    int tentative, omax;
    PyObject **pick = NULL;
    double *dist = NULL;
    if (!PyArg_ParseTuple(args, "Opi", &objects, &tentative, &omax))
        return NULL;
    if (pk_strings() < 0)
        return NULL;
    Py_ssize_t no = objects != Py_None && PyList_Check(objects) ? PyList_Size(objects) : 0, n = 0;
    pick = (PyObject **)PyMem_Malloc((size_t)(no + 1) * sizeof(PyObject *));
    dist = (double *)PyMem_Malloc((size_t)(no + 1) * sizeof(double));
    if (!pick || !dist) {
        PyErr_NoMemory();
        goto done;
    }
    for (Py_ssize_t i = 0; i < no; i++) {
        PyObject *o = PyList_GetItem(objects, i);
        double x, y;
        if (!PyDict_Check(o) || !py_fin(pk_get(o, K_x_m), &x) || !py_fin(pk_get(o, K_y_m), &y))
            continue;
        if (!tentative && pk_is(pk_get(o, K_state), K_tentative))
            continue;
        pick[n] = o;
        dist[n++] = x * x + y * y;
    }
    if (n > omax && omax >= 0) {                  /* the nearest (a stable order, as sorted()) */
        for (Py_ssize_t i = 1; i < n; i++) {
            PyObject *o = pick[i];
            double d = dist[i];
            Py_ssize_t j = i;
            while (j > 0 && dist[j - 1] > d) {
                pick[j] = pick[j - 1];
                dist[j] = dist[j - 1];
                j--;
            }
            pick[j] = o;
            dist[j] = d;
        }
        n = omax;
    }
    rows = PyBytes_FromStringAndSize(NULL, n * P_NO * (Py_ssize_t)sizeof(double));
    ids = PyList_New(n);
    taken = PyList_New(n);
    sel = PyDict_New();
    if (!rows || !ids || !taken || !sel)
        goto done;
    double *A = (double *)PyBytes_AsString(rows);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *o = pick[i], *v;
        double *a = A + i * P_NO;
        int bad = 0;
        bad |= pk_num(pk_get(o, K_x_m), 0.0, &a[PO_X]) < 0 || pk_num(pk_get(o, K_y_m), 0.0, &a[PO_Y]) < 0;
        bad |= pk_or0(pk_get(o, K_vx_mps), &a[PO_VX]) < 0 || pk_or0(pk_get(o, K_vy_mps), &a[PO_VY]) < 0 ||
               pk_or0(pk_get(o, K_ax_mps2), &a[PO_AX]) < 0 || pk_or0(pk_get(o, K_ay_mps2), &a[PO_AY]) < 0;
        if (bad)
            goto done;
        v = pk_get(o, K_std);                     /* std = o.get("std") or (2, 1, 2, 1) */
        int given = v != NULL && v != Py_None ? PyObject_IsTrue(v) : 0;
        if (given < 0)
            goto done;
        if (given) {
            for (int q = 0; q < 4; q++) {
                PyObject *it = PySequence_GetItem(v, q);
                if (it == NULL)
                    goto done;
                a[PO_SX + q] = PyFloat_AsDouble(it);
                Py_DECREF(it);
                if (a[PO_SX + q] == -1.0 && PyErr_Occurred())
                    goto done;
            }
        } else {
            a[PO_SX] = 2.0;
            a[PO_SY] = 1.0;
            a[PO_SVX] = 2.0;
            a[PO_SVY] = 1.0;
        }
        a[PO_KIND] = 8.0;
        v = pk_get(o, K_class);
        if (v != NULL && v != Py_None) {
            int t = PyObject_IsTrue(v);
            if (t < 0)
                goto done;
            if (t) {
                if (!PyDict_Check(v)) {
                    PyErr_SetString(PyExc_TypeError, "predict_pack: an object's class is not a dict");
                    goto done;
                }
                PyObject *lab = pk_get(v, K_label);
                if (lab != NULL && PyUnicode_Check(lab))
                    for (int q = 0; q < 8; q++)
                        if (PyUnicode_CompareWithASCIIString(lab, PK_KIND[q]) == 0) {
                            a[PO_KIND] = q;
                            break;
                        }
            }
        }
        PyObject *hd = pk_get(o, K_hdg), *hs = pk_get(o, K_hdg_std), *h0 = NULL, *h1 = NULL;
        if (hd != NULL && PyList_CheckExact(hd)) {                 /* [value, std] */
            Py_ssize_t hn = PyList_Size(hd);
            if (hn < 1) {
                PyErr_SetString(PyExc_IndexError, "predict_pack: an empty hdg");
                goto done;
            }
            h0 = PyList_GetItem(hd, 0);
            h1 = hn > 1 ? PyList_GetItem(hd, 1) : NULL;
            hd = h0;
            hs = h1;
        }
        if (pk_num(pk_get(o, K_p_stationary), 0.5, &a[PO_PSTAT]) < 0 || pk_num(pk_get(o, K_len), NAN, &a[PO_LEN]) < 0 ||
            pk_num(pk_get(o, K_wid), NAN, &a[PO_WID]) < 0 || pk_num(hd, NAN, &a[PO_HDG]) < 0 || pk_num(hs, NAN, &a[PO_HDGSD]) < 0 ||
            pk_num(pk_get(o, K_exist), 1.0, &a[PO_EXIST]) < 0)
            goto done;
        a[PO_SEEN] = pk_is(pk_get(o, K_state), K_memory) ? 0.0 : 1.0;
        v = pk_get(o, K_indicator);
        a[PO_IND] = v == NULL || v == Py_None ? NAN : pk_is(v, K_left) ? 1.0 : pk_is(v, K_right) ? -1.0 : 0.0;
        v = pk_get(o, K_brake_lights);
        if (v == NULL || v == Py_None)
            a[PO_BRAKE] = NAN;
        else {
            int t = PyObject_IsTrue(v);
            if (t < 0)
                goto done;
            a[PO_BRAKE] = t ? 1.0 : 0.0;
        }
        PyObject *fl = pk_get(o, K_flags);
        int parked = 0, with = 0;
        if (fl != NULL && fl != Py_None) {
            int t = PyObject_IsTrue(fl);
            if (t < 0)
                goto done;
            if (t) {
                int w1, w2;
                if ((parked = PySequence_Contains(fl, pk_s[K_parked])) < 0 || (w1 = PySequence_Contains(fl, pk_s[K_with_traffic])) < 0 ||
                    (w2 = PySequence_Contains(fl, pk_s[K_oncoming])) < 0)
                    goto done;
                with = w1 || w2;
            }
        }
        a[PO_PARKED] = parked ? 1.0 : 0.0;
        a[PO_ID] = (double)i;
        v = pk_get(o, K_vis);
        if (v == NULL || v == Py_None)
            a[PO_VIS] = NAN;
        else {
            int f = pk_is(v, K_occluded) || pk_is(v, K_hidden);
            if (!f && (f = PyObject_RichCompareBool(v, Py_False, Py_EQ)) < 0)
                goto done;
            a[PO_VIS] = f ? 0.0 : 1.0;
        }
        v = pk_get(o, K_motion);
        a[PO_MOTION] = pk_is(v, K_stationary) ? 1.0 : pk_is(v, K_moving) ? -1.0 : 0.0;
        a[PO_WITH] = with ? 1.0 : 0.0;
        a[PO_CX] = a[PO_CY] = NAN;
        v = pk_get(o, K_ctr);                     /* the centre: a list or a tuple of two finite numbers */
        if (v != NULL && (PyList_CheckExact(v) || PyTuple_CheckExact(v)) && PySequence_Size(v) == 2) {
            double c0, c1;
            PyObject *i0 = PyList_CheckExact(v) ? PyList_GetItem(v, 0) : PyTuple_GetItem(v, 0);
            PyObject *i1 = PyList_CheckExact(v) ? PyList_GetItem(v, 1) : PyTuple_GetItem(v, 1);
            if (py_fin(i0, &c0) && py_fin(i1, &c1)) {
                a[PO_CX] = c0;
                a[PO_CY] = c1;
            }
        }
        a[PO_RHO] = NAN;
        {
            double rho;                           /* the correlation of the place's errors: a finite number within +-1 */
            if (py_fin(pk_get(o, K_std_rho), &rho) && rho >= -1.0 && rho <= 1.0)
                a[PO_RHO] = rho < -0.999 ? -0.999 : rho > 0.999 ? 0.999 : rho;
        }
        int fin = 1;
        for (int q = 0; q < 12; q++)
            fin &= isfinite(a[q]);
        if (!fin) {                               /* a number that is none: the object's row is made harmless */
            a[PO_VX] = a[PO_VY] = a[PO_AX] = a[PO_AY] = 0.0;
            a[PO_SX] = 2.0;
            a[PO_SY] = 1.0;
            a[PO_SVX] = 2.0;
            a[PO_SVY] = 1.0;
            a[PO_PSTAT] = 0.5;
            a[PO_RHO] = NAN;
        }
        for (int q = PO_SX; q <= PO_SVY; q++)
            if (!(a[q] >= 0.05 || isnan(a[q])))
                a[q] = 0.05;
        PyObject *id = pk_get(o, K_id), *ix = PyLong_FromSsize_t(i);
        if (id == NULL)
            id = Py_None;
        if (ix == NULL || PyDict_SetItem(sel, id, ix) < 0) {
            Py_XDECREF(ix);
            goto done;
        }
        Py_DECREF(ix);
        Py_INCREF(id);
        PyList_SetItem(ids, i, id);
        Py_INCREF(o);
        PyList_SetItem(taken, i, o);
    }
    res = PyTuple_Pack(4, rows, ids, taken, sel);
done:
    PyMem_Free(pick);
    PyMem_Free(dist);
    Py_XDECREF(rows);
    Py_XDECREF(ids);
    Py_XDECREF(taken);
    Py_XDECREF(sel);
    return res;
}

/* ---- the message form (Prediction.message and its parts) ---- */
static PyObject *pm_list(const double *v, Py_ssize_t n, Py_ssize_t stride, double sc)     /* [rint(v * sc) / sc ...] */
{
    PyObject *l = PyList_New(n);
    for (Py_ssize_t j = 0; l != NULL && j < n; j++) {
        PyObject *f = PyFloat_FromDouble(sc > 0.0 ? rint(v[j * stride] * sc) / sc : v[j * stride]);
        if (f == NULL)
            Py_CLEAR(l);
        else
            PyList_SetItem(l, j, f);
    }
    return l;
}

static int pm_set(PyObject *d, PyObject *key, PyObject *v)       /* d[key] = v; takes v */
{
    if (v == NULL)
        return -1;
    int rc = PyDict_SetItem(d, key, v);
    Py_DECREF(v);
    return rc;
}

/* the futures of every object: [[{"p", "p0", "m", "x", "y", "sx", "sy", "rxy", "v"}, ...], ...] */
static PyObject *pm_futs(const double *t, const double *M, Py_ssize_t n, Py_ssize_t nt, PyObject *names, int dec)
{
    static const int KEYS[6] = {K_x, K_y, K_sx, K_sy, K_rxy, K_v};
    Py_ssize_t nn = PyTuple_Size(names);
    double sc = POW10[dec];
    PyObject *res = PyList_New(n);
    if (res == NULL)
        return NULL;
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
            if (pm_set(d, pk_s[K_p], PyFloat_FromDouble(rint(m[3 * P_KMAX + k] * 1000.0) / 1000.0)) < 0 ||
                pm_set(d, pk_s[K_p0], PyFloat_FromDouble(rint(m[4 * P_KMAX + k] * 1000.0) / 1000.0)) < 0)
                goto fail;
            if (code < 0 || code >= nn || PyDict_SetItem(d, pk_s[K_m], PyTuple_GetItem(names, code)) < 0) {
                if (!PyErr_Occurred())
                    PyErr_SetString(PyExc_ValueError, "predict: a name code without a name");
                goto fail;
            }
            const double *tr = t + ((size_t)i * P_KMAX + k) * nt * 6;
            for (int q = 0; q < 6; q++)
                if (pm_set(d, pk_s[KEYS[q]], pm_list(tr + q, nt, 6, sc)) < 0)
                    goto fail;
        }
    }
    return res;
fail:
    Py_DECREF(res);
    return NULL;
}

/* {lane name: probability to three decimals} of object i */
static PyObject *pm_lane(const double *L, PyObject *lanes)
{
    PyObject *d = PyDict_New();
    for (int q = 0; d != NULL && q < P_NLANE; q++)
        if (pm_set(d, PyTuple_GetItem(lanes, q), PyFloat_FromDouble(rint(L[q] * 1000.0) / 1000.0)) < 0)
            Py_CLEAR(d);
    return d;
}

/* {manoeuvre name: probability} of an object: the intent filter's slots under the names its futures are published
 * with (a slot without a future keeps its slot name); three decimals, what rounds to nothing left out */
static PyObject *pm_intent(const double *I, const double *m, PyObject *names, PyObject *slots)
{
    PyObject *d = PyDict_New(), *named[P_NM];
    int nm = (int)m[0];
    Py_ssize_t nn = PyTuple_Size(names);
    if (d == NULL)
        return NULL;
    for (int s = 0; s < P_NM; s++)
        named[s] = NULL;
    for (int k = 0; k < nm && k < P_KMAX; k++) {
        int slot = (int)m[P_KMAX + k], code = (int)m[2 * P_KMAX + k];
        if (slot >= 0 && slot < P_NM && code >= 0 && code < nn)
            named[slot] = PyTuple_GetItem(names, code);
    }
    for (int s = 0; s < P_NM; s++) {
        double v = rint(I[s] * 1000.0) / 1000.0;
        if (!(v > 0.0005))
            continue;
        PyObject *key = named[s] != NULL ? named[s] : PyTuple_GetItem(slots, s);
        if (named[s] != NULL) {                   /* (named.get(s) or SLOTS[s]) */
            int t = PyObject_IsTrue(named[s]);
            if (t < 0)
                goto fail;
            if (!t)
                key = PyTuple_GetItem(slots, s);
        }
        PyObject *was = PyDict_GetItemWithError(d, key), *val;
        if (was == NULL && PyErr_Occurred())
            goto fail;
        if (was != NULL) {
            double sum = PyFloat_AsDouble(was) + v;
            int ok;
            double r = round_fast(sum, 3, &ok);
            if (ok)
                val = PyFloat_FromDouble(r);
            else {
                PyObject *a = Py_BuildValue("(di)", sum, 3);
                val = a ? PyObject_CallObject(builtin_round, a) : NULL;
                Py_XDECREF(a);
            }
        } else
            val = PyFloat_FromDouble(v);
        if (pm_set(d, key, val) < 0)
            goto fail;
    }
    return d;
fail:
    Py_DECREF(d);
    return NULL;
}

/* {"modes": [{"p", "m", "x", "y", "v"}, ...]}: the car's own futures */
static PyObject *pm_ego(const double *EO, const double *ET, int ego_n, Py_ssize_t nt, PyObject *names, int dec)
{
    PyObject *d = PyDict_New(), *l = PyList_New(ego_n);
    Py_ssize_t nn = PyTuple_Size(names);
    double sc = POW10[dec];
    static const int KEYS[3] = {K_x, K_y, K_v};
    if (d == NULL || l == NULL)
        goto fail;
    for (int k = 0; k < ego_n; k++) {
        PyObject *m = PyDict_New();
        if (m == NULL)
            goto fail;
        PyList_SetItem(l, k, m);
        int code = (int)EO[2 * k + 1];
        if (pm_set(m, pk_s[K_p], PyFloat_FromDouble(rint(EO[2 * k] * 1000.0) / 1000.0)) < 0)
            goto fail;
        if (code < 0 || code >= nn || PyDict_SetItem(m, pk_s[K_m], PyTuple_GetItem(names, code)) < 0) {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_ValueError, "predict: a name code without a name");
            goto fail;
        }
        for (int q = 0; q < 3; q++)
            if (pm_set(m, pk_s[KEYS[q]], pm_list(ET + (size_t)k * nt * 3 + q, nt, 3, sc)) < 0)
                goto fail;
    }
    if (PyDict_SetItem(d, pk_s[K_modes], l) < 0)
        goto fail;
    Py_DECREF(l);
    return d;
fail:
    Py_XDECREF(d);
    Py_XDECREF(l);
    return NULL;
}

/* predict_message(radar_msg, sel, t, traj, modes, lane, intent, ego_out, ego_traj, ego_n, names, lanes, slots,
 *                 decimals, source, part) -> the `radar_fut` message (part 0), or a part of it for the Prediction's
 * accessors: 1 the futures of every object, 2 their lane dicts, 3 their intent dicts, 4 `ego_fut`.
 * The message: the radar message of the same stamp with the prediction added: top level `fut` {"t"}, `ego_fut`
 * {"modes"}, `fut_ctx` (source); per object of `sel` ({id: row}) `fut` [modes], `lane`, `intent`; the other objects
 * are passed on as they are. Numbers rounded as round_lists rounds them (rint(v * 10^d) / 10^d; probabilities to
 * three decimals). */
static PyObject *n_predict_message(PyObject *self, PyObject *args)
{
    PyObject *radar, *sel, *o[7], *names, *lanes, *slots, *source, *res = NULL;
    PyObject *futs = NULL, *out = NULL, *objs = NULL;
    int ego_n, dec, part;
    View v[7];
    Py_ssize_t rows[7];
    static const int WIDTH[7] = {1, 6, 5 * P_KMAX, P_NLANE, P_NM, 2, 3};
    static const char *const WHAT[7] = {"t", "traj", "modes", "lane", "intent", "ego_out", "ego_traj"};
    for (int i = 0; i < 7; i++)
        v[i].held = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOiO!O!O!iOi", &radar, &sel, &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &ego_n,
                          &PyTuple_Type, &names, &PyTuple_Type, &lanes, &PyTuple_Type, &slots, &dec, &source, &part))
        return NULL;
    if (pk_strings() < 0)
        return NULL;
    for (int i = 0; i < 7; i++)
        if (pview(o[i], &v[i], 0, WIDTH[i], &rows[i], WHAT[i]) < 0)
            goto done;
    Py_ssize_t nt = rows[0], n = rows[2];
    if (dec < 0 || dec > 9 || nt < 1 || rows[1] != n * P_KMAX * nt || rows[3] != n || rows[4] != n || ego_n < 0 ||
        ego_n > rows[5] || rows[6] < ego_n * nt || PyTuple_Size(lanes) != P_NLANE || PyTuple_Size(slots) != P_NM) {
        PyErr_SetString(PyExc_ValueError, "predict_message: array sizes do not fit together");
        goto done;
    }
    const double *T = (const double *)v[0].b.buf, *TR = (const double *)v[1].b.buf, *M = (const double *)v[2].b.buf;
    const double *LN = (const double *)v[3].b.buf, *IN = (const double *)v[4].b.buf, *EO = (const double *)v[5].b.buf;
    const double *ET = (const double *)v[6].b.buf;
    if (part == 4) {
        res = pm_ego(EO, ET, ego_n, nt, names, dec);
        goto done;
    }
    if (part == 2 || part == 3) {
        res = PyList_New(n);
        for (Py_ssize_t i = 0; res != NULL && i < n; i++) {
            PyObject *d = part == 2 ? pm_lane(LN + i * P_NLANE, lanes) : pm_intent(IN + i * P_NM, M + i * 5 * P_KMAX, names, slots);
            if (d == NULL)
                Py_CLEAR(res);
            else
                PyList_SetItem(res, i, d);
        }
        goto done;
    }
    futs = pm_futs(TR, M, n, nt, names, dec);
    if (futs == NULL)
        goto done;
    if (part == 1) {
        res = futs;
        futs = NULL;
        goto done;
    }
    if (!PyDict_Check(radar) || !PyDict_Check(sel)) {
        PyErr_SetString(PyExc_TypeError, "predict_message: the radar message and the objects' rows: dicts");
        goto done;
    }
    out = PyDict_Copy(radar);
    PyObject *ro = PyDict_GetItemWithError(radar, pk_s[K_objects]);
    Py_ssize_t no = ro != NULL && ro != Py_None && PyList_Check(ro) ? PyList_Size(ro) : 0;
    objs = PyList_New(no);
    if (out == NULL || objs == NULL)
        goto done;
    for (Py_ssize_t j = 0; j < no; j++) {
        PyObject *ob = PyList_GetItem(ro, j), *item = NULL;
        if (PyDict_Check(ob)) {
            PyObject *id = PyDict_GetItemWithError(ob, pk_s[K_id]);
            PyObject *row = PyDict_GetItemWithError(sel, id != NULL ? id : Py_None);
            if (row == NULL && PyErr_Occurred())
                goto done;
            if (row != NULL) {
                Py_ssize_t i = PyLong_AsSsize_t(row);
                if (i < 0 || i >= n) {
                    if (!PyErr_Occurred())
                        PyErr_SetString(PyExc_IndexError, "predict_message: an object's row beyond the arrays");
                    goto done;
                }
                item = PyDict_Copy(ob);
                if (item == NULL || PyDict_SetItem(item, pk_s[K_fut], PyList_GetItem(futs, i)) < 0 ||
                    pm_set(item, pk_s[K_lane], pm_lane(LN + i * P_NLANE, lanes)) < 0 ||
                    pm_set(item, pk_s[K_intent], pm_intent(IN + i * P_NM, M + i * 5 * P_KMAX, names, slots)) < 0 ||
                    (M[i * 5 * P_KMAX + 1] > 0.0 && pm_set(item, pk_s[K_lane_res], PyFloat_FromDouble(rint(M[i * 5 * P_KMAX + 1] * 1000.0) / 1000.0)) < 0)) {
                    Py_XDECREF(item);
                    goto done;
                }
            }
        }
        if (item == NULL) {
            item = ob;
            Py_INCREF(item);
        }
        PyList_SetItem(objs, j, item);
    }
    {
        PyObject *ft = PyDict_New();
        if (ft == NULL || pm_set(ft, pk_s[K_t], pm_list(T, nt, 1, 0.0)) < 0) {
            Py_XDECREF(ft);
            goto done;
        }
        if (PyDict_SetItem(out, pk_s[K_type], pk_s[K_radar_fut]) < 0 || PyDict_SetItem(out, pk_s[K_objects], objs) < 0 ||
            pm_set(out, pk_s[K_fut], ft) < 0 || pm_set(out, pk_s[K_ego_fut], pm_ego(EO, ET, ego_n, nt, names, dec)) < 0 ||
            PyDict_SetItem(out, pk_s[K_fut_ctx], source) < 0)
            goto done;
    }
    res = out;
    out = NULL;
done:
    Py_XDECREF(futs);
    Py_XDECREF(out);
    Py_XDECREF(objs);
    for (int i = 0; i < 7; i++)
        view_drop(&v[i]);
    return res;
}

/* ---- the self-check's sums ---- */
static volatile double pp_two = 2.0;             /* x ** 2 of Python: the C library's pow, not folded */

/* one scored prediction into the statistics (selfcheck.py SelfCheck.add): out: predict_score's rows (nt + 2, 6),
 * val (nt), p (K), win; hk: the horizons' sample indices; n (4, nh), sums (4, nh, 5), cal (10, 2), brier (2) */
static void score_add(int kind, const double *out, Py_ssize_t nt, const double *val, const double *p, int K, int win,
                      const Py_ssize_t *hk, int nh, double *n, double *sums, double *cal, double *brier)
{
    static const int COLS[5] = {0, 1, 3, 4, 5};   /* min_ade, min_fde, top_fde, nll, maha2 */
    int g = kind == 7 ? 2 : kind == 6 ? 1 : 0;
    for (int h = 0; h < nh; h++) {
        if (!(val[hk[h]] > 0.5))
            continue;
        n[g * nh + h] += 1.0;
        n[3 * nh + h] += 1.0;
        for (int c = 0; c < 5; c++) {
            double x = out[hk[h] * 6 + COLS[c]];
            sums[(g * nh + h) * 5 + c] += x;
            sums[(3 * nh + h) * 5 + c] += x;
        }
    }
    if (K >= 2 && win >= 0) {
        const double *ade = out + nt * 6;         /* the two further rows: every future's mean error */
        double best = ade[win];
        for (int m = 0; m < K && m < 12; m++) {
            int hit = m == win || ade[m] <= 1.1 * best + 0.3;
            double b10 = p[m] * 10.0;
            int bin = b10 >= 9.0 ? 9 : (b10 > 0.0 ? (int)b10 : 0);
            cal[2 * bin] += 1.0;
            cal[2 * bin + 1] += hit;
            brier[0] += pow(p[m] - hit, pp_two);
            brier[1] += 1.0;
        }
    }
}

/* predict_score_add(kind, out (nt + 2, 6), val (nt), p (K), win, hk, n (4, nh), sums (4, nh, 5), cal (10, 2),
 *                   brier (2)): one scored prediction into the self-check's statistics (in place) */
static PyObject *n_predict_score_add(PyObject *self, PyObject *args)
{
    PyObject *oo, *ov, *op, *ohk, *on, *os, *oc, *ob, *res = NULL;
    int kind, win;
    View vo, vv, vp, vn, vs, vc, vb;
    Py_ssize_t ro, nt, K, rn, rs, rc, rb, hk[16];
    vo.held = vv.held = vp.held = vn.held = vs.held = vc.held = vb.held = 0;
    if (!PyArg_ParseTuple(args, "iOOOiO!OOOO", &kind, &oo, &ov, &op, &win, &PyList_Type, &ohk, &on, &os, &oc, &ob))
        return NULL;
    if (pview(oo, &vo, 0, 6, &ro, "out") < 0 || pview(ov, &vv, 0, 1, &nt, "valid") < 0 || pview(op, &vp, 0, 1, &K, "p") < 0 ||
        pview(on, &vn, 1, 1, &rn, "n") < 0 || pview(os, &vs, 1, 5, &rs, "sums") < 0 || pview(oc, &vc, 1, 2, &rc, "cal") < 0 ||
        pview(ob, &vb, 1, 2, &rb, "brier") < 0)
        goto done;
    Py_ssize_t nh = PyList_Size(ohk);
    if (nh > 16 || ro != nt + 2 || rn != 4 * nh || rs != 4 * nh || rc != 10 || rb != 1 || win >= K) {
        PyErr_SetString(PyExc_ValueError, "predict_score_add: array sizes do not fit together");
        goto done;
    }
    for (Py_ssize_t h = 0; h < nh; h++) {
        hk[h] = PyLong_AsSsize_t(PyList_GetItem(ohk, h));
        if (hk[h] < 0 || hk[h] >= nt) {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_IndexError, "predict_score_add: a horizon beyond the samples");
            goto done;
        }
    }
    score_add(kind, (const double *)vo.b.buf, nt, (const double *)vv.b.buf, (const double *)vp.b.buf, (int)K, win, hk, (int)nh,
              (double *)vn.b.buf, (double *)vs.b.buf, (double *)vc.b.buf, (double *)vb.b.buf);
    res = Py_None;
    Py_INCREF(res);
done:
    view_drop(&vo);
    view_drop(&vv);
    view_drop(&vp);
    view_drop(&vn);
    view_drop(&vs);
    view_drop(&vc);
    view_drop(&vb);
    return res;
}

/* predict_check_take(ids, sel, objs (n, P_NO), ground (n, 2), actual (R, nt, 2), valid (R, nt), k) -> taken
 * A kept batch of predictions takes where its objects are now (sample k of its times): for each of its ids that
 * this frame has (`sel`: {id: row of objs}) and sees (PO_SEEN), the place on the ground. */
static PyObject *n_predict_check_take(PyObject *self, PyObject *args)
{
    PyObject *ids, *sel, *oobj, *og, *oa, *ov, *res = NULL;
    Py_ssize_t k, n, ng, ra, rv;
    View vobj, vg, va, vv;
    vobj.held = vg.held = va.held = vv.held = 0;
    if (!PyArg_ParseTuple(args, "O!O!OOOOn", &PyList_Type, &ids, &PyDict_Type, &sel, &oobj, &og, &oa, &ov, &k))
        return NULL;
    if (pview(oobj, &vobj, 0, P_NO, &n, "objs") < 0 || pview(og, &vg, 0, 2, &ng, "ground") < 0 ||
        pview(oa, &va, 1, 2, &ra, "actual") < 0 || pview(ov, &vv, 1, 1, &rv, "valid") < 0)
        goto done;
    Py_ssize_t R = PyList_Size(ids), nt = R > 0 ? rv / R : 0;
    if (ng != n || (R > 0 && (rv % R || ra != rv || k < 0 || k >= nt))) {
        PyErr_SetString(PyExc_ValueError, "predict_check_take: array sizes do not fit together");
        goto done;
    }
    const double *O = (const double *)vobj.b.buf, *G = (const double *)vg.b.buf;
    double *A = (double *)va.b.buf, *V = (double *)vv.b.buf;
    long taken = 0;
    for (Py_ssize_t r = 0; r < R; r++) {
        PyObject *row = PyDict_GetItemWithError(sel, PyList_GetItem(ids, r));
        if (row == NULL) {
            if (PyErr_Occurred())
                goto done;
            continue;
        }
        Py_ssize_t j = PyLong_AsSsize_t(row);
        if (j < 0 || j >= n) {
            if (PyErr_Occurred())
                goto done;
            continue;
        }
        if (!(O[j * P_NO + PO_SEEN] > 0.5))
            continue;
        A[(r * nt + k) * 2] = G[2 * j];
        A[(r * nt + k) * 2 + 1] = G[2 * j + 1];
        V[r * nt + k] = 1.0;
        taken++;
    }
    res = PyLong_FromLong(taken);
done:
    view_drop(&vobj);
    view_drop(&vg);
    view_drop(&va);
    view_drop(&vv);
    return res;
}

/* predict_score_batch(traj (R, P_KMAX, nt, 6), nm (R), p (R, P_KMAX), actual (R, nt, 2), valid (R, nt), kind (R),
 *                     min_seen, hk, n, sums, cal, brier) -> predictions scored
 * A kept batch whose horizon has passed: every prediction that was seen at min_seen of its samples at least is
 * scored against what its object did (predict_score) and folded into the statistics (predict_score_add). */
static PyObject *n_predict_score_batch(PyObject *self, PyObject *args)
{
    PyObject *o[10], *ohk, *res = NULL;
    double min_seen, *out = NULL;
    View v[10];
    Py_ssize_t rows[10], hk[16];
    static const int WIDTH[10] = {6, 1, P_KMAX, 2, 1, 1, 1, 5, 2, 2};
    static const int WR[10] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1};
    static const char *const WHAT[10] = {"traj", "nm", "p", "actual", "valid", "kind", "n", "sums", "cal", "brier"};
    for (int i = 0; i < 10; i++)
        v[i].held = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOdO!OOOO", &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &min_seen, &PyList_Type, &ohk, &o[6],
                          &o[7], &o[8], &o[9]))
        return NULL;
    for (int i = 0; i < 10; i++)
        if (pview(o[i], &v[i], WR[i], WIDTH[i], &rows[i], WHAT[i]) < 0)
            goto done;
    Py_ssize_t R = rows[1], nt = R > 0 ? rows[4] / R : 0, nh = PyList_Size(ohk);
    if (nh > 16 || rows[2] != R || rows[5] != R || (R > 0 && (rows[4] % R || rows[3] != rows[4] || rows[0] != R * P_KMAX * nt)) ||
        rows[6] != 4 * nh || rows[7] != 4 * nh || rows[8] != 10 || rows[9] != 1) {
        PyErr_SetString(PyExc_ValueError, "predict_score_batch: array sizes do not fit together");
        goto done;
    }
    for (Py_ssize_t h = 0; h < nh; h++) {
        hk[h] = PyLong_AsSsize_t(PyList_GetItem(ohk, h));
        if (hk[h] < 0 || hk[h] >= nt) {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_IndexError, "predict_score_batch: a horizon beyond the samples");
            goto done;
        }
    }
    out = (double *)PyMem_Calloc((size_t)(nt + 2) * 6, sizeof(double));
    if (out == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    const double *TR = (const double *)v[0].b.buf, *NM = (const double *)v[1].b.buf, *P = (const double *)v[2].b.buf;
    const double *ACT = (const double *)v[3].b.buf, *VAL = (const double *)v[4].b.buf, *KIND = (const double *)v[5].b.buf;
    long scored = 0;
    for (Py_ssize_t r = 0; r < R; r++) {
        const double *val = VAL + r * nt;
        double seen = 0.0;
        for (Py_ssize_t k = 0; k < nt; k++)
            seen += val[k];
        int K = (int)NM[r];
        if (seen < min_seen)
            continue;
        if (K < 1 || K > P_KMAX) {
            PyErr_SetString(PyExc_ValueError, "predict_score: array sizes do not fit together");
            goto done;
        }
        double pw;
        int win = score_one(TR + (size_t)r * P_KMAX * nt * 6, K, nt, P + r * P_KMAX, ACT + r * nt * 2, val, out, 1, &pw);
        score_add((int)KIND[r], out, nt, val, P + r * P_KMAX, K, win, hk, (int)nh, (double *)v[6].b.buf, (double *)v[7].b.buf,
                  (double *)v[8].b.buf, (double *)v[9].b.buf);
        scored++;
    }
    res = PyLong_FromLong(scored);
done:
    PyMem_Free(out);
    for (int i = 0; i < 10; i++)
        view_drop(&v[i]);
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
    vt.held = vm.held = 0;
    if (!PyArg_ParseTuple(args, "OOO!i", &ot, &om, &PyTuple_Type, &names, &dec))
        return NULL;
    if (pk_strings() < 0 || pview(ot, &vt, 0, 6, &rt, "traj") < 0 || pview(om, &vm, 0, 5 * P_KMAX, &n, "modes") < 0)
        goto done;
    if (dec < 0 || dec > 9 || (n > 0 && rt % (n * P_KMAX) != 0) || (n > 0 && rt / (n * P_KMAX) < 1)) {
        PyErr_SetString(PyExc_ValueError, "predict_futs: array sizes do not fit together");
        goto done;
    }
    res = pm_futs((const double *)vt.b.buf, (const double *)vm.b.buf, n, n > 0 ? rt / (n * P_KMAX) : 0, names, dec);
done:
    view_drop(&vt);
    view_drop(&vm);
    return res;
}

/* predict_handoff(traj (n, P_KMAX, nt, 6), modes (n, 5, P_KMAX), lane (n, 7), ego_out (P_EMODES, 2),
 *                 ego_traj (P_EMODES, nt, 3), ego_n, names, decimals)
 *   -> (modes, names, rows, lanes, ego | None, ego_names, ego_v_bad)
 * The prediction as the decision layer takes it in process (chris_integrator/decide/futures.py `Futures`), with the
 * numbers the `radar_fut` message would carry (predict_futs, Prediction.ego_fut, lane_dict: rint(v * 10^d) / 10^d,
 * probabilities and lanes to three decimals), so a decision is the same from the arrays and from the message:
 *   modes   bytes of float64 rows (2 + 5 nt): p0, p, x[nt], y[nt], sx[nt], sy[nt], rxy[nt], of the objects whose
 *           futures are whole (every number finite), in the objects' order
 *   names   their manoeuvre names
 *   rows    bytes of float64 (n, 3): first mode, modes, state (1: futures as the contract says; 2: not)
 *   lanes   bytes of float64 (n, 7)
 *   ego     bytes of float64 rows (1 + 3 nt): p, x, y, v of the car's futures; None without one, or where a place
 *           is not finite
 *   ego_v_bad  1 where a speed of the car's futures is not finite (the caller derives it from the places) */
static PyObject *n_predict_handoff(PyObject *self, PyObject *args)
{
    PyObject *ot, *om, *ol, *oeo, *oet, *names, *res = NULL;
    PyObject *bm = NULL, *lnames = NULL, *br = NULL, *bl = NULL, *be = NULL, *enames = NULL;
    int dec, ego_n;
    View vt, vm, vl, veo, vet;
    Py_ssize_t rt, n, nl, neo, ret;
    vt.held = vm.held = vl.held = veo.held = vet.held = 0;
    if (!PyArg_ParseTuple(args, "OOOOOiO!i", &ot, &om, &ol, &oeo, &oet, &ego_n, &PyTuple_Type, &names, &dec))
        return NULL;
    if (pview(ot, &vt, 0, 6, &rt, "traj") < 0 || pview(om, &vm, 0, 5 * P_KMAX, &n, "modes") < 0 ||
        pview(ol, &vl, 0, 7, &nl, "lane") < 0 || pview(oeo, &veo, 0, 2, &neo, "ego_out") < 0 ||
        pview(oet, &vet, 0, 3, &ret, "ego_traj") < 0)
        goto done;
    if (dec < 0 || dec > 9 || (n > 0 && (rt % (n * P_KMAX) != 0 || rt / (n * P_KMAX) < 1)) || nl != n || ego_n < 0 || ego_n > neo || (neo > 0 && ret % neo != 0)) {
        PyErr_SetString(PyExc_ValueError, "predict_handoff: array sizes do not fit together");
        goto done;
    }
    Py_ssize_t nt = n > 0 ? rt / (n * P_KMAX) : (neo > 0 ? ret / neo : 0), nn = PyTuple_Size(names);
    if (ego_n > 0 && ret / neo != nt) {
        PyErr_SetString(PyExc_ValueError, "predict_handoff: the car's futures and the objects' differ in length");
        goto done;
    }
    const double *t = (const double *)vt.b.buf, *M = (const double *)vm.b.buf, *LN = (const double *)vl.b.buf;
    const double *EO = (const double *)veo.b.buf, *ET = (const double *)vet.b.buf;
    const double sc = POW10[dec];
    const Py_ssize_t w = 2 + 5 * nt;
    Py_ssize_t K = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        int nm = (int)M[i * 5 * P_KMAX];
        K += nm > 0 && nm <= P_KMAX ? nm : 0;
    }
    bm = PyBytes_FromStringAndSize(NULL, K * w * (Py_ssize_t)sizeof(double));
    br = PyBytes_FromStringAndSize(NULL, n * 3 * (Py_ssize_t)sizeof(double));
    bl = PyBytes_FromStringAndSize(NULL, n * 7 * (Py_ssize_t)sizeof(double));
    lnames = PyList_New(0);
    enames = PyList_New(0);
    if (!bm || !br || !bl || !lnames || !enames)
        goto done;
    double *MO = (double *)PyBytes_AsString(bm), *RO = (double *)PyBytes_AsString(br), *LO = (double *)PyBytes_AsString(bl);
    Py_ssize_t k_out = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        const double *m = M + i * 5 * P_KMAX;
        int nm = (int)m[0], ok = nm > 0 && nm <= P_KMAX;
        for (int q = 0; q < 7; q++)
            LO[7 * i + q] = rint(LN[7 * i + q] * 1000.0) / 1000.0;
        for (int k = 0; ok && k < nm; k++) {
            const double *tr = t + ((size_t)i * P_KMAX + k) * nt * 6;
            double *row = MO + (k_out + k) * w;
            row[0] = rint(m[4 * P_KMAX + k] * 1000.0) / 1000.0;
            row[1] = rint(m[3 * P_KMAX + k] * 1000.0) / 1000.0;
            for (int q = 0; q < 5 && ok; q++)
                for (Py_ssize_t j = 0; j < nt; j++) {
                    double v = rint(tr[6 * j + q] * sc) / sc;
                    if (!isfinite(v)) {
                        ok = 0;
                        break;
                    }
                    row[2 + q * nt + j] = v;
                }
            int code = (int)m[2 * P_KMAX + k];
            if (code < 0 || code >= nn) {
                PyErr_SetString(PyExc_ValueError, "predict_handoff: a name code without a name");
                goto done;
            }
        }
        RO[3 * i] = (double)k_out;
        RO[3 * i + 1] = ok ? (double)nm : 0.0;
        RO[3 * i + 2] = ok ? (m[1] > 0.0 ? 3.0 + (m[1] < 0.999 ? rint(m[1] * 1000.0) / 1000.0 : 0.999) : 1.0) : 2.0;      /* 3: as the contract says, the lanes' weight in the futures */
        if (ok) {
            for (int k = 0; k < nm; k++)
                if (PyList_Append(lnames, PyTuple_GetItem(names, (int)m[2 * P_KMAX + k])) < 0)
                    goto done;
            k_out += nm;
        }
    }
    if (k_out < K) {                               /* an object's futures were not whole: the rows written are fewer */
        PyObject *cut = PyBytes_FromStringAndSize(PyBytes_AsString(bm), k_out * w * (Py_ssize_t)sizeof(double));
        if (cut == NULL)
            goto done;
        Py_DECREF(bm);
        bm = cut;
    }
    int ego_ok = ego_n > 0, v_bad = 0;
    if (ego_ok) {
        const Py_ssize_t we = 1 + 3 * nt;
        be = PyBytes_FromStringAndSize(NULL, ego_n * we * (Py_ssize_t)sizeof(double));
        if (be == NULL)
            goto done;
        double *E = (double *)PyBytes_AsString(be);
        for (int k = 0; k < ego_n; k++) {
            double *row = E + k * we;
            row[0] = rint(EO[2 * k] * 1000.0) / 1000.0;
            for (Py_ssize_t j = 0; j < nt; j++)
                for (int q = 0; q < 3; q++) {
                    double v = rint(ET[((size_t)k * nt + j) * 3 + q] * sc) / sc;
                    if (!isfinite(v)) {
                        if (q == 2)
                            v_bad = 1;
                        else
                            ego_ok = 0;
                    }
                    row[1 + q * nt + j] = v;
                }
            int code = (int)EO[2 * k + 1];
            if (code < 0 || code >= nn) {
                PyErr_SetString(PyExc_ValueError, "predict_handoff: a name code without a name");
                goto done;
            }
            if (PyList_Append(enames, PyTuple_GetItem(names, code)) < 0)
                goto done;
        }
    }
    res = Py_BuildValue("(OOOOOOi)", bm, lnames, br, bl, ego_ok ? be : Py_None, enames, v_bad);
done:
    Py_XDECREF(bm);
    Py_XDECREF(lnames);
    Py_XDECREF(br);
    Py_XDECREF(bl);
    Py_XDECREF(be);
    Py_XDECREF(enames);
    view_drop(&vt);
    view_drop(&vm);
    view_drop(&vl);
    view_drop(&veo);
    view_drop(&vet);
    return res;
}

#define PREDICT_METHODS                                                                                                \
    {"predict_cfg", n_predict_cfg, METH_NOARGS, "the predictor's constants: {name: default}, in the array's order"},  \
    {"predict_names", n_predict_names, METH_NOARGS, "(manoeuvre names, lane names, intent slots, array widths)"},     \
    {"predict_frame", n_predict_frame, METH_VARARGS, "one frame of the predictor: lanes, intents, futures"},          \
    {"predict_futs", n_predict_futs, METH_VARARGS, "the futures' message form: per object a list of dicts"},      \
    {"predict_handoff", n_predict_handoff, METH_VARARGS, "the prediction as the decision layer takes it in process"}, \
    {"predict_pack", n_predict_pack, METH_VARARGS, "a radar message's objects as predict_frame reads them"},        \
    {"predict_message", n_predict_message, METH_VARARGS, "the radar_fut message of a prediction, or a part of it"},  \
    {"predict_check_take", n_predict_check_take, METH_VARARGS, "a kept batch takes where its objects are now"},      \
    {"predict_score_batch", n_predict_score_batch, METH_VARARGS, "a kept batch scored into the self-check's sums"},  \
    {"predict_score_add", n_predict_score_add, METH_VARARGS, "one scored prediction into the self-check's sums"},    \
    {"predict_score", n_predict_score, METH_VARARGS, "a past prediction against what the object did"},
