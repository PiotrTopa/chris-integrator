/* egomotion.c: ego-motion hygiene of the estimator (docs/EGO_MOTION.md). Included by chris_native.c (one
 * translation unit, before the filter pass); its module functions are EGOMOTION_METHODS, its types are added by
 * egomotion_types().
 *
 *   Stand     is the car at rest? The bus's standstill flag (speed and wheels read zero: they do below a floor,
 *             while the car still rolls and turns) confirmed by the inertial sensors' quiet and by the camera.
 *   Lag       the time offset of one row class against the source that holds its quantity: a slow, robust,
 *             excitation-gated regression of the rows' first innovations on the reference signal's rate.
 *   ekf_h     the measurement functions of the estimator's rows (EH_* kinds), evaluated by the filter pass.
 *
 * Plain IEEE arithmetic in the order written; libm's sqrt, exp, sin, cos, fabs. */

/* ---- small helpers ------------------------------------------------------------------------------------------ */
static int eg_finite(double v) { return v == v && v - v == 0.0; }

static double eg_median(double *a, int n)             /* sorts a (n small) */
{
    for (int i = 1; i < n; i++) {
        double v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = v;
    }
    return n % 2 ? a[n / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
}

/* ---- Stand: the standstill, confirmed ------------------------------------------------------------------------
 * bus(t, flag, gyro, ax, ay, bias, bias_sd) -> state   one sample of the vehicle feed (NaN: the signal is not
 *                                                      there); bias: the gyro bias as estimated, and its sigma
 * cam(t, vx, yaw_rate)                                 one camera pose
 * state: 0 moving (the bus does not say standstill), 1 claimed (the bus says so, not confirmed), 2 confirmed.
 *
 * Witnesses:
 *   inertial   over the last `win` s of bus samples (at least 4, spanning 0.6 win): the gyro's span (max - min),
 *              its mean, and the accelerometers' spans.
 *                against   the gyro's span beyond `move_k` x its quiet span, or its mean beyond `gyro_max` (no
 *                          gyro bias is larger; a car turning on the spot reads more)
 *                for       every signal that is there within its quiet span, and the gyro's mean where its bias
 *                          is believed to be (within 3 sigma of the estimate, at least `mean_tol`)
 *                offset    quiet, but the gyro's mean is not where its bias is believed to be: a steady slow
 *                          turn under the speed sensor's floor, or a bias that has changed
 *                unsettled the gyro's span between quiet and moving: not yet
 *                silent    the gyro quiet where it should be, the accelerometers not (they never speak against:
 *                          they withhold the "for")
 *   camera     the last poses within `cam_fresh` s (at least 3): medians of |vx| and |yaw rate|. For: both
 *              under the still limits. Turning: the yaw rate over its moving limit (its speed never speaks
 *              against: a camera at night reports speeds of a standing car; and its quiet yaw rate does not
 *              settle an inertial "offset": on the reference car it read no turn while the gyro read 0.02 rad/s
 *              under the floor).
 * Confirmed, once the flag has held `dwell` s: inertial for and the camera not turning; inertial offset once
 * the gyro has been quiet about one mean for `offset_dwell` s (nobody turns that steadily for that long: then
 * the bias has changed, and is learnt anew); inertial silent and the camera for; no witness speaking either way after
 * `alone_dwell` s (a feed without inertial signals and no camera). */
#define ST_N 48
#define ST_C 12
enum { SE_INERTIAL_FOR = 1, SE_INERTIAL_AGAINST = 2, SE_CAMERA_FOR = 4, SE_CAMERA_AGAINST = 8, SE_BUS_ONLY = 16,
       SE_DWELL = 32, SE_INERTIAL_OFFSET = 64 };
typedef struct {
    PyObject_HEAD
    double dwell, alone_dwell, win, gyro_span, gyro_max, acc_span, move_k, cam_speed, cam_yaw, cam_move_speed,
        cam_move_yaw, cam_fresh, gap, mean_tol, offset_dwell;
    double bt[ST_N], bg[ST_N], bax[ST_N], bay[ST_N];
    int bn, bh;
    double ct[ST_C], cv[ST_C], cw[ST_C];
    int cn, ch;
    double flag_since, last_t, since, gyro_mean, quiet_since, quiet_mean;
    int has_flag, has_t, state, evidence, has_mean, was_quiet;
    long long n_bus, n_claimed, n_confirmed, n_events;
} Stand;

static PyObject *StandType;

static PyObject *stand_new(PyTypeObject *tp, PyObject *args, PyObject *kw)
{
    double c[15];
    if (!PyArg_ParseTuple(args, "ddddddddddddddd", c, c + 1, c + 2, c + 3, c + 4, c + 5, c + 6, c + 7, c + 8, c + 9,
                          c + 10, c + 11, c + 12, c + 13, c + 14))
        return NULL;
    allocfunc alloc = (allocfunc)PyType_GetSlot(tp, Py_tp_alloc);
    Stand *s = (Stand *)alloc(tp, 0);
    if (s == NULL)
        return NULL;
    s->dwell = c[0], s->alone_dwell = c[1], s->win = c[2], s->gyro_span = c[3], s->gyro_max = c[4], s->acc_span = c[5];
    s->move_k = c[6], s->cam_speed = c[7], s->cam_yaw = c[8], s->cam_move_speed = c[9], s->cam_move_yaw = c[10];
    s->cam_fresh = c[11], s->gap = c[12], s->mean_tol = c[13], s->offset_dwell = c[14];
    return (PyObject *)s;
}

static void eg_dealloc(PyObject *self)
{
    PyTypeObject *tp = Py_TYPE(self);
    freefunc f = (freefunc)PyType_GetSlot(tp, Py_tp_free);
    f(self);
    Py_DECREF(tp);
}

/* one signal's verdict over the window: 1 quiet, -1 moving, 0 between, 2 not there, 3 not yet. *mean: its mean */
static int stand_signal(const Stand *s, const double *v, double t, double quiet, double *mean)
{
    double lo = 0.0, hi = 0.0, sum = 0.0, t0 = 0.0, t1 = 0.0;
    int n = 0;
    for (int k = 0; k < s->bn; k++) {
        int i = (s->bh - 1 - k + 2 * ST_N) % ST_N;
        if (s->bt[i] < t - s->win)
            break;
        if (!eg_finite(v[i]))
            continue;
        if (n == 0)
            lo = hi = v[i], t1 = s->bt[i];
        if (v[i] < lo)
            lo = v[i];
        if (v[i] > hi)
            hi = v[i];
        sum += v[i];
        t0 = s->bt[i];
        n++;
    }
    if (n == 0)
        return 2;
    if (n < 4 || t1 - t0 < 0.6 * s->win)
        return 3;                                      /* there, the window not yet full */
    if (mean != NULL)
        *mean = sum / n;
    double span = hi - lo;
    return span <= quiet ? 1 : (span > s->move_k * quiet ? -1 : 0);
}

static int stand_eval(Stand *s, double t, double bias, double bias_sd)
{
    int ev = 0;
    if (!s->has_flag) {
        if (s->state != 0)
            s->since = t;
        s->state = 0;
        s->evidence = 0;
        s->was_quiet = 0;
        return 0;
    }
    double gm = 0.0;
    int g = stand_signal(s, s->bg, t, s->gyro_span, &gm);
    int ax = stand_signal(s, s->bax, t, s->acc_span, NULL), ay = stand_signal(s, s->bay, t, s->acc_span, NULL);
    enum { I_NONE, I_SILENT, I_FOR, I_OFFSET, I_UNSETTLED, I_AGAINST } inertial = I_NONE;
    if (g != 2 || ax != 2 || ay != 2) {
        int accel_ok = (ax == 1 || ax == 2) && (ay == 1 || ay == 2);
        inertial = I_SILENT;
        if (g == -1 || ((g == 0 || g == 1) && fabs(gm) > s->gyro_max))
            inertial = I_AGAINST;
        else if (g == 1) {
            double tol = 3.0 * bias_sd;
            if (!(tol > s->mean_tol))
                tol = s->mean_tol;
            int mean_ok = !eg_finite(bias) || fabs(gm - bias) <= tol;
            if (!mean_ok)
                inertial = I_OFFSET;
            else if (accel_ok)
                inertial = I_FOR;
        } else if (g == 0 || g == 3 || ax == 3 || ay == 3)
            inertial = I_UNSETTLED;                    /* the gyro is neither quiet nor clearly moving: not yet */
        else if (g == 2 && accel_ok)
            inertial = I_FOR;                          /* (no gyro: the accelerometers' quiet is what there is) */
    }
    s->has_mean = (g == 1);
    if (g == 1 && inertial != I_AGAINST) {
        /* the gyro's quiet about one mean: it begins anew when the mean has moved by half a quiet span (a turn
         * that fades is not a quiet), and ends with a moving span or a mean beyond any bias; a span just over the
         * quiet one does not end it */
        s->gyro_mean = gm;
        if (!s->was_quiet || fabs(gm - s->quiet_mean) > 0.5 * s->gyro_span) {
            s->quiet_since = t;
            s->quiet_mean = gm;
        }
        s->was_quiet = 1;
    } else if (g != 0 || inertial == I_AGAINST)
        s->was_quiet = 0;
    int cam_for = 0, cam_turn = 0;
    double av[ST_C], aw[ST_C];
    int nc = 0;
    for (int k = 0; k < s->cn; k++) {
        int i = (s->ch - 1 - k + 2 * ST_C) % ST_C;
        if (s->ct[i] < t - s->cam_fresh)
            break;
        if (s->ct[i] > t + 1e-6)                       /* (a pose stamped after this sample: not yet) */
            continue;
        av[nc] = fabs(s->cv[i]);
        aw[nc] = fabs(s->cw[i]);
        nc++;
    }
    if (nc >= 3) {
        double mv = eg_median(av, nc), mw = eg_median(aw, nc);
        cam_turn = mw > s->cam_move_yaw;
        cam_for = !cam_turn && mv < s->cam_speed && mw < s->cam_yaw;
    }
    if (inertial == I_FOR)
        ev |= SE_INERTIAL_FOR;
    if (inertial == I_AGAINST)
        ev |= SE_INERTIAL_AGAINST;
    if (inertial == I_OFFSET)
        ev |= SE_INERTIAL_OFFSET;
    if (cam_for)
        ev |= SE_CAMERA_FOR;
    if (cam_turn)
        ev |= SE_CAMERA_AGAINST;
    double held = t - s->flag_since;
    int ok = 0;
    if (inertial == I_AGAINST || inertial == I_UNSETTLED || cam_turn)
        ok = 0;
    else if (inertial == I_FOR)
        ok = held >= s->dwell;
    else if (inertial == I_OFFSET)                     /* a slow steady turn, or a bias that has changed? */
        ok = held >= s->dwell && t - s->quiet_since >= s->offset_dwell;
    else if (cam_for)
        ok = held >= s->dwell;
    else {
        ev |= SE_BUS_ONLY;
        ok = held >= s->alone_dwell;
    }
    if (held >= s->dwell)
        ev |= SE_DWELL;
    int st = ok ? 2 : 1;
    if (st != s->state) {
        s->since = t;
        if (st == 2)
            s->n_events++;
    }
    s->state = st;
    s->evidence = ev;
    return st;
}

static PyObject *stand_bus(PyObject *self, PyObject *args)
{
    Stand *s = (Stand *)self;
    double t, g, ax, ay, bias, bias_sd;
    int flag;
    if (!PyArg_ParseTuple(args, "dpddddd", &t, &flag, &g, &ax, &ay, &bias, &bias_sd))
        return NULL;
    if (s->has_t && t < s->last_t - 1.0) {             /* the clock went back: forget */
        s->bn = s->bh = s->cn = s->ch = 0;
        s->has_flag = 0;
    }
    if (flag) {
        if (!s->has_flag || (s->has_t && t - s->last_t > s->gap))
            s->flag_since = t;
        s->has_flag = 1;
    } else
        s->has_flag = 0;
    s->bt[s->bh] = t, s->bg[s->bh] = g, s->bax[s->bh] = ax, s->bay[s->bh] = ay;
    s->bh = (s->bh + 1) % ST_N;
    if (s->bn < ST_N)
        s->bn++;
    s->last_t = t;
    s->has_t = 1;
    s->n_bus++;
    int st = stand_eval(s, t, bias, bias_sd);
    if (st == 1)
        s->n_claimed++;
    else if (st == 2)
        s->n_confirmed++;
    return PyLong_FromLong(st);
}

static PyObject *stand_cam(PyObject *self, PyObject *args)
{
    Stand *s = (Stand *)self;
    double t, v, w;
    if (!PyArg_ParseTuple(args, "ddd", &t, &v, &w))
        return NULL;
    if (eg_finite(t) && eg_finite(v) && eg_finite(w)) {
        s->ct[s->ch] = t, s->cv[s->ch] = v, s->cw[s->ch] = w;
        s->ch = (s->ch + 1) % ST_C;
        if (s->cn < ST_C)
            s->cn++;
    }
    Py_RETURN_NONE;
}

static PyObject *stand_reset(PyObject *self, PyObject *unused)
{
    Stand *s = (Stand *)self;
    s->bn = s->bh = s->cn = s->ch = 0;
    s->has_flag = s->has_t = s->state = s->evidence = s->has_mean = 0;
    Py_RETURN_NONE;
}

static PyObject *stand_get_state(PyObject *self, void *c) { return PyLong_FromLong(((Stand *)self)->state); }
static PyObject *stand_get_evidence(PyObject *self, void *c) { return PyLong_FromLong(((Stand *)self)->evidence); }
static PyObject *stand_get_since(PyObject *self, void *c) { return PyFloat_FromDouble(((Stand *)self)->since); }
static PyObject *stand_get_counts(PyObject *self, void *c)
{
    Stand *s = (Stand *)self;
    return Py_BuildValue("(LLLL)", s->n_bus, s->n_claimed, s->n_confirmed, s->n_events);
}
static PyObject *stand_get_gyro(PyObject *self, void *c)
{
    Stand *s = (Stand *)self;
    if (!s->has_mean || s->state != 2)
        Py_RETURN_NONE;
    return PyFloat_FromDouble(s->gyro_mean);
}

static PyMethodDef stand_methods[] = {
    {"bus", stand_bus, METH_VARARGS, "bus(t, flag, gyro, accel_long, accel_lat, bias, bias_sd) -> 0 moving | 1 claimed | 2 confirmed"},
    {"cam", stand_cam, METH_VARARGS, "cam(t, vx, yaw_rate): one camera pose"},
    {"reset", stand_reset, METH_NOARGS, "forget (a new epoch)"},
    {NULL, NULL, 0, NULL}};
static PyGetSetDef stand_getset[] = {
    {"state", stand_get_state, NULL, "0 moving | 1 claimed | 2 confirmed", NULL},
    {"evidence", stand_get_evidence, NULL, "bits: 1 inertial for, 2 inertial against, 4 camera for, 8 camera turning, 16 bus only, 32 dwell, 64 inertial offset", NULL},
    {"since", stand_get_since, NULL, "the time the state began", NULL},
    {"counts", stand_get_counts, NULL, "(bus samples, claimed only, confirmed, confirmations)", NULL},
    {"gyro_mean", stand_get_gyro, NULL, "the gyro's mean over the window while confirmed (its bias as read), else None", NULL},
    {NULL, NULL, NULL, NULL, NULL}};
static PyType_Slot stand_slots[] = {
    {Py_tp_new, stand_new}, {Py_tp_dealloc, eg_dealloc}, {Py_tp_methods, stand_methods}, {Py_tp_getset, stand_getset},
    {Py_tp_doc, "Stand(dwell, alone_dwell, win, gyro_span, gyro_max, acc_span, move_k, cam_speed, cam_yaw, cam_move_speed, "
                "cam_move_yaw, cam_fresh, gap, mean_tol, offset_dwell): the bus's standstill confirmed by the inertial "
                "sensors and the camera"},
    {0, NULL}};
static PyType_Spec stand_spec = {"chris_integrator.native._chris_native.Stand", sizeof(Stand), 0, Py_TPFLAGS_DEFAULT,
                                 stand_slots};

/* ---- Lag: the time offset of a row class ---------------------------------------------------------------------
 * A row stamped t by a source whose rows are `o` s late is inserted at t - applied; if o is not `applied`, its
 * innovation is nu = -(o - applied) r + noise, with r the rate of the measured quantity at the row's time. r is
 * taken from the reference signal itself (the samples of the source that holds the quantity: `ref`), as the
 * slope of a line through its samples within +- `span` s of the row: a row waits until the reference has
 * samples on both sides.
 *
 *   offer(t, nu, s, applied)   a row's first innovation, its variance, the offset it was inserted with
 *   ref(t, value)              a sample of the reference signal
 *
 * Only rows under excitation count: |r| at least `r_min` and `clear` sigma clear of the slope's own noise (a
 * noisy regressor biases what it teaches: the regressor is shrunk by its variance as the filter's scale rows
 * do). The estimate is the posterior of a Gaussian prior (the configured value and its sigma) and the weighted
 * rows (Huber weights, redescending beyond 3 k; the innovation variance inflated by `corr`: rows are correlated in time); the rows'
 * information halves every `half_life` s of excitation, so the estimate follows a slow change and falls back
 * to the configured value when the rows stop saying anything. Clamped to [lo, hi]. */
#define LG_R 96
#define LG_P 48
typedef struct {
    PyObject_HEAD
    double prior, prior_var, lo, hi, half_life, span, r_min, clear, huber, corr, ref_sigma;
    double rt[LG_R], rv[LG_R];
    int rn, rh;
    double pt[LG_P], pnu[LG_P], ps[LG_P], pa[LG_P];
    int pn;
    double A, b, value, last_t, excited_s, r_rms;
    int has_last;
    long long n, n_gated, n_dropped;
} Lag;

static PyObject *LagType;

static PyObject *lag_new(PyTypeObject *tp, PyObject *args, PyObject *kw)
{
    double c[11];
    if (!PyArg_ParseTuple(args, "ddddddddddd", c, c + 1, c + 2, c + 3, c + 4, c + 5, c + 6, c + 7, c + 8, c + 9, c + 10))
        return NULL;
    if (!(c[1] > 0.0) || !(c[4] > 0.0) || !(c[5] > 0.0) || !(c[2] <= c[3])) {
        PyErr_SetString(PyExc_ValueError, "Lag: prior sigma, half life and span must be positive, lo <= hi");
        return NULL;
    }
    allocfunc alloc = (allocfunc)PyType_GetSlot(tp, Py_tp_alloc);
    Lag *g = (Lag *)alloc(tp, 0);
    if (g == NULL)
        return NULL;
    g->prior = c[0], g->prior_var = c[1] * c[1], g->lo = c[2], g->hi = c[3], g->half_life = c[4], g->span = c[5];
    g->r_min = c[6], g->clear = c[7], g->huber = c[8], g->corr = c[9], g->ref_sigma = c[10];
    g->value = c[0] < c[2] ? c[2] : (c[0] > c[3] ? c[3] : c[0]);
    return (PyObject *)g;
}

static void lag_solve(Lag *g)
{
    /* the prior moved by what the rows say (with no row: the prior itself, to the bit) */
    double v = g->prior + (g->b - g->A * g->prior) / (1.0 / g->prior_var + g->A);
    g->value = v < g->lo ? g->lo : (v > g->hi ? g->hi : v);
}

/* the slope of the reference about time t: 1 with (r, var_r), 0 not yet (no sample late enough), -1 never */
static int lag_slope(const Lag *g, double t, double *r, double *var_r)
{
    if (g->rn == 0)
        return 0;
    int newest = (g->rh - 1 + LG_R) % LG_R;
    if (g->rt[newest] < t + g->span)
        return 0;
    double st = 0.0, sv = 0.0, tmin = 0.0, tmax = 0.0, prev = 0.0, gap = 0.0;
    int n = 0;
    for (int k = 0; k < g->rn; k++) {
        int i = (g->rh - 1 - k + 2 * LG_R) % LG_R;
        double d = g->rt[i] - t;
        if (d > g->span)
            continue;
        if (d < -g->span)
            break;
        if (n == 0)
            tmax = d;
        else if (prev - d > gap)
            gap = prev - d;
        prev = tmin = d;
        st += d;
        sv += g->rv[i];
        n++;
    }
    if (n < 4 || tmax < 0.5 * g->span || tmin > -0.5 * g->span || gap > 0.75 * g->span)
        return -1;
    double mt = st / n, mv = sv / n, stt = 0.0, stv = 0.0;
    for (int k = 0; k < g->rn; k++) {
        int i = (g->rh - 1 - k + 2 * LG_R) % LG_R;
        double d = g->rt[i] - t;
        if (d > g->span)
            continue;
        if (d < -g->span)
            break;
        stt += (d - mt) * (d - mt);
        stv += (d - mt) * (g->rv[i] - mv);
    }
    if (!(stt > 0.0))
        return -1;
    *r = stv / stt;
    *var_r = g->ref_sigma * g->ref_sigma / stt;
    return 1;
}

static void lag_process(Lag *g)
{
    int w = 0;
    for (int k = 0; k < g->pn; k++) {
        double t = g->pt[k], r = 0.0, vr = 0.0;
        int ok = lag_slope(g, t, &r, &vr);
        if (ok == 0) {                                 /* waits for the reference */
            int newest = (g->rh - 1 + LG_R) % LG_R;
            if (g->rn > 0 && g->rt[newest] - t < -5.0) {
                g->n_dropped++;
                continue;
            }
            g->pt[w] = g->pt[k], g->pnu[w] = g->pnu[k], g->ps[w] = g->ps[k], g->pa[w] = g->pa[k];
            w++;
            continue;
        }
        if (ok < 0) {
            g->n_dropped++;
            continue;
        }
        double lim = g->clear * g->clear * vr;
        if (fabs(r) < g->r_min || r * r <= lim) {
            g->n_gated++;
            continue;
        }
        double re = r * (1.0 - lim / (r * r));
        double nu = g->pnu[k], applied = g->pa[k], s = g->ps[k] * g->corr;
        if (!(s > 0.0))
            continue;
        double e = nu + (g->value - applied) * re;
        /* Huber's weight to 3 k sigma, falling off as 1 / u^2 beyond (a wild row's pull goes to nothing) */
        double u = fabs(e) / sqrt(s), kh = g->huber;
        double wh = (u <= kh) ? 1.0 : (u <= 3.0 * kh ? kh / u : 3.0 * kh * kh / (u * u));
        double dtx = 0.25;
        if (g->has_last) {
            dtx = t - g->last_t;
            dtx = dtx < 0.0 ? 0.0 : (dtx > 0.25 ? 0.25 : dtx);
        }
        double f = exp(-0.6931471805599453 * dtx / g->half_life);
        g->A *= f;
        g->b *= f;
        double wt = wh / s;
        g->A += wt * re * re;
        g->b += wt * re * (applied * re - nu);
        g->excited_s += dtx;
        g->r_rms = sqrt(0.98 * g->r_rms * g->r_rms + 0.02 * re * re);
        g->last_t = t;
        g->has_last = 1;
        g->n++;
        lag_solve(g);
    }
    g->pn = w;
}

static PyObject *lag_offer(PyObject *self, PyObject *args)
{
    Lag *g = (Lag *)self;
    double t, nu, s, applied;
    if (!PyArg_ParseTuple(args, "dddd", &t, &nu, &s, &applied))
        return NULL;
    if (!eg_finite(t) || !eg_finite(nu) || !eg_finite(s) || !eg_finite(applied))
        Py_RETURN_NONE;
    if (g->pn == LG_P) {                               /* full: the oldest goes */
        for (int k = 1; k < LG_P; k++)
            g->pt[k - 1] = g->pt[k], g->pnu[k - 1] = g->pnu[k], g->ps[k - 1] = g->ps[k], g->pa[k - 1] = g->pa[k];
        g->pn--;
        g->n_dropped++;
    }
    g->pt[g->pn] = t, g->pnu[g->pn] = nu, g->ps[g->pn] = s, g->pa[g->pn] = applied;
    g->pn++;
    lag_process(g);
    Py_RETURN_NONE;
}

static PyObject *lag_ref(PyObject *self, PyObject *args)
{
    Lag *g = (Lag *)self;
    double t, v;
    if (!PyArg_ParseTuple(args, "dd", &t, &v))
        return NULL;
    if (!eg_finite(t) || !eg_finite(v))
        Py_RETURN_NONE;
    if (g->rn > 0) {
        int newest = (g->rh - 1 + LG_R) % LG_R;
        if (t <= g->rt[newest]) {
            if (t < g->rt[newest] - 1.0)               /* the clock went back: the buffers are of another time */
                g->rn = g->rh = g->pn = 0, g->has_last = 0;
            else
                Py_RETURN_NONE;
        }
    }
    g->rt[g->rh] = t, g->rv[g->rh] = v;
    g->rh = (g->rh + 1) % LG_R;
    if (g->rn < LG_R)
        g->rn++;
    if (g->pn)
        lag_process(g);
    Py_RETURN_NONE;
}

/* seed(value, sigma): what was learnt before, as the prior of this run */
static PyObject *lag_seed(PyObject *self, PyObject *args)
{
    Lag *g = (Lag *)self;
    double v, sd;
    if (!PyArg_ParseTuple(args, "dd", &v, &sd))
        return NULL;
    if (eg_finite(v) && sd > 0.0 && eg_finite(sd)) {
        g->prior = v, g->prior_var = sd * sd;
        g->A = g->b = 0.0;
        lag_solve(g);
    }
    Py_RETURN_NONE;
}

static PyObject *lag_clear(PyObject *self, PyObject *unused)
{
    Lag *g = (Lag *)self;
    g->rn = g->rh = g->pn = 0;
    g->has_last = 0;
    Py_RETURN_NONE;
}

static PyObject *lag_get_value(PyObject *self, void *c) { return PyFloat_FromDouble(((Lag *)self)->value); }
static PyObject *lag_get_std(PyObject *self, void *c)
{
    Lag *g = (Lag *)self;
    return PyFloat_FromDouble(sqrt(1.0 / (1.0 / g->prior_var + g->A)));
}
static PyObject *lag_get_n(PyObject *self, void *c) { return PyLong_FromLongLong(((Lag *)self)->n); }
static PyObject *lag_get_excited(PyObject *self, void *c) { return PyFloat_FromDouble(((Lag *)self)->excited_s); }
static PyObject *lag_get_counts(PyObject *self, void *c)
{
    Lag *g = (Lag *)self;
    return Py_BuildValue("(LLLi)", g->n, g->n_gated, g->n_dropped, g->pn);
}
static PyObject *lag_get_info(PyObject *self, void *c)
{
    Lag *g = (Lag *)self;                              /* the rows' share of the estimate's information */
    return PyFloat_FromDouble(g->A / (1.0 / g->prior_var + g->A));
}

static PyMethodDef lag_methods[] = {
    {"offer", lag_offer, METH_VARARGS, "offer(t, nu, s, applied): a row's first innovation"},
    {"ref", lag_ref, METH_VARARGS, "ref(t, value): a sample of the reference signal"},
    {"seed", lag_seed, METH_VARARGS, "seed(value, sigma): the prior of this run (a value learnt before)"},
    {"clear", lag_clear, METH_NOARGS, "forget the buffers (a new epoch); the estimate stays"},
    {NULL, NULL, 0, NULL}};
static PyGetSetDef lag_getset[] = {
    {"value", lag_get_value, NULL, "the offset, s", NULL},
    {"std", lag_get_std, NULL, "its standard deviation", NULL},
    {"n", lag_get_n, NULL, "rows used", NULL},
    {"excited_s", lag_get_excited, NULL, "seconds of excitation seen", NULL},
    {"counts", lag_get_counts, NULL, "(used, gated: no excitation, dropped: no reference, waiting)", NULL},
    {"info", lag_get_info, NULL, "the rows' share of the information (0: the configured value alone)", NULL},
    {NULL, NULL, NULL, NULL, NULL}};
static PyType_Slot lag_slots[] = {
    {Py_tp_new, lag_new}, {Py_tp_dealloc, eg_dealloc}, {Py_tp_methods, lag_methods}, {Py_tp_getset, lag_getset},
    {Py_tp_doc, "Lag(prior, prior_sigma, lo, hi, half_life_s, span_s, r_min, clear, huber, corr, ref_sigma): the time "
                "offset of a row class against the source that holds its quantity"},
    {0, NULL}};
static PyType_Spec lag_spec = {"chris_integrator.native._chris_native.Lag", sizeof(Lag), 0, Py_TPFLAGS_DEFAULT, lag_slots};

static int egomotion_types(PyObject *m)
{
    if ((StandType = PyType_FromSpec(&stand_spec)) == NULL || (LagType = PyType_FromSpec(&lag_spec)) == NULL)
        return -1;
    Py_INCREF(StandType);
    Py_INCREF(LagType);
    if (PyModule_AddObject(m, "Stand", StandType) < 0 || PyModule_AddObject(m, "Lag", LagType) < 0)
        return -1;
    return 0;
}

/* ---- the measurement functions of the estimator's rows -------------------------------------------------------
 * A row's model is (kind, idx, par): the prediction and the sparse Jacobian row (over the states `idx`) are
 * computed here, in the filter pass, from the state of the pass: a late row costs no call into Python when the
 * pass is run again. The arithmetic is, operation for operation, what the sources' Python closures did (the
 * results are theirs to the last bit; tests/test_ego_motion.py holds each kind against its expression).
 *
 *   EH_LIN      par [const, c...]          const + sum c_j x[idx_j]                                  | c
 *   EH_SCALE    par [var, floor]           k x[i0] (+ x[i2]),  k = 1 + x[i1]                         | k, reg(x[i0]) (, 1)
 *               par [var, floor, d, nb]    k (x[i0] - d x[ir]) (+ x[i2] where nb),  ir the last index: | k, reg(x[i0]) (, 1), -k d
 *                                          the quantity as it was d s ago, by its rate x[ir] (a source
 *                                          whose value is that old at the time the row is inserted)
 *   EH_ACCEL    par [g]                    x[i0] + g sin(x[i1]) + x[i2]                              | 1, g cos(x[i1]), 1
 *   EH_PRODUCT  par []                     x[i0] x[i1] + x[i2]                                       | x[i1], x[i0], 1
 *   EH_VSPEED   par [tau0, var, floor, cs, cf]   k (x[i0] - a d) + cs x[i2] + cf x[i3],  k = 1 + x[i1],
 *                                          d = x[i4] - tau0, a = the acceleration state               | k, reg(x[i0]), cs, cf, -k a
 *   EH_HEADING  par [tau0]                 x[i0] - w (x[i1] - tau0),  w = the yaw-rate state          | 1, -w
 *
 * reg(v) is the regressor of a scale state shrunk by its own uncertainty (sources/vehicle.py `clear_reg`):
 * 0 where v^2 <= max(9 var, floor^2), else v (1 - that / v^2). */
enum { EH_LIN, EH_SCALE, EH_ACCEL, EH_PRODUCT, EH_VSPEED, EH_HEADING, EH_KINDS };
#define EH_MAXM 8

static double eh_reg(double value, double var, double floor)
{
    double a = value * value;
    double lim = 9.0 * var, f2 = floor * floor;
    if (f2 > lim)
        lim = f2;
    return a <= lim ? 0.0 : value * (1.0 - lim / a);
}

/* 0 with *pred and cf[0..m-1]; -1 with an exception (a model that does not fit its kind) */
static int eh_eval(long kind, const double *x, const int64_t *ix, Py_ssize_t m, const double *par, Py_ssize_t np_,
                   double *pred, double *cf)
{
    static const signed char want_m[EH_KINDS] = {-1, -2, 3, 3, 5, 2}, want_p[EH_KINDS] = {-1, -2, 1, 0, 5, 1};
    int bad = kind < 0 || kind >= EH_KINDS || m < 1 || m > EH_MAXM;
    if (!bad && kind == EH_SCALE)
        bad = !((np_ == 2 && (m == 2 || m == 3)) || (np_ == 4 && m == (par[3] != 0.0 ? 4 : 3)));
    else if (!bad)
        bad = (want_m[kind] > 0 && m != want_m[kind]) || (want_p[kind] >= 0 ? np_ != want_p[kind] : np_ != m + 1);
    if (bad) {
        PyErr_SetString(PyExc_ValueError, "ekf_run: a row's model does not fit its kind");
        return -1;
    }
    switch (kind) {
    case EH_LIN: {
        double acc = 0.0;
        for (Py_ssize_t j = 0; j < m; j++) {
            acc += par[1 + j] * x[ix[j]];
            cf[j] = par[1 + j];
        }
        *pred = par[0] + acc;
        break;
    }
    case EH_SCALE: {
        double k = 1.0 + x[ix[1]], v = x[ix[0]];
        if (np_ == 2) {
            double p = k * v;
            if (m == 3) {
                p = p + x[ix[2]];
                cf[2] = 1.0;
            }
            *pred = p;
        } else {                                       /* the quantity as it was d s ago */
            double d = par[2];
            double p = k * (v - d * x[ix[m - 1]]);
            if (m == 4) {
                p = p + x[ix[2]];
                cf[2] = 1.0;
            }
            *pred = p;
            cf[m - 1] = -k * d;
        }
        cf[0] = k;
        cf[1] = eh_reg(v, par[0], par[1]);
        break;
    }
    case EH_ACCEL: {
        double th = x[ix[1]];
        *pred = x[ix[0]] + par[0] * sin(th) + x[ix[2]];
        cf[0] = 1.0;
        cf[1] = par[0] * cos(th);
        cf[2] = 1.0;
        break;
    }
    case EH_PRODUCT:
        *pred = x[ix[0]] * x[ix[1]] + x[ix[2]];
        cf[0] = x[ix[1]];
        cf[1] = x[ix[0]];
        cf[2] = 1.0;
        break;
    case EH_VSPEED: {
        double k = 1.0 + x[ix[1]], d = x[ix[4]] - par[0];
        *pred = k * (x[ix[0]] - x[A] * d) + par[3] * x[ix[2]] + par[4] * x[ix[3]];
        cf[0] = k;
        cf[1] = eh_reg(x[ix[0]], par[1], par[2]);
        cf[2] = par[3];
        cf[3] = par[4];
        cf[4] = -k * x[A];
        break;
    }
    default:                                           /* EH_HEADING */
        *pred = x[ix[0]] - x[W] * (x[ix[1]] - par[0]);
        cf[0] = 1.0;
        cf[1] = -x[W];
        break;
    }
    return 0;
}

/* ekf_h(kind, idx, par, x) -> (prediction, coefficients): a row's model evaluated (tests, tools) */
static PyObject *n_ekf_h(PyObject *self, PyObject *args)
{
    long kind;
    PyObject *oi, *op, *ox, *res = NULL;
    View iv, pv, xv;
    iv.held = pv.held = xv.held = 0;
    if (!PyArg_ParseTuple(args, "lOOO", &kind, &oi, &op, &ox))
        return NULL;
    if (view_get(oi, &iv, 0, 'i', 1, "idx") < 0 || view_get(op, &pv, 0, 'd', 1, "par") < 0 ||
        view_get(ox, &xv, 0, 'd', 1, "x") < 0)
        goto done;
    Py_ssize_t m = iv.b.shape[0], n = xv.b.shape[0];
    const int64_t *ix = (const int64_t *)iv.b.buf;
    for (Py_ssize_t j = 0; j < m; j++)
        if (ix[j] < 0 || ix[j] >= n) {
            PyErr_SetString(PyExc_IndexError, "ekf_h: an index beyond the state");
            goto done;
        }
    if (n < NCORE) {
        PyErr_SetString(PyExc_ValueError, "ekf_h: a state shorter than the kinematic one");
        goto done;
    }
    double pred, cf[EH_MAXM];
    if (eh_eval(kind, DATA(xv), ix, m, DATA(pv), pv.b.shape[0], &pred, cf) < 0)
        goto done;
    PyObject *lst = PyList_New(m);
    if (lst == NULL)
        goto done;
    for (Py_ssize_t j = 0; j < m; j++)
        PyList_SetItem(lst, j, PyFloat_FromDouble(cf[j]));
    res = Py_BuildValue("(dN)", pred, lst);
done:
    view_drop(&iv); view_drop(&pv); view_drop(&xv);
    return res;
}

#define EGOMOTION_METHODS \
    {"ekf_h", n_ekf_h, METH_VARARGS, "a row's native model evaluated: (kind, idx, par, x) -> (prediction, coefficients)"},
