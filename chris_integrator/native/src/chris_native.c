/* chris-integrator's native core: the arithmetic and the text work of the per-message hot path.
 *
 * One C99 file, the CPython limited API (one binary for every CPython >= 3.11), no numpy headers (arrays are
 * read and written through the buffer protocol), no other dependency than libm. The package requires it, and
 * every function here exists only here (docs/NATIVE.md: the rules); tests/test_native.py holds each to known
 * results (CPython's json and round; golden data frozen from the pure-Python references they replaced).
 *
 *   dumps(obj), loads(raw)     json.dumps(obj, separators=(",", ":")), json.loads: the same text / objects;
 *                              what is not certain raises `Fallback` and json decides (errors are CPython's)
 *   round(x, n), round_list    builtins.round: the same result (builtins.round itself next to a tie)
 *   trim_road, slim_road_line  the road line without what the default stack does not read; the recorder's
 *                              reduced road (fastjson.py, io/recorder.py)
 *   interp_all, view_prep, view_blend     the view's table walk and drawn lines (trackers/view.py)
 *   lanes_measure, lanes_predict, lanes_fit, lanes_apply, round_lists     the lane tracker's array work
 *                              (trackers/lanes.py): numpy's elementwise steps, the same bits
 *   geom_centre, geom_reach    the road geometry's centre line from the camera (trackers/roadgeom.py)
 *   ekf_predict, ekf_project, ekf_update, ekf_sym, ekf_rts     the estimator's filter (core/ekf.py,
 *                              core/ego.py); elementwise steps exact, sums of products in index order
 *   radar_evolve, radar_combine, radar_p_still     the radar's motion model (radar/models.py, tracker.py):
 *                              interaction, prediction, combination of the hypotheses; agree to rounding
 *
 * Compile with -ffp-contract=off: a fused multiply-add would change the bits.
 *
 * SPDX-License-Identifier: MIT
 */
#define Py_LIMITED_API 0x030B0000
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define NATIVE_VERSION 5

static PyObject *Fallback;          /* raised where CPython's json (or the radar's reference) must decide */
static PyObject *builtin_round;

static const double POW10[23] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14,
                                 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};

/* ====================================================================================================
 * Text buffer
 * ==================================================================================================== */
typedef struct {
    char *p;
    size_t n, cap;
} Buf;

static int buf_room(Buf *b, size_t more)
{
    if (b->n + more <= b->cap)
        return 0;
    size_t cap = b->cap ? b->cap : 1024;
    while (cap < b->n + more) {
        if (cap > ((size_t)1 << 40)) {
            PyErr_NoMemory();
            return -1;
        }
        cap *= 2;
    }
    char *q = (char *)PyMem_Realloc(b->p, cap);
    if (q == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    b->p = q;
    b->cap = cap;
    return 0;
}

static inline int buf_put(Buf *b, const char *s, size_t n)
{
    if (n == 0)                                    /* (nothing to copy; the buffer may not exist yet) */
        return 0;
    if (buf_room(b, n) < 0)
        return -1;
    memcpy(b->p + b->n, s, n);
    b->n += n;
    return 0;
}

static inline int buf_ch(Buf *b, char c)
{
    if (b->n == b->cap && buf_room(b, 1) < 0)
        return -1;
    b->p[b->n++] = c;
    return 0;
}

/* ====================================================================================================
 * JSON: writing
 * ==================================================================================================== */
static int put_u64(Buf *b, uint64_t v)
{
    char t[24];
    int i = 24;
    do {
        t[--i] = (char)('0' + v % 10);
        v /= 10;
    } while (v);
    return buf_put(b, t + i, (size_t)(24 - i));
}

/* float.__repr__ for a finite double.
 * repr is the shortest decimal string that reads back as the same double. If the double is k / 10^n for an
 * integer k below 10^15, that string is k with n decimals: decimals of 15 significant digits or fewer and
 * doubles correspond one to one, so there is no second such string and no shorter one. Everything else
 * (16 or 17 digits, or an exponent form: below 1e-4) is left to CPython's own conversion. */
static int put_float(Buf *b, double v)
{
    double a = fabs(v);
    if (a == 0.0)
        return buf_put(b, signbit(v) ? "-0.0" : "0.0", signbit(v) ? 4 : 3);
    if (a >= 1e-4 && a < 1e15) {
        for (int n = 0; n <= 19; n++) {
            double y = a * POW10[n];
            if (!(y < 1e15))
                break;
            double r = rint(y);
            /* a * 10^n is within a few units in the last place of the integer k when a is k / 10^n: only then
             * is the exact test (one division) worth making */
            if (fabs(y - r) > y * 1e-15)
                continue;
            if (r / POW10[n] == a && r >= 1.0) {
                /* the digits of k = r, right to left, with the point n digits from the right */
                uint64_t k = (uint64_t)r;
                char t[48];
                int i = 48;
                if (n == 0)
                    t[--i] = '0';
                for (int d = 0; d < n; d++) {
                    t[--i] = (char)('0' + k % 10);
                    k /= 10;
                }
                t[--i] = '.';
                do {
                    t[--i] = (char)('0' + k % 10);
                    k /= 10;
                } while (k);
                if (v < 0.0)
                    t[--i] = '-';
                return buf_put(b, t + i, (size_t)(48 - i));
            }
        }
    }
    char *s = PyOS_double_to_string(v, 'r', 0, Py_DTSF_ADD_DOT_0, NULL);
    if (s == NULL)
        return -1;
    int rc = buf_put(b, s, strlen(s));
    PyMem_Free(s);
    return rc;
}

static int put_double(Buf *b, double v)
{
    if (v != v)
        return buf_put(b, "NaN", 3);
    if (isinf(v))
        return v > 0 ? buf_put(b, "Infinity", 8) : buf_put(b, "-Infinity", 9);
    return put_float(b, v);
}

static int put_int(Buf *b, PyObject *o)
{
    int over = 0;
    long long v = PyLong_AsLongLongAndOverflow(o, &over);
    if (v == -1 && PyErr_Occurred())
        return -1;
    if (over) {                                    /* a big integer: int.__repr__ as json writes it */
        PyErr_SetNone(Fallback);
        return -1;
    }
    if (v < 0) {
        if (buf_ch(b, '-') < 0)
            return -1;
        return put_u64(b, (uint64_t)0 - (uint64_t)v);
    }
    return put_u64(b, (uint64_t)v);
}

/* a str as json.dumps writes it with ensure_ascii (the default): \" \\ \n \r \t \b \f, \uXXXX for every
 * other character outside 0x20..0x7e, surrogate pairs above U+FFFF */
static int put_str(Buf *b, PyObject *o)
{
    static const char HEX[] = "0123456789abcdef";
    Py_ssize_t len;
    const char *s = PyUnicode_AsUTF8AndSize(o, &len);
    if (s == NULL) {                               /* a lone surrogate: json writes it, UTF-8 cannot hold it */
        PyErr_Clear();
        PyErr_SetNone(Fallback);
        return -1;
    }
    const unsigned char *u = (const unsigned char *)s, *end = u + len;
    if (buf_room(b, (size_t)len + 2) < 0)
        return -1;
    b->p[b->n++] = '"';
    while (u < end) {
        const unsigned char *run = u;
        while (u < end && *u >= 0x20 && *u < 0x7f && *u != '"' && *u != '\\')
            u++;
        if (u > run && buf_put(b, (const char *)run, (size_t)(u - run)) < 0)
            return -1;
        if (u >= end)
            break;
        unsigned c = *u;
        char e[12];
        size_t n = 0;
        if (c < 0x80) {
            u++;
            e[n++] = '\\';
            switch (c) {
            case '"': e[n++] = '"'; break;
            case '\\': e[n++] = '\\'; break;
            case '\n': e[n++] = 'n'; break;
            case '\r': e[n++] = 'r'; break;
            case '\t': e[n++] = 't'; break;
            case '\b': e[n++] = 'b'; break;
            case '\f': e[n++] = 'f'; break;
            default:
                e[n++] = 'u'; e[n++] = '0'; e[n++] = '0';
                e[n++] = HEX[c >> 4]; e[n++] = HEX[c & 15];
            }
        } else {
            /* UTF-8 from CPython is well formed; the lengths are checked all the same */
            uint32_t cp;
            int k;
            if ((c & 0xe0) == 0xc0) { cp = c & 0x1f; k = 1; }
            else if ((c & 0xf0) == 0xe0) { cp = c & 0x0f; k = 2; }
            else if ((c & 0xf8) == 0xf0) { cp = c & 0x07; k = 3; }
            else { PyErr_SetNone(Fallback); return -1; }
            if (end - u <= k) { PyErr_SetNone(Fallback); return -1; }
            for (int i = 1; i <= k; i++)
                cp = (cp << 6) | (u[i] & 0x3f);
            u += k + 1;
            uint32_t units[2];
            int nu = 1;
            if (cp >= 0x10000) {
                cp -= 0x10000;
                units[0] = 0xd800 | (cp >> 10);
                units[1] = 0xdc00 | (cp & 0x3ff);
                nu = 2;
            } else
                units[0] = cp;
            for (int i = 0; i < nu; i++) {
                e[n++] = '\\'; e[n++] = 'u';
                e[n++] = HEX[(units[i] >> 12) & 15]; e[n++] = HEX[(units[i] >> 8) & 15];
                e[n++] = HEX[(units[i] >> 4) & 15]; e[n++] = HEX[units[i] & 15];
            }
        }
        if (buf_put(b, e, n) < 0)
            return -1;
    }
    return buf_ch(b, '"');
}

#define MAX_DEPTH 100

static int put_value(Buf *b, PyObject *o, int depth);

/* builtins.round(x, n) for a float, or -1 (with `*ok` 0) where only CPython's own rounding is certain */
static inline double round_fast(double x, int n, int *ok)
{
    double y = x * POW10[n];
    double r = rint(y);
    /* round() is correct in decimal; rint(x * 10^n) agrees wherever the product is not next to a tie (its own
     * rounding error is below 1e-7 for products below 1e9), and r / 10^n is the double nearest to that decimal */
    *ok = (fabs(y - r) < 0.5 - 1e-6) && (fabs(y) < 1e9);
    return r / POW10[n];
}

/* round(x, n) written as float.__repr__ writes it, from the integer rint(x * 10^n) itself: the text of that
 * integer with n decimals, trailing zeros dropped, is the shortest decimal of the rounded double (15 significant
 * digits or fewer: see put_float). 1 when written; 0 where round_fast is not certain or repr takes another form
 * (an exponent below 1e-4), and nothing is written. */
static int put_round_fast(Buf *b, double x, int n, int *err)
{
    double y = x * POW10[n];
    double r = rint(y);
    *err = 0;
    if (!((fabs(y - r) < 0.5 - 1e-6) && (fabs(y) < 1e9)))
        return 0;
    if (r == 0.0) {
        *err = buf_put(b, signbit(r) ? "-0.0" : "0.0", signbit(r) ? 4 : 3);
        return 1;
    }
    if (fabs(r / POW10[n]) < 1e-4)
        return 0;
    uint64_t k = (uint64_t)fabs(r);
    while (n > 0 && k % 10 == 0) {
        k /= 10;
        n--;
    }
    char t[48];
    int i = 48;
    if (n == 0)
        t[--i] = '0';
    for (int d = 0; d < n; d++) {
        t[--i] = (char)('0' + k % 10);
        k /= 10;
    }
    t[--i] = '.';
    do {
        t[--i] = (char)('0' + k % 10);
        k /= 10;
    } while (k);
    if (r < 0.0)
        t[--i] = '-';
    *err = buf_put(b, t + i, (size_t)(48 - i));
    return 1;
}

static int put_value(Buf *b, PyObject *o, int depth)
{
    if (o == Py_None)
        return buf_put(b, "null", 4);
    if (o == Py_True)
        return buf_put(b, "true", 4);
    if (o == Py_False)
        return buf_put(b, "false", 5);
    if (PyFloat_Check(o)) {
        double v = PyFloat_AsDouble(o);
        if (v == -1.0 && PyErr_Occurred())
            return -1;
        return put_double(b, v);
    }
    if (PyUnicode_Check(o))
        return put_str(b, o);
    if (PyLong_Check(o))
        return put_int(b, o);
    if (depth >= MAX_DEPTH) {                      /* deep or circular: json.dumps says which */
        PyErr_SetNone(Fallback);
        return -1;
    }
    if (PyDict_CheckExact(o)) {                    /* (a subclass may order its items its own way: json asks it) */
        Py_ssize_t pos = 0;
        PyObject *k, *v;
        int first = 1;
        if (buf_ch(b, '{') < 0)
            return -1;
        /* nothing in this loop runs Python code (no callbacks, no __repr__): the dict cannot change under it */
        while (PyDict_Next(o, &pos, &k, &v)) {
            if (!PyUnicode_Check(k)) {             /* json converts int, float, bool and None keys */
                PyErr_SetNone(Fallback);
                return -1;
            }
            if (!first && buf_ch(b, ',') < 0)
                return -1;
            first = 0;
            if (put_str(b, k) < 0 || buf_ch(b, ':') < 0 || put_value(b, v, depth + 1) < 0)
                return -1;
        }
        return buf_ch(b, '}');
    }
    if (PyList_Check(o)) {
        Py_ssize_t n = PyList_Size(o);
        if (buf_ch(b, '[') < 0)
            return -1;
        for (Py_ssize_t i = 0; i < n; i++) {
            if (i && buf_ch(b, ',') < 0)
                return -1;
            PyObject *v = PyList_GetItem(o, i);
            if (v == NULL || put_value(b, v, depth + 1) < 0)
                return -1;
        }
        return buf_ch(b, ']');
    }
    if (PyTuple_Check(o)) {
        Py_ssize_t n = PyTuple_Size(o);
        if (buf_ch(b, '[') < 0)
            return -1;
        for (Py_ssize_t i = 0; i < n; i++) {
            if (i && buf_ch(b, ',') < 0)
                return -1;
            PyObject *v = PyTuple_GetItem(o, i);
            if (v == NULL || put_value(b, v, depth + 1) < 0)
                return -1;
        }
        return buf_ch(b, ']');
    }
    PyErr_SetNone(Fallback);                       /* any other type: json.dumps decides (and raises its error) */
    return -1;
}

static PyObject *n_dumps(PyObject *self, PyObject *o)
{
    Buf b = {NULL, 0, 0};
    PyObject *res = NULL;
    if (put_value(&b, o, 0) == 0)
        res = PyUnicode_DecodeLatin1(b.p, (Py_ssize_t)b.n, NULL);     /* ASCII only */
    PyMem_Free(b.p);
    return res;
}

/* ====================================================================================================
 * The recorder's reduced `road` (io/recorder.py `slim_road_line`; names follow `_slim` there)
 * ==================================================================================================== */
static PyObject *json_dumps_compact;   /* functools.partial(json.dumps, separators=(",", ":")): set by the package */

/* a value json.dumps writes: the native writer, or json.dumps itself for what that one leaves to json */
static int slim_put(Buf *b, PyObject *v)
{
    size_t mark = b->n;
    if (put_value(b, v, 1) == 0)
        return 0;
    if (!PyErr_ExceptionMatches(Fallback) || json_dumps_compact == NULL)
        return -1;
    PyErr_Clear();
    b->n = mark;
    PyObject *t = PyObject_CallFunctionObjArgs(json_dumps_compact, v, NULL);
    if (t == NULL)
        return -1;
    Py_ssize_t n;
    const char *s = PyUnicode_AsUTF8AndSize(t, &n);
    int rc = s == NULL ? -1 : buf_put(b, s, (size_t)n);
    Py_DECREF(t);
    return rc;
}

/* round(v, n) written as json.dumps writes it */
static int slim_round(Buf *b, PyObject *v, int n)
{
    if (PyFloat_CheckExact(v)) {
        int err;
        if (put_round_fast(b, PyFloat_AsDouble(v), n, &err))
            return err;
    } else if (PyLong_CheckExact(v))
        return put_int(b, v);
    PyObject *r = PyObject_CallFunction(builtin_round, "Oi", v, n);
    if (r == NULL)
        return -1;
    int rc = slim_put(b, r);
    Py_DECREF(r);
    return rc;
}

/* `put`: a list written with every number rounded to n digits (None as null), anything else as it is */
static int slim_member(Buf *b, PyObject *v, int n)
{
    if (!PyList_Check(v))
        return slim_put(b, v);
    Py_INCREF(v);                                  /* (a __round__ of an odd item is Python code) */
    int rc = buf_ch(b, '[');
    for (Py_ssize_t i = 0; rc == 0 && i < PyList_Size(v); i++) {
        PyObject *x = PyList_GetItem(v, i);
        Py_INCREF(x);
        if (i)
            rc = buf_ch(b, ',');
        if (rc == 0)
            rc = x == Py_None ? buf_put(b, "null", 4) : slim_round(b, x, n);
        Py_DECREF(x);
    }
    if (rc == 0)
        rc = buf_ch(b, ']');
    Py_DECREF(v);
    return rc;
}

/* a dict's key as json.dumps writes it: a str as it is; a float, an int, a bool, None: its JSON text, quoted */
static int slim_dict_key(Buf *b, PyObject *k)
{
    if (PyUnicode_Check(k))
        return put_str(b, k);
    if (k == Py_True || k == Py_False || k == Py_None || PyFloat_Check(k) || PyLong_Check(k)) {
        PyObject *t = PyObject_CallFunctionObjArgs(json_dumps_compact, k, NULL);
        if (t == NULL)
            return -1;
        int rc = put_str(b, t);
        Py_DECREF(t);
        return rc;
    }
    PyObject *nm = PyType_GetName(Py_TYPE(k));
    if (nm != NULL) {
        PyErr_Format(PyExc_TypeError, "keys must be str, int, float, bool or None, not %U", nm);
        Py_DECREF(nm);
    }
    return -1;
}

/* `"key":` with a comma before it unless it is the first member */
static int slim_key(Buf *b, const char *key, int *first)
{
    if (!*first && buf_ch(b, ',') < 0)
        return -1;
    *first = 0;
    if (buf_ch(b, '"') < 0 || buf_put(b, key, strlen(key)) < 0 || buf_ch(b, '"') < 0)
        return -1;
    return buf_ch(b, ':');
}

/* d.get(key) as a new reference, NULL (no error) when absent */
static PyObject *slim_get(PyObject *d, const char *key, int *err)
{
    PyObject *v = PyDict_GetItemString(d, key);
    if (v == NULL) {
        if (PyErr_Occurred())
            *err = 1;
        return NULL;
    }
    Py_INCREF(v);
    return v;
}

static const char *SLIM_KEYS[] = {"type", "seq", "ts_mono_ns", "ts_real_ns", "model_ms", "v_ego", "valid", "frames_skipped",
                                  "pair_dt_ms", "road_model", "road_model_fallback", "policy_hz", "frame_seq_prev",
                                  "camera", NULL};

static int slim_write(Buf *b, PyObject *road)
{
    int first = 1, err = 0;
    PyObject *v;
    if (buf_ch(b, '{') < 0)
        return -1;
    for (int k = 0; SLIM_KEYS[k]; k++) {
        if ((v = slim_get(road, SLIM_KEYS[k], &err)) == NULL) {
            if (err)
                return -1;
            continue;
        }
        int rc = slim_key(b, SLIM_KEYS[k], &first) < 0 ? -1 : slim_put(b, v);
        Py_DECREF(v);
        if (rc < 0)
            return -1;
    }
    if ((v = slim_get(road, "pose", &err)) != NULL && PyDict_Check(v)) {      /* numbers to 4 digits */
        int rc = slim_key(b, "pose", &first) < 0 || buf_ch(b, '{') < 0 ? -1 : 0;
        PyObject *items = rc == 0 ? PyDict_Items(v) : NULL;
        if (items == NULL)
            rc = -1;
        for (Py_ssize_t i = 0; rc == 0 && i < PyList_Size(items); i++) {
            PyObject *kv = PyList_GetItem(items, i), *k = PyTuple_GetItem(kv, 0), *x = PyTuple_GetItem(kv, 1);
            if (i)
                rc = buf_ch(b, ',');
            if (rc == 0)
                rc = slim_dict_key(b, k) < 0 || buf_ch(b, ':') < 0 ? -1 : 0;
            if (rc == 0)
                rc = PyFloat_Check(x) ? slim_round(b, x, 4) : slim_put(b, x);
        }
        Py_XDECREF(items);
        if (rc == 0)
            rc = buf_ch(b, '}');
        Py_DECREF(v);
        if (rc < 0)
            return -1;
    } else {
        Py_XDECREF(v);
        if (err)
            return -1;
    }
    static const char *LINES[] = {"lane_lines", "road_edges"};
    for (int L = 0; L < 2; L++) {
        if ((v = slim_get(road, LINES[L], &err)) == NULL) {
            if (err)
                return -1;
            continue;
        }
        if (!PyList_Check(v)) {
            Py_DECREF(v);
            continue;
        }
        int rc = slim_key(b, LINES[L], &first) < 0 || buf_ch(b, '[') < 0 ? -1 : 0;
        int fl = 1;
        for (Py_ssize_t i = 0; rc == 0 && i < PyList_Size(v); i++) {
            PyObject *ln = PyList_GetItem(v, i);
            if (!PyDict_Check(ln))
                continue;
            Py_INCREF(ln);
            if (!fl)
                rc = buf_ch(b, ',');
            fl = 0;
            int f2 = 1;
            if (rc == 0)
                rc = buf_ch(b, '{');
            static const char *M[] = {"x", "y", "y_std"};
            static const int D[] = {2, 3, 3};
            for (int m = 0; rc == 0 && m < 3; m++) {
                PyObject *x = slim_get(ln, M[m], &err);
                if (err)
                    rc = -1;
                else
                    rc = slim_key(b, M[m], &f2) < 0 ? -1 : (x == NULL ? buf_put(b, "null", 4) : slim_member(b, x, D[m]));
                Py_XDECREF(x);
            }
            if (rc == 0) {
                PyObject *p = slim_get(ln, "prob", &err);
                if (err)
                    rc = -1;
                else if (p != NULL && (PyLong_Check(p) || PyFloat_Check(p)))      /* isinstance(p, (int, float)) */
                    rc = slim_key(b, "prob", &f2) < 0 ? -1 : slim_round(b, p, 4);
                Py_XDECREF(p);
            }
            if (rc == 0)
                rc = buf_ch(b, '}');
            Py_DECREF(ln);
        }
        if (rc == 0)
            rc = buf_ch(b, ']');
        Py_DECREF(v);
        if (rc < 0)
            return -1;
    }
    if ((v = slim_get(road, "leads", &err)) != NULL && PyList_Check(v) && PyList_Size(v) > 0 &&
        PyDict_Check(PyList_GetItem(v, 0))) {
        PyObject *ld = PyList_GetItem(v, 0);
        Py_INCREF(ld);
        int rc = slim_key(b, "leads", &first) < 0 || buf_put(b, "[{", 2) < 0 ? -1 : 0;
        PyObject *items = rc == 0 ? PyDict_Items(ld) : NULL;
        if (items == NULL)
            rc = -1;
        for (Py_ssize_t i = 0; rc == 0 && i < PyList_Size(items); i++) {
            PyObject *kv = PyList_GetItem(items, i), *k = PyTuple_GetItem(kv, 0), *x = PyTuple_GetItem(kv, 1);
            if (i)
                rc = buf_ch(b, ',');
            if (rc == 0)
                rc = slim_dict_key(b, k) < 0 || buf_ch(b, ':') < 0 ? -1 : slim_member(b, x, 3);
        }
        Py_XDECREF(items);
        Py_DECREF(ld);
        if (rc == 0)
            rc = buf_put(b, "}]", 2);
        Py_DECREF(v);
        if (rc < 0)
            return -1;
    } else {
        Py_XDECREF(v);
        if (err)
            return -1;
    }
    if ((v = slim_get(road, "plan", &err)) != NULL && PyDict_Check(v)) {
        int rc = slim_key(b, "plan", &first) < 0 || buf_ch(b, '{') < 0 ? -1 : 0;
        int f2 = 1;
        static const char *P[] = {"x", "y", "z", "held", "age_ms"};
        for (int m = 0; rc == 0 && m < 5; m++) {
            PyObject *x = slim_get(v, P[m], &err);
            if (err) {
                rc = -1;
                break;
            }
            if (x == NULL)
                continue;
            rc = slim_key(b, P[m], &f2) < 0 ? -1 : (m < 3 ? slim_member(b, x, 2) : slim_put(b, x));
            Py_DECREF(x);
        }
        if (rc == 0)
            rc = buf_ch(b, '}');
        Py_DECREF(v);
        if (rc < 0)
            return -1;
    } else {
        Py_XDECREF(v);
        if (err)
            return -1;
    }
    return buf_ch(b, '}');
}

/* slim_road_line(road) -> str: json.dumps(_slim(road) with every list rounded, separators=(",", ":")) */
static PyObject *n_slim_road_line(PyObject *self, PyObject *road)
{
    if (!PyDict_Check(road)) {
        PyErr_SetString(PyExc_TypeError, "slim_road_line: a dict is needed");
        return NULL;
    }
    Buf b = {NULL, 0, 0};
    PyObject *res = NULL;
    if (slim_write(&b, road) == 0)
        res = PyUnicode_FromStringAndSize(b.p, (Py_ssize_t)b.n);
    PyMem_Free(b.p);
    return res;
}

static PyObject *n_set_json_dumps(PyObject *self, PyObject *f)
{
    Py_INCREF(f);
    Py_XDECREF(json_dumps_compact);
    json_dumps_compact = f;
    Py_RETURN_NONE;
}

/* ====================================================================================================
 * JSON: reading. The grammar of json.loads (RFC 8259 plus NaN, Infinity, -Infinity). Every read is
 * bounds-checked against `end`; nesting is limited; whatever is not plain (an escape that needs
 * surrogate handling, a number beyond the exact fast path, invalid text) raises Fallback and json.loads
 * decides, so errors and corner cases are CPython's own.
 * ==================================================================================================== */
typedef struct {
    const unsigned char *p, *end;
    PyObject *memo;                 /* key text -> key object: one str per distinct key, as json does */
} Rd;

#define RD_MAX_DEPTH 64

static inline void rd_ws(Rd *r)
{
    while (r->p < r->end && (*r->p == ' ' || *r->p == '\t' || *r->p == '\n' || *r->p == '\r'))
        r->p++;
}

static PyObject *rd_fail(void)
{
    if (!PyErr_Occurred())
        PyErr_SetNone(Fallback);
    return NULL;
}

static int hex4(const unsigned char *p, uint32_t *out)
{
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        unsigned c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else return -1;
    }
    *out = v;
    return 0;
}

/* a string with escapes, from its first backslash on: the text is rebuilt as UTF-8. A \uXXXX that is half of
 * a surrogate pair without its partner is left to the reference (it makes a lone surrogate). */
static PyObject *rd_string_esc(Rd *r, const unsigned char *s)
{
    Buf b = {NULL, 0, 0};
    PyObject *o = NULL;
    if (buf_put(&b, (const char *)s, (size_t)(r->p - s)) < 0)
        goto out;
    while (r->p < r->end) {
        unsigned c = *r->p;
        if (c == '"') {
            o = PyUnicode_DecodeUTF8(b.p ? b.p : "", (Py_ssize_t)b.n, "strict");
            r->p++;
            if (o == NULL)
                PyErr_Clear();
            goto out;
        }
        if (c < 0x20)
            goto out;
        if (c != '\\') {
            if (buf_ch(&b, (char)c) < 0)
                goto out;
            r->p++;
            continue;
        }
        if (r->end - r->p < 2)
            goto out;
        unsigned e = r->p[1];
        r->p += 2;
        char ch;
        switch (e) {
        case '"': ch = '"'; break;
        case '\\': ch = '\\'; break;
        case '/': ch = '/'; break;
        case 'b': ch = '\b'; break;
        case 'f': ch = '\f'; break;
        case 'n': ch = '\n'; break;
        case 'r': ch = '\r'; break;
        case 't': ch = '\t'; break;
        case 'u': {
            uint32_t cp, lo;
            if (r->end - r->p < 4 || hex4(r->p, &cp) < 0)
                goto out;
            r->p += 4;
            if (cp >= 0xd800 && cp <= 0xdbff) {
                if (r->end - r->p < 6 || r->p[0] != '\\' || r->p[1] != 'u' || hex4(r->p + 2, &lo) < 0 ||
                    lo < 0xdc00 || lo > 0xdfff)
                    goto out;
                r->p += 6;
                cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
            } else if (cp >= 0xdc00 && cp <= 0xdfff)
                goto out;
            char u[4];
            size_t n;
            if (cp < 0x80) { u[0] = (char)cp; n = 1; }
            else if (cp < 0x800) { u[0] = (char)(0xc0 | (cp >> 6)); u[1] = (char)(0x80 | (cp & 0x3f)); n = 2; }
            else if (cp < 0x10000) {
                u[0] = (char)(0xe0 | (cp >> 12)); u[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
                u[2] = (char)(0x80 | (cp & 0x3f)); n = 3;
            } else {
                u[0] = (char)(0xf0 | (cp >> 18)); u[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
                u[2] = (char)(0x80 | ((cp >> 6) & 0x3f)); u[3] = (char)(0x80 | (cp & 0x3f)); n = 4;
            }
            if (buf_put(&b, u, n) < 0)
                goto out;
            continue;
        }
        default:
            goto out;
        }
        if (buf_ch(&b, ch) < 0)
            goto out;
    }
out:
    PyMem_Free(b.p);
    return o ? o : rd_fail();
}

/* a string: the bytes between the quotes, which must be valid UTF-8 without control characters (json.loads
 * refuses those in its default strict mode) */
static PyObject *rd_string(Rd *r)
{
    const unsigned char *s = ++r->p;               /* past the opening quote */
    while (r->p < r->end) {
        unsigned c = *r->p;
        if (c == '"') {
            PyObject *o = PyUnicode_DecodeUTF8((const char *)s, (Py_ssize_t)(r->p - s), "strict");
            r->p++;
            if (o == NULL) {
                PyErr_Clear();
                return rd_fail();
            }
            return o;
        }
        if (c == '\\')
            return rd_string_esc(r, s);
        if (c < 0x20)
            return rd_fail();                      /* a control character: the reference names the error */
        r->p++;
    }
    return rd_fail();
}

static PyObject *rd_number(Rd *r)
{
    const unsigned char *s = r->p, *p = r->p, *end = r->end;
    int neg = 0, is_float = 0;
    if (p < end && *p == '-') {
        neg = 1;
        p++;
    }
    if (p >= end)
        return rd_fail();
    if (*p == 'I' || *p == 'N')
        return rd_fail();                          /* Infinity, NaN: rare, the reference reads them */
    uint64_t mant = 0;
    int digits = 0, frac = 0;                      /* digits read into `mant`; how many of them after the point */
    const unsigned char *ip = p;
    if (*p == '0') {
        p++;
    } else if (*p >= '1' && *p <= '9') {
        while (p < end && *p >= '0' && *p <= '9') {
            if (digits < 19)
                mant = mant * 10 + (uint64_t)(*p - '0');
            digits++;
            p++;
        }
    } else
        return rd_fail();
    int int_len = (int)(p - ip);
    if (p < end && *p == '.') {
        const unsigned char *fp = ++p;
        while (p < end && *p >= '0' && *p <= '9') {
            if (digits < 19)
                mant = mant * 10 + (uint64_t)(*p - '0');
            digits++;
            frac++;
            p++;
        }
        if (p == fp)
            return rd_fail();
        is_float = 1;
    }
    int has_exp = 0;
    if (p < end && (*p == 'e' || *p == 'E')) {
        const unsigned char *q = p + 1;
        if (q < end && (*q == '+' || *q == '-'))
            q++;
        const unsigned char *ep = q;
        while (q < end && *q >= '0' && *q <= '9')
            q++;
        if (q == ep)
            return rd_fail();
        p = q;
        has_exp = is_float = 1;
    }
    if (!is_float) {
        /* up to 19 digits fit 64 bits unsigned; the value must fit a signed 64-bit integer */
        if (int_len > 19 || mant > (uint64_t)INT64_MAX)
            return rd_fail();                      /* a big integer: the reference reads it */
        r->p = p;
        long long v = (long long)mant;
        return PyLong_FromLongLong(neg ? -v : v);
    }
    /* mant / 10^frac with mant <= 2^53 and frac <= 22: both are exact doubles, and one correctly rounded
     * division gives the double nearest to the decimal, which is what CPython's strtod returns */
    if (!has_exp && digits <= 19 && frac <= 22 && mant <= ((uint64_t)1 << 53)) {
        r->p = p;
        double v = (double)mant / POW10[frac];
        return PyFloat_FromDouble(neg ? -v : v);
    }
    /* any other float (an exponent, 17 digits): CPython's own conversion of the token, as json does */
    char tok[64];
    size_t n = (size_t)(p - s);
    if (n >= sizeof tok)
        return rd_fail();
    memcpy(tok, s, n);
    tok[n] = 0;
    char *stop = NULL;
    double v = PyOS_string_to_double(tok, &stop, NULL);
    if (v == -1.0 && PyErr_Occurred()) {
        PyErr_Clear();
        return rd_fail();
    }
    if (stop != tok + n)
        return rd_fail();
    r->p = p;
    return PyFloat_FromDouble(v);
}

static PyObject *rd_value(Rd *r, int depth)
{
    rd_ws(r);
    if (r->p >= r->end)
        return rd_fail();
    unsigned c = *r->p;
    if (c == '"')
        return rd_string(r);
    if (c == '{') {
        if (depth >= RD_MAX_DEPTH)
            return rd_fail();
        PyObject *d = PyDict_New();
        if (d == NULL)
            return NULL;
        r->p++;
        rd_ws(r);
        if (r->p < r->end && *r->p == '}') {
            r->p++;
            return d;
        }
        for (;;) {
            rd_ws(r);
            if (r->p >= r->end || *r->p != '"')
                goto bad_d;
            /* the key: looked up in the memo by its bytes, so equal keys are one object */
            PyObject *k = rd_string(r);
            if (k == NULL)
                goto bad_d;
            PyObject *kk = PyDict_GetItemWithError(r->memo, k);   /* borrowed */
            if (kk == NULL) {
                if (PyErr_Occurred() || PyDict_SetItem(r->memo, k, k) < 0) {
                    Py_DECREF(k);
                    goto bad_d;
                }
                kk = k;
            }
            Py_INCREF(kk);
            Py_DECREF(k);
            rd_ws(r);
            if (r->p >= r->end || *r->p != ':') {
                Py_DECREF(kk);
                goto bad_d;
            }
            r->p++;
            PyObject *v = rd_value(r, depth + 1);
            if (v == NULL) {
                Py_DECREF(kk);
                goto bad_d;
            }
            int rc = PyDict_SetItem(d, kk, v);
            Py_DECREF(kk);
            Py_DECREF(v);
            if (rc < 0)
                goto bad_d;
            rd_ws(r);
            if (r->p >= r->end)
                goto bad_d;
            if (*r->p == ',') {
                r->p++;
                continue;
            }
            if (*r->p == '}') {
                r->p++;
                return d;
            }
            goto bad_d;
        }
    bad_d:
        Py_DECREF(d);
        return rd_fail();
    }
    if (c == '[') {
        if (depth >= RD_MAX_DEPTH)
            return rd_fail();
        PyObject *l = PyList_New(0);
        if (l == NULL)
            return NULL;
        r->p++;
        rd_ws(r);
        if (r->p < r->end && *r->p == ']') {
            r->p++;
            return l;
        }
        for (;;) {
            PyObject *v = rd_value(r, depth + 1);
            if (v == NULL)
                goto bad_l;
            int rc = PyList_Append(l, v);
            Py_DECREF(v);
            if (rc < 0)
                goto bad_l;
            rd_ws(r);
            if (r->p >= r->end)
                goto bad_l;
            if (*r->p == ',') {
                r->p++;
                continue;
            }
            if (*r->p == ']') {
                r->p++;
                return l;
            }
            goto bad_l;
        }
    bad_l:
        Py_DECREF(l);
        return rd_fail();
    }
    if (c == '-' || (c >= '0' && c <= '9'))
        return rd_number(r);
    size_t left = (size_t)(r->end - r->p);
    if (c == 't' && left >= 4 && memcmp(r->p, "true", 4) == 0) {
        r->p += 4;
        Py_RETURN_TRUE;
    }
    if (c == 'f' && left >= 5 && memcmp(r->p, "false", 5) == 0) {
        r->p += 5;
        Py_RETURN_FALSE;
    }
    if (c == 'n' && left >= 4 && memcmp(r->p, "null", 4) == 0) {
        r->p += 4;
        Py_RETURN_NONE;
    }
    return rd_fail();
}

static PyObject *n_loads(PyObject *self, PyObject *arg)
{
    const char *s;
    Py_ssize_t len;
    Py_buffer view;
    int have_view = 0;
    if (PyUnicode_Check(arg)) {
        s = PyUnicode_AsUTF8AndSize(arg, &len);
        if (s == NULL) {
            PyErr_Clear();
            PyErr_SetNone(Fallback);
            return NULL;
        }
    } else {
        if (!PyBytes_Check(arg) && !PyByteArray_Check(arg)) {     /* json.loads takes str, bytes, bytearray */
            PyErr_SetNone(Fallback);
            return NULL;
        }
        if (PyObject_GetBuffer(arg, &view, PyBUF_SIMPLE) < 0) {
            PyErr_Clear();
            PyErr_SetNone(Fallback);
            return NULL;
        }
        have_view = 1;
        s = (const char *)view.buf;
        len = view.len;
    }
    Rd r;
    r.p = (const unsigned char *)s;
    r.end = r.p + len;
    r.memo = PyDict_New();
    PyObject *o = NULL;
    if (r.memo != NULL) {
        o = rd_value(&r, 0);
        if (o != NULL) {
            rd_ws(&r);
            if (r.p != r.end) {                    /* text after the value: the reference names the error */
                Py_DECREF(o);
                o = NULL;
                PyErr_SetNone(Fallback);
            }
        }
        Py_DECREF(r.memo);
    }
    if (have_view)
        PyBuffer_Release(&view);
    return o;
}

/* ====================================================================================================
 * The `road` line trimmed of what the default stack does not read (fastjson.trim_road: the cut on the bytes)
 * ==================================================================================================== */

/* bytes.find(sub, start, end): the first index >= start where sub lies entirely before end, or -1 */
static Py_ssize_t b_find(const char *s, Py_ssize_t n, const char *sub, Py_ssize_t m, Py_ssize_t start, Py_ssize_t end)
{
    if (end > n)
        end = n;
    if (start < 0)
        start = 0;
    if (m == 0)
        return start <= end ? start : -1;
    /* by the last byte of `sub` (a bracket in every pattern here: rarer in the text than a comma or a quote) */
    char last = sub[m - 1];
    for (Py_ssize_t i = start + m - 1; i < end;) {
        const char *q = memchr(s + i, last, (size_t)(end - i));
        if (q == NULL)
            return -1;
        i = q - s;
        if (memcmp(s + i - (m - 1), sub, (size_t)m) == 0)
            return i - (m - 1);
        i++;
    }
    return -1;
}

typedef struct {
    Py_ssize_t a, b;
} Cut;

static int cut_add(Cut **cuts, Py_ssize_t *n, Py_ssize_t *cap, Py_ssize_t a, Py_ssize_t b)
{
    if (*n == *cap) {
        Py_ssize_t nc = *cap ? 2 * *cap : 32;
        Cut *c = PyMem_Realloc(*cuts, (size_t)nc * sizeof(Cut));
        if (c == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        *cuts = c;
        *cap = nc;
    }
    (*cuts)[*n].a = a;
    (*cuts)[*n].b = b;
    (*n)++;
    return 0;
}

/* trim_road(line, plan_key, plan_pats, list_keys, line_pats) -> bytes: fastjson.trim_road with its patterns
 * (bytes, a tuple of bytes, a tuple of bytes, a tuple of bytes); the line itself when nothing is cut */
static PyObject *n_trim_road(PyObject *self, PyObject *args)
{
    PyObject *line, *plan_key, *plan_pats, *list_keys, *line_pats;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &PyBytes_Type, &line, &PyBytes_Type, &plan_key, &PyTuple_Type, &plan_pats,
                          &PyTuple_Type, &list_keys, &PyTuple_Type, &line_pats))
        return NULL;
    char *s;
    Py_ssize_t n;
    if (PyBytes_AsStringAndSize(line, &s, &n) < 0)
        return NULL;
    Cut *cuts = NULL;
    Py_ssize_t nc = 0, cap = 0;
    PyObject *res = NULL;
    char *pk;
    Py_ssize_t pkn;
    if (PyBytes_AsStringAndSize(plan_key, &pk, &pkn) < 0)
        goto done;
    Py_ssize_t i = b_find(s, n, pk, pkn, 0, n);
    if (i >= 0) {
        Py_ssize_t e = b_find(s, n, "}", 1, i, n);
        if (e > 0) {
            Py_ssize_t at = i;
            for (Py_ssize_t k = 0; k < PyTuple_Size(plan_pats); k++) {
                char *p;
                Py_ssize_t pn;
                if (PyBytes_AsStringAndSize(PyTuple_GetItem(plan_pats, k), &p, &pn) < 0)
                    goto done;
                Py_ssize_t a = b_find(s, n, p, pn, at, e);
                if (a < 0 && at > i)
                    a = b_find(s, n, p, pn, i, e);
                if (a >= 0) {
                    Py_ssize_t b = b_find(s, n, "]", 1, a + pn, e);
                    if (b > 0) {
                        if (cut_add(&cuts, &nc, &cap, a, b + 1) < 0)
                            goto done;
                        at = b + 1;
                    }
                }
            }
        }
    }
    for (Py_ssize_t kk = 0; kk < PyTuple_Size(list_keys); kk++) {
        char *key;
        Py_ssize_t kn;
        if (PyBytes_AsStringAndSize(PyTuple_GetItem(list_keys, kk), &key, &kn) < 0)
            goto done;
        i = b_find(s, n, key, kn, 0, n);
        if (i < 0)
            continue;
        Py_ssize_t e = b_find(s, n, "}]", 2, i, n);
        if (e < 0)
            continue;
        for (Py_ssize_t k = 0; k < PyTuple_Size(line_pats); k++) {
            char *p;
            Py_ssize_t pn;
            if (PyBytes_AsStringAndSize(PyTuple_GetItem(line_pats, k), &p, &pn) < 0)
                goto done;
            Py_ssize_t a = b_find(s, n, p, pn, i, e);
            while (a >= 0) {
                Py_ssize_t b = b_find(s, n, "]", 1, a + pn, e);
                if (b < 0)
                    break;
                if (cut_add(&cuts, &nc, &cap, a, b + 1) < 0)
                    goto done;
                a = b_find(s, n, p, pn, b + 1, e);
            }
        }
    }
    if (nc == 0) {
        Py_INCREF(line);
        res = line;
        goto done;
    }
    for (Py_ssize_t x = 1; x < nc; x++) {             /* cuts.sort() */
        Cut v = cuts[x];
        Py_ssize_t y = x - 1;
        while (y >= 0 && (cuts[y].a > v.a || (cuts[y].a == v.a && cuts[y].b > v.b))) {
            cuts[y + 1] = cuts[y];
            y--;
        }
        cuts[y + 1] = v;
    }
    Py_ssize_t at = 0, len = 0;
    for (Py_ssize_t x = 0; x < nc; x++) {
        if (cuts[x].a < at) {                          /* overlapping spans: the line as it is */
            Py_INCREF(line);
            res = line;
            goto done;
        }
        len += cuts[x].a - at;
        at = cuts[x].b;
    }
    len += n - at;
    res = PyBytes_FromStringAndSize(NULL, len);
    if (res == NULL)
        goto done;
    char *o = PyBytes_AsString(res);
    at = 0;
    for (Py_ssize_t x = 0; x < nc; x++) {
        memcpy(o, s + at, (size_t)(cuts[x].a - at));
        o += cuts[x].a - at;
        at = cuts[x].b;
    }
    memcpy(o, s + at, (size_t)(n - at));
done:
    PyMem_Free(cuts);
    return res;
}

/* ====================================================================================================
 * round
 * ==================================================================================================== */
static PyObject *n_round(PyObject *self, PyObject *args)
{
    Py_ssize_t na = PyTuple_Size(args);
    if (na == 2) {
        PyObject *x = PyTuple_GetItem(args, 0), *nd = PyTuple_GetItem(args, 1);
        if (PyFloat_CheckExact(x) && PyLong_CheckExact(nd)) {
            int over = 0;
            long n = PyLong_AsLongAndOverflow(nd, &over);
            if (!over && n >= 0 && n <= 22) {
                int ok;
                double r = round_fast(PyFloat_AsDouble(x), (int)n, &ok);
                if (ok)
                    return PyFloat_FromDouble(r);
            }
        }
    }
    return PyObject_CallObject(builtin_round, args);
}

/* round_list(values, n): [v if v is None else round(v, n) for v in values] for a list */
static PyObject *n_round_list(PyObject *self, PyObject *args)
{
    PyObject *vals;
    long n;
    if (!PyArg_ParseTuple(args, "Ol", &vals, &n))
        return NULL;
    if (!PyList_CheckExact(vals)) {
        PyErr_SetString(PyExc_TypeError, "round_list: a list is needed");
        return NULL;
    }
    Py_ssize_t len = PyList_Size(vals);
    PyObject *out = PyList_New(len);
    if (out == NULL)
        return NULL;
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject *v = PyList_GetItem(vals, i), *r = NULL;
        if (v == NULL)
            goto fail;
        if (v == Py_None) {
            r = v;
            Py_INCREF(r);
        } else if (PyFloat_CheckExact(v) && n >= 0 && n <= 22) {
            int ok;
            double d = round_fast(PyFloat_AsDouble(v), (int)n, &ok);
            if (ok)
                r = PyFloat_FromDouble(d);
        }
        if (r == NULL && !PyErr_Occurred()) {
            Py_INCREF(v);
            r = PyObject_CallFunction(builtin_round, "Ol", v, n);
            Py_DECREF(v);
        }
        if (r == NULL)
            goto fail;
        PyList_SetItem(out, i, r);
    }
    return out;
fail:
    Py_DECREF(out);
    return NULL;
}

/* ====================================================================================================
 * Arrays through the buffer protocol
 * ==================================================================================================== */
typedef struct {
    Py_buffer b;
    int held;
} View;

static int view_get(PyObject *o, View *v, int writable, char kind, int ndim, const char *what)
{
    v->held = 0;
    if (PyObject_GetBuffer(o, &v->b, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0)) < 0)
        return -1;
    v->held = 1;
    const char *f = v->b.format ? v->b.format : "B";
    if (*f == '@' || *f == '=' || *f == '<')
        f++;
    int ok = 0;
    if (kind == 'd')
        ok = (f[0] == 'd' && f[1] == 0 && v->b.itemsize == 8);
    else if (kind == 'b')                           /* a bool array */
        ok = (f[0] == '?' && f[1] == 0 && v->b.itemsize == 1);
    else                                            /* an index array: the platform's signed 64-bit integer */
        ok = ((f[0] == 'l' || f[0] == 'q' || f[0] == 'n') && f[1] == 0 && v->b.itemsize == 8);
    if (!ok || v->b.ndim != ndim) {
        PyBuffer_Release(&v->b);
        v->held = 0;
        PyErr_Format(PyExc_TypeError, "%s: a C-contiguous %s array of %d dimension(s) is needed", what,
                     kind == 'd' ? "float64" : kind == 'b' ? "bool" : "int64", ndim);
        return -1;
    }
    return 0;
}

static void view_drop(View *v)
{
    if (v->held) {
        PyBuffer_Release(&v->b);
        v->held = 0;
    }
}

#define DATA(v) ((double *)(v).b.buf)

/* ====================================================================================================
 * The estimator's filter (core/ekf.py, core/ego.py; names follow them)
 * ==================================================================================================== */
enum { PE, PN, PSI, V, A, W, AL, Z, TH, SV, NCORE };
#define PAR0 13                  /* scalars at the head of the model's parameter block */
#define PI 3.141592653589793     /* math.pi */

/* out[:NCORE, :] = Fc @ M[:NCORE, :], the other rows of M as they are; terms whose coefficient is exactly
 * zero are skipped (the kinematic block is mostly zeros), the others are summed in index order */
static void fc_rows(const double *Fc, const double *M, double *out, Py_ssize_t n)
{
    for (int i = 0; i < NCORE; i++) {
        double *o = out + (Py_ssize_t)i * n;
        int first = 1;
        for (int k = 0; k < NCORE; k++) {
            double f = Fc[i * NCORE + k];
            if (f == 0.0)
                continue;
            const double *m = M + (Py_ssize_t)k * n;
            if (first) {
                for (Py_ssize_t j = 0; j < n; j++)
                    o[j] = f * m[j];
                first = 0;
            } else {
                for (Py_ssize_t j = 0; j < n; j++)
                    o[j] += f * m[j];
            }
        }
        if (first)
            for (Py_ssize_t j = 0; j < n; j++)
                o[j] = 0.0;
    }
    if (out != M)
        memcpy(out + (Py_ssize_t)NCORE * n, M + (Py_ssize_t)NCORE * n, (size_t)((n - NCORE) * n) * sizeof(double));
}

/* M[:, :NCORE] = M[:, :NCORE] @ Fc.T, in place */
static void fc_cols(const double *Fc, double *M, Py_ssize_t n)
{
    double t[NCORE];
    for (Py_ssize_t r = 0; r < n; r++) {
        double *m = M + r * n;
        for (int j = 0; j < NCORE; j++) {
            double acc = 0.0;
            int first = 1;
            for (int k = 0; k < NCORE; k++) {
                double f = Fc[j * NCORE + k];
                if (f == 0.0)
                    continue;
                if (first) {
                    acc = m[k] * f;
                    first = 0;
                } else
                    acc += m[k] * f;
            }
            t[j] = acc;
        }
        for (int j = 0; j < NCORE; j++)
            m[j] = t[j];
    }
}

/* ekf_predict(x, P, dt, par, rho_out | None, x_out, P_out, Fc_out)
 * par: [jerk_psd, yaw_jerk_psd, pos_psd, alt_psd, grade_psd, scale_psd, the five caps of ego._CAPS (V, A, W,
 *       AL, PSI), n_clamp, n_gm, psd of the aux states ..., (index, lo, hi) of every clamp ... (NaN: no bound),
 *       (index, tau, sigma) of every Gauss-Markov state ...]   (EgoModel._native_par)
 * rho_out: with Gauss-Markov states, the array that receives their decay over dt (1 for every other state):
 * the diagonal part of the transition, which the smoother needs. */
static PyObject *n_ekf_predict(PyObject *self, PyObject *args)
{
    PyObject *ox, *oP, *opar, *orr, *oxo, *oPo, *oF;
    double dt;
    View x, P, par, rr, xo, Po, F;
    x.held = P.held = par.held = rr.held = xo.held = Po.held = F.held = 0;
    double *gq = NULL;
    PyObject *res = NULL;
    if (!PyArg_ParseTuple(args, "OOdOOOOO", &ox, &oP, &dt, &opar, &orr, &oxo, &oPo, &oF))
        return NULL;
    if (view_get(ox, &x, 0, 'd', 1, "x") < 0 || view_get(oP, &P, 0, 'd', 2, "P") < 0 ||
        view_get(opar, &par, 0, 'd', 1, "par") < 0 || view_get(oxo, &xo, 1, 'd', 1, "x_out") < 0 ||
        view_get(oPo, &Po, 1, 'd', 2, "P_out") < 0 || view_get(oF, &F, 1, 'd', 2, "Fc_out") < 0)
        goto done;
    int has_rho = (orr != Py_None);
    if (has_rho && view_get(orr, &rr, 1, 'd', 1, "rho_out") < 0)
        goto done;
    Py_ssize_t n = x.b.shape[0];
    Py_ssize_t npar = par.b.shape[0];
    if (n < NCORE || P.b.shape[0] != n || P.b.shape[1] != n || xo.b.shape[0] != n || Po.b.shape[0] != n ||
        Po.b.shape[1] != n || F.b.shape[0] != NCORE || F.b.shape[1] != NCORE || npar < PAR0 ||
        (has_rho && rr.b.shape[0] != n) || DATA(Po) == DATA(P) || DATA(xo) == DATA(x)) {
        PyErr_SetString(PyExc_ValueError, "ekf_predict: shapes");
        goto done;
    }
    const double *c = DATA(par);
    Py_ssize_t nclamp = (Py_ssize_t)c[11], ngm = (Py_ssize_t)c[12];
    if (nclamp < 0 || ngm < 0 || npar != PAR0 + (n - NCORE) + 3 * nclamp + 3 * ngm || has_rho != (ngm > 0)) {
        PyErr_SetString(PyExc_ValueError, "ekf_predict: parameter block");
        goto done;
    }
    const double *psd = c + PAR0, *clamp = c + PAR0 + (n - NCORE), *gm = clamp + 3 * nclamp;
    if (has_rho) {
        /* Gauss-Markov states: decay r = exp(-dt / tau), added variance q = sigma^2 (1 - exp(-2 dt / tau)) */
        double *r = DATA(rr);
        gq = (double *)PyMem_Calloc((size_t)n, sizeof(double));
        if (gq == NULL) {
            PyErr_NoMemory();
            goto done;
        }
        for (Py_ssize_t i = 0; i < n; i++)
            r[i] = 1.0;
        for (Py_ssize_t q = 0; q < ngm; q++) {
            Py_ssize_t i = (Py_ssize_t)gm[3 * q];
            double tau = gm[3 * q + 1], sig = gm[3 * q + 2];
            if (i < 0 || i >= n) {
                PyErr_SetString(PyExc_ValueError, "ekf_predict: Gauss-Markov index");
                goto done;
            }
            r[i] = exp(-dt / tau);
            gq[i] = sig * sig * (1.0 - exp(-2.0 * dt / tau));
        }
    }
    const double *xi = DATA(x);
    double *xn = DATA(xo), *Pn = DATA(Po), *Fc = DATA(F);

    memcpy(xn, xi, (size_t)n * sizeof(double));
    double v = xi[V], a = xi[A], w = xi[W], al = xi[AL], psi = xi[PSI], th = xi[TH];
    double dt2 = dt * dt;
    double g = 1.0 / (1.0 + xi[SV]);
    double ds = (v * dt + 0.5 * a * dt2) * g;
    double pm = psi + 0.5 * w * dt + al * dt2 / 6.0;
    double cm = cos(pm), sm = sin(pm);
    double sth = sin(th), cth = cos(th);
    xn[PE] += ds * cm;
    xn[PN] += ds * sm;
    xn[PSI] = psi + w * dt + 0.5 * al * dt2;
    xn[V] = v + a * dt;
    xn[W] = w + al * dt;
    xn[Z] += ds * sth;

    double dsm = -ds * sm, dcm = ds * cm, h = 0.5 * dt;
    memset(Fc, 0, NCORE * NCORE * sizeof(double));
#define FC(i, j) Fc[(i) * NCORE + (j)]
    for (int i = 0; i < NCORE; i++)
        FC(i, i) = 1.0;
    FC(0, 2) = dsm; FC(0, 3) = dt * cm * g; FC(0, 4) = 0.5 * dt2 * cm * g; FC(0, 5) = dsm * h;
    FC(0, 6) = dsm * dt2 / 6.0; FC(0, 9) = -ds * g * cm;
    FC(1, 2) = dcm; FC(1, 3) = dt * sm * g; FC(1, 4) = 0.5 * dt2 * sm * g; FC(1, 5) = dcm * h;
    FC(1, 6) = dcm * dt2 / 6.0; FC(1, 9) = -ds * g * sm;
    FC(2, 5) = dt; FC(2, 6) = 0.5 * dt2;
    FC(3, 4) = dt;
    FC(5, 6) = dt;
    FC(7, 3) = dt * sth * g; FC(7, 4) = 0.5 * dt2 * sth * g; FC(7, 8) = ds * cth; FC(7, 9) = -ds * g * sth;
#undef FC
    if (has_rho) {
        const double *r = DATA(rr);
        for (Py_ssize_t i = 0; i < n; i++)
            xn[i] *= r[i];
    }
    fc_rows(Fc, DATA(P), Pn, n);
    fc_cols(Fc, Pn, n);
#define PP(i, j) Pn[(Py_ssize_t)(i) * n + (j)]
    if (has_rho) {
        const double *r = DATA(rr);
        for (Py_ssize_t i = 0; i < n; i++)
            for (Py_ssize_t j = 0; j < n; j++)
                PP(i, j) *= r[i];
        for (Py_ssize_t i = 0; i < n; i++)
            for (Py_ssize_t j = 0; j < n; j++)
                PP(i, j) *= r[j];
    }
    /* process noise (white jerk and yaw jerk integrated over dt; position noise along the heading) */
    double dt3 = dt2 * dt, dt4 = dt2 * dt2, dt5 = dt2 * dt2 * dt;
    double qj = c[0], qy = c[1];
    double q55 = qj * dt5 / 20.0, q44 = qj * dt4 / 8.0, q33 = qj * dt3 / 6.0;
    double pp = c[2] * dt;
    double Qk[7][7] = {
        {q55 * cm * cm + pp, q55 * cm * sm, 0.0, q44 * cm, q33 * cm, 0.0, 0.0},
        {q55 * cm * sm, q55 * sm * sm + pp, 0.0, q44 * sm, q33 * sm, 0.0, 0.0},
        {0.0, 0.0, qy * dt5 / 20.0, 0.0, 0.0, qy * dt4 / 8.0, qy * dt3 / 6.0},
        {q44 * cm, q44 * sm, 0.0, qj * dt3 / 3.0, qj * dt2 / 2.0, 0.0, 0.0},
        {q33 * cm, q33 * sm, 0.0, qj * dt2 / 2.0, qj * dt, 0.0, 0.0},
        {0.0, 0.0, qy * dt4 / 8.0, 0.0, 0.0, qy * dt3 / 3.0, qy * dt2 / 2.0},
        {0.0, 0.0, qy * dt3 / 6.0, 0.0, 0.0, qy * dt2 / 2.0, qy * dt}};
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++)
            PP(i, j) += Qk[i][j];
    PP(Z, Z) += c[3] * dt;
    PP(TH, TH) += c[4] * dt;
    PP(SV, SV) += c[5] * dt;
    for (Py_ssize_t i = NCORE; i < n; i++)
        PP(i, i) += psd[i - NCORE] * dt;
    if (has_rho) {
        for (Py_ssize_t i = NCORE; i < n; i++)
            PP(i, i) += gq[i];
    }
    if (dt > 0.5) {
        /* a gap without data: bounded quantities keep a bounded uncertainty (ego._CAPS) */
        static const int cap_i[5] = {V, A, W, AL, PSI};
        const double *cap_lim = c + 6;
        for (int q = 0; q < 5; q++) {
            int i = cap_i[q];
            if (PP(i, i) > cap_lim[q]) {
                double k = sqrt(cap_lim[q] / PP(i, i));
                for (Py_ssize_t j = 0; j < n; j++)
                    PP(i, j) *= k;
                for (Py_ssize_t j = 0; j < n; j++)
                    PP(j, i) *= k;
            }
        }
    }
#undef PP
    for (Py_ssize_t q = 0; q < nclamp; q++) {
        Py_ssize_t i = (Py_ssize_t)clamp[3 * q];
        double lo = clamp[3 * q + 1], hi = clamp[3 * q + 2];
        if (i < 0 || i >= n)
            continue;
        if (lo == lo && xn[i] < lo)
            xn[i] = lo;
        else if (hi == hi && xn[i] > hi)
            xn[i] = hi;
    }
    res = Py_None;
    Py_INCREF(res);
done:
    PyMem_Free(gq);
    view_drop(&x); view_drop(&P); view_drop(&par); view_drop(&rr);
    view_drop(&xo); view_drop(&Po); view_drop(&F);
    return res;
}

/* ekf_project(P, idx, coef, ph_out) -> s0:   ph = P[:, idx] @ coef;  s0 = coef @ ph[idx] */
static PyObject *n_ekf_project(PyObject *self, PyObject *args)
{
    PyObject *oP, *oi, *oc, *oph;
    View P, idx, coef, ph;
    P.held = idx.held = coef.held = ph.held = 0;
    PyObject *res = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &oP, &oi, &oc, &oph))
        return NULL;
    if (view_get(oP, &P, 0, 'd', 2, "P") < 0 || view_get(oi, &idx, 0, 'i', 1, "idx") < 0 ||
        view_get(oc, &coef, 0, 'd', 1, "coef") < 0 || view_get(oph, &ph, 1, 'd', 1, "ph") < 0)
        goto done;
    Py_ssize_t n = P.b.shape[0], m = idx.b.shape[0];
    const int64_t *ix = (const int64_t *)idx.b.buf;
    if (P.b.shape[1] != n || coef.b.shape[0] != m || ph.b.shape[0] != n || m < 1) {
        PyErr_SetString(PyExc_ValueError, "ekf_project: shapes");
        goto done;
    }
    for (Py_ssize_t j = 0; j < m; j++)
        if (ix[j] < 0 || ix[j] >= n) {
            PyErr_SetString(PyExc_IndexError, "ekf_project: an index beyond the state");
            goto done;
        }
    const double *p = DATA(P), *cf = DATA(coef);
    double *o = DATA(ph);
    for (Py_ssize_t i = 0; i < n; i++) {
        const double *row = p + i * n;
        double acc = row[ix[0]] * cf[0];
        for (Py_ssize_t j = 1; j < m; j++)
            acc += row[ix[j]] * cf[j];
        o[i] = acc;
    }
    double s0 = cf[0] * o[ix[0]];
    for (Py_ssize_t j = 1; j < m; j++)
        s0 += cf[j] * o[ix[j]];
    res = PyFloat_FromDouble(s0);
done:
    view_drop(&P); view_drop(&idx); view_drop(&coef); view_drop(&ph);
    return res;
}

/* ekf_update(x, P, ph, s, nu, freeze | None): one scalar row applied in place.
 *   k = ph / s;  x += k * nu;  P = P - k[:, None] * ph[None, :]
 * with `freeze` (Schmidt-Kalman: those states are not corrected), the Joseph form:
 *   k[freeze] = 0;  P = P - kp - kp.T + s * (k[:, None] * k[None, :]);  P = 0.5 * (P + P.T)   (kp = k ph^T) */
static PyObject *n_ekf_update(PyObject *self, PyObject *args)
{
    PyObject *ox, *oP, *oph, *ofz;
    double s, nu;
    View x, P, ph, fz;
    x.held = P.held = ph.held = fz.held = 0;
    PyObject *res = NULL;
    double *k = NULL;
    if (!PyArg_ParseTuple(args, "OOOddO", &ox, &oP, &oph, &s, &nu, &ofz))
        return NULL;
    if (view_get(ox, &x, 1, 'd', 1, "x") < 0 || view_get(oP, &P, 1, 'd', 2, "P") < 0 ||
        view_get(oph, &ph, 0, 'd', 1, "ph") < 0)
        goto done;
    int frozen = (ofz != Py_None);
    if (frozen && view_get(ofz, &fz, 0, 'i', 1, "freeze") < 0)
        goto done;
    Py_ssize_t n = x.b.shape[0];
    if (P.b.shape[0] != n || P.b.shape[1] != n || ph.b.shape[0] != n) {
        PyErr_SetString(PyExc_ValueError, "ekf_update: shapes");
        goto done;
    }
    k = (double *)PyMem_Malloc((size_t)(n ? n : 1) * sizeof(double));
    if (k == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    const double *h = DATA(ph);
    double *xx = DATA(x), *p = DATA(P);
    for (Py_ssize_t i = 0; i < n; i++)
        k[i] = h[i] / s;
    if (frozen) {
        const int64_t *f = (const int64_t *)fz.b.buf;
        Py_ssize_t m = fz.b.shape[0];
        for (Py_ssize_t j = 0; j < m; j++) {
            if (f[j] < 0 || f[j] >= n) {
                PyErr_SetString(PyExc_ValueError, "ekf_update: freeze index");
                goto done;
            }
        }
        for (Py_ssize_t j = 0; j < m; j++)
            k[f[j]] = 0.0;
    }
    for (Py_ssize_t i = 0; i < n; i++)
        xx[i] += k[i] * nu;
    if (!frozen) {
        for (Py_ssize_t i = 0; i < n; i++) {
            double ki = k[i];
            double *row = p + i * n;
            for (Py_ssize_t j = 0; j < n; j++)
                row[j] = row[j] - ki * h[j];
        }
    } else {
        /* ((P - kp) - kp.T) + s * (k k^T), then the symmetric part; pairs (i, j), (j, i) are taken together
         * so that the in-place result is the one of the out-of-place expression */
        for (Py_ssize_t i = 0; i < n; i++) {
            for (Py_ssize_t j = i; j < n; j++) {
                double a = ((p[i * n + j] - k[i] * h[j]) - k[j] * h[i]) + s * (k[i] * k[j]);
                double b = ((p[j * n + i] - k[j] * h[i]) - k[i] * h[j]) + s * (k[j] * k[i]);
                p[i * n + j] = 0.5 * (a + b);
                p[j * n + i] = 0.5 * (b + a);
            }
        }
    }
    res = Py_None;
    Py_INCREF(res);
done:
    PyMem_Free(k);
    view_drop(&x); view_drop(&P); view_drop(&ph); view_drop(&fz);
    return res;
}

/* ekf_sym(P): P = 0.5 * (P + P.T) in place */
static PyObject *n_ekf_sym(PyObject *self, PyObject *oP)
{
    View P;
    if (view_get(oP, &P, 1, 'd', 2, "P") < 0)
        return NULL;
    Py_ssize_t n = P.b.shape[0];
    if (P.b.shape[1] != n) {
        view_drop(&P);
        PyErr_SetString(PyExc_ValueError, "ekf_sym: shape");
        return NULL;
    }
    double *p = DATA(P);
    for (Py_ssize_t i = 0; i < n; i++)
        for (Py_ssize_t j = i; j < n; j++) {
            double a = p[i * n + j], b = p[j * n + i];
            p[i * n + j] = 0.5 * (a + b);
            p[j * n + i] = 0.5 * (b + a);
        }
    view_drop(&P);
    Py_RETURN_NONE;
}

/* LU with partial pivoting of A (n x n, overwritten), then the solution of A X = B for the n columns of B
 * (overwritten). Returns -1 on an exactly zero pivot (numpy: LinAlgError). */
static int lu_solve(double *A, double *B, Py_ssize_t n, Py_ssize_t *piv)
{
    for (Py_ssize_t c = 0; c < n; c++) {
        Py_ssize_t best = c;
        double bv = fabs(A[c * n + c]);
        for (Py_ssize_t r = c + 1; r < n; r++) {
            double t = fabs(A[r * n + c]);
            if (t > bv) {
                bv = t;
                best = r;
            }
        }
        if (!(bv > 0.0))                            /* zero or NaN */
            return -1;
        piv[c] = best;
        if (best != c) {
            for (Py_ssize_t j = 0; j < n; j++) {
                double t = A[c * n + j]; A[c * n + j] = A[best * n + j]; A[best * n + j] = t;
                t = B[c * n + j]; B[c * n + j] = B[best * n + j]; B[best * n + j] = t;
            }
        }
        double d = A[c * n + c];
        for (Py_ssize_t r = c + 1; r < n; r++) {
            double f = A[r * n + c] / d;
            if (f == 0.0)
                continue;
            A[r * n + c] = f;
            for (Py_ssize_t j = c + 1; j < n; j++)
                A[r * n + j] -= f * A[c * n + j];
            for (Py_ssize_t j = 0; j < n; j++)
                B[r * n + j] -= f * B[c * n + j];
        }
    }
    for (Py_ssize_t r = n - 1; r >= 0; r--) {
        double d = A[r * n + r];
        for (Py_ssize_t j = 0; j < n; j++) {
            double acc = B[r * n + j];
            for (Py_ssize_t q = r + 1; q < n; q++)
                acc -= A[r * n + q] * B[q * n + j];
            B[r * n + j] = acc / d;
        }
    }
    return 0;
}

static double wrap_pi(double a)
{
    /* Python: (a + pi) % (2 pi) - pi, the remainder with the divisor's sign */
    double b = 2.0 * PI;
    double m = fmod(a + PI, b);
    if (m != 0.0) {
        if (m < 0.0)
            m += b;
    } else
        m = 0.0;
    return m - PI;
}

/* ekf_rts(Pp, Fc, rho | None, Pa, xa, xp, xs, Ps, wrap, xs_out, Ps_out) -> bool
 * One Rauch-Tung-Striebel step back over a prediction (core/ekf.py `at`):
 *   C = solve(Pp, F @ Pa).T;  xs_out = xa + C @ diff(xs, xp);  Ps_out = Pa + C @ (Ps - Pp) @ C.T
 * F is the model's transition (the kinematic block Fc and the diagonal rho, ego.apply_F); `wrap`: a tuple of
 * angle state indices whose difference is wrapped. False: Pp is singular (the caller keeps the filtered state). */
static PyObject *n_ekf_rts(PyObject *self, PyObject *args)
{
    PyObject *oPp, *oF, *orho, *oPa, *oxa, *oxp, *oxs, *oPs, *owrap, *oxo, *oPo;
    View Pp, F, rho, Pa, xa, xp, xs, Ps, xo, Po;
    Pp.held = F.held = rho.held = Pa.held = xa.held = xp.held = xs.held = Ps.held = xo.held = Po.held = 0;
    PyObject *res = NULL;
    double *work = NULL;
    Py_ssize_t *piv = NULL;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOO", &oPp, &oF, &orho, &oPa, &oxa, &oxp, &oxs, &oPs, &owrap, &oxo, &oPo))
        return NULL;
    if (view_get(oPp, &Pp, 0, 'd', 2, "Pp") < 0 || view_get(oF, &F, 0, 'd', 2, "Fc") < 0 ||
        view_get(oPa, &Pa, 0, 'd', 2, "Pa") < 0 || view_get(oxa, &xa, 0, 'd', 1, "xa") < 0 ||
        view_get(oxp, &xp, 0, 'd', 1, "xp") < 0 || view_get(oxs, &xs, 0, 'd', 1, "xs") < 0 ||
        view_get(oPs, &Ps, 0, 'd', 2, "Ps") < 0 || view_get(oxo, &xo, 1, 'd', 1, "xs_out") < 0 ||
        view_get(oPo, &Po, 1, 'd', 2, "Ps_out") < 0)
        goto done;
    int has_rho = (orho != Py_None);
    if (has_rho && view_get(orho, &rho, 0, 'd', 1, "rho") < 0)
        goto done;
    Py_ssize_t n = xa.b.shape[0];
    if (n < NCORE || Pp.b.shape[0] != n || Pp.b.shape[1] != n || Pa.b.shape[0] != n || Pa.b.shape[1] != n ||
        Ps.b.shape[0] != n || Ps.b.shape[1] != n || Po.b.shape[0] != n || Po.b.shape[1] != n ||
        xp.b.shape[0] != n || xs.b.shape[0] != n || xo.b.shape[0] != n || F.b.shape[0] != NCORE ||
        F.b.shape[1] != NCORE || (has_rho && rho.b.shape[0] != n) || !PyTuple_Check(owrap)) {
        PyErr_SetString(PyExc_ValueError, "ekf_rts: shapes");
        goto done;
    }
    size_t nn = (size_t)n * (size_t)n;
    work = (double *)PyMem_Malloc((4 * nn + (size_t)n) * sizeof(double));
    piv = (Py_ssize_t *)PyMem_Malloc((size_t)n * sizeof(Py_ssize_t));
    if (work == NULL || piv == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    double *Aw = work, *X = work + nn, *D = work + 2 * nn, *T = work + 3 * nn, *d = work + 4 * nn;
    memcpy(Aw, DATA(Pp), nn * sizeof(double));
    fc_rows(DATA(F), DATA(Pa), X, n);                       /* X = F @ Pa */
    if (has_rho) {
        const double *r = DATA(rho);
        for (Py_ssize_t i = 0; i < n; i++)
            for (Py_ssize_t j = 0; j < n; j++)
                X[i * n + j] *= r[i];
    }
    if (lu_solve(Aw, X, n, piv) < 0) {
        res = Py_False;
        Py_INCREF(res);
        goto done;
    }
    /* C = X.T:  C[i][j] = X[j][i] */
    const double *vxs = DATA(xs), *vxp = DATA(xp), *vxa = DATA(xa), *vPs = DATA(Ps), *vPp = DATA(Pp), *vPa = DATA(Pa);
    for (Py_ssize_t i = 0; i < n; i++)
        d[i] = vxs[i] - vxp[i];
    Py_ssize_t nw = PyTuple_Size(owrap);
    for (Py_ssize_t q = 0; q < nw; q++) {
        Py_ssize_t i = PyLong_AsSsize_t(PyTuple_GetItem(owrap, q));
        if (i == -1 && PyErr_Occurred())
            goto done;
        if (i < 0 || i >= n) {
            PyErr_SetString(PyExc_ValueError, "ekf_rts: wrap index");
            goto done;
        }
        d[i] = wrap_pi(d[i]);
    }
    double *vxo = DATA(xo), *vPo = DATA(Po);
    for (Py_ssize_t i = 0; i < n; i++) {
        double acc = X[0 * n + i] * d[0];
        for (Py_ssize_t j = 1; j < n; j++)
            acc += X[j * n + i] * d[j];
        vxo[i] = vxa[i] + acc;
    }
    for (size_t i = 0; i < nn; i++)
        D[i] = vPs[i] - vPp[i];
    /* T = C @ D:  T[i][j] = sum_k X[k][i] D[k][j] */
    for (Py_ssize_t i = 0; i < n; i++) {
        double *t = T + i * n;
        for (Py_ssize_t j = 0; j < n; j++)
            t[j] = X[0 * n + i] * D[0 * n + j];
        for (Py_ssize_t k = 1; k < n; k++) {
            double c = X[k * n + i];
            const double *dr = D + k * n;
            for (Py_ssize_t j = 0; j < n; j++)
                t[j] += c * dr[j];
        }
    }
    /* Ps_out = Pa + T @ C.T:  (T @ C.T)[i][j] = sum_k T[i][k] X[k][j] */
    for (Py_ssize_t i = 0; i < n; i++) {
        const double *t = T + i * n;
        double *o = vPo + i * n;
        for (Py_ssize_t j = 0; j < n; j++)
            o[j] = t[0] * X[0 * n + j];
        for (Py_ssize_t k = 1; k < n; k++) {
            double c = t[k];
            const double *xr = X + k * n;
            for (Py_ssize_t j = 0; j < n; j++)
                o[j] += c * xr[j];
        }
        for (Py_ssize_t j = 0; j < n; j++)
            o[j] = vPa[i * n + j] + o[j];
    }
    res = Py_True;
    Py_INCREF(res);
done:
    PyMem_Free(work);
    PyMem_Free(piv);
    view_drop(&Pp); view_drop(&F); view_drop(&rho); view_drop(&Pa); view_drop(&xa); view_drop(&xp);
    view_drop(&xs); view_drop(&Ps); view_drop(&xo); view_drop(&Po);
    return res;
}

/* ====================================================================================================
 * The view (trackers/view.py): the table walk `_interp_all`, and the per-line steps of `ViewBuilder._lines`
 * ==================================================================================================== */
/* a number as a double: a float (or a float subclass), an int (a bool is an int) small enough to be exact as a
 * double; 0 for anything else */
static int get_num(PyObject *o, double *v)
{
    if (PyFloat_Check(o)) {
        *v = PyFloat_AsDouble(o);
        return 1;
    }
    if (PyLong_Check(o)) {
        int over = 0;
        long long k = PyLong_AsLongLongAndOverflow(o, &over);
        if (over || k > 9007199254740992LL || k < -9007199254740992LL)
            return 0;
        *v = (double)k;
        return 1;
    }
    return 0;
}

/* item i of a list or a tuple (borrowed) */
static inline PyObject *seq_item(PyObject *o, int is_list, Py_ssize_t i)
{
    return is_list ? PyList_GetItem(o, i) : PyTuple_GetItem(o, i);
}

static int seq_len(PyObject *o, const char *fn, Py_ssize_t *n, int *is_list)
{
    *is_list = PyList_Check(o);
    if (!*is_list && !PyTuple_Check(o)) {
        PyErr_Format(PyExc_TypeError, "%s: a list or a tuple is needed", fn);
        return -1;
    }
    *n = *is_list ? PyList_Size(o) : PyTuple_Size(o);
    return 0;
}

static int not_num(const char *fn)
{
    PyErr_Format(PyExc_TypeError, "%s: numbers are needed (floats, ints up to 2**53)", fn);
    return -1;
}

/* The table (xs, ys), lists or tuples, as doubles over min(len(xs), len(ys)) points; none[i]: ys[i] is None;
 * y0: ys[0] itself (what a query at or below xs[0] gives). The arrays live in `small` or `heap` (PyMem_Free it). */
typedef struct {
    double *X, *Y;
    char *none;
    Py_ssize_t n;
    PyObject *y0;
    void *heap;                    /* NULL while the points fit `small` */
    double small[2 * 64 + 8];
} Table;

static int table_get(PyObject *xs, PyObject *ys, Table *t, const char *fn)
{
    Py_ssize_t nx, ny;
    int lx, ly;
    t->heap = NULL;
    if (seq_len(xs, fn, &nx, &lx) < 0 || seq_len(ys, fn, &ny, &ly) < 0)
        return -1;
    Py_ssize_t n = nx < ny ? nx : ny;
    if (n <= 64)
        t->X = t->small;
    else {
        t->heap = PyMem_Malloc((size_t)n * (2 * sizeof(double) + 1));
        if (t->heap == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        t->X = (double *)t->heap;
    }
    t->Y = t->X + n;
    t->none = (char *)(t->Y + n);
    t->n = n;
    t->y0 = n > 0 ? seq_item(ys, ly, 0) : Py_None;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *y = seq_item(ys, ly, i);
        t->none[i] = (y == Py_None);
        t->Y[i] = 0.0;
        if (!get_num(seq_item(xs, lx, i), &t->X[i]) || (!t->none[i] && !get_num(y, &t->Y[i])))
            return not_num(fn);
    }
    return 0;
}

/* `_interp_all(xs, ys, qs)` into items (new references) and their values (val[q], none[q]): for a query at or
 * below xs[0], ys[0]; else the segment it falls in (searched on from the last query when the queries rise, from
 * the start when one falls back), interpolated, None when an end of it is None or the query is beyond the
 * table; an empty table: None for every query */
static int table_interp(const Table *t, const double *Q, Py_ssize_t nq, PyObject **items, double *val, char *none)
{
    Py_ssize_t i = 1, n = t->n;
    int have_prev = 0;
    double prev = 0.0, y0v = 0.0;
    int y0none = (t->y0 == Py_None);
    if (n > 0 && !y0none)
        y0v = t->Y[0];
    for (Py_ssize_t q = 0; q < nq; q++) {
        double x = Q[q];
        if (n == 0) {
            items[q] = Py_None;
            none[q] = 1;
        } else if (x <= t->X[0]) {
            items[q] = t->y0;
            none[q] = (char)y0none;
            val[q] = y0v;
        } else {
            if (have_prev && !(x >= prev))
                i = 1;
            have_prev = 1;
            prev = x;
            while (i < n && !(x <= t->X[i]))
                i++;
            if (i >= n || t->none[i - 1] || t->none[i]) {
                items[q] = Py_None;
                none[q] = 1;
            } else {
                double a = t->Y[i - 1], b = t->Y[i];
                double den = t->X[i] - t->X[i - 1];
                if (den == 0.0)                       /* (xs[i] - xs[i - 1]) or 1.0 */
                    den = 1.0;
                double tt = (x - t->X[i - 1]) / den;
                val[q] = a + (b - a) * tt;
                none[q] = 0;
                items[q] = PyFloat_FromDouble(val[q]);
                if (items[q] == NULL) {
                    for (Py_ssize_t k = 0; k < q; k++)
                        Py_DECREF(items[k]);
                    return -1;
                }
                continue;
            }
        }
        Py_INCREF(items[q]);
    }
    return 0;
}

/* a list or tuple of numbers (no None) as doubles: its values in t->Y (t->X too) */
static int nums_get(PyObject *o, Table *t, const char *fn)
{
    if (table_get(o, o, t, fn) < 0)
        return -1;
    for (Py_ssize_t i = 0; i < t->n; i++)
        if (t->none[i])
            return not_num(fn);
    return 0;
}

/* the queries as doubles, into a new heap block of nq doubles (+ `more` bytes per query) */
static double *queries(PyObject *qs, const char *fn, Py_ssize_t *nq, size_t more)
{
    int lq;
    if (seq_len(qs, fn, nq, &lq) < 0)
        return NULL;
    double *Q = PyMem_Malloc((size_t)(*nq > 0 ? *nq : 1) * (sizeof(double) + more));
    if (Q == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t q = 0; q < *nq; q++)
        if (!get_num(seq_item(qs, lq, q), &Q[q])) {
            PyMem_Free(Q);
            not_num(fn);
            return NULL;
        }
    return Q;
}

/* interp_all(xs, ys, qs) -> list: trackers/view.py `_interp_all` (see table_interp) */
static PyObject *n_interp_all(PyObject *self, PyObject *args)
{
    PyObject *xs, *ys, *qs, *out = NULL;
    if (!PyArg_ParseTuple(args, "OOO", &xs, &ys, &qs))
        return NULL;
    Table t;
    Py_ssize_t nq;
    double *Q = NULL;
    if (table_get(xs, ys, &t, "interp_all") < 0 || (Q = queries(qs, "interp_all", &nq, sizeof(double) + sizeof(PyObject *) + 1)) == NULL)
        goto done;
    double *val = Q + nq;
    PyObject **items = (PyObject **)(val + nq);
    char *none = (char *)(items + nq);
    if (table_interp(&t, Q, nq, items, val, none) < 0)
        goto done;
    out = PyList_New(nq);
    for (Py_ssize_t q = 0; q < nq; q++) {
        if (out != NULL)
            PyList_SetItem(out, q, items[q]);
        else
            Py_DECREF(items[q]);
    }
done:
    PyMem_Free(t.heap);
    PyMem_Free(Q);
    return out;
}

/* sum(values) of CPython for a sequence of floats: Neumaier's compensated sum, the compensation added at the
 * end when it is non-zero and finite (bltinmodule.c, builtin_sum_impl, since 3.12) */
static double py_fsum_seq(const double *a, Py_ssize_t n)
{
    double f = 0.0, c = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        double x = a[i], t = f + x;
        if (fabs(f) >= fabs(x))
            c += (f - t) + x;
        else
            c += (x - t) + f;
        f = t;
    }
    if (c != 0.0 && isfinite(c))
        f += c;
    return f;
}

/* view_prep(gx, y, y_std, xs, reach_std, b) -> None | (ys, reach, jump): a drawn line's points over xs.
 *     ys = _interp_all(gx, y, xs), stds = _interp_all(gx, y_std, xs); None when ys[0] or ys[1] has no value.
 *     reach: xs[i - 1] of the first point i without a value or with a std above reach_std beyond 10 m (else
 *     xs[-1]); ys: the points beyond it carry the last value (never drawn); jump: how far the near part (5 to
 *     20 m) moved against b, the drawn line: abs(sum(near) / len(near)), near = [y - b for y, b, x in
 *     zip(ys, b, xs) if 5 <= x <= 20] (None without b; ZeroDivisionError when no point is near). */
static PyObject *n_view_prep(PyObject *self, PyObject *args)
{
    PyObject *gx, *y, *ystd, *xs, *b;
    double reach_std;
    if (!PyArg_ParseTuple(args, "OOOOdO", &gx, &y, &ystd, &xs, &reach_std, &b))
        return NULL;
    Table ty, ts;
    ty.heap = ts.heap = NULL;
    PyObject *res = NULL, *ys = NULL;
    double *Q = NULL;
    Py_ssize_t nq = 0;
    if (table_get(gx, y, &ty, "view_prep") < 0 || table_get(gx, ystd, &ts, "view_prep") < 0)
        goto done;
    /* per query: its value, the std's value, the near difference (doubles), the two items, the two none flags */
    Q = queries(xs, "view_prep", &nq, 3 * sizeof(double) + 2 * sizeof(PyObject *) + 2);
    if (Q == NULL)
        goto done;
    if (nq < 2) {
        PyErr_SetString(PyExc_IndexError, "list index out of range");     /* ys[1] */
        goto done;
    }
    double *yv = Q + nq, *sv = yv + nq, *near = sv + nq;
    PyObject **yi = (PyObject **)(near + nq), **si = yi + nq;
    char *yn = (char *)(si + nq), *sn = yn + nq;
    if (table_interp(&ty, Q, nq, yi, yv, yn) < 0)
        goto done;
    if (table_interp(&ts, Q, nq, si, sv, sn) < 0) {
        for (Py_ssize_t q = 0; q < nq; q++)
            Py_DECREF(yi[q]);
        goto done;
    }
    for (Py_ssize_t q = 0; q < nq; q++)
        Py_DECREF(si[q]);                              /* the stds' values are all that is needed */
    if (yn[0] || yn[1]) {
        for (Py_ssize_t q = 0; q < nq; q++)
            Py_DECREF(yi[q]);
        res = Py_None;
        Py_INCREF(res);
        goto done;
    }
    Py_ssize_t ir = nq - 1;
    for (Py_ssize_t q = 0; q < nq; q++)
        if (yn[q] || (!sn[q] && sv[q] > reach_std && Q[q] > 10.0)) {
            ir = q == 0 ? nq - 1 : q - 1;              /* (xs[i - 1] of i = 0 is xs[-1]) */
            break;
        }
    PyObject *last = yi[0];
    double lastv = yv[0];
    for (Py_ssize_t q = 0; q < nq; q++) {             /* beyond the reach: carry the last value */
        if (yn[q]) {
            Py_DECREF(yi[q]);
            Py_INCREF(last);
            yi[q] = last;
            yv[q] = lastv;
            yn[q] = 0;
        }
        last = yi[q];
        lastv = yv[q];
    }
    ys = PyList_New(nq);
    for (Py_ssize_t q = 0; q < nq; q++) {
        if (ys != NULL)
            PyList_SetItem(ys, q, yi[q]);              /* steals */
        else
            Py_DECREF(yi[q]);
    }
    if (ys == NULL)
        goto done;
    PyObject *jump;
    if (b == Py_None) {
        jump = Py_None;
        Py_INCREF(jump);
    } else {
        Table tb;
        if (nums_get(b, &tb, "view_prep") < 0) {       /* (b as numbers: its values in tb.Y) */
            PyMem_Free(tb.heap);
            goto done;
        }
        Py_ssize_t m = tb.n < nq ? tb.n : nq, k = 0;
        for (Py_ssize_t q = 0; q < m; q++)
            if (5.0 <= Q[q] && Q[q] <= 20.0)
                near[k++] = yv[q] - tb.Y[q];
        PyMem_Free(tb.heap);
        if (k == 0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
            goto done;
        }
        jump = PyFloat_FromDouble(fabs(py_fsum_seq(near, k) / (double)k));
        if (jump == NULL)
            goto done;
    }
    int lq = PyList_Check(xs);
    res = Py_BuildValue("(OON)", ys, seq_item(xs, lq, ir), jump);
done:
    Py_XDECREF(ys);
    PyMem_Free(ty.heap);
    PyMem_Free(ts.heap);
    PyMem_Free(Q);
    return res;
}

static PyObject *float_list(const double *v, Py_ssize_t n)
{
    PyObject *out = PyList_New(n);
    if (out == NULL)
        return NULL;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *f = PyFloat_FromDouble(v[i]);
        if (f == NULL) {
            Py_DECREF(out);
            return NULL;
        }
        PyList_SetItem(out, i, f);
    }
    return out;
}

/* view_blend(a, b, dy, ys, ks, dt, shifting) -> (a, b, dy): a drawn line's two smoothing stages towards ys
 *     a = [p + (y - p) * k for p, y, k in zip(a, ys, ks)]
 *     b = [p + (y - p) * k for p, y, k in zip(b, a, ks)]   (shifting: list(a))
 *     dy = [0.7 * d0 + 0.3 * (b1 - b0) / dt for d0, b1, b0 in zip(dy, b, old b)] when dt > 0 (else dy itself)
 * Lists or tuples of numbers; the results are floats. */
static PyObject *n_view_blend(PyObject *self, PyObject *args)
{
    PyObject *oa, *ob, *ody, *oys, *oks;
    double dt;
    int shifting;
    if (!PyArg_ParseTuple(args, "OOOOOdp", &oa, &ob, &ody, &oys, &oks, &dt, &shifting))
        return NULL;
    Table A, B, D, Y, K;                               /* (each a list of numbers: its values in .Y) */
    A.heap = B.heap = D.heap = Y.heap = K.heap = NULL;
    PyObject *res = NULL, *ra = NULL, *rb = NULL, *rd = NULL;
    double *buf = NULL;
    if (nums_get(oa, &A, "view_blend") < 0 || nums_get(ob, &B, "view_blend") < 0 || nums_get(oys, &Y, "view_blend") < 0
        || nums_get(oks, &K, "view_blend") < 0 || (dt > 0.0 && nums_get(ody, &D, "view_blend") < 0))
        goto done;
    Py_ssize_t m1 = A.n < Y.n ? A.n : Y.n;
    m1 = m1 < K.n ? m1 : K.n;
    buf = PyMem_Malloc((size_t)(m1 > 0 ? m1 : 1) * 3 * sizeof(double));
    if (buf == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    double *A2 = buf, *B2 = buf + m1, *D2 = buf + 2 * m1;
    for (Py_ssize_t i = 0; i < m1; i++)
        A2[i] = A.Y[i] + (Y.Y[i] - A.Y[i]) * K.Y[i];
    Py_ssize_t m2;
    if (shifting) {
        m2 = m1;
        memcpy(B2, A2, sizeof(double) * (size_t)m1);
    } else {
        m2 = B.n < m1 ? B.n : m1;
        for (Py_ssize_t i = 0; i < m2; i++)
            B2[i] = B.Y[i] + (A2[i] - B.Y[i]) * K.Y[i];
    }
    if ((ra = float_list(A2, m1)) == NULL || (rb = float_list(B2, m2)) == NULL)
        goto done;
    if (dt > 0.0) {
        Py_ssize_t m3 = D.n < m2 ? D.n : m2;
        m3 = m3 < B.n ? m3 : B.n;
        for (Py_ssize_t i = 0; i < m3; i++)
            D2[i] = 0.7 * D.Y[i] + 0.3 * (B2[i] - B.Y[i]) / dt;
        if ((rd = float_list(D2, m3)) == NULL)
            goto done;
    } else {
        rd = ody;
        Py_INCREF(rd);
    }
    res = Py_BuildValue("(NNN)", ra, rb, rd);
    ra = rb = rd = NULL;
done:
    Py_XDECREF(ra);
    Py_XDECREF(rb);
    Py_XDECREF(rd);
    PyMem_Free(A.heap);
    PyMem_Free(B.heap);
    PyMem_Free(D.heap);
    PyMem_Free(Y.heap);
    PyMem_Free(K.heap);
    PyMem_Free(buf);
    return res;
}

/* ====================================================================================================
 * The lane tracker's array work (trackers/lanes.py; names follow it). Elementwise steps as numpy does them;
 * `mean` over the gate points adds in numpy's order (seq_sum below).
 * ==================================================================================================== */
#define N_LANE 4
#define N_LINES 6

/* The mean over the gate points, as numpy took it in the tracker's numpy form: `a[:, gate].mean(axis=1)`. The
 * selection a[:, gate] comes out in column order, and numpy adds each line's values one after the other, in
 * order (not pairwise); the golden results of tests/test_native.py were made that way. */
static double seq_sum(const double *a, Py_ssize_t n)
{
    double res = a[0];
    for (Py_ssize_t i = 1; i < n; i++)
        res += a[i];
    return res;
}

static inline double clamp01(double v)
{
    double m = (v > 0.0) ? v : 0.0;              /* Python: min(1.0, max(0.0, v)) */
    return (m < 1.0) ? m : 1.0;
}

static void sort_small(double *a, Py_ssize_t n)
{
    for (Py_ssize_t i = 1; i < n; i++) {
        double v = a[i];
        Py_ssize_t j = i - 1;
        while (j >= 0 && a[j] > v) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = v;
    }
}

/* np.asarray(v, float) of a lane line's list as the tracker takes it (`_arr`): 1 and the n values in `out` when v
 * is a list or tuple of n items that convert to float (float(), as numpy converts them: ints, bools, numeric
 * text) and all finite; 0 when not (no error); -1 with the error of a conversion that is not a refusal (an int
 * beyond the float range: OverflowError, as numpy raises it). */
static int seq_row(PyObject *v, Py_ssize_t n, double *out)
{
    int is_list = v != NULL && PyList_Check(v);
    if (v == NULL || (!is_list && !PyTuple_Check(v)))
        return 0;
    if ((is_list ? PyList_Size(v) : PyTuple_Size(v)) != n)
        return 0;
    int fin = 1;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *o = seq_item(v, is_list, i);
        double d;
        if (PyFloat_CheckExact(o))
            d = PyFloat_AsDouble(o);
        else {
            PyObject *f = PyNumber_Float(o);
            if (f == NULL) {
                if (PyErr_ExceptionMatches(PyExc_TypeError) || PyErr_ExceptionMatches(PyExc_ValueError)) {
                    PyErr_Clear();
                    return 0;
                }
                return -1;
            }
            d = PyFloat_AsDouble(f);
            Py_DECREF(f);
        }
        if (!isfinite(d))
            fin = 0;
        out[i] = d;
    }
    return fin;
}

/* lanes_measure(srcs, x, std_x0, std_x1, std_bad, std_good, y, s, prob, ok)
 * RoadGeometryTracker._measure after the distances x are known: srcs, the six lines (far left, left, right, far
 * right, left edge, right edge: a dict, or anything else for a line that is not there). A line is taken when its
 * "y" is n finite numbers (`seq_row`); its "y_std" the same, else 0.3 m (lines) / 1.0 m (edges); the deviations
 * floored at 1 mm. prob: a line's own "prob" (a finite int / float, a bool is an int) clipped to [0, 1], else 0;
 * an edge's from the median of its near-range deviations (all of them when no point is near), between
 * std_bad (0) and std_good (1). Fills y[6,n], s[6,n], prob[6], ok[6]; a line not taken: zeros, ones, 0, false. */
static PyObject *n_lanes_measure(PyObject *self, PyObject *args)
{
    PyObject *srcs, *ox, *oy, *os, *oprob, *ook;
    double x0, x1, bad, good;
    View x, y, sv, prob, ok;
    x.held = y.held = sv.held = prob.held = ok.held = 0;
    PyObject *res = NULL;
    double *rows = NULL;
    if (!PyArg_ParseTuple(args, "OOddddOOOO", &srcs, &ox, &x0, &x1, &bad, &good, &oy, &os, &oprob, &ook))
        return NULL;
    if (!PyList_CheckExact(srcs) || PyList_Size(srcs) != N_LINES) {
        PyErr_SetString(PyExc_TypeError, "lanes_measure: a list of the six lines is needed");
        return NULL;
    }
    if (view_get(ox, &x, 0, 'd', 1, "x") < 0 || view_get(oy, &y, 1, 'd', 2, "y") < 0 ||
        view_get(os, &sv, 1, 'd', 2, "s") < 0 || view_get(oprob, &prob, 1, 'd', 1, "prob") < 0 ||
        view_get(ook, &ok, 1, 'b', 1, "ok") < 0)
        goto done;
    Py_ssize_t n = x.b.shape[0];
    if (n < 1 || y.b.shape[0] != N_LINES || y.b.shape[1] != n || sv.b.shape[0] != N_LINES ||
        sv.b.shape[1] != n || prob.b.shape[0] != N_LINES || ok.b.shape[0] != N_LINES) {
        PyErr_SetString(PyExc_ValueError, "lanes_measure: shapes");
        goto done;
    }
    rows = (double *)PyMem_Malloc((size_t)(2 * N_LINES + 1) * (size_t)n * sizeof(double));
    if (rows == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    int have[2 * N_LINES];
    double pv[N_LANE];
    for (int i = 0; i < N_LINES; i++) {
        PyObject *d = PyList_GetItem(srcs, i);
        have[i] = have[N_LINES + i] = 0;
        if (i < N_LANE)
            pv[i] = 0.0;
        if (d == NULL || !PyDict_Check(d))
            continue;
        PyObject *ly = PyDict_GetItemString(d, "y"), *ls = PyDict_GetItemString(d, "y_std");
        int r = seq_row(ly, n, rows + (Py_ssize_t)i * n);
        if (r < 0)
            goto done;
        have[i] = r;
        if (!r)
            continue;
        r = seq_row(ls, n, rows + (Py_ssize_t)(N_LINES + i) * n);
        if (r < 0)
            goto done;
        have[N_LINES + i] = r;
        if (i < N_LANE) {
            /* min(1, max(0, float(p))) if p is a finite int / float (a bool is an int), else 0 */
            PyObject *p = PyDict_GetItemString(d, "prob");
            if (p != NULL && (PyFloat_Check(p) || PyLong_Check(p))) {
                double v = PyFloat_Check(p) ? PyFloat_AsDouble(p) : PyLong_AsDouble(p);
                if (v == -1.0 && PyErr_Occurred())
                    goto done;                         /* an int beyond the float range: OverflowError, as Python */
                pv[i] = isfinite(v) ? clamp01(v) : 0.0;
            }
        }
    }
    const double *xv = DATA(x);
    double *Y = DATA(y), *S = DATA(sv), *PR = DATA(prob);
    char *OK = (char *)ok.b.buf;
    double *tmp = rows + (Py_ssize_t)(2 * N_LINES) * n;
    Py_ssize_t n_near = 0;
    for (Py_ssize_t j = 0; j < n; j++)
        if (xv[j] >= x0 && xv[j] <= x1)
            n_near++;
    for (int i = 0; i < N_LINES; i++) {
        double *yo = Y + (Py_ssize_t)i * n, *so = S + (Py_ssize_t)i * n;
        if (!have[i]) {                              /* no offsets (or not finite): the line is not there this frame */
            for (Py_ssize_t j = 0; j < n; j++) {
                yo[j] = 0.0;
                so[j] = 1.0;
            }
            PR[i] = 0.0;
            OK[i] = 0;
            continue;
        }
        const double *yi = rows + (Py_ssize_t)i * n, *si = rows + (Py_ssize_t)(N_LINES + i) * n;
        double dflt = (i < N_LANE) ? 0.3 : 1.0;
        int have_s = have[N_LINES + i];
        for (Py_ssize_t j = 0; j < n; j++) {
            double sj = have_s ? si[j] : dflt;
            yo[j] = yi[j];
            so[j] = (sj >= 1e-3) ? sj : 1e-3;        /* np.maximum(si, 1e-3) */
        }
        if (i < N_LANE)
            PR[i] = pv[i];
        else {
            /* the median of the near-range deviations (all of them when no point is near) */
            Py_ssize_t m = 0;
            for (Py_ssize_t j = 0; j < n; j++)
                if (n_near == 0 || (xv[j] >= x0 && xv[j] <= x1))
                    tmp[m++] = have_s ? si[j] : dflt;
            sort_small(tmp, m);
            Py_ssize_t h = m / 2;
            double sd = (m % 2) ? tmp[h] : (tmp[h - 1] + tmp[h]) / 2.0;
            PR[i] = clamp01((bad - sd) / (bad - good));
        }
        OK[i] = 1;
    }
    res = Py_None;
    Py_INCREF(res);
done:
    PyMem_Free(rows);
    view_drop(&x); view_drop(&y); view_drop(&sv); view_drop(&prob); view_drop(&ok);
    return res;
}

/* lanes_predict(p, x, q0, q1, dt, cap):  p = minimum(p + (q0 + q1 * x) ** 2 * dt, cap)   in place */
static PyObject *n_lanes_predict(PyObject *self, PyObject *args)
{
    PyObject *op, *ox;
    double q0, q1, dt, cap;
    View p, x;
    p.held = x.held = 0;
    PyObject *res = NULL;
    if (!PyArg_ParseTuple(args, "OOdddd", &op, &ox, &q0, &q1, &dt, &cap))
        return NULL;
    if (view_get(op, &p, 1, 'd', 2, "p") < 0 || view_get(ox, &x, 0, 'd', 1, "x") < 0)
        goto done;
    Py_ssize_t n = x.b.shape[0], m = p.b.shape[0];
    if (p.b.shape[1] != n) {
        PyErr_SetString(PyExc_ValueError, "lanes_predict: shapes");
        goto done;
    }
    const double *xv = DATA(x);
    double *P = DATA(p);
    for (Py_ssize_t j = 0; j < n; j++) {
        double b = q0 + q1 * xv[j];
        double q = b * b * dt;
        for (Py_ssize_t i = 0; i < m; i++) {
            double v = P[i * n + j] + q;
            P[i * n + j] = (v <= cap || v != v) ? v : cap;      /* np.minimum (a NaN stays) */
        }
    }
    res = Py_None;
    Py_INCREF(res);
done:
    view_drop(&p); view_drop(&x);
    return res;
}

/* lanes_fit(x, zy, zs, prob, y, p, par, r, nu, gate, jumps, nsigs)
 * par = [r_floor, r_scale, edge_r_floor, edge_r_scale, gate_x0, gate_x1, on_conf, low_prob_inflate]
 *   r     = maximum(floor, scale * zs) per line kind, times low_prob_inflate where prob < on_conf
 *   gate  = (x >= gate_x0) & (x <= gate_x1), all True when none is
 *   nu    = zy - y
 *   jumps = mean over the gate points of |nu|;  nsigs = mean of |nu| / sqrt(p + r^2) */
static PyObject *n_lanes_fit(PyObject *self, PyObject *args)
{
    PyObject *ox, *ozy, *ozs, *oprob, *oy, *op, *opar, *orr, *onu, *og, *oj, *ons;
    View x, zy, zs, prob, y, p, par, r, nu, g, jm, ns;
    x.held = zy.held = zs.held = prob.held = y.held = p.held = par.held = r.held = nu.held = g.held = jm.held = ns.held = 0;
    PyObject *res = NULL;
    double *tmp = NULL;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOO", &ox, &ozy, &ozs, &oprob, &oy, &op, &opar, &orr, &onu, &og, &oj, &ons))
        return NULL;
    if (view_get(ox, &x, 0, 'd', 1, "x") < 0 || view_get(ozy, &zy, 0, 'd', 2, "zy") < 0 ||
        view_get(ozs, &zs, 0, 'd', 2, "zs") < 0 || view_get(oprob, &prob, 0, 'd', 1, "prob") < 0 ||
        view_get(oy, &y, 0, 'd', 2, "y") < 0 || view_get(op, &p, 0, 'd', 2, "p") < 0 ||
        view_get(opar, &par, 0, 'd', 1, "par") < 0 || view_get(orr, &r, 1, 'd', 2, "r") < 0 ||
        view_get(onu, &nu, 1, 'd', 2, "nu") < 0 || view_get(og, &g, 1, 'b', 1, "gate") < 0 ||
        view_get(oj, &jm, 1, 'd', 1, "jumps") < 0 || view_get(ons, &ns, 1, 'd', 1, "nsigs") < 0)
        goto done;
    Py_ssize_t n = x.b.shape[0];
#define SHAPE2(v) ((v).b.shape[0] == N_LINES && (v).b.shape[1] == n)
    if (n < 1 || !SHAPE2(zy) || !SHAPE2(zs) || !SHAPE2(y) || !SHAPE2(p) || !SHAPE2(r) || !SHAPE2(nu) ||
        prob.b.shape[0] != N_LINES || par.b.shape[0] != 8 || g.b.shape[0] != n || jm.b.shape[0] != N_LINES ||
        ns.b.shape[0] != N_LINES) {
        PyErr_SetString(PyExc_ValueError, "lanes_fit: shapes");
        goto done;
    }
#undef SHAPE2
    tmp = (double *)PyMem_Malloc((size_t)(2 * n) * sizeof(double));
    if (tmp == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    const double *c = DATA(par), *xv = DATA(x), *ZY = DATA(zy), *ZS = DATA(zs), *PR = DATA(prob), *Y = DATA(y), *P = DATA(p);
    double *R = DATA(r), *NU = DATA(nu), *J = DATA(jm), *NS = DATA(ns);
    char *G = (char *)g.b.buf;
    Py_ssize_t ng = 0;
    for (Py_ssize_t j = 0; j < n; j++) {
        G[j] = (xv[j] >= c[4] && xv[j] <= c[5]);
        ng += G[j];
    }
    if (ng == 0) {
        for (Py_ssize_t j = 0; j < n; j++)
            G[j] = 1;
        ng = n;
    }
    for (int i = 0; i < N_LINES; i++) {
        double floor_ = (i < N_LANE) ? c[0] : c[2], scale = (i < N_LANE) ? c[1] : c[3];
        double infl = (PR[i] < c[6]) ? c[7] : 1.0;
        Py_ssize_t m = 0;
        for (Py_ssize_t j = 0; j < n; j++) {
            double v = scale * ZS[i * n + j];
            double rr = (floor_ >= v) ? floor_ : v;             /* np.maximum(floor, scale * zs) */
            rr = rr * infl;
            R[i * n + j] = rr;
            double d = ZY[i * n + j] - Y[i * n + j];
            NU[i * n + j] = d;
            if (G[j]) {
                double an = fabs(d);
                tmp[m] = an;
                tmp[n + m] = an / sqrt(P[i * n + j] + rr * rr);
                m++;
            }
        }
        J[i] = seq_sum(tmp, ng) / (double)ng;
        NS[i] = seq_sum(tmp + n, ng) / (double)ng;
    }
    res = Py_None;
    Py_INCREF(res);
done:
    PyMem_Free(tmp);
    view_drop(&x); view_drop(&zy); view_drop(&zs); view_drop(&prob); view_drop(&y); view_drop(&p); view_drop(&par);
    view_drop(&r); view_drop(&nu); view_drop(&g); view_drop(&jm); view_drop(&ns);
    return res;
}

/* lanes_apply(y, p, r, nu, upd): for every line i in `upd` (a list of ints)
 *   k = p / (p + r * r);  y += k * nu;  p = p * (1.0 - k) */
static PyObject *n_lanes_apply(PyObject *self, PyObject *args)
{
    PyObject *oy, *op, *orr, *onu, *upd;
    View y, p, r, nu;
    y.held = p.held = r.held = nu.held = 0;
    PyObject *res = NULL;
    if (!PyArg_ParseTuple(args, "OOOOO", &oy, &op, &orr, &onu, &upd))
        return NULL;
    if (!PyList_CheckExact(upd)) {
        PyErr_SetString(PyExc_TypeError, "lanes_apply: upd is a list");
        return NULL;
    }
    if (view_get(oy, &y, 1, 'd', 2, "y") < 0 || view_get(op, &p, 1, 'd', 2, "p") < 0 ||
        view_get(orr, &r, 0, 'd', 2, "r") < 0 || view_get(onu, &nu, 0, 'd', 2, "nu") < 0)
        goto done;
    Py_ssize_t m = y.b.shape[0], n = y.b.shape[1];
    if (p.b.shape[0] != m || p.b.shape[1] != n || r.b.shape[0] != m || r.b.shape[1] != n || nu.b.shape[0] != m ||
        nu.b.shape[1] != n) {
        PyErr_SetString(PyExc_ValueError, "lanes_apply: shapes");
        goto done;
    }
    Py_ssize_t nu_ = PyList_Size(upd);
    for (Py_ssize_t q = 0; q < nu_; q++) {
        Py_ssize_t i = PyLong_AsSsize_t(PyList_GetItem(upd, q));
        if (i == -1 && PyErr_Occurred())
            goto done;
        if (i < 0 || i >= m) {
            PyErr_SetString(PyExc_ValueError, "lanes_apply: line index");
            goto done;
        }
    }
    double *Y = DATA(y), *P = DATA(p);
    const double *R = DATA(r), *NU = DATA(nu);
    for (Py_ssize_t q = 0; q < nu_; q++) {
        Py_ssize_t i = PyLong_AsSsize_t(PyList_GetItem(upd, q));
        for (Py_ssize_t j = 0; j < n; j++) {
            double pu = P[i * n + j], ru = R[i * n + j];
            double k = pu / (pu + ru * ru);
            Y[i * n + j] += k * NU[i * n + j];
            P[i * n + j] = pu * (1.0 - k);
        }
    }
    res = Py_None;
    Py_INCREF(res);
done:
    view_drop(&y); view_drop(&p); view_drop(&r); view_drop(&nu);
    return res;
}

/* round_lists(a, decimals, root): np.round(a, decimals).tolist() of a float array of one or two dimensions
 * (root true: of np.sqrt(a)); numpy rounds as rint(v * 10^decimals) / 10^decimals */
static PyObject *n_round_lists(PyObject *self, PyObject *args)
{
    PyObject *oa;
    int dec, root;
    View a;
    a.held = 0;
    if (!PyArg_ParseTuple(args, "Oip", &oa, &dec, &root))
        return NULL;
    if (dec < 1 || dec > 22) {
        PyErr_SetString(PyExc_ValueError, "round_lists: decimals 1..22");
        return NULL;
    }
    if (PyObject_GetBuffer(oa, &a.b, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;
    a.held = 1;
    const char *f = a.b.format ? a.b.format : "B";
    if (*f == '@' || *f == '=' || *f == '<')
        f++;
    if (!(f[0] == 'd' && f[1] == 0) || a.b.itemsize != 8 || a.b.ndim < 1 || a.b.ndim > 2) {
        view_drop(&a);
        PyErr_SetString(PyExc_TypeError, "round_lists: a float64 array of one or two dimensions");
        return NULL;
    }
    Py_ssize_t rows = a.b.ndim == 2 ? a.b.shape[0] : 1, n = a.b.ndim == 2 ? a.b.shape[1] : a.b.shape[0];
    const double *v = DATA(a);
    double sc = POW10[dec];
    PyObject *out = a.b.ndim == 2 ? PyList_New(rows) : NULL;
    if (a.b.ndim == 2 && out == NULL) {
        view_drop(&a);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < rows; i++) {
        PyObject *row = PyList_New(n);
        if (row == NULL)
            goto fail;
        for (Py_ssize_t j = 0; j < n; j++) {
            double d = v[i * n + j];
            if (root)
                d = sqrt(d);
            PyObject *o = PyFloat_FromDouble(rint(d * sc) / sc);
            if (o == NULL) {
                Py_DECREF(row);
                goto fail;
            }
            PyList_SetItem(row, j, o);
        }
        if (a.b.ndim == 2)
            PyList_SetItem(out, i, row);
        else
            out = row;
    }
    view_drop(&a);
    return out;
fail:
    Py_XDECREF(out);
    view_drop(&a);
    return NULL;
}

/* ====================================================================================================
 * The road geometry's centre line from the camera (trackers/roadgeom.py: `on_lanes` and `_centre`; numpy's
 * elementwise steps in its order, cumsum in sequence, np.round as rint(v * 10^d) / 10^d)
 * ==================================================================================================== */
static PyObject *rint_list(const double *v, Py_ssize_t n, int dec)
{
    PyObject *out = PyList_New(n);
    if (out == NULL)
        return NULL;
    double sc = POW10[dec];
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *o = PyFloat_FromDouble(rint(v[i] * sc) / sc);
        if (o == NULL) {
            Py_DECREF(out);
            return NULL;
        }
        PyList_SetItem(out, i, o);
    }
    return out;
}

/* geom_centre(x, y, c, reach) -> (s, x, y, conf) as lists, rounded to 1, 1, 2, 2 decimals:
 *     reach None:  `_centre(x, y, c, ...)` (c: the confidence per point)
 *     else:        m = x <= reach + 1e-6;  `_centre(x[m], y[m], np.clip(1.0 - c[m] / 1.5, 0.1, 1.0), ...)` (c: std)
 * x, y, c: float64 arrays of one length. No point (left): IndexError, as x[0] of an empty array. */
static PyObject *n_geom_centre(PyObject *self, PyObject *args)
{
    PyObject *ox, *oy, *oc, *oreach;
    if (!PyArg_ParseTuple(args, "OOOO", &ox, &oy, &oc, &oreach))
        return NULL;
    double reach = 0.0;
    int masked = oreach != Py_None;
    if (masked) {
        reach = PyFloat_AsDouble(oreach);
        if (reach == -1.0 && PyErr_Occurred())
            return NULL;
    }
    View vx, vy, vc;
    vx.held = vy.held = vc.held = 0;
    PyObject *res = NULL;
    double *buf = NULL;
    if (view_get(ox, &vx, 0, 'd', 1, "x") < 0 || view_get(oy, &vy, 0, 'd', 1, "y") < 0 || view_get(oc, &vc, 0, 'd', 1, "c") < 0)
        goto done;
    Py_ssize_t n = vx.b.shape[0];
    if (vy.b.shape[0] != n || vc.b.shape[0] != n) {
        PyErr_SetString(PyExc_ValueError, "geom_centre: arrays of one length");
        goto done;
    }
    buf = PyMem_Malloc((size_t)(n > 0 ? n : 1) * 5 * sizeof(double));
    if (buf == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    double *X = buf, *Y = buf + n, *C = buf + 2 * n, *S = buf + 3 * n;
    const double *x = DATA(vx), *y = DATA(vy), *c = DATA(vc);
    Py_ssize_t m = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        if (masked && !(x[i] <= reach + 1e-6))
            continue;
        X[m] = x[i];
        Y[m] = y[i];
        if (masked) {
            double v = 1.0 - c[i] / 1.5;
            if (!isnan(v)) {                           /* np.clip: NaN stays NaN */
                v = v > 0.1 ? v : 0.1;
                v = v < 1.0 ? v : 1.0;
            }
            C[m] = v;
        } else
            C[m] = c[i];
        m++;
    }
    if (m == 0) {
        PyErr_SetString(PyExc_IndexError, "index 0 is out of bounds for axis 0 with size 0");
        goto done;
    }
    /* s = concatenate([0], cumsum(hypot(diff(x), diff(y)))) + x[0] */
    double acc = 0.0;
    S[0] = 0.0;
    for (Py_ssize_t i = 1; i < m; i++) {
        double h = hypot(X[i] - X[i - 1], Y[i] - Y[i - 1]);
        acc = i == 1 ? h : acc + h;
        S[i] = acc;
    }
    for (Py_ssize_t i = 0; i < m; i++)
        S[i] = S[i] + X[0];
    PyObject *ls = rint_list(S, m, 1), *lx = rint_list(X, m, 1), *ly = rint_list(Y, m, 2), *lc = rint_list(C, m, 2);
    if (ls && lx && ly && lc)
        res = Py_BuildValue("(NNNN)", ls, lx, ly, lc);
    else {
        Py_XDECREF(ls);
        Py_XDECREF(lx);
        Py_XDECREF(ly);
        Py_XDECREF(lc);
    }
done:
    PyMem_Free(buf);
    view_drop(&vx);
    view_drop(&vy);
    view_drop(&vc);
    return res;
}

/* geom_reach(x, sd, reach_std, reach_max, reach_min): RoadGeometryFusion._reach (the min / max of Python on the
 * same values, so the result is the same object kind: a float, or one of the configured numbers) */
static PyObject *n_geom_reach(PyObject *self, PyObject *args)
{
    PyObject *ox, *os, *ostd, *omax, *omin;
    if (!PyArg_ParseTuple(args, "OOOOO", &ox, &os, &ostd, &omax, &omin))
        return NULL;
    double rstd, rmax, rmin;
    if (!get_num(ostd, &rstd) || !get_num(omax, &rmax) || !get_num(omin, &rmin)) {
        PyErr_SetString(PyExc_TypeError, "geom_reach: plain numbers are needed");
        return NULL;
    }
    View vx, vs;
    vx.held = vs.held = 0;
    PyObject *res = NULL;
    if (view_get(ox, &vx, 0, 'd', 1, "x") < 0 || view_get(os, &vs, 0, 'd', 1, "sd") < 0)
        goto done;
    Py_ssize_t n = vx.b.shape[0];
    if (vs.b.shape[0] != n) {
        PyErr_SetString(PyExc_ValueError, "geom_reach: arrays of one length");
        goto done;
    }
    if (n == 0) {
        PyErr_SetString(PyExc_IndexError, "index -1 is out of bounds for axis 0 with size 0");
        goto done;
    }
    const double *x = DATA(vx), *sd = DATA(vs);
    double xl = x[n - 1], r = xl;
    for (Py_ssize_t i = 0; i < n; i++)
        if (sd[i] > rstd || x[i] > rmax) {
            r = x[i > 0 ? i - 1 : 0];
            break;
        }
    /* a = min(r, reach_max, x[-1]); b = min(reach_min, x[-1]); max(a, b) */
    double a = r, b = rmin;
    PyObject *ao = NULL, *bo = omin;                   /* NULL: a float of the value */
    if (rmax < a) {
        a = rmax;
        ao = omax;
    }
    if (xl < a) {
        a = xl;
        ao = NULL;
    }
    if (xl < b) {
        b = xl;
        bo = NULL;
    }
    if (b > a) {
        a = b;
        ao = bo;
    }
    if (ao != NULL) {
        Py_INCREF(ao);
        res = ao;
    } else
        res = PyFloat_FromDouble(a);
done:
    view_drop(&vx);
    view_drop(&vs);
    return res;
}

/* ====================================================================================================
 * The radar's motion model (radar/models.py is the reference: still_b, transition_b, mix_b, predict_b,
 * combine_b; radar/tracker.py `_evolve`, `p_still_many`). State [px, py, vx, vy, ax, ay, dp, lh], three
 * hypotheses (stand, cv, ca). The reference builds dense 8x8 transitions and multiplies them; here only the
 * entries that are not zero are propagated (F has at most three per row), summed in index order. Elementwise
 * steps are written in numpy's order of operations; sums of products differ from numpy's (BLAS, einsum) in
 * the order of addition: the results agree to rounding. Inputs that are not finite are left to the reference
 * (`Fallback`): a dense product turns 0 * inf into NaN where skipping the zero does not.
 * ==================================================================================================== */
enum { R_PX, R_PY, R_VX, R_VY, R_AX, R_AY, R_DP, R_LH, RN };
enum { R_STAND, R_CV, R_CA };
#define RNN (RN * RN)
#define RNPAR 12

typedef struct {
    double dpw2, sd2, sv2, cvn, caj, tau, qacc, r_s2m, r_m2s, r_fast, r_cv2ca, r_ca2cv;
} RPar;

/* a C-contiguous float64 buffer of exactly n doubles (any shape) */
static int dview_get(PyObject *o, View *v, int writable, Py_ssize_t n, const char *what)
{
    v->held = 0;
    if (PyObject_GetBuffer(o, &v->b, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0)) < 0)
        return -1;
    v->held = 1;
    const char *f = v->b.format ? v->b.format : "B";
    if (*f == '@' || *f == '=' || *f == '<')
        f++;
    if (!(f[0] == 'd' && f[1] == 0 && v->b.itemsize == 8) || v->b.len != n * (Py_ssize_t)sizeof(double)) {
        view_drop(v);
        PyErr_Format(PyExc_TypeError, "%s: a C-contiguous float64 array of %zd values is needed", what, n);
        return -1;
    }
    return 0;
}

/* a batch of B items of `len` doubles: one buffer of B * len, or a list / tuple of B buffers of len each */
typedef struct {
    View one;
    View *items;
    Py_ssize_t nitems;
    const double **p;
} Bat;

static void bat_init(Bat *bt)
{
    bt->one.held = 0;
    bt->items = NULL;
    bt->nitems = 0;
    bt->p = NULL;
}

static void bat_drop(Bat *bt)
{
    view_drop(&bt->one);
    for (Py_ssize_t i = 0; i < bt->nitems; i++)
        view_drop(&bt->items[i]);
    PyMem_Free(bt->items);
    PyMem_Free((void *)bt->p);
    bat_init(bt);
}

static int bat_get(PyObject *o, Py_ssize_t B, Py_ssize_t len, Bat *bt, const char *what)
{
    bt->p = (const double **)PyMem_Malloc((size_t)(B ? B : 1) * sizeof(double *));
    if (bt->p == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    int is_list = PyList_Check(o);
    if (is_list || PyTuple_Check(o)) {
        Py_ssize_t n = is_list ? PyList_Size(o) : PyTuple_Size(o);
        if (n != B) {
            PyErr_Format(PyExc_ValueError, "%s: %zd items, %zd expected", what, n, B);
            return -1;
        }
        bt->items = (View *)PyMem_Calloc((size_t)(B ? B : 1), sizeof(View));
        if (bt->items == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        for (Py_ssize_t b = 0; b < B; b++) {
            if (dview_get(seq_item(o, is_list, b), &bt->items[b], 0, len, what) < 0)
                return -1;
            bt->nitems = b + 1;
            bt->p[b] = (const double *)bt->items[b].b.buf;
        }
        return 0;
    }
    if (dview_get(o, &bt->one, 0, B * len, what) < 0)
        return -1;
    for (Py_ssize_t b = 0; b < B; b++)
        bt->p[b] = (const double *)bt->one.b.buf + b * len;
    return 0;
}

/* B rows of `width` numbers into out: a float64 buffer of B * width, or a list / tuple of B numbers
 * (width 1) or of B sequences of `width` numbers */
static int rows_get(PyObject *o, Py_ssize_t B, int width, double *out, const char *what)
{
    int is_list = PyList_Check(o);
    if (is_list || PyTuple_Check(o)) {
        Py_ssize_t n = is_list ? PyList_Size(o) : PyTuple_Size(o);
        if (n != B) {
            PyErr_Format(PyExc_ValueError, "%s: %zd items, %zd expected", what, n, B);
            return -1;
        }
        for (Py_ssize_t b = 0; b < B; b++) {
            PyObject *it = seq_item(o, is_list, b);
            if (width == 1) {
                double v = PyFloat_AsDouble(it);
                if (v == -1.0 && PyErr_Occurred())
                    return -1;
                out[b] = v;
                continue;
            }
            int il = PyList_Check(it);
            if (!(il || PyTuple_Check(it)) || (il ? PyList_Size(it) : PyTuple_Size(it)) != width) {
                PyErr_Format(PyExc_TypeError, "%s: rows of %d numbers are needed", what, width);
                return -1;
            }
            for (int k = 0; k < width; k++) {
                double v = PyFloat_AsDouble(seq_item(it, il, k));
                if (v == -1.0 && PyErr_Occurred())
                    return -1;
                out[b * width + k] = v;
            }
        }
        return 0;
    }
    View v;
    if (dview_get(o, &v, 0, B * width, what) < 0)
        return -1;
    memcpy(out, v.b.buf, (size_t)(B * width) * sizeof(double));
    view_drop(&v);
    return 0;
}

/* the batch size: the length of a list / tuple, or of a float64 buffer of one value per item */
static Py_ssize_t batch_len(PyObject *o)
{
    if (PyList_Check(o))
        return PyList_Size(o);
    if (PyTuple_Check(o))
        return PyTuple_Size(o);
    Py_buffer v;
    if (PyObject_GetBuffer(o, &v, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return -1;
    Py_ssize_t n = v.ndim >= 1 ? v.shape[0] : -1;
    PyBuffer_Release(&v);
    if (n < 0)
        PyErr_SetString(PyExc_TypeError, "radar: a batch (an array of one row per object, or a list) is needed");
    return n;
}

static int all_finite(const double *a, Py_ssize_t n)
{
    for (Py_ssize_t i = 0; i < n; i++)
        if (!isfinite(a[i]))
            return 0;
    return 1;
}

/* still_b: the weight of the velocity (mean and uncertainty) inside exp(-|v|^2 / v0^2) */
static double r_still(const double *x, const double *P, double v0)
{
    double v2 = v0 * v0;
    double a00 = 2.0 * P[R_VX * RN + R_VX] + v2, a11 = 2.0 * P[R_VY * RN + R_VY] + v2, a01 = 2.0 * P[R_VX * RN + R_VY];
    double det = a00 * a11 - a01 * (2.0 * P[R_VY * RN + R_VX]);
    if (!(det > 0.0))
        return 0.0;
    double vx = x[R_VX], vy = x[R_VY];
    double q = (a11 * vx * vx - 2.0 * a01 * vx * vy + a00 * vy * vy) / det;
    double r = v2 / sqrt(det) * exp(-(q > 60.0 ? 60.0 : q));
    return r > 1.0 ? 1.0 : r;
}

/* transition_b: switching probabilities over dt, T[from * 3 + to] */
static void r_transition(double dt, double q0, double q1, const RPar *p, double *T)
{
    if (!(dt > 0.0))
        dt = 0.0;
    double R[9] = {0.0, p->r_s2m * 0.7, p->r_s2m * 0.3,
                   p->r_fast + p->r_m2s * q0, 0.0, p->r_cv2ca,
                   p->r_fast + p->r_m2s * q1, p->r_ca2cv, 0.0};
    for (int k = 0; k < 9; k++)
        T[k] = 1.0 - exp(-R[k] * dt);
    for (int i = 0; i < 3; i++) {
        double *t = T + 3 * i;
        double off = (t[0] + t[1]) + t[2];
        double s = off > 0.9 ? 0.9 / off : 1.0;
        t[0] *= s;
        t[1] *= s;
        t[2] *= s;
        double d = 1.0 - ((t[0] + t[1]) + t[2]);
        t[i] = d;
    }
}

/* mix_b with the free estimate: "it stands" hands (xf, Pf) to a moving hypothesis. Out: xo (3, N), Po (3, N, N),
 * c (3) the predicted model probabilities */
static void r_mix(const double *xs, const double *Ps, const double *mu, const double *T, const double *xf,
                  const double *Pf, double *xo, double *Po, double *c)
{
    double w[9];
    for (int j = 0; j < 3; j++)
        c[j] = (T[j] * mu[0] + T[3 + j] * mu[1]) + T[6 + j] * mu[2];
    for (int j = 0; j < 3; j++) {
        int low = c[j] < 1e-12;
        for (int i = 0; i < 3; i++)
            w[3 * i + j] = low ? (i == j ? 1.0 : 0.0) : T[3 * i + j] * mu[i] / c[j];
    }
    for (int j = 0; j < 3; j++) {
        const double *sx[3], *sp[3];
        for (int i = 0; i < 3; i++) {
            int hand = (i == R_STAND && j != R_STAND);
            sx[i] = hand ? xf : xs + i * RN;
            sp[i] = hand ? Pf : Ps + i * RNN;
        }
        double w0 = w[j], w1 = w[3 + j], w2 = w[6 + j];
        double *x = xo + j * RN, *P = Po + j * RNN;
        double d0[RN], d1[RN], d2[RN];
        for (int n = 0; n < RN; n++)
            x[n] = (w0 * sx[0][n] + w1 * sx[1][n]) + w2 * sx[2][n];
        for (int n = 0; n < RN; n++) {
            d0[n] = sx[0][n] - x[n];
            d1[n] = sx[1][n] - x[n];
            d2[n] = sx[2][n] - x[n];
        }
        for (int m = 0; m < RN; m++)
            for (int n = 0; n < RN; n++) {
                int k = m * RN + n;
                double a = (w0 * sp[0][k] + w1 * sp[1][k]) + w2 * sp[2][k];
                double b = ((w0 * d0[m]) * d0[n] + (w1 * d1[m]) * d1[n]) + (w2 * d2[m]) * d2[n];
                P[k] = a + b;
            }
    }
}

/* _shape_b: (s00, s11, s01), q_lon along the travel and q_lat across it, isotropic below 1 m/s */
static void r_shape(double ql, double qt, const double *x, double *s)
{
    double vx = x[R_VX], vy = x[R_VY];
    double sp = hypot(vx, vy);
    if (sp >= 1.0) {
        double c = vx / sp, sn = vy / sp;
        s[0] = c * c * ql + sn * sn * qt;
        s[1] = sn * sn * ql + c * c * qt;
        s[2] = c * sn * (ql - qt);
    } else {
        s[0] = s[1] = 0.5 * (ql + qt);
        s[2] = 0.0;
    }
}

/* qa * k into the 2x2 block (r, c) of Q, and its mirror (c, r) */
static void r_qblock(double *Q, int r, int c, const double *s, double k)
{
    double a = s[0] * k, b = s[1] * k, o = s[2] * k;
    Q[r * RN + c] = a;
    Q[(r + 1) * RN + c + 1] = b;
    Q[r * RN + c + 1] = o;
    Q[(r + 1) * RN + c] = o;
    if (r != c) {
        Q[c * RN + r] = a;
        Q[(c + 1) * RN + r + 1] = b;
        Q[(c + 1) * RN + r] = o;
        Q[c * RN + r + 1] = o;
    }
}

/* predict_b for one hypothesis: (x, P) forward by dt into (xo, Po), with the kind's speed bound */
static void r_predict(int model, const double *x, const double *P, double dt, double a_lon, double a_lat,
                      double v_max, const RPar *p, double *xo, double *Po)
{
    double adt = fabs(dt);
    int nz[RN], col[RN][3];
    double val[RN][3];
    double Q[RNN];
    memset(Q, 0, sizeof(Q));
    for (int i = 0; i < RN; i++) {
        nz[i] = 1;
        col[i][0] = i;
        val[i][0] = 1.0;
    }
    Q[R_DP * RN + R_DP] = p->dpw2 * adt;
    if (model == R_STAND) {
        for (int i = R_VX; i <= R_AY; i++) {
            nz[i] = 0;
            Q[i * RN + i] = p->sv2;
        }
        Q[R_PX * RN + R_PX] = Q[R_PY * RN + R_PY] = p->sd2 * adt;
    } else if (model == R_CV) {
        double dt2 = adt * adt, dt3 = pow(adt, 3.0), s[3];
        for (int i = 0; i < 2; i++) {
            nz[R_PX + i] = 2;
            col[R_PX + i][1] = R_VX + i;
            val[R_PX + i][1] = dt;
        }
        nz[R_AX] = nz[R_AY] = 0;
        double t1 = p->cvn * a_lon, t2 = p->cvn * a_lat;
        r_shape(t1 * t1, t2 * t2, x, s);
        r_qblock(Q, 0, 0, s, dt3 / 3.0);
        r_qblock(Q, 0, 2, s, dt2 / 2.0);
        r_qblock(Q, 2, 2, s, adt);
        Q[R_AX * RN + R_AX] = Q[R_AY * RN + R_AY] = p->qacc;
    } else {
        double rho = exp(-adt / p->tau), s[3];
        double dt2 = adt * adt, dt3 = pow(adt, 3.0), dt4 = pow(adt, 4.0), dt5 = pow(adt, 5.0);
        for (int i = 0; i < 2; i++) {
            nz[R_PX + i] = 3;
            col[R_PX + i][1] = R_VX + i;
            val[R_PX + i][1] = dt;
            col[R_PX + i][2] = R_AX + i;
            val[R_PX + i][2] = 0.5 * dt * dt;          /* also for a step back: x - v|dt| + a dt^2 / 2 */
            nz[R_VX + i] = 2;
            col[R_VX + i][1] = R_AX + i;
            val[R_VX + i][1] = dt;
            val[R_AX + i][0] = rho;
        }
        double t1 = p->caj * a_lon, t2 = p->caj * a_lat;
        r_shape(t1 * t1, t2 * t2, x, s);
        r_qblock(Q, 0, 0, s, dt5 / 20.0);
        r_qblock(Q, 0, 2, s, dt4 / 8.0);
        r_qblock(Q, 0, 4, s, dt3 / 6.0);
        r_qblock(Q, 2, 2, s, dt3 / 3.0);
        r_qblock(Q, 2, 4, s, dt2 / 2.0);
        r_qblock(Q, 4, 4, s, adt);
    }
    /* coefficients that are exactly zero (dt = 0) are no terms */
    for (int i = 0; i < RN; i++) {
        int k = 0;
        for (int e = 0; e < nz[i]; e++)
            if (val[i][e] != 0.0) {
                col[i][k] = col[i][e];
                val[i][k] = val[i][e];
                k++;
            }
        nz[i] = k;
    }
    double x1[RN], A[RNN], P1[RNN];
    for (int i = 0; i < RN; i++) {
        double acc = 0.0;
        for (int e = 0; e < nz[i]; e++)
            acc = e ? acc + val[i][e] * x[col[i][e]] : val[i][e] * x[col[i][e]];
        x1[i] = acc;
    }
    for (int i = 0; i < RN; i++)                         /* A = F P */
        for (int j = 0; j < RN; j++) {
            double acc = 0.0;
            for (int e = 0; e < nz[i]; e++)
                acc = e ? acc + val[i][e] * P[col[i][e] * RN + j] : val[i][e] * P[col[i][e] * RN + j];
            A[i * RN + j] = acc;
        }
    for (int i = 0; i < RN; i++)                         /* P1 = A F^T + Q */
        for (int j = 0; j < RN; j++) {
            double acc = 0.0;
            for (int e = 0; e < nz[j]; e++)
                acc = e ? acc + A[i * RN + col[j][e]] * val[j][e] : A[i * RN + col[j][e]] * val[j][e];
            P1[i * RN + j] = acc + Q[i * RN + j];
        }
    if (model != R_STAND) {
        double sp = hypot(x1[R_VX], x1[R_VY]);
        if (sp > v_max) {
            double f = v_max / sp, d = sp - v_max, ex = d * d;
            x1[R_VX] *= f;
            x1[R_VY] *= f;
            P1[R_VX * RN + R_VX] += ex;
            P1[R_VY * RN + R_VY] += ex;
        }
    }
    memcpy(xo, x1, sizeof(x1));
    for (int i = 0; i < RN; i++)
        for (int j = 0; j < RN; j++)
            Po[i * RN + j] = 0.5 * (P1[i * RN + j] + P1[j * RN + i]);
}

/* combine_b: the moment-matched single estimate of the three hypotheses */
static void r_combine(const double *xs, const double *Ps, const double *mu, double *x, double *P)
{
    double d[3][RN];
    for (int n = 0; n < RN; n++)
        x[n] = (mu[0] * xs[n] + mu[1] * xs[RN + n]) + mu[2] * xs[2 * RN + n];
    for (int i = 0; i < 3; i++)
        for (int n = 0; n < RN; n++)
            d[i][n] = xs[i * RN + n] - x[n];
    for (int m = 0; m < RN; m++)
        for (int n = 0; n < RN; n++) {
            int k = m * RN + n;
            double a = (mu[0] * Ps[k] + mu[1] * Ps[RNN + k]) + mu[2] * Ps[2 * RNN + k];
            double b = ((mu[0] * d[0][m]) * d[0][n] + (mu[1] * d[1][m]) * d[1][n]) + (mu[2] * d[2][m]) * d[2][n];
            P[k] = a + b;
        }
}

static int rpar_get(PyObject *o, RPar *p)
{
    double v[RNPAR];
    if (rows_get(o, RNPAR, 1, v, "par") < 0)           /* 12 numbers: a tuple or a buffer */
        return -1;
    if (!all_finite(v, RNPAR)) {
        PyErr_SetString(Fallback, "radar: parameters not finite");
        return -1;
    }
    p->dpw2 = v[0]; p->sd2 = v[1]; p->sv2 = v[2]; p->cvn = v[3]; p->caj = v[4]; p->tau = v[5]; p->qacc = v[6];
    p->r_s2m = v[7]; p->r_m2s = v[8]; p->r_fast = v[9]; p->r_cv2ca = v[10]; p->r_ca2cv = v[11];
    return 0;
}

/* an output buffer of n doubles, or None (*out = NULL) where `optional` */
static int out_get(PyObject *o, View *v, Py_ssize_t n, int optional, double **out, const char *what)
{
    v->held = 0;
    *out = NULL;
    if (optional && o == Py_None)
        return 0;
    if (dview_get(o, v, 1, n, what) < 0)
        return -1;
    *out = (double *)v->b.buf;
    return 0;
}

/* radar_evolve(xs, Ps, mu, xf, Pf, dt, kin, par, xs_out, Ps_out, mu_out, x_out | None, P_out | None,
 *              xf_out | None, Pf_out | None)
 * Tracker._evolve for B objects: stillness of the moving hypotheses, switching matrix, interaction with the
 * free estimate's hand-over, prediction of the three hypotheses by dt[b]; with x_out, P_out also combine_b of
 * the result with the predicted probabilities; with xf_out, Pf_out the free estimate predicted by the steady
 * hypothesis (Tracker._advance_all).
 *   xs (B, 3, N), Ps (B, 3, N, N), mu (B, 3), xf (B, N), Pf (B, N, N): one buffer each, or a list / tuple of B
 *   per-object buffers;  dt: (B) buffer or list of numbers;  kin: (B, 4) buffer or list of
 *   (a_lon, a_lat, v_max, v_still);  par: 12 numbers (models.py `_native_par`). Outputs: float64 buffers. */
static PyObject *n_radar_evolve(PyObject *self, PyObject *args)
{
    PyObject *oxs, *oPs, *omu, *oxf, *oPf, *odt, *okin, *opar, *o1, *o2, *o3, *o4, *o5, *o6, *o7;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOO", &oxs, &oPs, &omu, &oxf, &oPf, &odt, &okin, &opar, &o1, &o2, &o3,
                          &o4, &o5, &o6, &o7))
        return NULL;
    PyObject *res = NULL;
    Bat xs, Ps, mu, xf, Pf;
    bat_init(&xs); bat_init(&Ps); bat_init(&mu); bat_init(&xf); bat_init(&Pf);
    View v1, v2, v3, v4, v5, v6, v7;
    v1.held = v2.held = v3.held = v4.held = v5.held = v6.held = v7.held = 0;
    double *dk = NULL;
    double *xso, *Pso, *muo, *xo, *Po, *xfo, *Pfo;
    RPar par;
    Py_ssize_t B = batch_len(odt);
    if (B < 0 || rpar_get(opar, &par) < 0)
        goto done;
    dk = (double *)PyMem_Malloc((size_t)(B ? B : 1) * 5 * sizeof(double));
    if (dk == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    if (rows_get(odt, B, 1, dk, "dt") < 0 || rows_get(okin, B, 4, dk + B, "kin") < 0 ||
        bat_get(oxs, B, 3 * RN, &xs, "xs") < 0 || bat_get(oPs, B, 3 * RNN, &Ps, "Ps") < 0 ||
        bat_get(omu, B, 3, &mu, "mu") < 0 || bat_get(oxf, B, RN, &xf, "xf") < 0 ||
        bat_get(oPf, B, RNN, &Pf, "Pf") < 0 ||
        out_get(o1, &v1, B * 3 * RN, 0, &xso, "xs_out") < 0 || out_get(o2, &v2, B * 3 * RNN, 0, &Pso, "Ps_out") < 0 ||
        out_get(o3, &v3, B * 3, 0, &muo, "mu_out") < 0 || out_get(o4, &v4, B * RN, 1, &xo, "x_out") < 0 ||
        out_get(o5, &v5, B * RNN, 1, &Po, "P_out") < 0 || out_get(o6, &v6, B * RN, 1, &xfo, "xf_out") < 0 ||
        out_get(o7, &v7, B * RNN, 1, &Pfo, "Pf_out") < 0)
        goto done;
    if ((xo == NULL) != (Po == NULL) || (xfo == NULL) != (Pfo == NULL)) {
        PyErr_SetString(PyExc_ValueError, "radar_evolve: x_out and P_out (xf_out and Pf_out) go together");
        goto done;
    }
    if (!all_finite(dk, B * 5))
        goto fallback;
    for (Py_ssize_t b = 0; b < B; b++)
        if (!all_finite(xs.p[b], 3 * RN) || !all_finite(Ps.p[b], 3 * RNN) || !all_finite(mu.p[b], 3) ||
            !all_finite(xf.p[b], RN) || !all_finite(Pf.p[b], RNN))
            goto fallback;
    for (Py_ssize_t b = 0; b < B; b++) {
        const double *K = dk + B + 4 * b;            /* a_lon, a_lat, v_max, v_still */
        double dt = dk[b], T[9], c[3], xm[3 * RN], Pm[3 * RNN];
        const double *X = xs.p[b], *P = Ps.p[b];
        double q0 = r_still(X + RN, P + RNN, K[3]), q1 = r_still(X + 2 * RN, P + 2 * RNN, K[3]);
        r_transition(dt, q0, q1, &par, T);
        r_mix(X, P, mu.p[b], T, xf.p[b], Pf.p[b], xm, Pm, c);
        double *xb = xso + b * 3 * RN, *Pb = Pso + b * 3 * RNN;
        for (int j = 0; j < 3; j++)
            r_predict(j, xm + j * RN, Pm + j * RNN, dt, K[0], K[1], K[2], &par, xb + j * RN, Pb + j * RNN);
        muo[3 * b] = c[0];
        muo[3 * b + 1] = c[1];
        muo[3 * b + 2] = c[2];
        if (xo)
            r_combine(xb, Pb, c, xo + b * RN, Po + b * RNN);
        if (xfo)
            r_predict(R_CV, xf.p[b], Pf.p[b], dt, K[0], K[1], K[2], &par, xfo + b * RN, Pfo + b * RNN);
    }
    res = Py_None;
    Py_INCREF(res);
    goto done;
fallback:
    PyErr_SetString(Fallback, "radar_evolve: an input is not finite");
done:
    PyMem_Free(dk);
    bat_drop(&xs); bat_drop(&Ps); bat_drop(&mu); bat_drop(&xf); bat_drop(&Pf);
    view_drop(&v1); view_drop(&v2); view_drop(&v3); view_drop(&v4); view_drop(&v5); view_drop(&v6); view_drop(&v7);
    return res;
}

/* radar_combine(xs, Ps, mu, x_out, P_out): combine_b for B objects (inputs as in radar_evolve) */
static PyObject *n_radar_combine(PyObject *self, PyObject *args)
{
    PyObject *oxs, *oPs, *omu, *o1, *o2;
    if (!PyArg_ParseTuple(args, "OOOOO", &oxs, &oPs, &omu, &o1, &o2))
        return NULL;
    PyObject *res = NULL;
    Bat xs, Ps, mu;
    bat_init(&xs); bat_init(&Ps); bat_init(&mu);
    View v1, v2;
    v1.held = v2.held = 0;
    double *xo, *Po;
    Py_ssize_t B = batch_len(omu);
    if (B < 0)
        goto done;
    if (bat_get(oxs, B, 3 * RN, &xs, "xs") < 0 || bat_get(oPs, B, 3 * RNN, &Ps, "Ps") < 0 ||
        bat_get(omu, B, 3, &mu, "mu") < 0 || out_get(o1, &v1, B * RN, 0, &xo, "x_out") < 0 ||
        out_get(o2, &v2, B * RNN, 0, &Po, "P_out") < 0)
        goto done;
    for (Py_ssize_t b = 0; b < B; b++)
        if (!all_finite(xs.p[b], 3 * RN) || !all_finite(Ps.p[b], 3 * RNN) || !all_finite(mu.p[b], 3)) {
            PyErr_SetString(Fallback, "radar_combine: an input is not finite");
            goto done;
        }
    for (Py_ssize_t b = 0; b < B; b++)
        r_combine(xs.p[b], Ps.p[b], mu.p[b], xo + b * RN, Po + b * RNN);
    res = Py_None;
    Py_INCREF(res);
done:
    bat_drop(&xs); bat_drop(&Ps); bat_drop(&mu);
    view_drop(&v1); view_drop(&v2);
    return res;
}

/* radar_p_still(xs, Ps, mu, v_still) -> [min(1, mu_stand + mu_cv q_cv + mu_ca q_ca)] (Tracker.p_still_many) */
static PyObject *n_radar_p_still(PyObject *self, PyObject *args)
{
    PyObject *oxs, *oPs, *omu, *ov;
    if (!PyArg_ParseTuple(args, "OOOO", &oxs, &oPs, &omu, &ov))
        return NULL;
    PyObject *res = NULL;
    Bat xs, Ps, mu;
    bat_init(&xs); bat_init(&Ps); bat_init(&mu);
    double *v0 = NULL;
    Py_ssize_t B = batch_len(ov);
    if (B < 0)
        goto done;
    v0 = (double *)PyMem_Malloc((size_t)(B ? B : 1) * sizeof(double));
    if (v0 == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    if (rows_get(ov, B, 1, v0, "v_still") < 0 || bat_get(oxs, B, 3 * RN, &xs, "xs") < 0 ||
        bat_get(oPs, B, 3 * RNN, &Ps, "Ps") < 0 || bat_get(omu, B, 3, &mu, "mu") < 0)
        goto done;
    for (Py_ssize_t b = 0; b < B; b++)
        if (!isfinite(v0[b]) || !all_finite(xs.p[b] + RN, 2 * RN) || !all_finite(Ps.p[b] + RNN, 2 * RNN) ||
            !all_finite(mu.p[b], 3)) {
            PyErr_SetString(Fallback, "radar_p_still: an input is not finite");
            goto done;
        }
    res = PyList_New(B);
    if (res == NULL)
        goto done;
    for (Py_ssize_t b = 0; b < B; b++) {
        const double *X = xs.p[b], *P = Ps.p[b], *m = mu.p[b];
        double q0 = r_still(X + RN, P + RNN, v0[b]), q1 = r_still(X + 2 * RN, P + 2 * RNN, v0[b]);
        double s = (m[0] + m[1] * q0) + m[2] * q1;
        PyObject *f = PyFloat_FromDouble(s > 1.0 ? 1.0 : s);
        if (f == NULL) {
            Py_CLEAR(res);
            goto done;
        }
        PyList_SetItem(res, b, f);
    }
done:
    PyMem_Free(v0);
    bat_drop(&xs); bat_drop(&Ps); bat_drop(&mu);
    return res;
}

/* ==================================================================================================== */
static PyMethodDef methods[] = {
    {"dumps", n_dumps, METH_O, "json.dumps(obj, separators=(',', ':')); raises Fallback where json must decide"},
    {"loads", n_loads, METH_O, "json.loads(bytes | str); raises Fallback where json must decide"},
    {"trim_road", n_trim_road, METH_VARARGS, "fastjson.trim_road with its patterns: the same bytes"},
    {"slim_road_line", n_slim_road_line, METH_O, "the recorder's reduced road message, written"},
    {"set_json_dumps", n_set_json_dumps, METH_O, "json.dumps with the stack's separators, for what the writer leaves to json"},
    {"round", n_round, METH_VARARGS, "builtins.round"},
    {"round_list", n_round_list, METH_VARARGS, "[v if v is None else round(v, n) for v in values]"},
    {"interp_all", n_interp_all, METH_VARARGS, "trackers.view._interp_all for lists of floats"},
    {"view_prep", n_view_prep, METH_VARARGS, "a drawn line's points, reach and jump (ViewBuilder._lines)"},
    {"view_blend", n_view_blend, METH_VARARGS, "a drawn line's two smoothing stages (ViewBuilder._lines)"},
    {"ekf_predict", n_ekf_predict, METH_VARARGS, "EgoModel.predict into given arrays"},
    {"ekf_project", n_ekf_project, METH_VARARGS, "ph = P[:, idx] @ coef into ph; returns coef @ ph[idx]"},
    {"ekf_update", n_ekf_update, METH_VARARGS, "one scalar row applied to x and P in place"},
    {"ekf_sym", n_ekf_sym, METH_O, "P = 0.5 * (P + P.T) in place"},
    {"ekf_rts", n_ekf_rts, METH_VARARGS, "one smoother step back over a prediction"},
    {"lanes_measure", n_lanes_measure, METH_VARARGS, "the lane tracker's measurement arrays from a road message's lines"},
    {"lanes_predict", n_lanes_predict, METH_VARARGS, "the lane tracker's process noise, in place"},
    {"lanes_fit", n_lanes_fit, METH_VARARGS, "measurement noise, gate, innovation and fit of every line"},
    {"lanes_apply", n_lanes_apply, METH_VARARGS, "the scalar-gain update of the fitted lines, in place"},
    {"round_lists", n_round_lists, METH_VARARGS, "np.round(a, decimals).tolist(), optionally of np.sqrt(a)"},
    {"geom_centre", n_geom_centre, METH_VARARGS, "the road geometry's centre line lists (RoadGeometryFusion._centre)"},
    {"geom_reach", n_geom_reach, METH_VARARGS, "how far the camera's centre line reaches (RoadGeometryFusion._reach)"},
    {"radar_evolve", n_radar_evolve, METH_VARARGS, "Tracker._evolve (+ combine_b, + the free estimate's prediction) for B objects"},
    {"radar_combine", n_radar_combine, METH_VARARGS, "combine_b for B objects into given arrays"},
    {"radar_p_still", n_radar_p_still, METH_VARARGS, "Tracker.p_still_many: a list of floats"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "_chris_native",
                                       "chris-integrator's native core (see chris_integrator/native)", -1, methods,
                                       NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__chris_native(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;
    Fallback = PyErr_NewException("chris_integrator.native.Fallback", NULL, NULL);
    PyObject *bi = PyImport_ImportModule("builtins");
    if (Fallback == NULL || bi == NULL)
        goto fail;
    builtin_round = PyObject_GetAttrString(bi, "round");
    Py_DECREF(bi);
    if (builtin_round == NULL)
        goto fail;
    Py_INCREF(Fallback);
    if (PyModule_AddObject(m, "Fallback", Fallback) < 0 || PyModule_AddIntConstant(m, "VERSION", NATIVE_VERSION) < 0)
        goto fail;
#ifdef __FP_FAST_FMA
    /* informational: the target has a fused multiply-add; the build must not contract (see the Makefile) */
    PyModule_AddIntConstant(m, "TARGET_HAS_FMA", 1);
#else
    PyModule_AddIntConstant(m, "TARGET_HAS_FMA", 0);
#endif
    return m;
fail:
    Py_DECREF(m);
    return NULL;
}
