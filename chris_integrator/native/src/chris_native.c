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
 *   radar_*                    the radar (radar/): its motion model, the sensors' measurement functions, a scan's
 *                              association and update, the objects' upkeep and the message's objects; held to
 *                              what its Python reference gave before it was removed (tests/data/radar_golden.npz)
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

#define NATIVE_VERSION 10

static PyObject *Fallback;          /* raised where CPython's json must decide */
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
static int predict_check(const double *c, Py_ssize_t npar, Py_ssize_t n, int has_rho);
static int predict_core(const double *xi, const double *Pi, Py_ssize_t n, double dt, const double *c, double *rr,
                        double *xn, double *Pn, double *Fc);

static PyObject *n_ekf_predict(PyObject *self, PyObject *args)
{
    PyObject *ox, *oP, *opar, *orr, *oxo, *oPo, *oF;
    double dt;
    View x, P, par, rr, xo, Po, F;
    x.held = P.held = par.held = rr.held = xo.held = Po.held = F.held = 0;
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
    if (predict_check(DATA(par), npar, n, has_rho) < 0 ||
        predict_core(DATA(x), DATA(P), n, dt, DATA(par), has_rho ? DATA(rr) : NULL, DATA(xo), DATA(Po), DATA(F)) < 0)
        goto done;
    res = Py_None;
    Py_INCREF(res);
done:
    view_drop(&x); view_drop(&P); view_drop(&par); view_drop(&rr);
    view_drop(&xo); view_drop(&Po); view_drop(&F);
    return res;
}

/* the parameter block fits a state of n (and has Gauss-Markov states exactly when a decay array is given) */
static int predict_check(const double *c, Py_ssize_t npar, Py_ssize_t n, int has_rho)
{
    Py_ssize_t nclamp = npar >= PAR0 ? (Py_ssize_t)c[11] : 0, ngm = npar >= PAR0 ? (Py_ssize_t)c[12] : 0;
    if (npar < PAR0 || nclamp < 0 || ngm < 0 || npar != PAR0 + (n - NCORE) + 3 * nclamp + 3 * ngm ||
        has_rho != (ngm > 0)) {
        PyErr_SetString(PyExc_ValueError, "ekf_predict: parameter block");
        return -1;
    }
    return 0;
}

/* the process model on raw arrays (rr: the decay of the Gauss-Markov states, or NULL without them) */
static int predict_core(const double *xi, const double *Pi, Py_ssize_t n, double dt, const double *c, double *rr,
                        double *xn, double *Pn, double *Fc)
{
    Py_ssize_t nclamp = (Py_ssize_t)c[11], ngm = (Py_ssize_t)c[12];
    const double *psd = c + PAR0, *clamp = c + PAR0 + (n - NCORE), *gm = clamp + 3 * nclamp;
    int has_rho = (rr != NULL);
    double *gq = NULL;
    if (has_rho) {
        /* Gauss-Markov states: decay r = exp(-dt / tau), added variance q = sigma^2 (1 - exp(-2 dt / tau)) */
        double *r = rr;
        gq = (double *)PyMem_Calloc((size_t)n, sizeof(double));
        if (gq == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        for (Py_ssize_t i = 0; i < n; i++)
            r[i] = 1.0;
        for (Py_ssize_t q = 0; q < ngm; q++) {
            Py_ssize_t i = (Py_ssize_t)gm[3 * q];
            double tau = gm[3 * q + 1], sig = gm[3 * q + 2];
            if (i < 0 || i >= n) {
                PyErr_SetString(PyExc_ValueError, "ekf_predict: Gauss-Markov index");
                PyMem_Free(gq);
                return -1;
            }
            r[i] = exp(-dt / tau);
            gq[i] = sig * sig * (1.0 - exp(-2.0 * dt / tau));
        }
    }

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
        const double *r = rr;
        for (Py_ssize_t i = 0; i < n; i++)
            xn[i] *= r[i];
    }
    fc_rows(Fc, Pi, Pn, n);
    fc_cols(Fc, Pn, n);
#define PP(i, j) Pn[(Py_ssize_t)(i) * n + (j)]
    if (has_rho) {
        const double *r = rr;
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
    PyMem_Free(gq);
    return 0;
}

/* ph = P[:, idx] @ coef (into o); returns s0 = coef @ ph[idx]. Indices checked by the caller, m >= 1 */
static double project_core(const double *p, Py_ssize_t n, const int64_t *ix, const double *cf, Py_ssize_t m, double *o)
{
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
    return s0;
}

/* the row's correction of x and P in place (the doc of ekf_update); k: work of n. Freeze indices checked by
 * the caller */
static void update_core(double *xx, double *p, const double *h, Py_ssize_t n, double s, double nu, const int64_t *f,
                        Py_ssize_t nf, int frozen, double *k)
{
    for (Py_ssize_t i = 0; i < n; i++)
        k[i] = h[i] / s;
    if (frozen)
        for (Py_ssize_t j = 0; j < nf; j++)
            k[f[j]] = 0.0;
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
}

static void sym_core(double *p, Py_ssize_t n)
{
    for (Py_ssize_t i = 0; i < n; i++)
        for (Py_ssize_t j = i; j < n; j++) {
            double a = p[i * n + j], b = p[j * n + i];
            p[i * n + j] = 0.5 * (a + b);
            p[j * n + i] = 0.5 * (b + a);
        }
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
    res = PyFloat_FromDouble(project_core(DATA(P), n, ix, DATA(coef), m, DATA(ph)));
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
    const int64_t *f = frozen ? (const int64_t *)fz.b.buf : NULL;
    Py_ssize_t nf = frozen ? fz.b.shape[0] : 0;
    for (Py_ssize_t j = 0; j < nf; j++) {
        if (f[j] < 0 || f[j] >= n) {
            PyErr_SetString(PyExc_ValueError, "ekf_update: freeze index");
            goto done;
        }
    }
    update_core(DATA(x), DATA(P), DATA(ph), n, s, nu, f, nf, frozen, k);
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
    sym_core(DATA(P), n);
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
 * The estimator's filter pass and its monitors (core/ekf.py `Filter._run`, core/monitors.py `NisMonitor`,
 * `SelfCheck`, core/fdi.py `Trend`, `Stream`): the loop over the queued entries and their rows, with the first-pass
 * monitoring of every row, in one call. Python is called for what is not arithmetic: a row's model h(x), a
 * generic model's prediction, a row's own callback, and the integrator's hooks when a monitor's verdict
 * holds or changes.
 * ==================================================================================================== */
static PyObject *np_empty, *np_contig, *np_intp, *np_double, *stat_median;

static int np_ready(void)
{
    if (np_empty != NULL)
        return 0;
    PyObject *np = PyImport_ImportModule("numpy");
    if (np == NULL)
        return -1;
    PyObject *st = PyImport_ImportModule("statistics");
    if (st == NULL) {
        Py_DECREF(np);
        return -1;
    }
    np_contig = PyObject_GetAttrString(np, "ascontiguousarray");
    np_intp = PyObject_GetAttrString(np, "intp");
    np_double = PyObject_GetAttrString(np, "float64");
    stat_median = PyObject_GetAttrString(st, "median");
    np_empty = PyObject_GetAttrString(np, "empty");
    Py_DECREF(np);
    Py_DECREF(st);
    if (np_contig == NULL || np_intp == NULL || np_double == NULL || stat_median == NULL || np_empty == NULL) {
        Py_CLEAR(np_contig); Py_CLEAR(np_intp); Py_CLEAR(np_double); Py_CLEAR(stat_median); Py_CLEAR(np_empty);
        return -1;
    }
    return 0;
}

/* a new float64 array of shape (a,) or (a, b) */
static PyObject *np_new(Py_ssize_t a, Py_ssize_t b)
{
    PyObject *shape = b < 0 ? Py_BuildValue("(n)", a) : Py_BuildValue("(nn)", a, b);
    if (shape == NULL)
        return NULL;
    PyObject *r = PyObject_CallFunctionObjArgs(np_empty, shape, NULL);
    Py_DECREF(shape);
    return r;
}

/* a view of o as a C-contiguous array of `kind` ('d' float64, 'i' intp); what the buffer protocol does not give
 * as such (a list, a view, another dtype) is made one first, as np.ascontiguousarray(o, dtype=...) does (*conv
 * holds the made array) */
static int view_conv(PyObject *o, View *v, char kind, int ndim, const char *what, PyObject **conv)
{
    *conv = NULL;
    if (view_get(o, v, 0, kind, ndim, what) == 0)
        return 0;
    if (!PyErr_ExceptionMatches(PyExc_TypeError) && !PyErr_ExceptionMatches(PyExc_ValueError) &&
        !PyErr_ExceptionMatches(PyExc_BufferError))
        return -1;
    PyErr_Clear();
    PyObject *kw = Py_BuildValue("{sO}", "dtype", kind == 'd' ? np_double : np_intp), *a = PyTuple_Pack(1, o);
    if (kw == NULL || a == NULL) {
        Py_XDECREF(kw);
        Py_XDECREF(a);
        return -1;
    }
    *conv = PyObject_Call(np_contig, a, kw);
    Py_DECREF(kw);
    Py_DECREF(a);
    if (*conv == NULL)
        return -1;
    if (view_get(*conv, v, 0, kind, ndim, what) < 0) {
        Py_CLEAR(*conv);
        return -1;
    }
    return 0;
}

/* ---- the innovation window of one row class (monitors.NisMonitor's arithmetic) ---------------------------
 * The window holds (t, d2) over `window_s`, at most `max_n` of them; the median of its d2 is kept from an
 * ordered copy maintained as samples come and go (bisect's insort and the leftmost equal value removed, as the
 * Python had it); a window that holds or held a NaN is sorted afresh, and statistics.median decides while a
 * NaN is in it. The verdict is evaluated at most five times a second. */
typedef struct {
    PyObject_HEAD
    double window_s, min_n, fire, recover, med;
    Py_ssize_t maxn;
    double *qt, *qd, *srt;          /* the window (a ring, oldest at head), the ordered values */
    Py_ssize_t head, len, ns;
    int plain, high, has_value;
    double value, t_eval, peak;
    long long n_total;
} NisWin;

static PyObject *NisWinType;

static PyObject *niswin_new(PyTypeObject *tp, PyObject *args, PyObject *kw)
{
    allocfunc alloc = (allocfunc)PyType_GetSlot(tp, Py_tp_alloc);
    NisWin *m = (NisWin *)alloc(tp, 0);
    if (m == NULL)
        return NULL;
    m->plain = 1;
    m->t_eval = -1e18;
    return (PyObject *)m;
}

static void niswin_free(NisWin *m)
{
    PyMem_Free(m->qt);
    PyMem_Free(m->qd);
    PyMem_Free(m->srt);
    m->qt = m->qd = m->srt = NULL;
}

/* NisWindow(max_n, window_s, min_n, fire, recover, med): med is the median of the statistic when the stated
 * uncertainty is right (chi-square, 1 dof) */
static int niswin_init(PyObject *self, PyObject *args, PyObject *kw)
{
    NisWin *m = (NisWin *)self;
    Py_ssize_t maxn;
    double window_s, min_n, fire, recover, med;
    if (!PyArg_ParseTuple(args, "nddddd", &maxn, &window_s, &min_n, &fire, &recover, &med) || np_ready() < 0)
        return -1;
    if (maxn < 1) {
        PyErr_SetString(PyExc_ValueError, "NisWindow: max_n must be at least 1");
        return -1;
    }
    niswin_free(m);
    m->qt = (double *)PyMem_Malloc((size_t)maxn * sizeof(double));
    m->qd = (double *)PyMem_Malloc((size_t)maxn * sizeof(double));
    m->srt = (double *)PyMem_Malloc((size_t)(maxn + 1) * sizeof(double));
    if (m->qt == NULL || m->qd == NULL || m->srt == NULL) {
        niswin_free(m);
        PyErr_NoMemory();
        return -1;
    }
    m->maxn = maxn;
    m->window_s = window_s, m->min_n = min_n, m->fire = fire, m->recover = recover, m->med = med;
    m->head = m->len = m->ns = 0;
    m->plain = 1, m->high = 0, m->has_value = 0;
    m->value = 0.0, m->t_eval = -1e18, m->peak = 0.0;
    m->n_total = 0;
    return 0;
}

static void niswin_dealloc(PyObject *self)
{
    PyTypeObject *tp = Py_TYPE(self);
    niswin_free((NisWin *)self);
    freefunc f = (freefunc)PyType_GetSlot(tp, Py_tp_free);
    f(self);
    Py_DECREF(tp);
}

static void nis_insort(NisWin *m, double d)
{
    Py_ssize_t lo = 0, hi = m->ns;                  /* bisect.insort: after the equal values */
    while (lo < hi) {
        Py_ssize_t mid = (lo + hi) / 2;
        if (d < m->srt[mid])
            hi = mid;
        else
            lo = mid + 1;
    }
    memmove(m->srt + lo + 1, m->srt + lo, (size_t)(m->ns - lo) * sizeof(double));
    m->srt[lo] = d;
    m->ns++;
}

static void nis_drop(NisWin *m, double d)
{
    Py_ssize_t lo = 0, hi = m->ns;                  /* bisect_left: the leftmost equal value */
    while (lo < hi) {
        Py_ssize_t mid = (lo + hi) / 2;
        if (m->srt[mid] < d)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < m->ns && m->srt[lo] == d) {
        memmove(m->srt + lo, m->srt + lo + 1, (size_t)(m->ns - lo - 1) * sizeof(double));
        m->ns--;
    } else                                          /* not there: the copy is not in order (a NaN went through) */
        m->plain = 0;
}

static int nis_median(NisWin *m, double *out)
{
    Py_ssize_t n = m->len;
    if (!m->plain || m->ns != n) {
        int nan = 0;
        for (Py_ssize_t i = 0; i < n; i++)
            if (m->qd[(m->head + i) % m->maxn] != m->qd[(m->head + i) % m->maxn])
                nan = 1;
        if (nan || n == 0) {                        /* statistics.median decides (and says what an empty one is) */
            PyObject *l = PyList_New(n);
            if (l == NULL)
                return -1;
            for (Py_ssize_t i = 0; i < n; i++) {
                PyObject *f = PyFloat_FromDouble(m->qd[(m->head + i) % m->maxn]);
                if (f == NULL) {
                    Py_DECREF(l);
                    return -1;
                }
                PyList_SetItem(l, i, f);
            }
            PyObject *r = PyObject_CallFunctionObjArgs(stat_median, l, NULL);
            Py_DECREF(l);
            if (r == NULL)
                return -1;
            *out = PyFloat_AsDouble(r);
            Py_DECREF(r);
            return (*out == -1.0 && PyErr_Occurred()) ? -1 : 0;
        }
        /* the value that did not order has left the window: sorted afresh (stable, as sorted() is) */
        for (Py_ssize_t i = 0; i < n; i++) {
            double v = m->qd[(m->head + i) % m->maxn];
            Py_ssize_t j = i;
            while (j > 0 && v < m->srt[j - 1]) {
                m->srt[j] = m->srt[j - 1];
                j--;
            }
            m->srt[j] = v;
        }
        m->ns = n;
        m->plain = 1;
    }
    const double *s = m->srt;
    *out = (n % 2) ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
    return 0;
}

/* one sample; *changed: the verdict changed */
static int nis_offer(NisWin *m, double t, double d2, int *changed)
{
    *changed = 0;
    if (m->plain) {
        if (m->len == m->maxn)                      /* the oldest sample leaves with this one */
            nis_drop(m, m->qd[m->head]);
        if (d2 == d2)
            nis_insort(m, d2);
        else
            m->plain = 0;
    }
    if (m->len == m->maxn) {
        m->head = (m->head + 1) % m->maxn;
        m->len--;
    }
    Py_ssize_t k = (m->head + m->len) % m->maxn;
    m->qt[k] = t;
    m->qd[k] = d2;
    m->len++;
    while (m->len && m->qt[m->head] < t - m->window_s) {
        if (m->plain)
            nis_drop(m, m->qd[m->head]);
        m->head = (m->head + 1) % m->maxn;
        m->len--;
    }
    m->n_total++;
    if ((double)m->len >= m->min_n && (t - m->t_eval >= 0.2 || t < m->t_eval)) {     /* the median: 5 Hz */
        double med;
        m->t_eval = t;
        if (nis_median(m, &med) < 0)
            return -1;
        m->value = med / m->med;
        m->has_value = 1;
        double r = m->value / m->fire;
        if (r > m->peak)
            m->peak = r;
        int old = m->high;
        m->high = m->value > (old ? m->recover : m->fire);
        *changed = m->high != old;
    }
    return 0;
}

static PyObject *niswin_offer(PyObject *self, PyObject *args)
{
    double t, d2;
    int ch;
    if (!PyArg_ParseTuple(args, "dd", &t, &d2))
        return NULL;
    if (((NisWin *)self)->qt == NULL) {
        PyErr_SetString(PyExc_ValueError, "NisWindow: not initialised");
        return NULL;
    }
    if (nis_offer((NisWin *)self, t, d2, &ch) < 0)
        return NULL;
    return PyBool_FromLong(ch);
}

static PyObject *niswin_reset(PyObject *self, PyObject *unused)
{
    NisWin *m = (NisWin *)self;
    m->head = m->len = m->ns = 0;
    m->plain = 1, m->has_value = 0, m->high = 0;
    Py_RETURN_NONE;
}

static PyObject *niswin_value(PyObject *self, void *c)
{
    NisWin *m = (NisWin *)self;
    if (!m->has_value)
        Py_RETURN_NONE;
    return PyFloat_FromDouble(m->value);
}
static PyObject *niswin_high(PyObject *self, void *c) { return PyBool_FromLong(((NisWin *)self)->high); }
static PyObject *niswin_peak(PyObject *self, void *c) { return PyFloat_FromDouble(((NisWin *)self)->peak); }
static PyObject *niswin_n(PyObject *self, void *c) { return PyLong_FromSsize_t(((NisWin *)self)->len); }
static PyObject *niswin_ntotal(PyObject *self, void *c) { return PyLong_FromLongLong(((NisWin *)self)->n_total); }

static PyMethodDef niswin_methods[] = {
    {"offer", niswin_offer, METH_VARARGS, "offer(t, d2) -> True when the verdict changed"},
    {"reset_data", niswin_reset, METH_NOARGS, "forget the window and the verdict (the peak stays)"},
    {NULL, NULL, 0, NULL}};
static PyGetSetDef niswin_getset[] = {
    {"value", niswin_value, NULL, "the window's median over its expected value (None before the first evaluation)", NULL},
    {"high", niswin_high, NULL, "the verdict: innovations beyond the stated uncertainty", NULL},
    {"peak", niswin_peak, NULL, "the largest value over the firing level", NULL},
    {"n", niswin_n, NULL, "samples in the window", NULL},
    {"n_total", niswin_ntotal, NULL, "samples ever offered", NULL},
    {NULL, NULL, NULL, NULL, NULL}};
static PyType_Slot niswin_slots[] = {
    {Py_tp_new, niswin_new}, {Py_tp_init, niswin_init}, {Py_tp_dealloc, niswin_dealloc},
    {Py_tp_methods, niswin_methods}, {Py_tp_getset, niswin_getset},
    {Py_tp_doc, "The innovation window of one row class (core/monitors.py NisMonitor subclasses it)"}, {0, NULL}};
static PyType_Spec niswin_spec = {"chris_integrator.native._chris_native.NisWindow", sizeof(NisWin), 0,
                                  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, niswin_slots};

/* ---- the trend of one row class's signed innovations (fdi.Trend) --------------------------------------- */
typedef struct {
    PyObject_HEAD
    double tau, fire, clear, min_n, corr;
    double m, w, t, score;
    int has_t, high;
    long long n;
} Trend;

static PyObject *TrendType;

static PyObject *trend_new(PyTypeObject *tp, PyObject *args, PyObject *kw)
{
    double tau, fire, clear, min_n, corr;
    if (!PyArg_ParseTuple(args, "ddddd", &tau, &fire, &clear, &min_n, &corr))
        return NULL;
    allocfunc alloc = (allocfunc)PyType_GetSlot(tp, Py_tp_alloc);
    Trend *tr = (Trend *)alloc(tp, 0);
    if (tr == NULL)
        return NULL;
    tr->tau = tau, tr->fire = fire, tr->clear = clear, tr->min_n = min_n, tr->corr = corr;
    return (PyObject *)tr;
}

static void trend_dealloc(PyObject *self)
{
    PyTypeObject *tp = Py_TYPE(self);
    freefunc f = (freefunc)PyType_GetSlot(tp, Py_tp_free);
    f(self);
    Py_DECREF(tp);
}

/* the exponential mean of the clipped z, its score, the verdict with hysteresis once min_n samples are in */
static int trend_offer(Trend *tr, double t, double z)
{
    double dt = 0.0;
    if (tr->has_t) {
        dt = t - tr->t;
        if (0.0 > dt)                               /* max(t - self.t, 0.0) */
            dt = 0.0;
    }
    tr->t = t;
    tr->has_t = 1;
    double a = exp(-dt / tr->tau);
    double zc = (z < 4.0) ? z : 4.0;                /* max(-4.0, min(4.0, z)): a wild sample is not a trend */
    zc = (zc > -4.0) ? zc : -4.0;
    tr->m = a * tr->m + zc;
    tr->w = a * tr->w + 1.0;
    tr->n++;
    double q = tr->w * tr->corr;
    if (1.0 > q)                                    /* max(w * corr, 1.0) */
        q = 1.0;
    tr->score = tr->m / sqrt(q);
    if ((double)tr->n >= tr->min_n)
        tr->high = fabs(tr->score) > (tr->high ? tr->clear : tr->fire);
    return tr->high;
}

static PyObject *trend_offer_py(PyObject *self, PyObject *args)
{
    double t, z;
    if (!PyArg_ParseTuple(args, "dd", &t, &z))
        return NULL;
    return PyBool_FromLong(trend_offer((Trend *)self, t, z));
}

static PyObject *trend_clear(PyObject *self, PyObject *unused)
{
    Trend *tr = (Trend *)self;
    tr->m = tr->w = 0.0;
    tr->has_t = 0;
    tr->n = 0;
    tr->high = 0;
    Py_RETURN_NONE;
}

static PyObject *trend_get_t(PyObject *self, void *c)
{
    Trend *tr = (Trend *)self;
    if (!tr->has_t)
        Py_RETURN_NONE;
    return PyFloat_FromDouble(tr->t);
}
static PyObject *trend_get_m(PyObject *self, void *c) { return PyFloat_FromDouble(((Trend *)self)->m); }
static PyObject *trend_get_w(PyObject *self, void *c) { return PyFloat_FromDouble(((Trend *)self)->w); }
static PyObject *trend_get_n(PyObject *self, void *c) { return PyLong_FromLongLong(((Trend *)self)->n); }
static PyObject *trend_get_high(PyObject *self, void *c) { return PyBool_FromLong(((Trend *)self)->high); }
static PyObject *trend_get_score(PyObject *self, void *c) { return PyFloat_FromDouble(((Trend *)self)->score); }

static PyMethodDef trend_methods[] = {
    {"offer", trend_offer_py, METH_VARARGS, "offer(t, z) -> the verdict"},
    {"clear", trend_clear, METH_NOARGS, "forget (a new epoch)"},
    {NULL, NULL, 0, NULL}};
static PyGetSetDef trend_getset[] = {
    {"t", trend_get_t, NULL, "time of the last sample (None before the first)", NULL},
    {"m", trend_get_m, NULL, "exponential sum of the clipped z", NULL},
    {"w", trend_get_w, NULL, "exponential sum of the weights", NULL},
    {"n", trend_get_n, NULL, "samples", NULL},
    {"high", trend_get_high, NULL, "the verdict", NULL},
    {"score", trend_get_score, NULL, "m / sqrt(max(w corr, 1))", NULL},
    {NULL, NULL, NULL, NULL, NULL}};
static PyType_Slot trend_slots[] = {
    {Py_tp_new, trend_new}, {Py_tp_dealloc, trend_dealloc}, {Py_tp_methods, trend_methods},
    {Py_tp_getset, trend_getset},
    {Py_tp_doc, "Trend(tau_s, fire, clear, min_n, corr): a signed, normalised innovation averaged over time"},
    {0, NULL}};
static PyType_Spec trend_spec = {"chris_integrator.native._chris_native.Trend", sizeof(Trend), 0, Py_TPFLAGS_DEFAULT,
                                 trend_slots};

/* ---- a source's stated uncertainty over its characterised one (monitors.SelfCheck's arithmetic) -------------
 * The window runs on evidence time (the clock advances by the time between samples, capped at max_step); it
 * keeps recover_window_s of it. The verdict uses the median of the last fire_window_s and of the whole window.
 * Both medians come from ordered copies kept as samples come and go (a new sample after its equals, the oldest
 * equal removed: the order sorted() gives); while a NaN is in the window statistics.median decides. */
typedef struct {
    PyObject_HEAD
    double fire_w, min_n, fire_ratio, rec_w, rec_min_n, rec_ratio, max_step;
    double *ev, *it;                /* the window, a ring of capacity cap (oldest at head) */
    Py_ssize_t head, len, cap, nfire; /* nfire: the newest nfire samples are within fire_window_s */
    double *sa, *sf;                /* ordered copies: the whole window, the fire part */
    Py_ssize_t nsa, nsf, nnan;
    double evnow, last_t, median, changed_t, peak;
    int has_last, fault, has_median, has_changed;
    Py_ssize_t n;
} SelfWin;

static PyObject *SelfWinType;

static PyObject *selfwin_new(PyTypeObject *tp, PyObject *args, PyObject *kw)
{
    allocfunc alloc = (allocfunc)PyType_GetSlot(tp, Py_tp_alloc);
    return alloc(tp, 0);
}

static void selfwin_free(SelfWin *m)
{
    PyMem_Free(m->ev); PyMem_Free(m->it); PyMem_Free(m->sa); PyMem_Free(m->sf);
    m->ev = m->it = m->sa = m->sf = NULL;
    m->cap = 0;
}

/* SelfWindow(fire_window_s, min_n, fire_ratio, recover_window_s, recover_min_n, recover_ratio, max_step_s) */
static int selfwin_init(PyObject *self, PyObject *args, PyObject *kw)
{
    SelfWin *m = (SelfWin *)self;
    double a[7];
    if (!PyArg_ParseTuple(args, "ddddddd", &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6]) || np_ready() < 0)
        return -1;
    selfwin_free(m);
    m->fire_w = a[0], m->min_n = a[1], m->fire_ratio = a[2], m->rec_w = a[3], m->rec_min_n = a[4];
    m->rec_ratio = a[5], m->max_step = a[6];
    m->head = m->len = m->nfire = m->nsa = m->nsf = m->nnan = m->n = 0;
    m->evnow = m->last_t = m->median = m->changed_t = m->peak = 0.0;
    m->has_last = m->fault = m->has_median = m->has_changed = 0;
    return 0;
}

static void selfwin_dealloc(PyObject *self)
{
    PyTypeObject *tp = Py_TYPE(self);
    selfwin_free((SelfWin *)self);
    freefunc f = (freefunc)PyType_GetSlot(tp, Py_tp_free);
    f(self);
    Py_DECREF(tp);
}

static void sorted_insert(double *s, Py_ssize_t *ns, double d)
{
    Py_ssize_t lo = 0, hi = *ns;                    /* after the equal values */
    while (lo < hi) {
        Py_ssize_t mid = (lo + hi) / 2;
        if (d < s[mid])
            hi = mid;
        else
            lo = mid + 1;
    }
    memmove(s + lo + 1, s + lo, (size_t)(*ns - lo) * sizeof(double));
    s[lo] = d;
    (*ns)++;
}

static void sorted_remove(double *s, Py_ssize_t *ns, double d)
{
    Py_ssize_t lo = 0, hi = *ns;                    /* the leftmost equal value (the oldest of them) */
    while (lo < hi) {
        Py_ssize_t mid = (lo + hi) / 2;
        if (s[mid] < d)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < *ns && s[lo] == d) {
        memmove(s + lo, s + lo + 1, (size_t)(*ns - lo - 1) * sizeof(double));
        (*ns)--;
    }
}

static double sorted_median(const double *s, Py_ssize_t n)
{
    return (n % 2) ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

#define SW_AT(m, i) (((m)->head + (i)) % (m)->cap)

/* statistics.median of the window's newest k samples (a NaN is among them) */
static int selfwin_pymedian(SelfWin *m, Py_ssize_t k, double *out)
{
    PyObject *l = PyList_New(k);
    if (l == NULL)
        return -1;
    for (Py_ssize_t i = 0; i < k; i++) {
        PyObject *f = PyFloat_FromDouble(m->it[SW_AT(m, m->len - k + i)]);
        if (f == NULL) {
            Py_DECREF(l);
            return -1;
        }
        PyList_SetItem(l, i, f);
    }
    PyObject *r = PyObject_CallFunctionObjArgs(stat_median, l, NULL);
    Py_DECREF(l);
    if (r == NULL)
        return -1;
    *out = PyFloat_AsDouble(r);
    Py_DECREF(r);
    return (*out == -1.0 && PyErr_Occurred()) ? -1 : 0;
}

static int selfwin_grow(SelfWin *m)
{
    Py_ssize_t cap = m->cap ? 2 * m->cap : 64;
    double *ev = PyMem_Malloc((size_t)cap * sizeof(double)), *it = PyMem_Malloc((size_t)cap * sizeof(double));
    double *sa = PyMem_Malloc((size_t)cap * sizeof(double)), *sf = PyMem_Malloc((size_t)cap * sizeof(double));
    if (ev == NULL || it == NULL || sa == NULL || sf == NULL) {
        PyMem_Free(ev); PyMem_Free(it); PyMem_Free(sa); PyMem_Free(sf);
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t i = 0; i < m->len; i++) {
        ev[i] = m->ev[SW_AT(m, i)];
        it[i] = m->it[SW_AT(m, i)];
    }
    if (m->nsa)
        memcpy(sa, m->sa, (size_t)m->nsa * sizeof(double));
    if (m->nsf)
        memcpy(sf, m->sf, (size_t)m->nsf * sizeof(double));
    PyMem_Free(m->ev); PyMem_Free(m->it); PyMem_Free(m->sa); PyMem_Free(m->sf);
    m->ev = ev, m->it = it, m->sa = sa, m->sf = sf;
    m->head = 0;
    m->cap = cap;
    return 0;
}

/* one sample; returns 1 when the verdict changed, 0 when not, -1 on an error */
static int selfwin_offer_c(SelfWin *m, double t, double r)
{
    if (m->has_last) {
        double d = t - m->last_t;
        if (0.0 > d)                                /* min(max(t - last_t, 0.0), max_step) */
            d = 0.0;
        if (m->max_step < d)
            d = m->max_step;
        m->evnow += d;
    }
    m->last_t = t;
    m->has_last = 1;
    if (m->len == m->cap && selfwin_grow(m) < 0)
        return -1;
    m->ev[SW_AT(m, m->len)] = m->evnow;
    m->it[SW_AT(m, m->len)] = r;
    m->len++;
    m->nfire++;
    if (r == r) {
        sorted_insert(m->sa, &m->nsa, r);
        sorted_insert(m->sf, &m->nsf, r);
    } else
        m->nnan++;
    while (m->len && m->ev[m->head] < m->evnow - m->rec_w) {      /* the window keeps recover_window_s */
        double v = m->it[m->head];
        if (v == v) {
            sorted_remove(m->sa, &m->nsa, v);
            if (m->nfire == m->len)
                sorted_remove(m->sf, &m->nsf, v);
        } else
            m->nnan--;
        if (m->nfire == m->len)
            m->nfire--;
        m->head = (m->head + 1) % m->cap;
        m->len--;
    }
    while (m->nfire && m->ev[SW_AT(m, m->len - m->nfire)] < m->evnow - m->fire_w) {
        double v = m->it[SW_AT(m, m->len - m->nfire)];
        if (v == v)
            sorted_remove(m->sf, &m->nsf, v);
        m->nfire--;
    }
    int nan_fire = 0;
    if (m->nnan)
        for (Py_ssize_t i = m->len - m->nfire; i < m->len; i++)
            if (m->it[SW_AT(m, i)] != m->it[SW_AT(m, i)])
                nan_fire = 1;
    m->n = m->nfire;
    if (m->n) {
        if (nan_fire) {
            if (selfwin_pymedian(m, m->n, &m->median) < 0)
                return -1;
        } else
            m->median = sorted_median(m->sf, m->nsf);
        m->has_median = 1;
    } else
        m->has_median = 0;
    if ((double)m->n >= m->min_n) {
        double q = m->median / m->fire_ratio;       /* max(self.peak, median / fire_ratio) */
        if (q > m->peak)
            m->peak = q;
    }
    int old = m->fault;
    if (!old) {
        if ((double)m->n >= m->min_n && m->median > m->fire_ratio)
            m->fault = 1;
    } else {
        double span = m->len ? m->evnow - m->ev[m->head] : 0.0;
        if ((double)m->len >= m->rec_min_n && span >= 0.9 * m->rec_w) {
            double mr;
            if (m->nnan) {
                if (selfwin_pymedian(m, m->len, &mr) < 0)
                    return -1;
            } else
                mr = sorted_median(m->sa, m->nsa);
            if (mr < m->rec_ratio && m->median < m->rec_ratio)
                m->fault = 0;
        }
    }
    if (m->fault != old) {
        m->changed_t = t;
        m->has_changed = 1;
        return 1;
    }
    return 0;
}

static PyObject *selfwin_offer(PyObject *self, PyObject *args)
{
    double t, r;
    if (!PyArg_ParseTuple(args, "dd", &t, &r))
        return NULL;
    int ch = selfwin_offer_c((SelfWin *)self, t, r);
    return ch < 0 ? NULL : PyBool_FromLong(ch);
}

static PyObject *selfwin_reset(PyObject *self, PyObject *unused)
{
    SelfWin *m = (SelfWin *)self;                   /* the window forgotten; evidence time and the verdict stay */
    m->head = m->len = m->nfire = m->nsa = m->nsf = m->nnan = 0;
    m->has_last = 0;
    Py_RETURN_NONE;
}

static PyObject *selfwin_median(PyObject *self, void *c)
{
    SelfWin *m = (SelfWin *)self;
    if (!m->has_median)
        Py_RETURN_NONE;
    return PyFloat_FromDouble(m->median);
}
static PyObject *selfwin_changed(PyObject *self, void *c)
{
    SelfWin *m = (SelfWin *)self;
    if (!m->has_changed)
        Py_RETURN_NONE;
    return PyFloat_FromDouble(m->changed_t);
}
static PyObject *selfwin_fault(PyObject *self, void *c) { return PyBool_FromLong(((SelfWin *)self)->fault); }
static PyObject *selfwin_n(PyObject *self, void *c) { return PyLong_FromSsize_t(((SelfWin *)self)->n); }
static PyObject *selfwin_peak(PyObject *self, void *c) { return PyFloat_FromDouble(((SelfWin *)self)->peak); }
static PyObject *selfwin_span(PyObject *self, void *c)
{
    SelfWin *m = (SelfWin *)self;
    return PyFloat_FromDouble(m->len ? m->evnow - m->ev[m->head] : 0.0);
}

static PyMethodDef selfwin_methods[] = {
    {"offer", selfwin_offer, METH_VARARGS, "offer(t, ratio) -> True when the verdict changed"},
    {"reset_data", selfwin_reset, METH_NOARGS, "forget the window (evidence time and the verdict stay)"},
    {NULL, NULL, 0, NULL}};
static PyGetSetDef selfwin_getset[] = {
    {"fault", selfwin_fault, NULL, "the verdict", NULL},
    {"median", selfwin_median, NULL, "median ratio of the last fire_window_s (None before a sample)", NULL},
    {"n", selfwin_n, NULL, "samples in the last fire_window_s", NULL},
    {"peak", selfwin_peak, NULL, "the largest median over the firing level", NULL},
    {"changed_t", selfwin_changed, NULL, "when the verdict last changed (None: never)", NULL},
    {"span", selfwin_span, NULL, "evidence time the window covers", NULL},
    {NULL, NULL, NULL, NULL, NULL}};
static PyType_Slot selfwin_slots[] = {
    {Py_tp_new, selfwin_new}, {Py_tp_init, selfwin_init}, {Py_tp_dealloc, selfwin_dealloc},
    {Py_tp_methods, selfwin_methods}, {Py_tp_getset, selfwin_getset},
    {Py_tp_doc, "A source's stated over characterised uncertainty, windowed (core/monitors.py SelfCheck subclasses it)"},
    {0, NULL}};
static PyType_Spec selfwin_spec = {"chris_integrator.native._chris_native.SelfWindow", sizeof(SelfWin), 0,
                                   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, selfwin_slots};

/* ---- signal integrity of one raw stream (fdi.Stream's arithmetic) -----------------------------------------
 * offer(t, v, s) for a finite v: None (an old sample, or a stream not yet known to vary), 1 (the windowed
 * statistics are not due: what was held last time holds), or (due, stuck, repeat, noise): stuck None or
 * (fault, duration, samples); repeat None or (repeats, pairs) when their share reaches repeat_share; noise None
 * or (ratio, wild, n) when it was computed. The texts and the held findings are the caller's (fdi.Stream). */
typedef struct {
    PyObject_HEAD
    double stuck_s, stuck_n, stuck_fault_s, learn_n, learn_repeat, repeat_share, noise_lim, outlier_z;
    Py_ssize_t repeat_w, noise_w, maxn;
    double *qt, *qv, *qs, *work;
    Py_ssize_t head, len;
    double run_t, run_v, h_rep, noise_ratio;
    int has_run;
    long long run_n, k, h_n;
} StreamSt;

static PyObject *StreamType;

static PyObject *stream_new(PyTypeObject *tp, PyObject *args, PyObject *kw)
{
    allocfunc alloc = (allocfunc)PyType_GetSlot(tp, Py_tp_alloc);
    StreamSt *m = (StreamSt *)alloc(tp, 0);
    if (m != NULL)
        m->noise_ratio = 1.0;
    return (PyObject *)m;
}

static void stream_free(StreamSt *m)
{
    PyMem_Free(m->qt); PyMem_Free(m->qv); PyMem_Free(m->qs); PyMem_Free(m->work);
    m->qt = m->qv = m->qs = m->work = NULL;
}

/* StreamState(stuck_s, stuck_n, stuck_fault_s, learn_n, learn_repeat, repeat_window, repeat_share, noise_window,
 *             noise_ratio, outlier_z) */
static int stream_init(PyObject *self, PyObject *args, PyObject *kw)
{
    StreamSt *m = (StreamSt *)self;
    double a[10];
    if (!PyArg_ParseTuple(args, "dddddddddd", &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7], &a[8], &a[9]) ||
        np_ready() < 0)
        return -1;
    if (!(a[5] >= 2 && a[7] >= 3 && a[5] < 1e6 && a[7] < 1e6) || a[5] != (double)(Py_ssize_t)a[5] ||
        a[7] != (double)(Py_ssize_t)a[7]) {
        PyErr_SetString(PyExc_ValueError, "StreamState: the windows must be whole numbers (repeat >= 2, noise >= 3)");
        return -1;
    }
    stream_free(m);
    m->stuck_s = a[0], m->stuck_n = a[1], m->stuck_fault_s = a[2], m->learn_n = a[3], m->learn_repeat = a[4];
    m->repeat_w = (Py_ssize_t)a[5], m->repeat_share = a[6], m->noise_w = (Py_ssize_t)a[7], m->noise_lim = a[8];
    m->outlier_z = a[9];
    m->maxn = (m->noise_w > m->repeat_w ? m->noise_w : m->repeat_w) + 2;
    m->qt = PyMem_Malloc((size_t)m->maxn * sizeof(double));
    m->qv = PyMem_Malloc((size_t)m->maxn * sizeof(double));
    m->qs = PyMem_Malloc((size_t)m->maxn * sizeof(double));
    m->work = PyMem_Malloc((size_t)m->maxn * sizeof(double));
    if (m->qt == NULL || m->qv == NULL || m->qs == NULL || m->work == NULL) {
        stream_free(m);
        PyErr_NoMemory();
        return -1;
    }
    m->head = m->len = 0;
    m->has_run = 0, m->run_n = 0, m->k = 0, m->h_n = 0;
    m->h_rep = 0.0, m->noise_ratio = 1.0;
    return 0;
}

static void stream_dealloc(PyObject *self)
{
    PyTypeObject *tp = Py_TYPE(self);
    stream_free((StreamSt *)self);
    freefunc f = (freefunc)PyType_GetSlot(tp, Py_tp_free);
    f(self);
    Py_DECREF(tp);
}

/* statistics.median of n values (sorted here when none is NaN; statistics.median itself otherwise) */
static int median_of(double *a, Py_ssize_t n, double *out)
{
    int nan = 0;
    for (Py_ssize_t i = 0; i < n; i++)
        if (a[i] != a[i])
            nan = 1;
    if (nan || n == 0) {
        PyObject *l = PyList_New(n);
        if (l == NULL)
            return -1;
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *f = PyFloat_FromDouble(a[i]);
            if (f == NULL) {
                Py_DECREF(l);
                return -1;
            }
            PyList_SetItem(l, i, f);
        }
        PyObject *r = PyObject_CallFunctionObjArgs(stat_median, l, NULL);
        Py_DECREF(l);
        if (r == NULL)
            return -1;
        *out = PyFloat_AsDouble(r);
        Py_DECREF(r);
        return (*out == -1.0 && PyErr_Occurred()) ? -1 : 0;
    }
    for (Py_ssize_t i = 1; i < n; i++) {            /* stable: the order sorted() gives */
        double v = a[i];
        Py_ssize_t j = i;
        while (j > 0 && v < a[j - 1]) {
            a[j] = a[j - 1];
            j--;
        }
        a[j] = v;
    }
    *out = sorted_median(a, n);
    return 0;
}

#define ST_AT(m, i) (((m)->head + (i)) % (m)->maxn)

static PyObject *stream_offer(PyObject *self, PyObject *args)
{
    StreamSt *m = (StreamSt *)self;
    double t, v, s;
    if (!PyArg_ParseTuple(args, "ddd", &t, &v, &s))
        return NULL;
    if (m->qt == NULL) {
        PyErr_SetString(PyExc_ValueError, "StreamState: not initialised");
        return NULL;
    }
    if (m->len && t <= m->qt[ST_AT(m, m->len - 1)])
        Py_RETURN_NONE;
    if (m->has_run && v == m->run_v)
        m->run_n++;
    else
        m->run_v = v, m->run_t = t, m->run_n = 1, m->has_run = 1;
    if (m->len == m->maxn) {
        m->head = (m->head + 1) % m->maxn;
        m->len--;
    }
    m->qt[ST_AT(m, m->len)] = t, m->qv[ST_AT(m, m->len)] = v, m->qs[ST_AT(m, m->len)] = s;
    m->len++;
    /* what this stream normally does: the share of samples equal to their predecessor (while no run goes on) */
    if (v != 0.0 && m->run_n <= 3) {
        m->h_n++;
        m->h_rep += ((m->run_n > 1 ? 1.0 : 0.0) - m->h_rep) / (double)(m->h_n < 500 ? m->h_n : 500);
    }
    if (!((double)m->h_n >= m->learn_n && m->h_rep < m->learn_repeat))
        Py_RETURN_NONE;
    PyObject *stuck = Py_None, *rep = Py_None, *noise = Py_None, *res = NULL;
    Py_INCREF(Py_None); Py_INCREF(Py_None); Py_INCREF(Py_None);
    if (fabs(v) > 4.0 * s && 4.0 * s > 0.0 && (double)m->run_n >= m->stuck_n && t - m->run_t >= m->stuck_s) {
        Py_DECREF(stuck);
        stuck = Py_BuildValue("(OdL)", t - m->run_t >= m->stuck_fault_s ? Py_True : Py_False, t - m->run_t, m->run_n);
        if (stuck == NULL)
            goto fail;
    }
    Py_ssize_t n = m->len;
    m->k++;
    if (m->k % 4) {                                 /* the windowed statistics: every fourth sample */
        if (stuck == Py_None) {
            Py_DECREF(stuck); Py_DECREF(rep); Py_DECREF(noise);
            return PyLong_FromLong(1);
        }
        res = Py_BuildValue("(ONOO)", Py_False, stuck, rep, noise);
        Py_DECREF(rep);
        Py_DECREF(noise);
        return res;
    }
    if (n >= m->repeat_w) {
        Py_ssize_t kk = n - m->repeat_w, cnt = 0;
        for (Py_ssize_t i = kk + 1; i < n; i++) {
            double a = m->qv[ST_AT(m, i)], b = m->qv[ST_AT(m, i - 1)];
            if (a == b && a != 0.0)
                cnt++;
        }
        if ((double)cnt / (double)(n - kk - 1) >= m->repeat_share) {
            Py_DECREF(rep);
            if ((rep = Py_BuildValue("(nn)", cnt, n - kk - 1)) == NULL)
                goto fail;
        }
    }
    if (n >= m->noise_w) {
        for (Py_ssize_t i = 0; i < n; i++)
            m->work[i] = m->qs[ST_AT(m, i)];
        double sig;
        if (median_of(m->work, n, &sig) < 0)
            goto fail;
        if (sig > 0.0) {
            Py_ssize_t nd = 0;
            for (Py_ssize_t i = 2; i < n; i++) {
                double a = m->qv[ST_AT(m, i)], b = m->qv[ST_AT(m, i - 1)], c = m->qv[ST_AT(m, i - 2)];
                if (a != b)
                    m->work[nd++] = fabs(a - 2.0 * b + c) / 2.449;
            }
            if (nd >= 10) {
                Py_ssize_t wild = 0;
                for (Py_ssize_t i = 0; i < nd; i++)
                    if (m->work[i] > m->outlier_z * sig)
                        wild++;
                double med;
                if (median_of(m->work, nd, &med) < 0)
                    goto fail;
                double ratio = 1.4826 * med / sig;
                m->noise_ratio = ratio;
                Py_DECREF(noise);
                if ((noise = Py_BuildValue("(dnn)", ratio, wild, nd)) == NULL)
                    goto fail;
            }
        }
    }
    res = Py_BuildValue("(ONNN)", Py_True, stuck, rep, noise);
    return res;
fail:
    Py_XDECREF(stuck); Py_XDECREF(rep); Py_XDECREF(noise);
    return NULL;
}

static PyObject *stream_clear(PyObject *self, PyObject *unused)
{
    StreamSt *m = (StreamSt *)self;                 /* the window and the run forgotten; what was learned stays */
    m->head = m->len = 0;
    m->has_run = 0;
    m->run_n = 0;
    Py_RETURN_NONE;
}

static PyObject *stream_noise(PyObject *self, void *c) { return PyFloat_FromDouble(((StreamSt *)self)->noise_ratio); }

static PyMethodDef stream_methods[] = {
    {"offer", stream_offer, METH_VARARGS, "offer(t, v, s) for a finite v: see the type's doc"},
    {"clear", stream_clear, METH_NOARGS, "forget the window and the run (a new epoch)"},
    {NULL, NULL, 0, NULL}};
static PyGetSetDef stream_getset[] = {
    {"noise_ratio", stream_noise, NULL, "the last measured noise over the characterised level", NULL},
    {NULL, NULL, NULL, NULL, NULL}};
static PyType_Slot stream_slots[] = {
    {Py_tp_new, stream_new}, {Py_tp_init, stream_init}, {Py_tp_dealloc, stream_dealloc},
    {Py_tp_methods, stream_methods}, {Py_tp_getset, stream_getset},
    {Py_tp_doc, "Signal integrity of one raw stream (core/fdi.py Stream subclasses it)"}, {0, NULL}};
static PyType_Spec stream_spec = {"chris_integrator.native._chris_native.StreamState", sizeof(StreamSt), 0,
                                  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, stream_slots};

#include "egomotion.c"               /* ego-motion hygiene (docs/EGO_MOTION.md): Stand, Lag, the rows' measurement functions; EGOMOTION_METHODS */

/* ---- the filter pass ------------------------------------------------------------------------------------ */
enum {
    A_t, A_rows, A_xp, A_Pp, A_F, A_x, A_P, A_anchor, A_usedq,
    A_rejected, A_used, A_group, A_q, A_h, A_z, A_wrap, A_var, A_counted, A_cb, A_robust, A_freeze, A_src,
    A_entries, A_disabled, A_gate, A_anchor_s, A_quarantine, A_model, A_predict, A_nis, A_nis_new, A_nis_note,
    A_trends, A_trend_new, A_trend_note, A_trend_skip,
    A_N
};
static const char *const ANAMES[A_N] = {
    "t", "rows", "xp", "Pp", "F", "x", "P", "anchor", "usedq",
    "rejected", "used", "group", "q", "h", "z", "wrap", "var", "counted", "cb", "robust", "freeze", "src",
    "entries", "disabled", "gate", "anchor_s", "quarantine", "model", "predict", "nis", "nis_new", "nis_note",
    "trends", "trend_new", "trend_note", "trend_skip"};
static PyObject *AN[A_N];

static int names_ready(void)
{
    if (AN[0] != NULL)
        return 0;
    for (int i = 0; i < A_N; i++)
        if ((AN[i] = PyUnicode_InternFromString(ANAMES[i])) == NULL)
            return -1;
    return 0;
}

#define T_NU 4.0                 /* degrees of freedom of the robust rows' Student-t (core/ekf.py T_NU) */

static double attr_double(PyObject *o, int a, int *err)
{
    PyObject *v = PyObject_GetAttr(o, AN[a]);
    if (v == NULL) {
        *err = 1;
        return 0.0;
    }
    double d = PyFloat_AsDouble(v);
    Py_DECREF(v);
    if (d == -1.0 && PyErr_Occurred())
        *err = 1;
    return d;
}

static int attr_true(PyObject *o, int a)
{
    PyObject *v = PyObject_GetAttr(o, AN[a]);
    if (v == NULL)
        return -1;
    int r = PyObject_IsTrue(v);
    Py_DECREF(v);
    return r;
}

static int set_attr(PyObject *o, int a, PyObject *v)
{
    return PyObject_SetAttr(o, AN[a], v);
}

/* the three items of what h(x) returned (an iterable of three, as `pred, idx, coef = r.h(x)` takes it) */
static int unpack3(PyObject *o, PyObject **a, PyObject **b, PyObject **c)
{
    *a = *b = *c = NULL;
    PyObject *it = PyObject_GetIter(o);
    if (it == NULL)
        return -1;
    PyObject **dst[3] = {a, b, c};
    for (int i = 0; i < 3; i++) {
        *dst[i] = PyIter_Next(it);
        if (*dst[i] == NULL) {
            if (!PyErr_Occurred())
                PyErr_Format(PyExc_ValueError, "not enough values to unpack (expected 3, got %d)", i);
            goto fail;
        }
    }
    PyObject *extra = PyIter_Next(it);
    if (extra != NULL) {
        Py_DECREF(extra);
        PyErr_SetString(PyExc_ValueError, "too many values to unpack (expected 3)");
        goto fail;
    }
    if (PyErr_Occurred())
        goto fail;
    Py_DECREF(it);
    return 0;
fail:
    Py_DECREF(it);
    Py_CLEAR(*a);
    Py_CLEAR(*b);
    Py_CLEAR(*c);
    return -1;
}

/* the filter's settings and the monitors' hooks, read from the Filter at the start of a pass */
typedef struct {
    PyObject *disabled, *quarantine, *rejected, *nis, *nis_new, *nis_note, *trends, *trend_new, *trend_note, *skip;
    double gate, anchor_s;
} Pass;

static void pass_drop(Pass *p)
{
    Py_CLEAR(p->disabled); Py_CLEAR(p->quarantine); Py_CLEAR(p->rejected); Py_CLEAR(p->nis); Py_CLEAR(p->nis_new);
    Py_CLEAR(p->nis_note); Py_CLEAR(p->trends); Py_CLEAR(p->trend_new); Py_CLEAR(p->trend_note); Py_CLEAR(p->skip);
}

/* a monitor of the key from its dict, made by the factory when there is none */
static PyObject *monitor_of(PyObject *dict, PyObject *factory, PyObject *key, PyObject *type)
{
    PyObject *m = PyDict_GetItemWithError(dict, key);
    if (m != NULL)
        Py_INCREF(m);
    else if (PyErr_Occurred())
        return NULL;
    else if ((m = PyObject_CallFunctionObjArgs(factory, key, NULL)) == NULL)
        return NULL;
    if (!PyObject_TypeCheck(m, (PyTypeObject *)type)) {
        Py_DECREF(m);
        PyErr_SetString(PyExc_TypeError, "ekf_run: a monitor of the wrong type");
        return NULL;
    }
    return m;
}

/* the first look at a row: its innovation window (the integrator's hook when the verdict changed or holds), its
 * trend (the hook while it holds or held), its own callback */
static int first_look(Pass *p, PyObject *flt, PyObject *row, PyObject *key, PyObject *q, PyObject *et_o, double et,
                      double nu, double s0, double var, double d2)
{
    if (p->nis != Py_None) {
        PyObject *m = monitor_of(p->nis, p->nis_new, key, NisWinType);
        if (m == NULL)
            return -1;
        if (((NisWin *)m)->qt == NULL) {
            Py_DECREF(m);
            PyErr_SetString(PyExc_ValueError, "NisWindow: not initialised");
            return -1;
        }
        int ch;
        if (nis_offer((NisWin *)m, et, d2, &ch) < 0) {
            Py_DECREF(m);
            return -1;
        }
        if (ch || ((NisWin *)m)->high) {
            PyObject *r = PyObject_CallFunctionObjArgs(p->nis_note, key, m, et_o, ch ? Py_True : Py_False, NULL);
            if (r == NULL) {
                Py_DECREF(m);
                return -1;
            }
            Py_DECREF(r);
        }
        Py_DECREF(m);
    }
    if (p->trends != Py_None) {
        Py_ssize_t pre = PyUnicode_Tailmatch(q, p->skip, 0, PY_SSIZE_T_MAX, -1);
        if (pre < 0)
            return -1;
        double s = s0 + var;
        if (!pre && !(s <= 0.0)) {
            PyObject *tr = monitor_of(p->trends, p->trend_new, key, TrendType);
            if (tr == NULL)
                return -1;
            int was = ((Trend *)tr)->high;
            int high = trend_offer((Trend *)tr, et, nu / sqrt(s));
            if (high || was) {
                PyObject *r = PyObject_CallFunctionObjArgs(p->trend_note, key, tr, et_o, NULL);
                if (r == NULL) {
                    Py_DECREF(tr);
                    return -1;
                }
                Py_DECREF(r);
            }
            Py_DECREF(tr);
        }
    }
    PyObject *cb = PyObject_GetAttr(row, AN[A_cb]);
    if (cb == NULL)
        return -1;
    if (cb != Py_None) {
        PyObject *r = PyObject_CallFunction(cb, "ddO", nu, s0, et_o);
        if (r == NULL) {
            Py_DECREF(cb);
            return -1;
        }
        Py_DECREF(r);
    }
    Py_DECREF(cb);
    /* (a hook may have set row classes aside: read again, as the Python read the attribute at every row) */
    Py_CLEAR(p->quarantine);
    if ((p->quarantine = PyObject_GetAttr(flt, AN[A_quarantine])) == NULL)
        return -1;
    Py_CLEAR(p->rejected);
    if ((p->rejected = PyObject_GetAttr(flt, AN[A_rejected])) == NULL)
        return -1;
    return 0;
}

/* a copy of the array held by *o (n values) into a new float64 array of the same shape: *o and *v are replaced */
static int take_copy(PyObject **o, View *v, Py_ssize_t a, Py_ssize_t b)
{
    PyObject *c = np_new(a, b);
    if (c == NULL)
        return -1;
    View cv;
    if (view_get(c, &cv, 1, 'd', b < 0 ? 1 : 2, "copy") < 0) {
        Py_DECREF(c);
        return -1;
    }
    memcpy(cv.b.buf, v->b.buf, (size_t)(a * (b < 0 ? 1 : b)) * sizeof(double));
    view_drop(v);
    *v = cv;
    Py_DECREF(*o);
    *o = c;
    return 0;
}

/* ekf_run(filter, i, t, x, P, anchor, par | None): Filter._run from entry i on, starting from the state (t, x, P)
 * and the anchors of the entry before i. par: the native model's parameter block (EgoModel), or None (the
 * model's own predict is called). Every entry gets xp, Pp, F, x, P, anchor, usedq; every row used, rejected,
 * counted. */
static PyObject *n_ekf_run(PyObject *self, PyObject *args)
{
    PyObject *flt, *x0, *P0, *anchor, *opar;
    Py_ssize_t i0;
    double t;
    if (!PyArg_ParseTuple(args, "OndOOO!O", &flt, &i0, &t, &x0, &P0, &PyDict_Type, &anchor, &opar))
        return NULL;
    if (names_ready() < 0 || np_ready() < 0)
        return NULL;
    Pass p;
    memset(&p, 0, sizeof(p));
    PyObject *slice = NULL, *x = x0, *P = P0, *predict = NULL, *xc = NULL, *Pc = NULL, *res = NULL;
    View par, xv, Pv;
    par.held = xv.held = Pv.held = 0;
    double *ph = NULL, *k = NULL;
    Py_INCREF(x);
    Py_INCREF(P);
    int err = 0;
    PyObject *entries = PyObject_GetAttr(flt, AN[A_entries]);
    if (entries == NULL)
        goto done;
    slice = PySequence_GetSlice(entries, i0, PY_SSIZE_T_MAX);
    Py_DECREF(entries);
    if (slice == NULL || !PyList_Check(slice)) {
        if (slice != NULL)
            PyErr_SetString(PyExc_TypeError, "ekf_run: the entries must be a list");
        goto done;
    }
    if ((p.disabled = PyObject_GetAttr(flt, AN[A_disabled])) == NULL ||
        (p.quarantine = PyObject_GetAttr(flt, AN[A_quarantine])) == NULL ||
        (p.rejected = PyObject_GetAttr(flt, AN[A_rejected])) == NULL ||
        (p.nis = PyObject_GetAttr(flt, AN[A_nis])) == NULL || (p.trends = PyObject_GetAttr(flt, AN[A_trends])) == NULL)
        goto done;
    p.gate = attr_double(flt, A_gate, &err);
    p.anchor_s = attr_double(flt, A_anchor_s, &err);
    if (err)
        goto done;
    if (p.nis != Py_None && (!PyDict_Check(p.nis) || (p.nis_new = PyObject_GetAttr(flt, AN[A_nis_new])) == NULL ||
                             (p.nis_note = PyObject_GetAttr(flt, AN[A_nis_note])) == NULL)) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_TypeError, "ekf_run: the monitors must be a dict");
        goto done;
    }
    if (p.trends != Py_None && (!PyDict_Check(p.trends) || (p.trend_new = PyObject_GetAttr(flt, AN[A_trend_new])) == NULL ||
                                (p.trend_note = PyObject_GetAttr(flt, AN[A_trend_note])) == NULL ||
                                (p.skip = PyObject_GetAttr(flt, AN[A_trend_skip])) == NULL)) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_TypeError, "ekf_run: the trends must be a dict");
        goto done;
    }
    if (view_conv(x, &xv, 'd', 1, "x", &xc) < 0)
        goto done;
    Py_ssize_t n = xv.b.shape[0];
    int has_rho = 0;
    if (opar != Py_None) {
        if (view_get(opar, &par, 0, 'd', 1, "par") < 0)
            goto done;
        if (n < NCORE) {
            PyErr_Format(PyExc_ValueError, "predict: a state of %zd", n);
            goto done;
        }
        has_rho = par.b.shape[0] >= PAR0 && DATA(par)[12] > 0;
        if (predict_check(DATA(par), par.b.shape[0], n, has_rho) < 0)
            goto done;
    } else {
        PyObject *model = PyObject_GetAttr(flt, AN[A_model]);
        if (model == NULL)
            goto done;
        predict = PyObject_GetAttr(model, AN[A_predict]);
        Py_DECREF(model);
        if (predict == NULL)
            goto done;
    }
    ph = (double *)PyMem_Malloc((size_t)(n ? n : 1) * sizeof(double));
    k = (double *)PyMem_Malloc((size_t)(n ? n : 1) * sizeof(double));
    if (ph == NULL || k == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    Py_ssize_t ne = PyList_Size(slice);
    for (Py_ssize_t ei = 0; ei < ne; ei++) {
        PyObject *e = PyList_GetItem(slice, ei);              /* (held by the slice) */
        PyObject *et_o = PyObject_GetAttr(e, AN[A_t]);
        if (et_o == NULL)
            goto done;
        double et = PyFloat_AsDouble(et_o);
        if (et == -1.0 && PyErr_Occurred()) {
            Py_DECREF(et_o);
            goto done;
        }
        double dt = et - t;
        PyObject *F;
        /* the state to predict from: a view of x (made contiguous first when it is not) */
        if (dt > 0.0) {
            if (opar != Py_None) {
                if (!Pv.held && view_conv(P, &Pv, 'd', 2, "P", &Pc) < 0)
                    goto fail_e;
                if (Pv.b.shape[0] != n || Pv.b.shape[1] != n || xv.b.shape[0] != n) {
                    PyErr_SetString(PyExc_ValueError, "ekf_predict: shapes");
                    goto fail_e;
                }
                PyObject *x1 = np_new(n, -1), *P1 = np_new(n, n), *Fc = np_new(NCORE, NCORE);
                PyObject *rho = has_rho ? np_new(n, -1) : (Py_INCREF(Py_None), Py_None);
                View x1v, P1v, Fv, rv;
                x1v.held = P1v.held = Fv.held = rv.held = 0;
                int bad = (x1 == NULL || P1 == NULL || Fc == NULL || rho == NULL ||
                           view_get(x1, &x1v, 1, 'd', 1, "x") < 0 || view_get(P1, &P1v, 1, 'd', 2, "P") < 0 ||
                           view_get(Fc, &Fv, 1, 'd', 2, "Fc") < 0 || (has_rho && view_get(rho, &rv, 1, 'd', 1, "rho") < 0) ||
                           predict_core(DATA(xv), DATA(Pv), n, dt, DATA(par), has_rho ? DATA(rv) : NULL, DATA(x1v),
                                        DATA(P1v), DATA(Fv)) < 0);
                view_drop(&Fv);
                view_drop(&rv);
                F = bad ? NULL : PyTuple_Pack(2, Fc, rho);
                Py_XDECREF(Fc);
                Py_XDECREF(rho);
                if (F == NULL) {
                    view_drop(&x1v);
                    view_drop(&P1v);
                    Py_XDECREF(x1);
                    Py_XDECREF(P1);
                    goto fail_e;
                }
                view_drop(&xv); view_drop(&Pv);
                Py_CLEAR(xc); Py_CLEAR(Pc);
                Py_DECREF(x);
                Py_DECREF(P);
                x = x1, P = P1;
                xv = x1v, Pv = P1v;
            } else {
                PyObject *r = PyObject_CallFunction(predict, "OOd", x, P, dt), *a, *b;
                if (r == NULL)
                    goto fail_e;
                int u = unpack3(r, &a, &b, &F);
                Py_DECREF(r);
                if (u < 0)
                    goto fail_e;
                view_drop(&xv); view_drop(&Pv);
                Py_CLEAR(xc); Py_CLEAR(Pc);
                Py_DECREF(x);
                Py_DECREF(P);
                x = a, P = b;
                if (view_conv(x, &xv, 'd', 1, "x", &xc) < 0) {
                    Py_DECREF(F);
                    goto fail_e;
                }
            }
        } else {
            F = Py_None;
            Py_INCREF(F);
        }
        int bad = set_attr(e, A_xp, x) < 0 || set_attr(e, A_Pp, P) < 0 || set_attr(e, A_F, F) < 0;
        Py_DECREF(F);
        if (bad)
            goto fail_e;
        int own = 0, pown = 0, dirty = 0;
        PyObject *rows = PyObject_GetAttr(e, AN[A_rows]);
        if (rows == NULL)
            goto fail_e;
        if (!PyList_Check(rows)) {
            Py_DECREF(rows);
            PyErr_SetString(PyExc_TypeError, "ekf_run: an entry's rows must be a list");
            goto fail_e;
        }
        for (Py_ssize_t ri = 0; ri < PyList_Size(rows); ri++) {
            PyObject *r = PyList_GetItem(rows, ri);
            PyObject *group = NULL, *q = NULL, *key = NULL, *h = NULL, *hr = NULL, *pred = NULL, *idx = NULL,
                     *coef = NULL, *ic = NULL, *cc = NULL, *fz = NULL, *fc = NULL;
            View iv, cv, fv;
            iv.held = cv.held = fv.held = 0;
            int ok = 0;
            Py_INCREF(r);
            int was = attr_true(r, A_rejected);
            if (was < 0 || set_attr(r, A_used, Py_False) < 0 || set_attr(r, A_rejected, Py_False) < 0)
                goto row_done;
            if ((group = PyObject_GetAttr(r, AN[A_group])) == NULL || (q = PyObject_GetAttr(r, AN[A_q])) == NULL ||
                (key = PyTuple_Pack(2, group, q)) == NULL)
                goto row_done;
            int c = PySequence_Contains(p.disabled, group);
            if (c == 0)
                c = PySequence_Contains(p.disabled, key);
            if (c < 0)
                goto row_done;
            if (c) {
                ok = 1;
                goto row_done;
            }
            /* the row's model: native (kind, idx, par), evaluated here (egomotion.c), or a callable h(x) */
            double cfb[EH_MAXM], pr;
            const double *cfp;
            Py_ssize_t m;
            const int64_t *ix;
            if ((h = PyObject_GetAttr(r, AN[A_model])) == NULL)
                goto row_done;
            if (!Pv.held && view_conv(P, &Pv, 'd', 2, "P", &Pc) < 0)
                goto row_done;
            if (h != Py_None) {
                if (!PyTuple_Check(h) || PyTuple_Size(h) != 3) {
                    PyErr_SetString(PyExc_TypeError, "ekf_run: a row's model must be (kind, idx, par)");
                    goto row_done;
                }
                long kind = PyLong_AsLong(PyTuple_GetItem(h, 0));
                if ((kind == -1 && PyErr_Occurred()) || view_get(PyTuple_GetItem(h, 1), &iv, 0, 'i', 1, "idx") < 0 ||
                    view_get(PyTuple_GetItem(h, 2), &cv, 0, 'd', 1, "par") < 0)
                    goto row_done;
                m = iv.b.shape[0];
                ix = (const int64_t *)iv.b.buf;
                if (Pv.b.shape[0] != n || Pv.b.shape[1] != n || n < NCORE) {
                    PyErr_SetString(PyExc_ValueError, "ekf_project: shapes");
                    goto row_done;
                }
                for (Py_ssize_t j = 0; j < m; j++)
                    if (ix[j] < 0 || ix[j] >= n) {
                        PyErr_SetString(PyExc_IndexError, "ekf_project: an index beyond the state");
                        goto row_done;
                    }
                if (eh_eval(kind, DATA(xv), ix, m, DATA(cv), cv.b.shape[0], &pr, cfb) < 0)
                    goto row_done;
                cfp = cfb;
            } else {
                Py_DECREF(h);
                if ((h = PyObject_GetAttr(r, AN[A_h])) == NULL || (hr = PyObject_CallFunctionObjArgs(h, x, NULL)) == NULL ||
                    unpack3(hr, &pred, &idx, &coef) < 0)
                    goto row_done;
                if (view_conv(idx, &iv, 'i', 1, "idx", &ic) < 0 || view_conv(coef, &cv, 'd', 1, "coef", &cc) < 0)
                    goto row_done;
                m = iv.b.shape[0];
                ix = (const int64_t *)iv.b.buf;
                if (Pv.b.shape[0] != n || Pv.b.shape[1] != n || cv.b.shape[0] != m || m < 1) {
                    PyErr_SetString(PyExc_ValueError, "ekf_project: shapes");
                    goto row_done;
                }
                for (Py_ssize_t j = 0; j < m; j++)
                    if (ix[j] < 0 || ix[j] >= n) {
                        PyErr_SetString(PyExc_IndexError, "ekf_project: an index beyond the state");
                        goto row_done;
                    }
                cfp = DATA(cv);
                pr = PyFloat_AsDouble(pred);
                if (pr == -1.0 && PyErr_Occurred())
                    goto row_done;
            }
            double s0 = project_core(DATA(Pv), n, ix, cfp, m, ph);
            double z = attr_double(r, A_z, &err);
            double var = attr_double(r, A_var, &err);
            if (err)
                goto row_done;
            double nu = z - pr;
            int wr = attr_true(r, A_wrap);
            if (wr < 0)
                goto row_done;
            if (wr)
                nu = wrap_pi(nu);
            double d2 = nu * nu / (s0 + var);
            int counted = attr_true(r, A_counted);
            if (counted < 0)
                goto row_done;
            if (!counted) {
                if (set_attr(r, A_counted, Py_True) < 0 || first_look(&p, flt, r, key, q, et_o, et, nu, s0, var, d2) < 0)
                    goto row_done;
            }
            if ((c = PyObject_IsTrue(p.quarantine)) < 0 || (c && (c = PySequence_Contains(p.quarantine, key)) < 0))
                goto row_done;
            if (c) {
                ok = 1;
                goto row_done;
            }
            int robust = attr_true(r, A_robust);
            if (robust < 0)
                goto row_done;
            if (robust) {
                if (d2 > p.gate) {
                    PyObject *at = PyDict_GetItemWithError(anchor, q);
                    double ta = -1e18;
                    if (at == NULL && PyErr_Occurred())
                        goto row_done;
                    if (at != NULL && (ta = PyFloat_AsDouble(at)) == -1.0 && PyErr_Occurred())
                        goto row_done;
                    if (et - ta <= p.anchor_s) {
                        if (set_attr(r, A_rejected, Py_True) < 0)
                            goto row_done;
                        if (!was) {
                            PyObject *src = PyObject_GetAttr(r, AN[A_src]), *cnt, *one = PyLong_FromLong(1), *nw = NULL;
                            if (src == NULL || one == NULL) {
                                Py_XDECREF(src);
                                Py_XDECREF(one);
                                goto row_done;
                            }
                            cnt = PyObject_CallMethod(p.rejected, "get", "Oi", src, 0);
                            if (cnt != NULL)
                                nw = PyNumber_Add(cnt, one);
                            int sbad = nw == NULL || PyObject_SetItem(p.rejected, src, nw) < 0;
                            Py_XDECREF(cnt);
                            Py_XDECREF(nw);
                            Py_DECREF(one);
                            Py_DECREF(src);
                            if (sbad)
                                goto row_done;
                        }
                        ok = 1;
                        goto row_done;
                    }
                }
                /* Student-t weight on the innovation: the whole innovation variance is inflated */
                double w = (T_NU + 1.0) / (T_NU + d2);
                if (w < 1.0)
                    var = (s0 + var) / w - s0;
            } else if (PyDict_SetItem(anchor, q, et_o) < 0)
                goto row_done;
            double s = s0 + var;
            if (!own) {                                   /* (the predicted state is kept in the entry) */
                if (take_copy(&x, &xv, n, -1) < 0)
                    goto row_done;
                Py_CLEAR(xc);
                own = 1;
            }
            if (!pown) {
                if (take_copy(&P, &Pv, n, n) < 0)
                    goto row_done;
                Py_CLEAR(Pc);
                pown = 1;
            }
            if ((fz = PyObject_GetAttr(r, AN[A_freeze])) == NULL)
                goto row_done;
            int frozen = fz != Py_None;
            if (frozen && view_conv(fz, &fv, 'i', 1, "freeze", &fc) < 0)
                goto row_done;
            const int64_t *f = frozen ? (const int64_t *)fv.b.buf : NULL;
            Py_ssize_t nf = frozen ? fv.b.shape[0] : 0;
            for (Py_ssize_t j = 0; j < nf; j++)
                if (f[j] < 0 || f[j] >= n) {
                    PyErr_SetString(PyExc_ValueError, "ekf_update: freeze index");
                    goto row_done;
                }
            update_core(DATA(xv), DATA(Pv), ph, n, s, nu, f, nf, frozen, k);
            if (set_attr(r, A_used, Py_True) < 0)
                goto row_done;
            dirty = 1;
            ok = 1;
        row_done:
            view_drop(&iv); view_drop(&cv); view_drop(&fv);
            Py_XDECREF(group); Py_XDECREF(q); Py_XDECREF(key); Py_XDECREF(h); Py_XDECREF(hr); Py_XDECREF(pred);
            Py_XDECREF(idx); Py_XDECREF(coef); Py_XDECREF(ic); Py_XDECREF(cc); Py_XDECREF(fz); Py_XDECREF(fc);
            Py_DECREF(r);
            if (!ok) {
                Py_DECREF(rows);
                goto fail_e;
            }
        }
        PyObject *usedq;
        if (dirty) {
            sym_core(DATA(Pv), n);
            usedq = PyList_New(0);
            for (Py_ssize_t ri = 0; usedq != NULL && ri < PyList_Size(rows); ri++) {
                PyObject *r = PyList_GetItem(rows, ri);
                int u = attr_true(r, A_used);
                if (u < 0) {
                    Py_CLEAR(usedq);
                    break;
                }
                if (!u)
                    continue;
                PyObject *q = PyObject_GetAttr(r, AN[A_q]), *src = q ? PyObject_GetAttr(r, AN[A_src]) : NULL;
                PyObject *pair = (q && src) ? PyTuple_Pack(2, q, src) : NULL;
                Py_XDECREF(q);
                Py_XDECREF(src);
                if (pair == NULL || PyList_Append(usedq, pair) < 0) {
                    Py_XDECREF(pair);
                    Py_CLEAR(usedq);
                    break;
                }
                Py_DECREF(pair);
            }
            if (usedq != NULL) {
                PyObject *tu = PyList_AsTuple(usedq);
                Py_DECREF(usedq);
                usedq = tu;
            }
        } else
            usedq = PyTuple_New(0);
        Py_DECREF(rows);
        PyObject *an = usedq ? PyDict_Copy(anchor) : NULL;
        bad = usedq == NULL || an == NULL || set_attr(e, A_usedq, usedq) < 0 || set_attr(e, A_x, x) < 0 ||
              set_attr(e, A_P, P) < 0 || set_attr(e, A_anchor, an) < 0;
        Py_XDECREF(usedq);
        Py_XDECREF(an);
        if (bad)
            goto fail_e;
        t = et;
        Py_DECREF(et_o);
        continue;
    fail_e:
        Py_DECREF(et_o);
        goto done;
    }
    res = Py_None;
    Py_INCREF(res);
done:
    view_drop(&par); view_drop(&xv); view_drop(&Pv);
    Py_XDECREF(xc); Py_XDECREF(Pc);
    PyMem_Free(ph);
    PyMem_Free(k);
    Py_XDECREF(slice);
    Py_XDECREF(predict);
    Py_DECREF(x);
    Py_DECREF(P);
    pass_drop(&p);
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
=======
 * The radar (chris_integrator/radar): its motion model, the sensors' measurement functions, one scan's
 * association and update, the objects' upkeep and the message's objects. The only implementation since the
 * Python reference was removed; tests/test_native.py holds it to what that reference gave (golden data).
 * State [px, py, vx, vy, ax, ay, dp, lh], three hypotheses (stand, cv, ca). Only the entries of the transition
 * that are not zero are propagated (F has at most three per row), summed in index order; the reference took
 * the elementwise steps in numpy's order of operations, which is kept, and summed products in BLAS's order:
 * the results agree to rounding. A state that is not finite is computed like any other (NaN and inf go where
 * the arithmetic takes them: skipping F's zeros, 0 * inf does not become NaN as a dense product would).
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
        for (int m = 0; m < RN; m++)                     /* the upper triangle, mirrored: symmetric by construction */
            for (int n = m; n < RN; n++) {
                int k = m * RN + n;
                double a = (w0 * sp[0][k] + w1 * sp[1][k]) + w2 * sp[2][k];
                double b = ((w0 * d0[m]) * d0[n] + (w1 * d1[m]) * d1[n]) + (w2 * d2[m]) * d2[n];
                P[k] = P[n * RN + m] = a + b;
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
    for (int m = 0; m < RN; m++)                         /* the upper triangle, mirrored */
        for (int n = m; n < RN; n++) {
            int k = m * RN + n;
            double a = (mu[0] * Ps[k] + mu[1] * Ps[RNN + k]) + mu[2] * Ps[2 * RNN + k];
            double b = ((mu[0] * d[0][m]) * d[0][n] + (mu[1] * d[1][m]) * d[1][n]) + (mu[2] * d[2][m]) * d[2][n];
            P[k] = P[n * RN + m] = a + b;
        }
}

static int rpar_get(PyObject *o, RPar *p)
{
    double v[RNPAR];
    if (rows_get(o, RNPAR, 1, v, "par") < 0)           /* 12 numbers: a tuple or a buffer */
        return -1;
    if (!all_finite(v, RNPAR)) {
        PyErr_SetString(PyExc_ValueError, "radar: the model's parameters must be finite");
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

/* ---- the objects' rows, the sensors' measurement functions, one scan's association (gate, likelihoods of the
 * hypotheses, the box as a feature, the pairing's probabilities, the assignment), the update of every pair of a
 * scan, the objects' upkeep and the message's objects. The formulas of the Python reference (radar/tracker.py,
 * sensors.py, box.py before its removal), sums of products in index order, small systems solved by Gaussian
 * elimination with partial pivoting (the reference: LAPACK): results agree with it to rounding. ---- */

/* an object's row in the tracker's state array (tracker.py `_LAY`) */
enum { RO_XS = 0, RO_PS = 24, RO_MU = 216, RO_XF = 219, RO_PF = 227, RO_X = 291, RO_P = 299, RO_E = 363, RO_PE = 366,
       RO_ST = 375, RO_M = 388, RROW = 408 };

/* a node's record (RO_M ...) and the step that led into it (RO_ST ...): see "the radar's hypotheses" */
enum { M_T, M_L, M_HITS, M_THIT, M_RANGE_T, M_BOXW, M_BOXT, M_BOXS, M_NIS, M_DL, M_DET, M_LEVEL, M_VIS, M_PD, M_SRC,
       M_OCC, M_COV, M_OCCBY, M_SENSE, M_HOLD, RMETA };
enum { ST_T = 0, ST_DT = 9, ST_E = 10 };

/* the state array (capacity, RROW), writable */
static int store_get(PyObject *o, View *v, Py_ssize_t *cap)
{
    v->held = 0;
    if (PyObject_GetBuffer(o, &v->b, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | PyBUF_WRITABLE) < 0)
        return -1;
    v->held = 1;
    const char *f = v->b.format ? v->b.format : "B";
    if (*f == '@' || *f == '=' || *f == '<')
        f++;
    if (!(f[0] == 'd' && f[1] == 0 && v->b.itemsize == 8) || v->b.len % (RROW * (Py_ssize_t)sizeof(double)) != 0) {
        view_drop(v);
        PyErr_Format(PyExc_TypeError, "radar: the state array must be a writable C-contiguous float64 array of rows of %d", RROW);
        return -1;
    }
    *cap = v->b.len / (RROW * (Py_ssize_t)sizeof(double));
    return 0;
}

/* a list / tuple of B row numbers, each < cap, into a new array (PyMem) */
static Py_ssize_t *idx_get(PyObject *o, Py_ssize_t cap, Py_ssize_t *B)
{
    int is_list = PyList_Check(o);
    if (!(is_list || PyTuple_Check(o))) {
        PyErr_SetString(PyExc_TypeError, "radar: the rows must be a list of ints");
        return NULL;
    }
    Py_ssize_t n = is_list ? PyList_Size(o) : PyTuple_Size(o);
    Py_ssize_t *out = (Py_ssize_t *)PyMem_Malloc((size_t)(n ? n : 1) * sizeof(Py_ssize_t));
    if (out == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        Py_ssize_t k = PyLong_AsSsize_t(seq_item(o, is_list, i));
        if (k == -1 && PyErr_Occurred()) {
            PyMem_Free(out);
            return NULL;
        }
        if (k < 0 || k >= cap) {
            PyMem_Free(out);
            PyErr_SetString(PyExc_IndexError, "radar: a row outside the state array");
            return NULL;
        }
        out[i] = k;
    }
    *B = n;
    return out;
}

/* a float64 buffer of exactly n values (any shape), or a list / tuple of n numbers, into out */
static int rnums_get(PyObject *o, Py_ssize_t n, double *out, const char *what)
{
    return rows_get(o, n, 1, out, what);
}

/* ---- the sensors (sensors.py): what a scan measures of a state ---- */
enum { SEN_CAMERA = 0, SEN_MICROWAVE = 1, SEN_POSITION = 2 };
enum { K_B = 0, K_RB = 1, K_RBH = 2, K_BT = 3, K_NONE = 4, K_SCALE = 8 };
#define RSENS 32

typedef struct {
    int type;
    /* camera */
    double hc, logf, bias;
    int cover;
    double bmin, bmax, gmin, max_range, p_d, p_cut, h_min, h_full, p_small, near, extra, vol[4], srspan, lhspan,
        born, born_s;
    double bias_var, dp_var, jit2, bias_floor2, top_rel, top_range, gm_var, gm_tau;
    /* microwave */
    double mx, my, yaw, fov, clutter, extended;
    /* the detector's further inputs for this scan (NULL: none; set by Hyp.cycle) */
    const struct RXc *xc;
    double grade_k;                  /* the road's vertical curvature, 1 sigma (1/m); 0: not modelled */
    int lcol;                        /* a lamp pair's colour says which end faces the camera */
} RSens;

/* sensors.py `Sensor.native()`: RSENS numbers */
static int sens_get(PyObject *o, RSens *s)
{
    double v[RSENS];
    if (rnums_get(o, RSENS, v, "sensor") < 0)
        return -1;
    memset(s, 0, sizeof(*s));
    s->type = (int)v[0];
    if (s->type == SEN_CAMERA) {
        s->hc = v[1]; s->logf = v[2]; s->bias = v[3]; s->cover = v[4] != 0.0; s->bmin = v[5]; s->bmax = v[6];
        s->gmin = v[7]; s->max_range = v[8]; s->p_d = v[9]; s->p_cut = v[10]; s->h_min = v[11]; s->h_full = v[12];
        s->p_small = v[13]; s->near = v[14]; s->extra = v[15];
        for (int k = 0; k < 4; k++)
            s->vol[k] = v[16 + k];
        s->srspan = v[20]; s->lhspan = v[21]; s->born = v[22]; s->born_s = v[23];
        s->bias_var = v[24]; s->dp_var = v[25]; s->jit2 = v[26]; s->bias_floor2 = v[27]; s->top_rel = v[28]; s->top_range = v[29];
        s->gm_var = v[30]; s->gm_tau = v[31];
    } else if (s->type == SEN_MICROWAVE) {
        s->mx = v[1]; s->my = v[2]; s->yaw = v[3]; s->fov = v[4]; s->max_range = v[5]; s->p_d = v[6]; s->clutter = v[7];
        s->extended = v[8];
    } else if (s->type == SEN_POSITION) {
        s->fov = v[1]; s->max_range = v[2]; s->p_d = v[3]; s->clutter = v[4];
    } else {
        PyErr_SetString(PyExc_ValueError, "radar: a sensor of a kind the native core does not know");
        return -1;
    }
    return 0;
}

/* math.hypot of two numbers as CPython computes it (Modules/mathmodule.c `vector_norm`: a scaled, compensated
 * sum of squares with one correction step); libm's hypot differs from it in the last bit now and then */
static double py_hypot(double a, double b)
{
    double v[2] = {fabs(a), fabs(b)};
    double mx = 0.0;
    int nan = 0;
    for (int i = 0; i < 2; i++) {
        nan |= isnan(v[i]);
        if (v[i] > mx)
            mx = v[i];
    }
    if (isinf(mx))
        return mx;
    if (nan)
        return NAN;
    if (mx == 0.0)
        return mx;
    int e;
    frexp(mx, &e);
    if (e < -1023) {                                    /* subnormal: scaled up first, as CPython does */
        return 2.2250738585072014e-308 * py_hypot(v[0] / 2.2250738585072014e-308, v[1] / 2.2250738585072014e-308);
    }
    double scale = ldexp(1.0, -e), csum = 1.0, frac1 = 0.0, frac2 = 0.0;
    for (int i = 0; i < 2; i++) {
        double x = v[i] * scale;
        double hi = x * x, lo = fma(x, x, -hi);
        double sh = csum + hi, sl = (csum - sh) + hi;
        csum = sh;
        frac1 += lo;
        frac2 += sl;
    }
    double h = sqrt(csum - 1.0 + (frac1 + frac2));
    double hi = -h * h, lo = fma(-h, h, -hi);
    double sh = csum + hi, sl = (csum - sh) + hi;
    csum = sh;
    frac1 += lo;
    frac2 += sl;
    double x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    return h / scale;
}

static inline double pymax(double a, double b) { return b > a ? b : a; }     /* Python's max(a, b) */
static inline double np_maximum(double a, double b) { return isnan(a) ? a : isnan(b) ? b : (a >= b ? a : b); }
static inline double np_minimum(double a, double b) { return isnan(a) ? a : isnan(b) ? b : (a <= b ? a : b); }
static inline double pymin(double a, double b) { return b < a ? b : a; }     /* Python's min(a, b) */

/* the measurement row that is an angle (wrapped), or -1 (sensors.py `rows`) */
static int r_angle_row(const RSens *s, int kind)
{
    if (s->type == SEN_MICROWAVE)
        return 1;
    if (s->type == SEN_POSITION)
        return -1;
    int b = kind & 7;
    return (b == K_B || b == K_BT) ? 0 : b == K_NONE ? -1 : 1;
}

/* CameraBoxes._scale_row */
static double r_scale_row(const double *x, const double *pose, double sdt, double *row, double near)
{
    double dx = x[R_PX] - pose[0], dy = x[R_PY] - pose[1];
    double c = cos(pose[2]), s = sin(pose[2]);
    double depth = pymax((c * dx + s * dy) * near, 0.5);
    double w = pose[5];
    double rate = (c * (x[R_VX] - pose[3]) + s * (x[R_VY] - pose[4])) + w * (-s * dx + c * dy);
    double dt = pymax(sdt, 1e-3);
    double u = pymax(1.0 - rate * dt / depth, 0.05);
    for (int k = 0; k < RN; k++)
        row[k] = 0.0;
    row[R_VX] = -c / (u * depth);
    row[R_VY] = -s / (u * depth);
    row[R_PX] = (w * s / depth + rate * c / (depth * depth)) / u;
    row[R_PY] = (-w * c / depth + rate * s / (depth * depth)) / u;
    return log(u) / dt;
}

/* half the depth of a rectangle (e: length, width, heading) along the line of sight `los` (a ground angle): how
 * far its near face is in front of its centre; *dda: the derivative by the heading */
static double r_half_depth(const double *e, double los, double *dda)
{
    double a = e[2] - los, sa = sin(a), ca = cos(a);
    if (dda)
        *dda = 0.5 * (-e[0] * (ca >= 0.0 ? sa : -sa) + e[1] * (sa >= 0.0 ? ca : -ca));
    return 0.5 * (e[0] * fabs(ca) + e[1] * fabs(sa));
}

/* the offset (*g0, *g1) from a rectangle's centre (e: length, width, heading) to its near face as seen by an
 * observer at (o0, o1) from the centre: where the line of sight from the observer to the centre enters the
 * rectangle. On the rectangle, continuous, and for a thing ahead its rear face at about the centre's lateral place
 * (as the message always had it). NOT the centre brought nearer along the line of sight by r_half_depth (the
 * rectangle's support): that point lies off the rectangle, towards the observer, by up to the half width when the
 * object is beside (a car parked 2.7 m to the side, its centre 2 m ahead: 0.75 m nearer the car's path than its
 * side; a long one beside the car at the origin itself). An observer inside the rectangle: the observer. */
static void r_near_offset(const double *e, double o0, double o1, double *g0, double *g1)
{
    double rc = hypot(o0, o1);
    if (!(rc > 1e-6)) {
        *g0 = *g1 = 0.0;
        return;
    }
    double c = cos(e[2]), s = sin(e[2]), hl = 0.5 * pymax(e[0], 0.0), hw = 0.5 * pymax(e[1], 0.0);
    double vx = (c * o0 + s * o1) / rc, vy = (-s * o0 + c * o1) / rc;     /* towards the observer, body frame */
    double t = rc;
    if (fabs(vx) > 1e-9)
        t = pymin(t, hl / fabs(vx));
    if (fabs(vy) > 1e-9)
        t = pymin(t, hw / fabs(vy));
    *g0 = t * o0 / rc;
    *g1 = t * o1 / rc;
}

/* sensor.h(x, pose, det): the predicted measurement z (m) and its Jacobian H (m x N); returns m. The state's place
 * is the object's CENTRE. A camera's box stands on the road at the object's near face and its height is that
 * face's: with the object's extent e (length, width, heading; NULL: a point) the ground angle, the box height, the
 * top edge and the scale rate are those of the near face, half the object's depth nearer than its centre. */
static int r_h(const RSens *s, int kind, double sdt, const double *x, const double *pose, double *z, double *H,
               const double *e)
{
    memset(H, 0, 4 * RN * sizeof(double));
    if (s->type == SEN_POSITION) {                      /* a place in the car frame */
        double c = cos(pose[2]), sn = sin(pose[2]);
        double dx = x[R_PX] - pose[0], dy = x[R_PY] - pose[1];
        H[R_PX] = c;
        H[R_PY] = sn;
        H[RN + R_PX] = -sn;
        H[RN + R_PY] = c;
        z[0] = c * dx + sn * dy;
        z[1] = -sn * dx + c * dy;
        return 2;
    }
    if (s->type == SEN_MICROWAVE) {
        double c = cos(pose[2]), sn = sin(pose[2]);
        double ox = c * s->mx - sn * s->my, oy = sn * s->mx + c * s->my;
        double sx = pose[0] + ox, sy = pose[1] + oy, svx = pose[3] - pose[5] * oy, svy = pose[4] + pose[5] * ox;
        double dx = x[R_PX] - sx, dy = x[R_PY] - sy;
        double r = pymax(py_hypot(dx, dy), 0.5);
        double rvx = x[R_VX] - svx, rvy = x[R_VY] - svy;
        double rr = (dx * rvx + dy * rvy) / r;
        H[R_PX] = dx / r;
        H[R_PY] = dy / r;
        H[RN + R_PX] = -dy / (r * r);
        H[RN + R_PY] = dx / (r * r);
        H[2 * RN + R_PX] = rvx / r - rr * dx / (r * r);
        H[2 * RN + R_PY] = rvy / r - rr * dy / (r * r);
        H[2 * RN + R_VX] = dx / r;
        H[2 * RN + R_VY] = dy / r;
        z[0] = r;
        z[1] = wrap_pi(atan2(dy, dx) - pose[2] - s->yaw);
        z[2] = rr;
        return 3;
    }
    int base = kind & 7, m = 0;
    double near = 1.0;                                  /* the near face's range over the centre's */
    if (base != K_NONE) {
        double dx = x[R_PX] - pose[0], dy = x[R_PY] - pose[1];
        double r = pymax(py_hypot(dx, dy), 0.5);
        double rc = r;                                  /* the centre's range; r: the near face's */
        if (e) {
            r = pymax(rc - r_half_depth(e, atan2(dy, dx), NULL), 0.5);
            near = r / rc;
        }
        double brow[RN] = {0};
        brow[R_PX] = -dy / (rc * rc);
        brow[R_PY] = dx / (rc * rc);
        double b = wrap_pi(atan2(dy, dx) - pose[2]);
        double hc = s->hc;
        if (base == K_B) {
            z[0] = b;
            memcpy(H, brow, sizeof(brow));
            m = 1;
        } else if (base == K_BT) {
            double ho = exp(x[R_LH]);
            double dh = hc - ho;
            double q = r * r + dh * dh;
            z[0] = b;
            memcpy(H, brow, sizeof(brow));
            H[RN + R_PX] = -dh / q * dx / rc;
            H[RN + R_PY] = -dh / q * dy / rc;
            H[RN + R_LH] = -ho * r / q;
            z[1] = atan2(dh, r) + s->bias;
            m = 2;
        } else {
            double g = -hc / (r * r + hc * hc);
            H[R_PX] = g * dx / rc;
            H[R_PY] = g * dy / rc;
            H[R_DP] = 1.0;
            z[0] = atan2(hc, r) + s->bias + x[R_DP];
            z[1] = b;
            memcpy(H + RN, brow, sizeof(brow));
            m = 2;
            if (base != K_RB) {
                double c = cos(pose[2]), sn = sin(pose[2]);
                double depth = pymax((c * dx + sn * dy) * near, 0.5);
                H[2 * RN + R_PX] = -c / depth;
                H[2 * RN + R_PY] = -sn / depth;
                H[2 * RN + R_LH] = 1.0;
                z[2] = s->logf + x[R_LH] - log(depth);
                m = 3;
            }
        }
    }
    if (kind & K_SCALE) {
        if (e && base == K_NONE) {
            double dx = x[R_PX] - pose[0], dy = x[R_PY] - pose[1], rc = pymax(py_hypot(dx, dy), 0.5);
            near = pymax(rc - r_half_depth(e, atan2(dy, dx), NULL), 0.5) / rc;
        }
        z[m] = r_scale_row(x, pose, sdt, H + m * RN, s->type == SEN_CAMERA ? near : 1.0);
        m++;
    }
    return m;
}

/* CameraBoxes.coverage / MicrowaveRadar.coverage at a place of the car frame */
static double r_coverage(const RSens *s, double xc, double yc)
{
    if (s->type == SEN_POSITION) {
        double r = py_hypot(xc, yc);
        return (1.0 < r && r < s->max_range && fabs(atan2(yc, xc)) < s->fov) ? s->p_d : 0.0;
    }
    if (s->type == SEN_MICROWAVE) {
        double dx = xc - s->mx, dy = yc - s->my;
        double r = py_hypot(dx, dy);
        if (r < 0.5 || r > s->max_range || fabs(wrap_pi(atan2(dy, dx) - s->yaw)) > s->fov)
            return 0.0;
        return s->p_d;
    }
    if (xc <= 0.5)
        return 0.0;
    if (s->cover) {
        double a = atan2(yc, xc);
        if (!(s->bmin <= a && a <= s->bmax))
            return 0.0;
    }
    if (py_hypot(xc, yc) > s->max_range)
        return 0.0;
    return s->p_d;
}

/* sensor.p_detect(xc, yc, x) */
static double r_p_detect(const RSens *s, double xc, double yc, const double *x)
{
    double p = r_coverage(s, xc, yc);
    if (s->type != SEN_CAMERA)
        return p;
    if (p > 0.0 && s->cover && py_hypot(xc, yc) < s->gmin)
        p *= s->p_cut;
    if (p <= 0.0 || isnan(s->h_min))
        return p;
    double h = exp(s->logf + x[R_LH]) / pymax(xc, 0.5);
    double ramp = (h / s->h_min - 1.0) / (s->h_full - 1.0);
    return p * pymin(1.0, pymax(s->p_small, ramp));
}

/* sensor.density(det) */
static double r_density(const RSens *s, int kind, const double *z)
{
    if (s->type != SEN_CAMERA)
        return s->clutter;
    int b = kind & 7;
    if (b == K_NONE)                                    /* the reference has no volume for this: never asked */
        return NAN;
    double d = s->extra / s->vol[b];
    if (kind & K_SCALE)
        d /= s->srspan;
    if (b != K_RBH)
        return d;
    double u = z[2] - ((s->h_min != 0.0 && isfinite(s->h_min)) ? log(s->h_min) : z[2]);
    double born = exp(-pymax(u, 0.0) / s->born_s) / s->born_s;
    return d * s->lhspan * ((1.0 - s->born) / s->lhspan + s->born * born);
}

/* ---- small dense algebra (m <= 4) ---- */
/* S = H P H^T (m x m) and HP = H P (m x N) */
static void r_hph(const double *H, int m, const double *P, double *HP, double *S)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < RN; j++) {
            double acc = 0.0;
            for (int k = 0; k < RN; k++)
                acc = k ? acc + H[i * RN + k] * P[k * RN + j] : H[i * RN + k] * P[k * RN + j];
            HP[i * RN + j] = acc;
        }
    for (int i = 0; i < m; i++)
        for (int l = 0; l < m; l++) {
            double acc = 0.0;
            for (int k = 0; k < RN; k++)
                acc = k ? acc + HP[i * RN + k] * H[l * RN + k] : HP[i * RN + k] * H[l * RN + k];
            S[i * m + l] = acc;
        }
}

/* LU with partial pivoting of A (m x m) in place; piv; returns 0, or -1 for a zero pivot (singular: numpy's
 * LinAlgError) */
static int r_lu(double *A, int m, int *piv)
{
    for (int c = 0; c < m; c++) {
        int p = c;
        double best = fabs(A[c * m + c]);
        for (int r = c + 1; r < m; r++)
            if (fabs(A[r * m + c]) > best) {
                best = fabs(A[r * m + c]);
                p = r;
            }
        piv[c] = p;
        if (p != c)
            for (int j = 0; j < m; j++) {
                double t = A[c * m + j];
                A[c * m + j] = A[p * m + j];
                A[p * m + j] = t;
            }
        double d = A[c * m + c];
        if (d == 0.0)
            return -1;
        for (int r = c + 1; r < m; r++) {
            double f = A[r * m + c] / d;
            A[r * m + c] = f;
            for (int j = c + 1; j < m; j++)
                A[r * m + j] -= f * A[c * m + j];
        }
    }
    return 0;
}

/* solve with an LU: b (m) in place */
static void r_lu_solve(const double *LU, int m, const int *piv, double *b)
{
    for (int c = 0; c < m; c++) {
        if (piv[c] != c) {
            double t = b[c];
            b[c] = b[piv[c]];
            b[piv[c]] = t;
        }
    }
    for (int r = 1; r < m; r++) {
        double acc = b[r];
        for (int k = 0; k < r; k++)
            acc -= LU[r * m + k] * b[k];
        b[r] = acc;
    }
    for (int r = m - 1; r >= 0; r--) {
        double acc = b[r];
        for (int k = r + 1; k < m; k++)
            acc -= LU[r * m + k] * b[k];
        b[r] = acc / LU[r * m + r];
    }
}

/* numpy's det from an LU: sign * exp(sum log |u_ii|) */
static double r_lu_det(const double *LU, int m, const int *piv)
{
    double sign = 1.0, logdet = 0.0;
    for (int c = 0; c < m; c++) {
        double d = LU[c * m + c];
        if (piv[c] != c)
            sign = -sign;
        if (d < 0.0)
            sign = -sign;
        logdet += log(fabs(d));
    }
    return sign * exp(logdet);
}

/* nu^T S^-1 nu (S: m x m, not changed); -1: singular. det of S into *det if det */
static int r_mahal(const double *S, int m, const double *nu, double *d2, double *det)
{
    double A[16], b[4];
    int piv[4];
    memcpy(A, S, (size_t)(m * m) * sizeof(double));
    if (r_lu(A, m, piv) < 0)
        return -1;
    memcpy(b, nu, (size_t)m * sizeof(double));
    r_lu_solve(A, m, piv, b);
    double acc = 0.0;
    for (int k = 0; k < m; k++)
        acc = k ? acc + nu[k] * b[k] : nu[k] * b[k];
    *d2 = acc;
    if (det)
        *det = r_lu_det(A, m, piv);
    return 0;
}

/* the inverse of S (m x m) and its det; -1: singular */
static int r_inv(const double *S, int m, double *Si, double *det)
{
    double A[16];
    int piv[4];
    memcpy(A, S, (size_t)(m * m) * sizeof(double));
    if (r_lu(A, m, piv) < 0)
        return -1;
    for (int j = 0; j < m; j++) {
        double e[4] = {0.0, 0.0, 0.0, 0.0};
        e[j] = 1.0;
        r_lu_solve(A, m, piv, e);
        for (int i = 0; i < m; i++)
            Si[i * m + j] = e[i];
    }
    *det = r_lu_det(A, m, piv);
    return 0;
}

/* numpy's sum of a contiguous row (pairwise summation, as np.add.reduce does) */
static double np_sum(const double *a, Py_ssize_t n, Py_ssize_t stride)
{
    if (n < 8) {
        double r = 0.0;
        for (Py_ssize_t i = 0; i < n; i++)
            r += a[i * stride];
        return r;
    }
    if (n <= 128) {
        double r[8];
        Py_ssize_t i;
        for (int k = 0; k < 8; k++)
            r[k] = a[k * stride];
        for (i = 8; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; k++)
                r[k] += a[(i + k) * stride];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++)
            res += a[i * stride];
        return res;
    }
    Py_ssize_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_sum(a, n2, stride) + np_sum(a + n2 * stride, n - n2, stride);
}

/* ---- marginals (tracker.py): loopy belief propagation on the assignment graph. L (nt x nd) -> out ---- */
static int r_marginals(const double *L, Py_ssize_t nt, Py_ssize_t nd, double *out)
{
    Py_ssize_t n = nt * nd;
    int any = 0;
    for (Py_ssize_t k = 0; k < n; k++)
        if (L[k] != 0.0)
            any = 1;
    if (!any) {
        for (Py_ssize_t k = 0; k < n; k++)
            out[k] = 0.0;
        return 0;
    }
    double *buf = (double *)PyMem_Malloc((size_t)(3 * n + nt + nd) * sizeof(double));
    if (buf == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    double *nu = buf, *val = buf + n, *mu = buf + 2 * n, *rs = buf + 3 * n, *cs = rs + nt;
    for (Py_ssize_t k = 0; k < n; k++)
        nu[k] = 1.0;
    for (int it = 0; it < 30; it++) {
        for (Py_ssize_t k = 0; k < n; k++)
            val[k] = L[k] * nu[k];
        for (Py_ssize_t i = 0; i < nt; i++)
            rs[i] = np_sum(val + i * nd, nd, 1);
        for (Py_ssize_t i = 0; i < nt; i++)
            for (Py_ssize_t j = 0; j < nd; j++)
                mu[i * nd + j] = L[i * nd + j] / (1.0 + rs[i] - val[i * nd + j]);
        /* the column sums: row by row (numpy reduces a non-contiguous axis that way), pairwise for one column */
        if (nd == 1)
            cs[0] = np_sum(mu, nt, 1);
        else {
            for (Py_ssize_t j = 0; j < nd; j++)
                cs[j] = 0.0 + mu[j];
            for (Py_ssize_t i = 1; i < nt; i++)
                for (Py_ssize_t j = 0; j < nd; j++)
                    cs[j] += mu[i * nd + j];
        }
        double worst = 0.0;
        int nan = 0;
        for (Py_ssize_t i = 0; i < nt; i++)
            for (Py_ssize_t j = 0; j < nd; j++) {
                double nw = 1.0 / (1.0 + cs[j] - mu[i * nd + j]);
                double d = fabs(nw - nu[i * nd + j]);
                if (isnan(d))
                    nan = 1;
                else if (d > worst)
                    worst = d;
                nu[i * nd + j] = nw;
            }
        if (!nan && worst < 1e-4)
            break;
    }
    for (Py_ssize_t k = 0; k < n; k++)
        val[k] = L[k] * nu[k];
    for (Py_ssize_t i = 0; i < nt; i++) {
        double r = 1.0 + np_sum(val + i * nd, nd, 1);
        for (Py_ssize_t j = 0; j < nd; j++)
            out[i * nd + j] = val[i * nd + j] / r;
    }
    PyMem_Free(buf);
    return 0;
}

/* ---- assign (tracker.py) with core/objects.py `assign` (Hungarian) for the groups that compete ---- */
/* core.objects.assign on cost (n x m, INFINITY: forbidden): pairs appended to (pi, pj), count returned */
static Py_ssize_t r_hungarian(const double *cost, Py_ssize_t n, Py_ssize_t m, Py_ssize_t *pi, Py_ssize_t *pj)
{
    Py_ssize_t cnt = 0;
    int anyf = 0;
    for (Py_ssize_t k = 0; k < n * m; k++)
        if (isfinite(cost[k]))
            anyf = 1;
    if (!anyf)
        return 0;
    /* the common case: no two rows compete: each row's best column is the optimum */
    {
        Py_ssize_t *best = (Py_ssize_t *)PyMem_Malloc((size_t)n * sizeof(Py_ssize_t));
        if (best == NULL)
            return -1;
        Py_ssize_t nr = 0;
        int uniq = 1;
        for (Py_ssize_t i = 0; i < n; i++) {
            Py_ssize_t b = -1;
            double bv = INFINITY;
            int fin = 0;
            for (Py_ssize_t j = 0; j < m; j++) {
                double c = cost[i * m + j];
                if (isfinite(c)) {
                    fin = 1;
                    if (b < 0 || c < bv) {
                        bv = c;
                        b = j;
                    }
                }
            }
            if (!fin)
                continue;
            best[nr] = b;
            pi[nr] = i;
            nr++;
        }
        /* argmin of np.where(fin, cost, inf): the first finite minimum (a row with finite entries all inf-equal
         * cannot occur: finite entries are below inf) */
        for (Py_ssize_t a = 0; a < nr && uniq; a++) {
            for (Py_ssize_t b2 = a + 1; b2 < nr; b2++)
                if (best[a] == best[b2]) {
                    uniq = 0;
                    break;
                }
            if (uniq) {
                int cntf = 0;
                for (Py_ssize_t i = 0; i < n; i++)
                    cntf += isfinite(cost[i * m + best[a]]) ? 1 : 0;
                if (cntf != 1)
                    uniq = 0;
            }
        }
        if (uniq) {
            for (Py_ssize_t a = 0; a < nr; a++)
                pj[a] = best[a];
            PyMem_Free(best);
            return nr;
        }
        PyMem_Free(best);
    }
    const double big = 1e6;
    Py_ssize_t k = n > m ? n : m;
    double *a = (double *)PyMem_Malloc((size_t)(k * k + 3 * (k + 1)) * sizeof(double));
    Py_ssize_t *p = (Py_ssize_t *)PyMem_Malloc((size_t)(2 * (k + 1)) * sizeof(Py_ssize_t));
    char *used = (char *)PyMem_Malloc((size_t)(k + 1));
    if (a == NULL || p == NULL || used == NULL) {
        PyMem_Free(a);
        PyMem_Free(p);
        PyMem_Free(used);
        return -1;
    }
    double *u = a + k * k, *v = u + k + 1, *minv = v + k + 1;
    Py_ssize_t *way = p + k + 1;
    for (Py_ssize_t i = 0; i < k; i++)
        for (Py_ssize_t j = 0; j < k; j++)
            a[i * k + j] = (i < n && j < m && isfinite(cost[i * m + j])) ? cost[i * m + j] : big;
    for (Py_ssize_t j = 0; j <= k; j++) {
        u[j] = v[j] = 0.0;
        p[j] = way[j] = 0;
    }
    for (Py_ssize_t i = 1; i <= k; i++) {
        p[0] = i;
        Py_ssize_t j0 = 0;
        for (Py_ssize_t j = 0; j <= k; j++) {
            minv[j] = INFINITY;
            used[j] = 0;
        }
        for (;;) {
            used[j0] = 1;
            Py_ssize_t i0 = p[j0], j1 = 0;
            double delta = INFINITY;
            for (Py_ssize_t j = 1; j <= k; j++)
                if (!used[j]) {
                    double cur = a[(i0 - 1) * k + j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            for (Py_ssize_t j = 0; j <= k; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else
                    minv[j] -= delta;
            }
            j0 = j1;
            if (p[j0] == 0)
                break;
        }
        for (;;) {
            Py_ssize_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if (j0 == 0)
                break;
        }
    }
    for (Py_ssize_t j = 1; j <= k; j++) {
        Py_ssize_t i = p[j] - 1;
        if (i < n && j - 1 < m && isfinite(cost[i * m + j - 1])) {
            pi[cnt] = i;
            pj[cnt] = j - 1;
            cnt++;
        }
    }
    PyMem_Free(a);
    PyMem_Free(p);
    PyMem_Free(used);
    return cnt;
}

/* tracker.assign: the pairs (object, detection) of the largest total gain (gain -inf: not allowed), settled per
 * group of objects and detections that compete. pairs appended to (pi, pj); count returned, -1 on error */
static Py_ssize_t r_assign(const double *gain, Py_ssize_t nt, Py_ssize_t nd, Py_ssize_t *pi, Py_ssize_t *pj)
{
    Py_ssize_t cnt = 0;
    char *seen_t = (char *)PyMem_Calloc((size_t)(nt + nd + 1), 1);
    char *in_t = (char *)PyMem_Calloc((size_t)(nt + nd + 1), 1);
    Py_ssize_t *stack = (Py_ssize_t *)PyMem_Malloc((size_t)(2 * (nt + nd) + 2) * sizeof(Py_ssize_t));
    Py_ssize_t *ts = (Py_ssize_t *)PyMem_Malloc((size_t)(nt + nd + 2) * sizeof(Py_ssize_t));
    double *cst = NULL;
    Py_ssize_t *hi = NULL, *hj = NULL;
    if (seen_t == NULL || in_t == NULL || stack == NULL || ts == NULL) {
        PyErr_NoMemory();
        cnt = -1;
        goto out;
    }
    char *in_d = in_t + nt;
    Py_ssize_t *ds = ts + nt + 1;
    (void)ds;
    for (Py_ssize_t i0 = 0; i0 < nt; i0++) {
        int any = 0;
        for (Py_ssize_t j = 0; j < nd; j++)
            if (isfinite(gain[i0 * nd + j]))
                any = 1;
        if (!any || seen_t[i0])
            continue;
        memset(in_t, 0, (size_t)(nt + nd));
        Py_ssize_t sp = 0;
        in_t[i0] = 1;
        stack[sp++] = i0;                                /* objects as i, detections as nt + j */
        while (sp) {
            Py_ssize_t k = stack[--sp];
            if (k < nt) {
                for (Py_ssize_t j = 0; j < nd; j++)
                    if (isfinite(gain[k * nd + j]) && !in_d[j]) {
                        in_d[j] = 1;
                        stack[sp++] = nt + j;
                    }
            } else {
                Py_ssize_t j = k - nt;
                for (Py_ssize_t i = 0; i < nt; i++)
                    if (isfinite(gain[i * nd + j]) && !in_t[i]) {
                        in_t[i] = 1;
                        stack[sp++] = i;
                    }
            }
        }
        Py_ssize_t nts = 0, nds = 0;
        Py_ssize_t *T = ts, *D = stack;                  /* the stack is free now */
        for (Py_ssize_t i = 0; i < nt; i++)
            if (in_t[i]) {
                seen_t[i] = 1;
                T[nts++] = i;
            }
        for (Py_ssize_t j = 0; j < nd; j++)
            if (in_d[j])
                D[nds++] = j;
        if (nts == 1) {                                  /* max(ds, key=gain): the first largest */
            Py_ssize_t bj = D[0];
            for (Py_ssize_t b = 1; b < nds; b++)
                if (gain[T[0] * nd + D[b]] > gain[T[0] * nd + bj])
                    bj = D[b];
            pi[cnt] = T[0];
            pj[cnt] = bj;
            cnt++;
            continue;
        }
        if (nds == 1) {
            Py_ssize_t bi = T[0];
            for (Py_ssize_t a = 1; a < nts; a++)
                if (gain[T[a] * nd + D[0]] > gain[bi * nd + D[0]])
                    bi = T[a];
            pi[cnt] = bi;
            pj[cnt] = D[0];
            cnt++;
            continue;
        }
        /* every object may also stay without a detection (its own column, cost 0) */
        Py_ssize_t w = nds + nts;
        cst = (double *)PyMem_Malloc((size_t)(nts * w) * sizeof(double));
        hi = (Py_ssize_t *)PyMem_Malloc((size_t)(nts + w + 1) * 2 * sizeof(Py_ssize_t));
        if (cst == NULL || hi == NULL) {
            PyErr_NoMemory();
            cnt = -1;
            goto out;
        }
        hj = hi + nts + w + 1;
        for (Py_ssize_t a = 0; a < nts; a++) {
            for (Py_ssize_t b = 0; b < w; b++)
                cst[a * w + b] = INFINITY;
            for (Py_ssize_t b = 0; b < nds; b++) {
                double g = gain[T[a] * nd + D[b]];
                if (isfinite(g))
                    cst[a * w + b] = -g;
            }
            cst[a * w + nds + a] = 0.0;
        }
        Py_ssize_t h = r_hungarian(cst, nts, w, hi, hj);
        if (h < 0) {
            PyErr_NoMemory();
            cnt = -1;
            goto out;
        }
        for (Py_ssize_t q = 0; q < h; q++)
            if (hj[q] < nds && isfinite(cst[hi[q] * w + hj[q]])) {
                pi[cnt] = T[hi[q]];
                pj[cnt] = D[hj[q]];
                cnt++;
            }
        PyMem_Free(cst);
        PyMem_Free(hi);
        cst = NULL;
        hi = NULL;
    }
out:
    PyMem_Free(seen_t);
    PyMem_Free(in_t);
    PyMem_Free(stack);
    PyMem_Free(ts);
    PyMem_Free(cst);
    PyMem_Free(hi);
    return cnt;
}

/* ---- the update of a scan's pairs (Tracker._update, its arithmetic) ---- */
/* K = P H^T Si (N x m), from HP = H P (P symmetric is NOT assumed: P H^T computed as such) */
static void r_gain(const double *P, const double *H, int m, const double *Si, double *K)
{
    double PH[RN * 4];
    for (int i = 0; i < RN; i++)
        for (int l = 0; l < m; l++) {
            double acc = 0.0;
            for (int k = 0; k < RN; k++)
                acc = k ? acc + P[i * RN + k] * H[l * RN + k] : P[i * RN + k] * H[l * RN + k];
            PH[i * m + l] = acc;
        }
    for (int i = 0; i < RN; i++)
        for (int l = 0; l < m; l++) {
            double acc = 0.0;
            for (int k = 0; k < m; k++)
                acc = k ? acc + PH[i * m + k] * Si[k * m + l] : PH[i * m + k] * Si[k * m + l];
            K[i * m + l] = acc;
        }
}

/* One hypothesis (or the free estimate) takes the detection with weight beta: x, P in place. The covariance in
 * Joseph form: with A = I - K H, the detection taken in full leaves A P A^T + K R K^T (a sum of two positive
 * semi-definite terms for ANY gain, so rounding in K or S cannot make it indefinite; P - K S K^T subtracts
 * nearly equal numbers where a detection takes most of a variance away), and taken with weight beta
 * (1 - beta) P + beta (A P A^T + K R K^T) + beta (1 - beta) dx dx^T: every term positive semi-definite, the
 * result symmetric by the way it is written out (the lower triangle is the upper one's mirror).
 * H: (m x N); R: the detection's variances (the diagonal); nu: the innovation. */
static void r_take(double *x, double *P, const double *K, const double *H, const double *R, const double *nu, int m,
                   double beta)
{
    double dx[RN], A[RNN], AP[RNN], Pn[RNN];
    for (int i = 0; i < RN; i++) {
        double acc = 0.0;
        for (int k = 0; k < m; k++)
            acc += K[i * m + k] * nu[k];
        dx[i] = acc;
    }
    for (int i = 0; i < RN; i++)                         /* A = I - K H */
        for (int j = 0; j < RN; j++) {
            double acc = 0.0;
            for (int k = 0; k < m; k++)
                acc += K[i * m + k] * H[k * RN + j];
            A[i * RN + j] = (i == j ? 1.0 : 0.0) - acc;
        }
    for (int i = 0; i < RN; i++)                         /* AP = A P */
        for (int j = 0; j < RN; j++) {
            double acc = 0.0;
            for (int k = 0; k < RN; k++)
                acc += A[i * RN + k] * P[k * RN + j];
            AP[i * RN + j] = acc;
        }
    double bb = beta * (1.0 - beta), keep = 1.0 - beta;
    for (int i = 0; i < RN; i++)
        for (int j = i; j < RN; j++) {
            double acc = 0.0, kr = 0.0;
            for (int k = 0; k < RN; k++)                 /* (A P A^T)[i][j] */
                acc += AP[i * RN + k] * A[j * RN + k];
            for (int k = 0; k < m; k++)                  /* (K R K^T)[i][j] */
                kr += K[i * m + k] * R[k] * K[j * m + k];
            /* P is taken as its symmetric part (it is symmetric but for rounding) */
            double v = keep * (0.5 * (P[i * RN + j] + P[j * RN + i])) + beta * (acc + kr) + bb * (dx[i] * dx[j]);
            Pn[i * RN + j] = v;
            Pn[j * RN + i] = v;
        }
    for (int i = 0; i < RN; i++)
        x[i] = x[i] + beta * dx[i];
    memcpy(P, Pn, sizeof(Pn));
}

/* nu = z - h(x) wrapped, S = H P H^T + diag(R); returns m */
static int r_innov(const RSens *s, int kind, double sdt, const double *x, const double *P, const double *pose,
                   const double *z, const double *R, double *nu, double *H, double *S, const double *e)
{
    double zp[4], HP[4 * RN];
    int m = r_h(s, kind, sdt, x, pose, zp, H, e);
    int ar = r_angle_row(s, kind);
    for (int r = 0; r < m; r++)
        nu[r] = r == ar ? wrap_pi(z[r] - zp[r]) : z[r] - zp[r];
    r_hph(H, m, P, HP, S);
    for (int r = 0; r < m; r++)
        S[r * m + r] = S[r * m + r] + R[r];
    return m;
}

/* ---- the motion model on the objects' rows ---- */
/* one object's hypotheses and free estimate after dt (Tracker._evolve): into xs1 (3N), Ps1 (3NN), c (3), and
 * the free estimate into xf1, Pf1 when given */
static void r_evolve_row(const double *row, double dt, const double *K, const RPar *par, double *xs1, double *Ps1,
                         double *c, double *xf1, double *Pf1, double *Tout)
{
    double Tl[9], *T = Tout ? Tout : Tl, xm[3 * RN], Pm[3 * RNN];
    const double *X = row + RO_XS, *P = row + RO_PS;
    double q0 = r_still(X + RN, P + RNN, K[3]), q1 = r_still(X + 2 * RN, P + 2 * RNN, K[3]);
    r_transition(dt, q0, q1, par, T);
    r_mix(X, P, row + RO_MU, T, row + RO_XF, row + RO_PF, xm, Pm, c);
    for (int j = 0; j < 3; j++)
        r_predict(j, xm + j * RN, Pm + j * RNN, dt, K[0], K[1], K[2], par, xs1 + j * RN, Ps1 + j * RNN);
    if (xf1)
        r_predict(R_CV, row + RO_XF, row + RO_PF, dt, K[0], K[1], K[2], par, xf1, Pf1);
}

/* where an object will be after dt, in part: the combination's x[0..6] (place, velocity, acceleration, offset)
 * and the top-left nb x nb block of its covariance (nb 2: the place, 4: place and velocity), the same bits as
 * r_evolve_row + r_combine give for those entries: only what they depend on is computed (the mixing of indices
 * 0..6, the transition's rows 0..nb-1 and 4..6), each entry the same way. Out: x (7), P (nb x nb, row-major),
 * the predicted probabilities (3) where muout. */
static void r_peek_part(const double *row, double dt, const double *K, const RPar *par, int nb, double *xout,
                        double *Pout, double *muout)
{
    double T[9], c[3], w[9];
    const double *X = row + RO_XS, *P = row + RO_PS, *mu = row + RO_MU, *xf = row + RO_XF, *Pf = row + RO_PF;
    double q0 = r_still(X + RN, P + RNN, K[3]), q1 = r_still(X + 2 * RN, P + 2 * RNN, K[3]);
    r_transition(dt, q0, q1, par, T);
    for (int j = 0; j < 3; j++)
        c[j] = (T[j] * mu[0] + T[3 + j] * mu[1]) + T[6 + j] * mu[2];
    for (int j = 0; j < 3; j++) {
        int low = c[j] < 1e-12;
        for (int i = 0; i < 3; i++)
            w[3 * i + j] = low ? (i == j ? 1.0 : 0.0) : T[3 * i + j] * mu[i] / c[j];
    }
    double adt = fabs(dt), x1[3][7], P1[3][16];
    for (int j = 0; j < 3; j++) {
        /* the mixing for target j, indices 0..6 (x) and 0..5 (P: all that rows 0..3 of F reach) */
        const double *sx[3], *sp[3];
        for (int i = 0; i < 3; i++) {
            int hand = (i == R_STAND && j != R_STAND);
            sx[i] = hand ? xf : X + i * RN;
            sp[i] = hand ? Pf : P + i * RNN;
        }
        double w0 = w[j], w1 = w[3 + j], w2 = w[6 + j], xm[7], d0[7], d1[7], d2[7], Pm[6][6];
        for (int n = 0; n < 7; n++)
            xm[n] = (w0 * sx[0][n] + w1 * sx[1][n]) + w2 * sx[2][n];
        for (int n = 0; n < 6; n++) {
            d0[n] = sx[0][n] - xm[n];
            d1[n] = sx[1][n] - xm[n];
            d2[n] = sx[2][n] - xm[n];
        }
        for (int m = 0; m < 6; m++)
            for (int n = m; n < 6; n++) {
                int k = m * RN + n;
                double a = (w0 * sp[0][k] + w1 * sp[1][k]) + w2 * sp[2][k];
                double b = ((w0 * d0[m]) * d0[n] + (w1 * d1[m]) * d1[n]) + (w2 * d2[m]) * d2[n];
                Pm[m][n] = Pm[n][m] = a + b;
            }
        /* r_predict: the transition's non-zero terms of rows 0..6, Q's top-left 4 x 4 */
        int nz[7], col[7][3];
        double val[7][3], Q[4][4] = {{0}};
        for (int i = 0; i < 7; i++) {
            nz[i] = 1;
            col[i][0] = i;
            val[i][0] = 1.0;
        }
        double s[3] = {0.0, 0.0, 0.0}, xr[RN] = {0};
        xr[R_VX] = xm[R_VX];
        xr[R_VY] = xm[R_VY];
        if (j == R_STAND) {
            for (int i = R_VX; i <= R_AY; i++)
                nz[i] = 0;
            Q[0][0] = Q[1][1] = par->sd2 * adt;
            Q[2][2] = Q[3][3] = par->sv2;
        } else if (j == R_CV) {
            double dt2 = adt * adt, dt3 = pow(adt, 3.0), t1 = par->cvn * K[0], t2 = par->cvn * K[1];
            for (int i = 0; i < 2; i++) {
                nz[R_PX + i] = 2;
                col[R_PX + i][1] = R_VX + i;
                val[R_PX + i][1] = dt;
            }
            nz[R_AX] = nz[R_AY] = 0;
            r_shape(t1 * t1, t2 * t2, xr, s);
            double ks[3] = {dt3 / 3.0, dt2 / 2.0, adt};          /* blocks (0,0), (0,2), (2,2) */
            for (int bl = 0; bl < 3; bl++) {
                int r0 = bl == 2 ? 2 : 0, c0 = bl == 0 ? 0 : 2;
                double aa = s[0] * ks[bl], bb = s[1] * ks[bl], oo = s[2] * ks[bl];
                Q[r0][c0] = Q[c0][r0] = aa;
                Q[r0 + 1][c0 + 1] = Q[c0 + 1][r0 + 1] = bb;
                Q[r0][c0 + 1] = Q[r0 + 1][c0] = Q[c0 + 1][r0] = Q[c0][r0 + 1] = oo;
            }
        } else {
            double rho = exp(-adt / par->tau), dt2 = adt * adt, dt3 = pow(adt, 3.0), dt4 = pow(adt, 4.0),
                   dt5 = pow(adt, 5.0), t1 = par->caj * K[0], t2 = par->caj * K[1];
            for (int i = 0; i < 2; i++) {
                nz[R_PX + i] = 3;
                col[R_PX + i][1] = R_VX + i;
                val[R_PX + i][1] = dt;
                col[R_PX + i][2] = R_AX + i;
                val[R_PX + i][2] = 0.5 * dt * dt;
                nz[R_VX + i] = 2;
                col[R_VX + i][1] = R_AX + i;
                val[R_VX + i][1] = dt;
                val[R_AX + i][0] = rho;
            }
            r_shape(t1 * t1, t2 * t2, xr, s);
            double ks[3] = {dt5 / 20.0, dt4 / 8.0, dt3 / 3.0};
            for (int bl = 0; bl < 3; bl++) {
                int r0 = bl == 2 ? 2 : 0, c0 = bl == 0 ? 0 : 2;
                double aa = s[0] * ks[bl], bb = s[1] * ks[bl], oo = s[2] * ks[bl];
                Q[r0][c0] = Q[c0][r0] = aa;
                Q[r0 + 1][c0 + 1] = Q[c0 + 1][r0 + 1] = bb;
                Q[r0][c0 + 1] = Q[r0 + 1][c0] = Q[c0 + 1][r0] = Q[c0][r0 + 1] = oo;
            }
            (void)dt2;
        }
        for (int i = 0; i < 7; i++) {
            int kk = 0;
            for (int e = 0; e < nz[i]; e++)
                if (val[i][e] != 0.0) {
                    col[i][kk] = col[i][e];
                    val[i][kk] = val[i][e];
                    kk++;
                }
            nz[i] = kk;
        }
        double A[4][6], Pr[4][4];
        for (int i = 0; i < 7; i++) {
            double acc = 0.0;
            for (int e = 0; e < nz[i]; e++)
                acc = e ? acc + val[i][e] * xm[col[i][e]] : val[i][e] * xm[col[i][e]];
            x1[j][i] = acc;
        }
        for (int i = 0; i < nb; i++)
            for (int q = 0; q < 6; q++) {
                double acc = 0.0;
                for (int e = 0; e < nz[i]; e++)
                    acc = e ? acc + val[i][e] * Pm[col[i][e]][q] : val[i][e] * Pm[col[i][e]][q];
                A[i][q] = acc;
            }
        for (int i = 0; i < nb; i++)
            for (int l = 0; l < nb; l++) {
                double acc = 0.0;
                for (int e = 0; e < nz[l]; e++)
                    acc = e ? acc + A[i][col[l][e]] * val[l][e] : A[i][col[l][e]] * val[l][e];
                Pr[i][l] = acc + Q[i][l];
            }
        if (j != R_STAND) {                              /* the kind's top speed */
            double sp = hypot(x1[j][R_VX], x1[j][R_VY]);
            if (sp > K[2]) {
                double f = K[2] / sp, dd = sp - K[2], ex = dd * dd;
                x1[j][R_VX] *= f;
                x1[j][R_VY] *= f;
                if (nb > 2) {
                    Pr[R_VX][R_VX] += ex;
                    Pr[R_VY][R_VY] += ex;
                }
            }
        }
        for (int i = 0; i < nb; i++)
            for (int l = 0; l < nb; l++)
                P1[j][i * nb + l] = 0.5 * (Pr[i][l] + Pr[l][i]);
    }
    /* r_combine of those entries, with the predicted probabilities */
    double d[3][7];
    for (int n = 0; n < 7; n++)
        xout[n] = (c[0] * x1[0][n] + c[1] * x1[1][n]) + c[2] * x1[2][n];
    for (int i = 0; i < 3; i++)
        for (int n = 0; n < nb; n++)
            d[i][n] = x1[i][n] - xout[n];
    for (int m = 0; m < nb; m++)
        for (int n = m; n < nb; n++) {
            int k = nb * m + n;
            double a = (c[0] * P1[0][k] + c[1] * P1[1][k]) + c[2] * P1[2][k];
            double b = ((c[0] * d[0][m]) * d[0][n] + (c[1] * d[1][m]) * d[1][n]) + (c[2] * d[2][m]) * d[2][n];
            Pout[k] = Pout[nb * n + m] = a + b;
        }
    if (muout)
        memcpy(muout, c, sizeof(c));
}

/* r_peek_part for the place: (x, y, P00, P01, P10, P11) */
static void r_peek_pos(const double *row, double dt, const double *K, const RPar *par, double *out)
{
    double x[7], P[4];
    r_peek_part(row, dt, K, par, 2, x, P, NULL);
    out[0] = x[0];
    out[1] = x[1];
    memcpy(out + 2, P, sizeof(P));
}

/* radar_peek(S, rows, dt, kin, par, x_out, P_out, mu_out) -> None. Tracker.peek_many: where each object is
 * believed to be after its dt (an object whose |dt| < 1e-6: its row as it is), nothing committed */
static PyObject *n_radar_peek(PyObject *self, PyObject *args)
{
    PyObject *oS, *orows, *odt, *okin, *opar, *o1, *o2, *o3;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &oS, &orows, &odt, &okin, &opar, &o1, &o2, &o3))
        return NULL;
    PyObject *res = NULL;
    View vS, v1, v2, v3;
    vS.held = v1.held = v2.held = v3.held = 0;
    Py_ssize_t cap, B = 0, *rows = NULL;
    double *dk = NULL, *xo, *Po, *muo;
    RPar par;
    if (store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &B)) == NULL || rpar_get(opar, &par) < 0)
        goto done;
    dk = (double *)PyMem_Malloc((size_t)(B ? B : 1) * 5 * sizeof(double));
    if (dk == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    if (rows_get(odt, B, 1, dk, "dt") < 0 || rows_get(okin, B, 4, dk + B, "kin") < 0 ||
        out_get(o1, &v1, B * RN, 0, &xo, "x_out") < 0 || out_get(o2, &v2, B * RNN, 0, &Po, "P_out") < 0 ||
        out_get(o3, &v3, B * 3, 0, &muo, "mu_out") < 0)
        goto done;
    double *A = (double *)vS.b.buf;
    for (Py_ssize_t b = 0; b < B; b++) {
        const double *row = A + rows[b] * RROW;
        double dt = dk[b];
        if (fabs(dt) < 1e-6) {
            memcpy(xo + b * RN, row + RO_X, RN * sizeof(double));
            memcpy(Po + b * RNN, row + RO_P, RNN * sizeof(double));
            memcpy(muo + 3 * b, row + RO_MU, 3 * sizeof(double));
            continue;
        }
        double xs1[3 * RN], Ps1[3 * RNN];
        r_evolve_row(row, dt, dk + B + 4 * b, &par, xs1, Ps1, muo + 3 * b, NULL, NULL, NULL);
        r_combine(xs1, Ps1, muo + 3 * b, xo + b * RN, Po + b * RNN);
    }
    res = Py_None;
    Py_INCREF(res);
done:
    PyMem_Free(rows);
    PyMem_Free(dk);
    view_drop(&vS); view_drop(&v1); view_drop(&v2); view_drop(&v3);
    return res;
}

/* radar_peek_models(S, row, dt, kin, par, xs_out (3, N), Ps_out (3, N, N), mu_out (3)) -> None: one row's motion
 * hypotheses after dt, each by itself (the interaction and the prediction of a scan's step), nothing committed
 * (for tools: what each hypothesis expects of a detection) */
static PyObject *n_radar_peek_models(PyObject *self, PyObject *args)
{
    PyObject *oS, *okin, *opar, *o1, *o2, *o3;
    Py_ssize_t slot;
    double dt;
    if (!PyArg_ParseTuple(args, "OndOOOOO", &oS, &slot, &dt, &okin, &opar, &o1, &o2, &o3))
        return NULL;
    PyObject *res = NULL;
    View vS, v1, v2, v3;
    vS.held = v1.held = v2.held = v3.held = 0;
    Py_ssize_t cap;
    double kin[4], *xo, *Po, *muo;
    RPar par;
    if (store_get(oS, &vS, &cap) < 0 || rpar_get(opar, &par) < 0 || rnums_get(okin, 4, kin, "kin") < 0 ||
        out_get(o1, &v1, 3 * RN, 0, &xo, "xs_out") < 0 || out_get(o2, &v2, 3 * RNN, 0, &Po, "Ps_out") < 0 ||
        out_get(o3, &v3, 3, 0, &muo, "mu_out") < 0)
        goto done;
    if (slot < 0 || slot >= cap) {
        PyErr_SetString(PyExc_IndexError, "radar: a row outside the state array");
        goto done;
    }
    {
        const double *row = (const double *)vS.b.buf + slot * RROW;
        if (fabs(dt) < 1e-6) {
            memcpy(xo, row + RO_XS, 3 * RN * sizeof(double));
            memcpy(Po, row + RO_PS, 3 * RNN * sizeof(double));
            memcpy(muo, row + RO_MU, 3 * sizeof(double));
        } else {
            double c[3];
            r_evolve_row(row, dt, kin, &par, xo, Po, c, NULL, NULL, NULL);
            memcpy(muo, dt > 0.0 ? c : row + RO_MU, 3 * sizeof(double));
        }
    }
    res = Py_None;
    Py_INCREF(res);
done:
    view_drop(&vS); view_drop(&v1); view_drop(&v2); view_drop(&v3);
    return res;
}

/* radar_combine_rows(S, rows) -> None: each object's combination (x, P) from its hypotheses and their
 * probabilities, in its row */
static PyObject *n_radar_combine_rows(PyObject *self, PyObject *args)
{
    PyObject *oS, *orows;
    if (!PyArg_ParseTuple(args, "OO", &oS, &orows))
        return NULL;
    View vS;
    Py_ssize_t cap, B = 0, *rows = NULL;
    if (store_get(oS, &vS, &cap) < 0)
        return NULL;
    PyObject *res = NULL;
    if ((rows = idx_get(orows, cap, &B)) != NULL) {
        double *A = (double *)vS.b.buf;
        for (Py_ssize_t b = 0; b < B; b++) {
            double *row = A + rows[b] * RROW;
            r_combine(row + RO_XS, row + RO_PS, row + RO_MU, row + RO_X, row + RO_P);
        }
        res = Py_None;
        Py_INCREF(res);
    }
    PyMem_Free(rows);
    view_drop(&vS);
    return res;
}

/* radar_predict1(model, x, P, dt, kin, par, x_out, P_out) -> None: one hypothesis forward by dt (no interaction):
 * models.predict */
static PyObject *n_radar_predict1(PyObject *self, PyObject *args)
{
    int model;
    double dt, x[RN], P[RNN], kin[4], *xo, *Po;
    PyObject *ox, *oP, *okin, *opar, *o1, *o2;
    if (!PyArg_ParseTuple(args, "iOOdOOOO", &model, &ox, &oP, &dt, &okin, &opar, &o1, &o2))
        return NULL;
    RPar par;
    View v1, v2;
    v1.held = v2.held = 0;
    PyObject *res = NULL;
    if (model < 0 || model > 2) {
        PyErr_SetString(PyExc_ValueError, "radar_predict1: the model is 0, 1 or 2");
        return NULL;
    }
    if (rnums_get(ox, RN, x, "x") < 0 || rnums_get(oP, RNN, P, "P") < 0 || rnums_get(okin, 4, kin, "kin") < 0 ||
        rpar_get(opar, &par) < 0 || out_get(o1, &v1, RN, 0, &xo, "x_out") < 0 || out_get(o2, &v2, RNN, 0, &Po, "P_out") < 0)
        goto done;
    r_predict(model, x, P, dt, kin[0], kin[1], kin[2], &par, xo, Po);
    res = Py_None;
    Py_INCREF(res);
done:
    view_drop(&v1);
    view_drop(&v2);
    return res;
}

/* radar_still1(x, P, v0) -> how much a hypothesis says "it does not move" (models.still) */
static PyObject *n_radar_still1(PyObject *self, PyObject *args)
{
    PyObject *ox, *oP;
    double v0, x[RN], P[RNN];
    if (!PyArg_ParseTuple(args, "OOd", &ox, &oP, &v0) || rnums_get(ox, RN, x, "x") < 0 || rnums_get(oP, RNN, P, "P") < 0)
        return NULL;
    return PyFloat_FromDouble(r_still(x, P, v0));
}

/* radar_transition1(dt, q0, q1, par) -> [[...] x 3]: the switching probabilities (models.transition) */
static PyObject *n_radar_transition1(PyObject *self, PyObject *args)
{
    double dt, q0, q1, T[9];
    PyObject *opar;
    RPar par;
    if (!PyArg_ParseTuple(args, "dddO", &dt, &q0, &q1, &opar) || rpar_get(opar, &par) < 0)
        return NULL;
    r_transition(dt, q0, q1, &par, T);
    return Py_BuildValue("[[ddd][ddd][ddd]]", T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7], T[8]);
}

/* radar_h(sensor, kind, sdt, x, pose, z_out (4), H_out (4 x N)) -> m: sensor.h */
static PyObject *n_radar_h(PyObject *self, PyObject *args)
{
    PyObject *osen, *ox, *opose, *oz, *oH;
    int kind;
    double sdt, pose[6], x[RN], *z, *H;
    if (!PyArg_ParseTuple(args, "OidOOOO", &osen, &kind, &sdt, &ox, &opose, &oz, &oH))
        return NULL;
    RSens sen;
    View vz, vH;
    vz.held = vH.held = 0;
    PyObject *res = NULL;
    if (sens_get(osen, &sen) < 0 || rnums_get(ox, RN, x, "x") < 0 || rnums_get(opose, 6, pose, "pose") < 0 ||
        out_get(oz, &vz, 4, 0, &z, "z_out") < 0 || out_get(oH, &vH, 4 * RN, 0, &H, "H_out") < 0)
        goto done;
    res = PyLong_FromLong(r_h(&sen, kind, sdt, x, pose, z, H, NULL));
done:
    view_drop(&vz);
    view_drop(&vH);
    return res;
}

/* radar_marginals(ratio (nt, nd), out (nt, nd)) -> None: tracker.marginals */
static PyObject *n_radar_marginals(PyObject *self, PyObject *args)
{
    PyObject *oL, *oo;
    if (!PyArg_ParseTuple(args, "OO", &oL, &oo))
        return NULL;
    View vL, vo;
    vL.held = vo.held = 0;
    PyObject *res = NULL;
    if (view_get(oL, &vL, 0, 'd', 2, "ratio") < 0)
        goto done;
    Py_ssize_t nt = vL.b.shape[0], nd = vL.b.shape[1];
    double *out;
    if (out_get(oo, &vo, nt * nd, 0, &out, "out") < 0 || r_marginals(DATA(vL), nt, nd, out) < 0)
        goto done;
    res = Py_None;
    Py_INCREF(res);
done:
    view_drop(&vL);
    view_drop(&vo);
    return res;
}

/* radar_assign(gain (nt, nd)) -> [(i, j)]: tracker.assign */
static PyObject *n_radar_assign(PyObject *self, PyObject *og)
{
    View vg;
    vg.held = 0;
    PyObject *res = NULL;
    Py_ssize_t *pi = NULL;
    if (view_get(og, &vg, 0, 'd', 2, "gain") < 0)
        goto done;
    Py_ssize_t nt = vg.b.shape[0], nd = vg.b.shape[1];
    pi = (Py_ssize_t *)PyMem_Malloc((size_t)(2 * (nt + nd) + 2) * sizeof(Py_ssize_t));
    if (pi == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    Py_ssize_t n = r_assign(DATA(vg), nt, nd, pi, pi + nt + nd + 1);
    if (n < 0)
        goto done;
    res = PyList_New(n);
    for (Py_ssize_t q = 0; res && q < n; q++) {
        PyObject *t = Py_BuildValue("(nn)", pi[q], pi[nt + nd + 1 + q]);
        if (t == NULL) {
            Py_CLEAR(res);
            break;
        }
        PyList_SetItem(res, q, t);
    }
done:
    PyMem_Free(pi);
    view_drop(&vg);
    return res;
}

/* ---- the output (box.py `Radar.frame`): every listed object's numbers in the car frame of the message ---- */
/* builtins.round(x, n) as a float object (the native `round`) */
static PyObject *round_obj(double x, int n)
{
    int ok;
    double r = round_fast(x, n, &ok);
    if (ok && isfinite(x))
        return PyFloat_FromDouble(r);
    PyObject *f = PyFloat_FromDouble(x);
    if (f == NULL)
        return NULL;
    PyObject *res = PyObject_CallFunction(builtin_round, "Oi", f, n);
    Py_DECREF(f);
    return res;
}

/* np.interp(q, xs, ys) for one point (xs not decreasing) */
static double np_interp1(double q, const double *xs, const double *ys, Py_ssize_t n)
{
    if (isnan(q))
        return q;
    if (q < xs[0])
        return ys[0];
    if (q >= xs[n - 1])
        return q > xs[n - 1] ? ys[n - 1] : ys[n - 1];
    Py_ssize_t lo = 0, hi = n - 1;                  /* xs[lo] <= q < xs[hi] */
    while (hi - lo > 1) {
        Py_ssize_t mid = (lo + hi) / 2;
        if (xs[mid] <= q)
            lo = mid;
        else
            hi = mid;
    }
    double slope = (ys[lo + 1] - ys[lo]) / (xs[lo + 1] - xs[lo]);
    double r = slope * (q - xs[lo]) + ys[lo];
    if (isnan(r)) {
        r = slope * (q - xs[lo + 1]) + ys[lo + 1];
        if (isnan(r) && ys[lo] == ys[lo + 1])
            r = ys[lo];
    }
    return r;
}

typedef struct {
    const double *xs, *ys;
    Py_ssize_t n;
    double v, w;
} RPath;

/* Radar._offset: the lateral offset of a car-frame point from the path */
static double r_offset(const RPath *p, double x, double y)
{
    if (p->n && x <= p->xs[p->n - 1] + 5.0)
        return y - np_interp1(x, p->xs, p->ys, p->n);
    double k = p->v > 2.0 ? p->w / p->v : 0.0;
    return y - 0.5 * k * x * x;
}

/* Radar._path_point: (x, y, direction) of the path at distance s ahead */
static void r_path_point(const RPath *p, double s, double *out)
{
    if (p->n && s <= p->xs[p->n - 1]) {
        double y = np_interp1(s, p->xs, p->ys, p->n), d = 1.0;
        out[0] = s;
        out[1] = y;
        out[2] = atan2(np_interp1(s + d, p->xs, p->ys, p->n) - np_interp1(s - d, p->xs, p->ys, p->n), 2.0 * d);
        return;
    }
    double k = p->v > 2.0 ? p->w / p->v : 0.0;
    if (fabs(k) < 1e-6) {
        out[0] = s;
        out[1] = 0.0;
        out[2] = 0.0;
        return;
    }
    out[0] = sin(k * s) / k;
    out[1] = (1.0 - cos(k * s)) / k;
    out[2] = k * s;
}

/* the keys of the message's object (box.py), interned once */
enum { RK_TS, RK_XM, RK_YM, RK_STDM, RK_STAND, RK_CV, RK_CA, RK_ID, RK_STATE, RK_EXIST, RK_AGE, RK_HITS, RK_SOURCES,
       RK_SEEN, RK_VX, RK_VY, RK_AX, RK_AY, RK_SPEED, RK_HEADING, RK_VREL, RK_STD, RK_MOTION, RK_PST, RK_MODELS,
       RK_CLASS, RK_FLAGS, RK_POFF, RK_TTC, RK_PTHR, RK_CONS, RK_PERR, RK_ANG, RK_PRED, RK_PPATH, RK_LABEL, RK_CONF,
       RK_SOURCE, RK_VOTES, RK_ON_PATH, RK_IN_PATH, RK_THREAT, RK_PARKED, RK_WITH, RK_STATIONARY, RK_HUMAN, RK_BIKE,
       RK_MOTORBIKE, RK_LEN, RK_WID, RK_HDG, RK_EXTSTD, RK_VIS, RK_OCC, RK_OCCBY, RK_OBJECT, RK_EDGE, RK_FOV, RK_VS,
       RK_VSSTD, RK_VSAGE, RK_GRP, RK_GRPKIND, RK_HYP, RK_CTR, RK_N };
static PyObject *rkey[RK_N + 1];
static const char *const RKEYS[RK_N + 1] = {
    "t_s", "x_m", "y_m", "std_m", "stand", "cv", "ca", "id", "state", "exist", "age_s", "hits", "sources", "seen",
    "vx_mps", "vy_mps", "ax_mps2", "ay_mps2", "speed_mps", "heading_rad", "v_rel_mps", "std", "motion",
    "p_stationary", "models", "class", "flags", "path_offset_m", "ttc_s", "p_threat", "consistency", "pred_err_m",
    "angle_offset_deg", "predict", "predict_path", "label", "confidence", "source", "votes", "on_path", "in_path",
    "threat", "parked", "with_traffic", "stationary", "human", "bike", "motorbike", "len", "wid", "hdg", "ext_std",
    "vis", "occ", "occ_by", "object", "edge", "fov", "vs", "vs_std", "vs_age_s", "grp", "grp_kind", "hyp", "ctr", NULL};

static int rkeys_init(void)
{
    for (int k = 0; RKEYS[k]; k++)
        if (rkey[k] == NULL && (rkey[k] = PyUnicode_InternFromString(RKEYS[k])) == NULL)
            return -1;
    return 0;
}

/* d[key] = value (stolen); -1 on error */
static int dset(PyObject *d, int key, PyObject *v)
{
    if (v == NULL)
        return -1;
    int r = PyDict_SetItem(d, rkey[key], v);
    Py_DECREF(v);
    return r;
}

/* d[rkey[key]] = v (borrowed); -1 on error */
static int dput(PyObject *d, int key, PyObject *v)
{
    return v == NULL ? -1 : PyDict_SetItem(d, rkey[key], v);
}

/* round(x, n) of a number object, or None for None */
static PyObject *round_or_none(PyObject *o, int n)
{
    if (o == Py_None) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    double v = PyFloat_AsDouble(o);
    if (v == -1.0 && PyErr_Occurred())
        return NULL;
    return round_obj(v, n);
}

/* a sorted list of a set's / dict's items */
static PyObject *sorted_list(PyObject *o)
{
    PyObject *l = PySequence_List(o);
    if (l != NULL && PyList_Sort(l) < 0)
        Py_CLEAR(l);
    return l;
}

#define RFLAGS 5                 /* in_path_hold_s, threat_p, threat_range_s, threat_hold_s, parked_offset_m */
#define RATTR 14                 /* id, state, logodds, t0, hits, sources, last_seen, verdict, flags, label, range_t,
                                    consistency, pred_err, class (classvote.result or None) */

/* the message's object (box.py `Radar.frame`) from radar_frame's numbers t and the object's attributes a; the
 * flags with their hysteresis (hin, hthr: the box's _in_since, _threat_since, updated here) */
static PyObject *r_obj_dict(PyObject *t, PyObject *a, PyObject *hin, PyObject *hthr, double T, const double *fc)
{
    if (!PyTuple_Check(a) || PyTuple_Size(a) != RATTR) {
        PyErr_SetString(PyExc_TypeError, "radar_frame: an object's attributes are a tuple of 14");
        return NULL;
    }
#define TI(o, k) PyTuple_GetItem(o, k)
    PyObject *id = TI(a, 0), *verdict = TI(a, 7), *label = TI(a, 9), *cls = TI(a, 13), *along = TI(t, 20);
    double logodds = PyFloat_AsDouble(TI(a, 2)), t0 = PyFloat_AsDouble(TI(a, 3)), range_t = PyFloat_AsDouble(TI(a, 10));
    double off = PyFloat_AsDouble(TI(t, 14)), p_thr = PyFloat_AsDouble(TI(t, 18));
    int on = TI(t, 16) == Py_True, sure_in = TI(t, 17) == Py_True;
    if (PyErr_Occurred())
        return NULL;
    PyObject *d = NULL, *flags = NULL, *fl = NULL, *tmp = NULL, *Tobj = NULL;
    int stationary = PyObject_RichCompareBool(verdict, rkey[RK_STATIONARY], Py_EQ);
    if (stationary < 0 || (flags = PySet_New(TI(a, 8))) == NULL || (Tobj = PyFloat_FromDouble(T)) == NULL)
        goto fail;
    if (on && PySet_Add(flags, rkey[RK_ON_PATH]) < 0)
        goto fail;
    /* "it stands in the way": inside the corridor beyond the place's own uncertainty, and stayed there */
    PyObject *hold = PyDict_GetItemWithError(hin, id);
    if (hold == NULL && PyErr_Occurred())
        goto fail;
    if (stationary && sure_in) {
        double hv = T;
        if (hold == NULL) {
            if (PyDict_SetItem(hin, id, Tobj) < 0)
                goto fail;
        } else
            hv = PyFloat_AsDouble(hold);
        if (T - hv >= fc[0] && PySet_Add(flags, rkey[RK_IN_PATH]) < 0)
            goto fail;
    } else if (hold != NULL && PyDict_DelItem(hin, id) < 0)
        goto fail;
    /* a threat is not started on an object whose distance has lately come from cut boxes only */
    PyObject *th0 = PyDict_GetItemWithError(hthr, id);
    if (th0 == NULL && PyErr_Occurred())
        goto fail;
    if (p_thr >= fc[1] && (th0 != NULL || T - range_t <= fc[2])) {
        double tv = T;
        if (th0 == NULL) {
            if (PyDict_SetItem(hthr, id, Tobj) < 0)
                goto fail;
        } else
            tv = PyFloat_AsDouble(th0);
        if (T - tv >= fc[3] && PySet_Add(flags, rkey[RK_THREAT]) < 0)
            goto fail;
    } else if (th0 != NULL && PyDict_DelItem(hthr, id) < 0)
        goto fail;
    if (stationary && !on && fabs(off) > fc[4]) {
        int person = 0;
        for (int k = RK_HUMAN; k <= RK_MOTORBIKE && !person; k++) {
            int e = PyObject_RichCompareBool(label, rkey[k], Py_EQ);
            if (e < 0)
                goto fail;
            person = e;
        }
        if (!person && PySet_Add(flags, rkey[RK_PARKED]) < 0)
            goto fail;
    }
    if (along != Py_None && PySet_Add(flags, rkey[RK_WITH]) < 0)
        goto fail;
    if ((fl = sorted_list(flags)) == NULL || (d = PyDict_New()) == NULL)
        goto fail;
    if (dput(d, RK_ID, id) < 0 || dput(d, RK_STATE, TI(a, 1)) < 0)
        goto fail;
    tmp = round_obj(1.0 / (1.0 + exp(-logodds)), 3);
    if (dput(d, RK_EXIST, tmp) < 0)
        goto fail;
    Py_CLEAR(tmp);
    tmp = round_obj(T - t0, 2);
    if (dput(d, RK_AGE, tmp) < 0 || dput(d, RK_HITS, TI(a, 4)) < 0)
        goto fail;
    Py_CLEAR(tmp);
    tmp = sorted_list(TI(a, 5));
    if (dput(d, RK_SOURCES, tmp) < 0)
        goto fail;
    Py_CLEAR(tmp);
    {
        PyObject *seen = PyDict_New(), *k, *v;
        Py_ssize_t pos = 0;
        if (seen == NULL)
            goto fail;
        tmp = seen;
        while (PyDict_Next(TI(a, 6), &pos, &k, &v)) {
            double tv = PyFloat_AsDouble(v);
            if (tv == -1.0 && PyErr_Occurred())
                goto fail;
            PyObject *r = round_obj(pymax(0.0, T - tv), 3);
            if (r == NULL || PyDict_SetItem(seen, k, r) < 0) {
                Py_XDECREF(r);
                goto fail;
            }
            Py_DECREF(r);
        }
        if (dput(d, RK_SEEN, seen) < 0)
            goto fail;
        Py_CLEAR(tmp);
    }
    static const int num_keys[9] = {RK_XM, RK_YM, RK_VX, RK_VY, RK_AX, RK_AY, RK_SPEED, RK_HEADING, RK_VREL};
    for (int k = 0; k < 9; k++)
        if (dput(d, num_keys[k], TI(t, k)) < 0)
            goto fail;
    if (dput(d, RK_STD, TI(t, 9)) < 0 || dput(d, RK_MOTION, verdict) < 0 || dput(d, RK_PST, TI(t, 11)) < 0 ||
        dput(d, RK_MODELS, TI(t, 10)) < 0)
        goto fail;
    if (cls == Py_None) {
        if (dput(d, RK_CLASS, Py_None) < 0)
            goto fail;
    } else {
        PyObject *c = PyDict_New();
        if (c == NULL)
            goto fail;
        tmp = c;
        for (int k = RK_LABEL; k <= RK_VOTES; k++) {
            PyObject *v = PyObject_GetItem(cls, rkey[k]);
            if (v == NULL || PyDict_SetItem(c, rkey[k], v) < 0) {
                Py_XDECREF(v);
                goto fail;
            }
            Py_DECREF(v);
        }
        if (dput(d, RK_CLASS, c) < 0)
            goto fail;
        Py_CLEAR(tmp);
    }
    if (dput(d, RK_FLAGS, fl) < 0 || dput(d, RK_POFF, TI(t, 15)) < 0 || dput(d, RK_TTC, TI(t, 19)) < 0)
        goto fail;
    tmp = round_obj(p_thr, 3);
    if (dput(d, RK_PTHR, tmp) < 0)
        goto fail;
    Py_CLEAR(tmp);
    tmp = round_or_none(TI(a, 11), 2);
    if (dput(d, RK_CONS, tmp) < 0)
        goto fail;
    Py_CLEAR(tmp);
    tmp = round_or_none(TI(a, 12), 2);
    if (dput(d, RK_PERR, tmp) < 0)
        goto fail;
    Py_CLEAR(tmp);
    if (dput(d, RK_ANG, TI(t, 12)) < 0 || dput(d, RK_PRED, TI(t, 13)) < 0 || dput(d, RK_PPATH, along) < 0)
        goto fail;
#undef TI
    Py_DECREF(flags);
    Py_DECREF(fl);
    Py_DECREF(Tobj);
    return d;
fail:
    Py_XDECREF(d);
    Py_XDECREF(flags);
    Py_XDECREF(fl);
    Py_XDECREF(tmp);
    Py_XDECREF(Tobj);
    return NULL;
}

#define RFCFG 12                 /* corridor_m, corridor_rel, path_max_m, ttc_min_closing, in_path_sigma,
                                    in_path_slack_m, threat_ttc_s, threat_min_mps, with_heading_deg, with_speed
                                    (low, high), with_min_mps */

/* radar_frame(S, rows, dt, kin, par, pose, ego, cfg, horizons, path_x | None, path_y | None, listed, v_still
 *             [, attrs, in_since, threat_since, T, flag_cfg])
 *   -> [(x_m, y_m, vx_mps, vy_mps, ax_mps2, ay_mps2, speed_mps, heading_rad, v_rel_mps, std, models,
 *        p_stationary, angle_offset_deg, predict, offset, path_offset_m, on, sure_in, p_threat, ttc_s, along)]
 *   dt: (B, 1 + H) the time from each object's own to the message's and to each horizon after it;
 *   pose: (X, Y, psi) of the car at the message's time; ego: (v, w); horizons: H numbers (predict_s);
 *   listed: per object 1 if it is not tentative (a threat is judged for those only).
 * Rounded as the message carries them; `offset` and `p_threat` also as they are (the flags' hysteresis needs them).
 * With attrs (one tuple of RATTR per object) the message's objects themselves, flags and all (r_obj_dict). */
static PyObject *n_radar_frame(PyObject *self, PyObject *args)
{
    PyObject *oS, *orows, *odt, *okin, *opar, *opose, *oego, *ocfg, *ohz, *opx, *opy, *olst, *ov0;
    PyObject *oattr = Py_None, *ohin = Py_None, *ohthr = Py_None, *ofc = Py_None;
    double Tm = 0.0, fc[RFLAGS];
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOO|OOOdO", &oS, &orows, &odt, &okin, &opar, &opose, &oego, &ocfg, &ohz,
                          &opx, &opy, &olst, &ov0, &oattr, &ohin, &ohthr, &Tm, &ofc))
        return NULL;
    PyObject *res = NULL;
    View vS, vpx, vpy;
    vS.held = vpx.held = vpy.held = 0;
    Py_ssize_t cap, B = 0, H, *rows = NULL;
    double *buf = NULL, pose[3], ego[2], cfg[RFCFG], hz[8];   /* hz: the horizons' values (speed x h) */
    RPar par;
    RPath path = {NULL, NULL, 0, 0.0, 0.0};
    if (rkeys_init() < 0 || store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &B)) == NULL ||
        rpar_get(opar, &par) < 0 || rnums_get(opose, 3, pose, "pose") < 0 || rnums_get(oego, 2, ego, "ego") < 0 ||
        rnums_get(ocfg, RFCFG, cfg, "cfg") < 0)
        goto done;
    if (!PyList_Check(ohz) && !PyTuple_Check(ohz)) {
        PyErr_SetString(PyExc_TypeError, "radar_frame: the horizons must be a list or tuple");
        goto done;
    }
    H = batch_len(ohz);
    if (H > 8) {
        PyErr_SetString(PyExc_ValueError, "radar_frame: at most 8 horizons");
        goto done;
    }
    if (rnums_get(ohz, H, hz, "horizons") < 0)
        goto done;
    if (opx != Py_None) {
        if (view_get(opx, &vpx, 0, 'd', 1, "path_x") < 0 || view_get(opy, &vpy, 0, 'd', 1, "path_y") < 0)
            goto done;
        if (vpx.b.shape[0] != vpy.b.shape[0] || vpx.b.shape[0] < 2) {
            PyErr_SetString(PyExc_ValueError, "radar_frame: a path of at least two points, x and y alike");
            goto done;
        }
        path.xs = DATA(vpx);
        path.ys = DATA(vpy);
        path.n = vpx.b.shape[0];
    }
    path.v = ego[0];
    path.w = ego[1];
    buf = (double *)PyMem_Malloc((size_t)(B ? B : 1) * (size_t)(1 + H + 4 + 1 + 1) * sizeof(double));
    if (buf == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    double *dts = buf, *kin = dts + B * (1 + H), *lst = kin + 4 * B, *v0 = lst + B;
    if (rows_get(odt, B, (int)(1 + H), dts, "dt") < 0 || rows_get(okin, B, 4, kin, "kin") < 0 ||
        rows_get(olst, B, 1, lst, "listed") < 0 || rows_get(ov0, B, 1, v0, "v_still") < 0)
        goto done;
    if (oattr != Py_None) {
        if (!PyList_Check(oattr) || PyList_Size(oattr) != B || !PyDict_Check(ohin) || !PyDict_Check(ohthr)) {
            PyErr_SetString(PyExc_TypeError, "radar_frame: attributes (a list, one per object) and two dicts");
            goto done;
        }
        if (rnums_get(ofc, RFLAGS, fc, "flag cfg") < 0)
            goto done;
    }
    res = PyList_New(B);
    if (res == NULL)
        goto done;
    double *A = (double *)vS.b.buf;
    double X = pose[0], Y = pose[1], ch = cos(pose[2]), sh = sin(pose[2]), v_ego = ego[0];
    for (Py_ssize_t b = 0; b < B; b++) {
        const double *row = A + rows[b] * RROW;
        double x[RN], P[RNN], mu[3], xp[8][RN], Pq[8][RNN];
        for (Py_ssize_t h = 0; h <= H; h++) {          /* now and ahead (Tracker.peek_many) */
            double dt = dts[b * (1 + H) + h], *xo = h ? xp[h - 1] : x, *Po = h ? Pq[h - 1] : P, mm[3];
            if (h && fabs(dt) >= 1e-6) {                /* ahead: the place is all that is asked */
                double o6[6];
                r_peek_pos(row, dt, kin + 4 * b, &par, o6);
                xo[R_PX] = o6[0];
                xo[R_PY] = o6[1];
                Po[0] = o6[2];
                Po[RN + 1] = o6[5];
                continue;
            }
            if (fabs(dt) < 1e-6) {
                memcpy(xo, row + RO_X, sizeof(double) * RN);
                memcpy(Po, row + RO_P, sizeof(double) * RNN);
                memcpy(mm, row + RO_MU, sizeof(mm));
            } else {                                     /* now: place, velocity, acceleration, offset */
                double x7[7], P4[16];
                r_peek_part(row, dt, kin + 4 * b, &par, 4, x7, P4, mm);
                memcpy(xo, x7, sizeof(x7));
                for (int i = 0; i < 4; i++)
                    for (int l = 0; l < 4; l++)
                        Po[i * RN + l] = P4[i * 4 + l];
            }
            if (h == 0)
                memcpy(mu, mm, sizeof(mm));
        }
        /* ground -> car axes */
        double dx = x[R_PX] - X, dy = x[R_PY] - Y;
        double pc0 = ch * dx + sh * dy, pc1 = -sh * dx + ch * dy;
        /* the state's place is the object's centre; the message's place (x_m, y_m, and where it will be) is its
         * near face: where the line of sight from the car's origin to the centre enters its rectangle (what is in
         * the way, and what a lead's distance refers to). The centre goes out as `ctr`. */
        double ctr0 = pc0, ctr1 = pc1, near0 = 0.0, near1 = 0.0;
        {
            double g0, g1;                               /* centre -> nearest point, ground axes */
            r_near_offset(row + RO_E, -dx, -dy, &g0, &g1);
            near0 = ch * g0 + sh * g1;
            near1 = -sh * g0 + ch * g1;
            pc0 += near0;
            pc1 += near1;
        }
        double vc0 = ch * x[R_VX] + sh * x[R_VY], vc1 = -sh * x[R_VX] + ch * x[R_VY];
        double ac0 = ch * x[R_AX] + sh * x[R_AY], ac1 = -sh * x[R_AX] + ch * x[R_AY];
        double Pp[4], Pv[4];
        for (int blk = 0; blk < 2; blk++) {             /* Rm @ P[blk] @ Rm.T */
            int o = 2 * blk;
            double a = P[o * RN + o], bb = P[o * RN + o + 1], c = P[(o + 1) * RN + o], d = P[(o + 1) * RN + o + 1];
            double m00 = ch * a + sh * c, m01 = ch * bb + sh * d, m10 = -sh * a + ch * c, m11 = -sh * bb + ch * d;
            double *out = blk ? Pv : Pp;
            out[0] = m00 * ch + m01 * sh;
            out[1] = m00 * -sh + m01 * ch;
            out[2] = m10 * ch + m11 * sh;
            out[3] = m10 * -sh + m11 * ch;
        }
        double speed = py_hypot(vc0, vc1);
        double off = r_offset(&path, pc0, pc1);
        double lim = cfg[0] + cfg[1] * pc0;
        int on = 0.3 < pc0 && pc0 <= cfg[2] && fabs(off) <= lim;
        double sy = sqrt(pymax(Pp[3], 0.0));
        int sure_in = on && fabs(off) + cfg[4] * sy <= lim + cfg[5];
        /* reached within threat_ttc_s at the present speeds, over every hypothesis (the object's own, at its time) */
        double p_thr = 0.0;
        if (sure_in && v_ego >= cfg[7] && lst[b] != 0.0) {
            double Tt = cfg[6], u[RN] = {0};
            u[R_PX] = ch;
            u[R_PY] = sh;
            u[R_VX] = Tt * ch;
            u[R_VY] = Tt * sh;
            for (int m = 0; m < 3; m++) {
                const double *xm = row + RO_XS + m * RN, *Pm = row + RO_PS + m * RNN;
                double g = ch * (xm[R_PX] - X) + sh * (xm[R_PY] - Y) + Tt * (ch * xm[R_VX] + sh * xm[R_VY]) - Tt * v_ego;
                double uPu = 0.0;
                for (int i = 0; i < RN; i++) {
                    double acc = 0.0;
                    for (int k = 0; k < RN; k++)
                        acc = k ? acc + u[k] * Pm[k * RN + i] : u[k] * Pm[k * RN + i];
                    uPu = i ? uPu + acc * u[i] : acc * u[i];
                }
                double sg = sqrt(pymax(uPu, 0.0) + 0.09 * Tt * Tt);
                p_thr += row[RO_MU + m] * 0.5 * (1.0 + erf(-g / (sg * 1.4142)));
            }
        }
        double v_rel = vc0 - v_ego;
        int has_ttc = on && v_rel < -cfg[3];
        double still;
        {
            const double *Xs = row + RO_XS, *Ps = row + RO_PS, *m = row + RO_MU;
            double q0 = r_still(Xs + RN, Ps + RNN, v0[b]), q1 = r_still(Xs + 2 * RN, Ps + 2 * RNN, v0[b]);
            double s = (m[0] + m[1] * q0) + m[2] * q1;
            still = s > 1.0 ? 1.0 : s;
        }
        PyObject *t = PyTuple_New(21), *pred = PyList_New(H), *std = PyList_New(4), *models = PyDict_New(), *along = NULL;
        if (t == NULL || pred == NULL || std == NULL || models == NULL)
            goto obj_fail;
        for (Py_ssize_t h = 0; h < H; h++) {
            double qx = xp[h][R_PX] - X, qy = xp[h][R_PY] - Y;
            double q0 = ch * qx + sh * qy + near0, q1 = -sh * qx + ch * qy + near1;
            PyObject *d = PyDict_New();
            if (d == NULL)
                goto obj_fail;
            PyList_SetItem(pred, h, d);
            PyObject *th = seq_item(ohz, PyList_Check(ohz), h);
            Py_INCREF(th);
            if (dset(d, 0, th) < 0 || dset(d, 1, round_obj(q0, 2)) < 0 ||
                dset(d, 2, round_obj(q1, 2)) < 0 ||
                dset(d, 3, round_obj(sqrt(pymax(Pq[h][0] + Pq[h][RN + 1], 0.0)), 2)) < 0)
                goto obj_fail;
        }
        {
            double sd[4] = {Pp[0], Pp[3], Pv[0], Pv[3]};
            for (int k = 0; k < 4; k++) {
                PyObject *f = round_obj(sqrt(pymax(sd[k], 0.0)), 2);
                if (f == NULL)
                    goto obj_fail;
                PyList_SetItem(std, k, f);
            }
        }
        for (int m = 0; m < 3; m++)
            if (dset(models, 4 + m, round_obj(mu[m], 3)) < 0)
                goto obj_fail;
        /* the same object if it keeps following the road as the car does */
        if (speed >= cfg[11] && v_ego >= cfg[11] && pc0 > 0.0 && cfg[9] <= speed / v_ego && speed / v_ego <= cfg[10]) {
            double pp[3];
            r_path_point(&path, pc0, pp);
            double dh = wrap_pi(atan2(vc1, vc0) - pp[2]);
            if (fabs(dh) <= cfg[8] * (PI / 180.0)) {
                along = PyList_New(H);
                if (along == NULL)
                    goto obj_fail;
                for (Py_ssize_t h = 0; h < H; h++) {
                    r_path_point(&path, pc0 + speed * hz[h], pp);
                    PyObject *d = PyDict_New();
                    if (d == NULL)
                        goto obj_fail;
                    PyList_SetItem(along, h, d);
                    PyObject *th = seq_item(ohz, PyList_Check(ohz), h);
                    Py_INCREF(th);
                    if (dset(d, 0, th) < 0 || dset(d, 1, round_obj(pp[0] - off * sin(pp[2]), 2)) < 0 ||
                        dset(d, 2, round_obj(pp[1] + off * cos(pp[2]), 2)) < 0)
                        goto obj_fail;
                }
            }
        }
        PyObject *items[21] = {
            round_obj(pc0, 2), round_obj(pc1, 2), round_obj(vc0, 2), round_obj(vc1, 2), round_obj(ac0, 2),
            round_obj(ac1, 2), round_obj(speed, 2), NULL, round_obj(v_rel, 2), std, models, round_obj(still, 3),
            round_obj(x[R_DP] * (180.0 / PI), 3), pred, PyFloat_FromDouble(off), round_obj(off, 2),
            PyBool_FromLong(on), PyBool_FromLong(sure_in), PyFloat_FromDouble(p_thr), NULL, NULL};
        std = models = pred = NULL;
        if (speed >= 1.0)
            items[7] = round_obj(atan2(vc1, vc0), 3);
        else {
            items[7] = Py_None;
            Py_INCREF(Py_None);
        }
        if (has_ttc)
            items[19] = round_obj(pc0 / -v_rel, 2);
        else {
            items[19] = Py_None;
            Py_INCREF(Py_None);
        }
        if (along)
            items[20] = along;
        else {
            items[20] = Py_None;
            Py_INCREF(Py_None);
        }
        along = NULL;
        int bad = 0;
        for (int k = 0; k < 21; k++) {
            if (items[k] == NULL)
                bad = 1;
            else
                PyTuple_SetItem(t, k, items[k]);
        }
        if (bad)
            goto obj_fail;
        if (oattr != Py_None) {
            PyObject *od = r_obj_dict(t, PyList_GetItem(oattr, b), ohin, ohthr, Tm, fc);
            Py_DECREF(t);
            t = od;
            if (t == NULL)
                goto obj_fail;
            /* its extent (the heading in the car's axes) and what the last scan could see of it */
            const double *e = row + RO_E, *Pe = row + RO_PE, *M = row + RO_M;
            PyObject *es = Py_BuildValue("[NNN]", round_obj(sqrt(pymax(Pe[0], 0.0)), 2), round_obj(sqrt(pymax(Pe[4], 0.0)), 2),
                                         round_obj(sqrt(pymax(Pe[8], 0.0)), 3));
            int oc = (int)M[M_OCC];
            PyObject *ocv = oc == 1 ? rkey[RK_OBJECT] : oc == 2 ? rkey[RK_EDGE] : oc == 3 ? rkey[RK_FOV] : Py_None;
            PyObject *by = oc == 1 && M[M_OCCBY] > 0.0 ? PyLong_FromDouble(M[M_OCCBY]) : (Py_INCREF(Py_None), Py_None);
            if (dset(t, RK_CTR, Py_BuildValue("[NN]", round_obj(ctr0, 2), round_obj(ctr1, 2))) < 0 ||
                dset(t, RK_LEN, round_obj(e[0], 2)) < 0 || dset(t, RK_WID, round_obj(e[1], 2)) < 0 ||
                dset(t, RK_HDG, round_obj(wrap_pi(e[2] - pose[2]), 3)) < 0 || dset(t, RK_EXTSTD, es) < 0 ||
                dset(t, RK_VIS, round_obj(M[M_VIS], 2)) < 0 || dput(t, RK_OCC, ocv) < 0 || dset(t, RK_OCCBY, by) < 0)
                goto obj_fail;
        }
        PyList_SetItem(res, b, t);
        continue;
    obj_fail:
        Py_XDECREF(t);
        Py_XDECREF(pred);
        Py_XDECREF(std);
        Py_XDECREF(models);
        Py_XDECREF(along);
        Py_CLEAR(res);
        goto done;
    }
done:
    PyMem_Free(rows);
    PyMem_Free(buf);
    view_drop(&vS); view_drop(&vpx); view_drop(&vpy);
    return res;
}

/* ---- the objects' life (Tracker.manage, Tracker._merge) ---- */
/* the larger eigenvalue of a symmetric 2 x 2 matrix from its lower triangle (np.linalg.eigvalsh(.)[-1]) */
static double r_eig2(double a, double b, double d)
{
    return 0.5 * (a + d) + hypot(0.5 * (a - d), b);
}

/* radar_life(S, rows, v_still, away, dt, kin, par, pose, stand_drift2) -> [(p_still, finite, mu_stand, sd, range)]
 * For every object: how still it is (p_still), whether its combined state is finite; for one that is out of
 * view (away: 1) and finite, where it will be at the message's time (dt from its own, kin, par: radar_peek):
 * the probability that it stands, how well its place is known (1 sigma of the larger axis; a standing one's
 * as it stands, with the standing drift over dt), its range from the car (pose: (X, Y) or None: 0). Others:
 * (p_still, finite, None, None, None). */
static PyObject *n_radar_life(PyObject *self, PyObject *args)
{
    PyObject *oS, *orows, *ov0, *oaway, *odt, *okin, *opar, *opose;
    double sd2;
    if (!PyArg_ParseTuple(args, "OOOOOOOOd", &oS, &orows, &ov0, &oaway, &odt, &okin, &opar, &opose, &sd2))
        return NULL;
    PyObject *res = NULL;
    View vS;
    vS.held = 0;
    Py_ssize_t cap, B = 0, *rows = NULL;
    double *buf = NULL, pose[2] = {0.0, 0.0};
    int has_pose = opose != Py_None;
    RPar par;
    if (store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &B)) == NULL || rpar_get(opar, &par) < 0 ||
        (has_pose && rnums_get(opose, 2, pose, "pose") < 0))
        goto done;
    buf = (double *)PyMem_Malloc((size_t)(B ? B : 1) * 7 * sizeof(double));
    if (buf == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    double *v0 = buf, *away = v0 + B, *dts = away + B, *kin = dts + B;
    if (rows_get(ov0, B, 1, v0, "v_still") < 0 || rows_get(oaway, B, 1, away, "away") < 0 ||
        rows_get(odt, B, 1, dts, "dt") < 0 || rows_get(okin, B, 4, kin, "kin") < 0)
        goto done;
    res = PyList_New(B);
    if (res == NULL)
        goto done;
    double *A = (double *)vS.b.buf;
    for (Py_ssize_t b = 0; b < B; b++) {
        const double *row = A + rows[b] * RROW;
        const double *Xs = row + RO_XS, *Ps = row + RO_PS, *m = row + RO_MU;
        double q0 = r_still(Xs + RN, Ps + RNN, v0[b]), q1 = r_still(Xs + 2 * RN, Ps + 2 * RNN, v0[b]);
        double s = (m[0] + m[1] * q0) + m[2] * q1;
        int fin = all_finite(row + RO_X, RN);
        PyObject *t;
        if (away[b] != 0.0 && fin) {
            double x[RN], P[RNN], mm[3], dt = dts[b];
            if (fabs(dt) < 1e-6) {
                memcpy(x, row + RO_X, sizeof(x));
                memcpy(P, row + RO_P, sizeof(P));
                memcpy(mm, row + RO_MU, sizeof(mm));
            } else {
                double xs1[3 * RN], Ps1[3 * RNN];
                r_evolve_row(row, dt, kin + 4 * b, &par, xs1, Ps1, mm, NULL, NULL, NULL);
                r_combine(xs1, Ps1, mm, x, P);
            }
            double lam;
            if (mm[R_STAND] >= 0.5) {
                /* believed standing: where it stands is known as well as ever */
                const double *Pst = row + RO_PS;           /* the standing hypothesis, at the object's own time */
                double grow = sd2 * pymax(dt, 0.0);
                lam = r_eig2(Pst[0] + grow, Pst[RN], Pst[RN + 1] + grow);
            } else
                lam = r_eig2(P[0], P[RN], P[RN + 1]);
            double sd = sqrt(pymax(lam, 0.0));
            double rng = has_pose ? py_hypot(x[R_PX] - pose[0], x[R_PY] - pose[1]) : 0.0;
            t = Py_BuildValue("(dOddd)", s > 1.0 ? 1.0 : s, Py_True, mm[R_STAND], sd, rng);
        } else
            t = Py_BuildValue("(dOOOO)", s > 1.0 ? 1.0 : s, fin ? Py_True : Py_False, Py_None, Py_None, Py_None);
        if (t == NULL) {
            Py_CLEAR(res);
            goto done;
        }
        PyList_SetItem(res, b, t);
    }
done:
    PyMem_Free(rows);
    PyMem_Free(buf);
    view_drop(&vS);
    return res;
}

#define RMCFG 4                  /* merge_m, merge_rel, merge_dv, merge_d2 */

/* radar_merge(S, rows, times, kin, par, here, ego, cfg) -> [(i, j, same_place)]
 * Tracker._merge's geometry: the pairs (i < j) worth a look (nearer than the limit can be at any range in reach,
 * with room for what the difference of their times may have moved one of them), and for each whether the two
 * are at one place within the limit at that range and move alike (the second taken to the first one's time).
 *   times: each object's own time;  here: (B, 2) the car's place at each object's time;  ego: (X, Y) now. */
static PyObject *n_radar_merge(PyObject *self, PyObject *args)
{
    PyObject *oS, *orows, *otm, *okin, *opar, *ohere, *oego, *ocfg;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &oS, &orows, &otm, &okin, &opar, &ohere, &oego, &ocfg))
        return NULL;
    PyObject *res = NULL;
    View vS;
    vS.held = 0;
    Py_ssize_t cap, B = 0, *rows = NULL;
    double *buf = NULL, ego[2], cfg[RMCFG];
    RPar par;
    if (store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &B)) == NULL || rpar_get(opar, &par) < 0 ||
        rnums_get(oego, 2, ego, "ego") < 0 || rnums_get(ocfg, RMCFG, cfg, "cfg") < 0)
        goto done;
    buf = (double *)PyMem_Malloc((size_t)(B ? B : 1) * 9 * sizeof(double));
    if (buf == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    double *tm = buf, *kin = tm + B, *here = kin + 4 * B, *sp = here + 2 * B, *far = sp + B;
    if (rows_get(otm, B, 1, tm, "times") < 0 || rows_get(okin, B, 4, kin, "kin") < 0 ||
        rows_get(ohere, B, 2, here, "here") < 0)
        goto done;
    res = PyList_New(0);
    if (res == NULL)
        goto done;
    double *A = (double *)vS.b.buf;
    for (Py_ssize_t i = 0; i < B; i++) {
        const double *x = A + rows[i] * RROW + RO_X;
        sp[i] = py_hypot(x[R_VX], x[R_VY]);
        far[i] = hypot(x[R_PX] - ego[0], x[R_PY] - ego[1]) + 10.0;    /* the car has not moved more than this */
    }
    for (Py_ssize_t i = 0; i < B; i++) {
        const double *ra = A + rows[i] * RROW, *xa = ra + RO_X, *Pa = ra + RO_P;
        for (Py_ssize_t j = i + 1; j < B; j++) {
            const double *rb = A + rows[j] * RROW, *xb0 = rb + RO_X, *Pb = rb + RO_P;
            double gap = hypot(xa[R_PX] - xb0[R_PX], xa[R_PY] - xb0[R_PY]);
            double room = cfg[0] + cfg[1] * far[i] + fabs(tm[i] - tm[j]) * (sp[j] + 30.0) * 2.0;
            if (!(gap <= room))
                continue;
            double xb[RN];
            if (fabs(tm[i] - tm[j]) < 1e-6)
                memcpy(xb, xb0, sizeof(xb));
            else {
                double xs1[3 * RN], Ps1[3 * RNN], mm[3], Pp[RNN];
                r_evolve_row(rb, tm[i] - tm[j], kin + 4 * j, &par, xs1, Ps1, mm, NULL, NULL, NULL);
                r_combine(xs1, Ps1, mm, xb, Pp);
            }
            double r = py_hypot(xa[R_PX] - here[2 * i], xa[R_PY] - here[2 * i + 1]);
            double lim = cfg[0] + cfg[1] * r;
            double dv = hypot(xb[R_VX] - xa[R_VX], xb[R_VY] - xa[R_VY]);
            double sv = sqrt(pymax((Pa[R_VX * RN + R_VX] + Pa[R_VY * RN + R_VY]) + (Pb[R_VX * RN + R_VX] + Pb[R_VY * RN + R_VY]), 0.0));
            int same = !(py_hypot(xb[R_PX] - xa[R_PX], xb[R_PY] - xa[R_PY]) > lim) && !(dv > cfg[2] + 2.0 * sv);
            if (same) {
                /* ... and within what the two know of their places: two things whose places are known and differ
                 * are two things, however near (cars side by side in a microwave radar's view) */
                double S[4] = {Pa[0] + Pb[0] + 0.05, Pa[1] + Pb[1], Pa[RN] + Pb[RN], Pa[RN + 1] + Pb[RN + 1] + 0.05};
                double nu[2] = {xb[R_PX] - xa[R_PX], xb[R_PY] - xa[R_PY]}, d2;
                if (r_mahal(S, 2, nu, &d2, NULL) == 0 && d2 > cfg[3])
                    same = 0;
                /* ... unless one's centre lies inside the other's rectangle: two things are not in one place.
                 * (Said only of places that are known: an overlap of two guesses is no evidence.) */
                if (!same && r_eig2(Pa[0], Pa[RN], Pa[RN + 1]) < 1.0 && r_eig2(Pb[0], Pb[RN], Pb[RN + 1]) < 1.0) {
                    const double *ea = ra + RO_E, *eb = rb + RO_E;
                    for (int w = 0; w < 2 && !same; w++) {
                        const double *e = w ? eb : ea;
                        double sgn = w ? -1.0 : 1.0, c = cos(e[2]), sn = sin(e[2]);
                        double bx = sgn * (c * nu[0] + sn * nu[1]), by = sgn * (-sn * nu[0] + c * nu[1]);
                        same = fabs(bx) <= 0.5 * e[0] && fabs(by) <= 0.5 * e[1];
                    }
                    /* ... or the two drive as one, end to end with no room between them: the parts of one long
                     * thing (things that drive keep a distance; things that stand may stand bumper to bumper) */
                    if (!same && ra[RO_M + M_HOLD] >= 2.0 && rb[RO_M + M_HOLD] >= 2.0 && sp[i] > 2.0 && sp[j] > 2.0 &&
                        dv <= 1.0 + 2.0 * sv) {
                        double c = cos(ea[2]), sn = sin(ea[2]);
                        double bx = c * nu[0] + sn * nu[1], by = -sn * nu[0] + c * nu[1];
                        double ch = fabs(cos(eb[2] - ea[2])), shh = fabs(sin(eb[2] - ea[2]));
                        double hx = 0.5 * (ea[0] + eb[0] * ch + eb[1] * shh), hy = 0.5 * (ea[1] + eb[0] * shh + eb[1] * ch);
                        same = fabs(bx) <= hx + 1.0 && fabs(by) <= hy + 0.3;
                    }
                }
            }
            PyObject *t = Py_BuildValue("(nnO)", i, j, same ? Py_True : Py_False);
            if (t == NULL || PyList_Append(res, t) < 0) {
                Py_XDECREF(t);
                Py_CLEAR(res);
                goto done;
            }
            Py_DECREF(t);
        }
    }
done:
    PyMem_Free(rows);
    PyMem_Free(buf);
    view_drop(&vS);
    return res;
}

/* the middle one by weight of v (n >= 1): argsort (stable, as numpy's for small arrays), cumulative weights, the
 * first whose cumulative weight reaches half of the total (np.searchsorted) */
static double r_wmedian(const double *v, const double *w, Py_ssize_t n, Py_ssize_t *k, double *cw)
{
    for (Py_ssize_t i = 0; i < n; i++) {
        Py_ssize_t j = i;
        while (j > 0 && v[k[j - 1]] > v[i]) {
            k[j] = k[j - 1];
            j--;
        }
        k[j] = i;
    }
    for (Py_ssize_t i = 0; i < n; i++)
        cw[i] = i ? cw[i - 1] + w[k[i]] : w[k[i]];
    double half = 0.5 * cw[n - 1];
    Py_ssize_t idx = 0;
    while (idx < n - 1 && cw[idx] < half)
        idx++;
    return v[k[idx]];
}

/* An expectation (px, py, its covariance Pp) against the object's row taken `back` seconds back: half the squared
 * distance in both uncertainties (+ 0.05 m2 each way) and the distance; -1: singular */
static int r_check(const double *row, double back, double px, double py, const double *Pp, double *m2, double *e)
{
    const double *x = row + RO_X, *P = row + RO_P;
    double ex = x[R_PX] - x[R_VX] * back - px, ey = x[R_PY] - x[R_VY] * back - py;
    double S[4] = {Pp[0] + P[0] + 0.05, Pp[1] + P[1] + 0.0, Pp[2] + P[RN] + 0.0, Pp[3] + P[RN + 1] + 0.05};
    double nu[2] = {ex, ey}, d2;
    if (r_mahal(S, 2, nu, &d2, NULL) < 0)
        return -1;
    *m2 = d2 / 2.0;
    *e = py_hypot(ex, ey);
    return 0;
}

/* radar_register(S, ripe, dp_var, rows, a) -> (m, sum of weights) | None. Tracker._register for one sensor:
 * what the ripe objects agree on in their angle offsets (weights 1 / (P[dp, dp] + dp_var); from three on the
 * middle one by weight, else the weighted mean), and that times `a` taken out of the offset of every object
 * in `rows` (its hypotheses, combination and free estimate). None (nothing changed): no ripe object. */
static PyObject *n_radar_register(PyObject *self, PyObject *args)
{
    PyObject *oS, *oripe, *orows;
    double dpv, a;
    if (!PyArg_ParseTuple(args, "OOdOd", &oS, &oripe, &dpv, &orows, &a))
        return NULL;
    View vS;
    vS.held = 0;
    PyObject *res = NULL;
    Py_ssize_t cap, n = 0, nr = 0, *ripe = NULL, *rows = NULL;
    double *w = NULL;
    if (store_get(oS, &vS, &cap) < 0 || (ripe = idx_get(oripe, cap, &n)) == NULL || (rows = idx_get(orows, cap, &nr)) == NULL)
        goto done;
    if (n == 0) {
        res = Py_None;
        Py_INCREF(res);
        goto done;
    }
    w = (double *)PyMem_Malloc((size_t)(4 * n) * sizeof(double));
    if (w == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    {
        double *A = (double *)vS.b.buf, *v = w + n, *cw = v + n, *wv = cw + n, m;
        for (Py_ssize_t i = 0; i < n; i++) {
            const double *row = A + ripe[i] * RROW;
            w[i] = 1.0 / (row[RO_P + R_DP * RN + R_DP] + dpv);
            v[i] = row[RO_X + R_DP];
        }
        if (n >= 3) {                                    /* the middle one by weight */
            Py_ssize_t *k = (Py_ssize_t *)PyMem_Malloc((size_t)n * sizeof(Py_ssize_t));
            if (k == NULL) {
                PyErr_NoMemory();
                goto done;
            }
            m = r_wmedian(v, w, n, k, cw);
            PyMem_Free(k);
        } else {
            for (Py_ssize_t i = 0; i < n; i++)
                wv[i] = w[i] * v[i];
            m = np_sum(wv, n, 1) / np_sum(w, n, 1);
        }
        double shift = a * m;
        for (Py_ssize_t i = 0; i < nr; i++) {
            double *row = A + rows[i] * RROW;
            for (int j = 0; j < 3; j++)
                row[RO_XS + j * RN + R_DP] -= shift;
            row[RO_X + R_DP] -= shift;
            row[RO_XF + R_DP] -= shift;
        }
        res = Py_BuildValue("(dd)", m, np_sum(w, n, 1));
    }
done:
    PyMem_Free(ripe);
    PyMem_Free(rows);
    PyMem_Free(w);
    view_drop(&vS);
    return res;
}

/* ====================================================================================================
 * The radar's hypotheses (radar/tracker.py `Tracker.process`): which detection is whose, decided over several
 * scans. Track-oriented multiple-hypothesis tracking in its Bernoulli form (docs/RADAR.md has the choice and
 * the derivation):
 *
 *   node      one row of the state array: an object's numbers after one scan under one story of what the scan's
 *             detections were (the object took detection j, or it was not reported). A node's parent is the same
 *             object one scan earlier; the nodes of an object form a tree, its leaves are the object's present
 *             hypotheses. The chain from a leaf to the root is that story's history, kept for the smoother.
 *   cluster   objects that compete for detections now or did within the last scans. A cluster holds its global
 *             hypotheses: each picks one leaf of every object of the cluster (or none: in that story the object
 *             does not exist, its first detection was another object's) such that no detection is used twice,
 *             with its probability (log weight, normalised in the cluster).
 *   scan      every leaf is stepped to the scan's time into a child (the "not reported" story); the leaves are
 *             compared with the detections (gate, likelihood ratio against "something else": the sensor's
 *             clutter and new objects); objects that share a gated detection are put into one cluster; for each
 *             global hypothesis of a cluster the K best assignments of its detections are enumerated (Murty's
 *             ranked assignment over the ratio matrix) and the best of all of them kept; a detection left
 *             over in a hypothesis starts an object there. Then the trees are cut N scans back to the branch
 *             the best hypothesis descends from (N-scan pruning): a doubtful pairing is decided by what the
 *             following scans show, not by a blend made on the spot.
 *
 * The object that is put out is the leaf of the cluster's best hypothesis (`primary`); the probability that it
 * exists is summed over all the cluster's hypotheses. An object's identity is its tree: it does not change
 * when the best hypothesis changes its mind about which branch is true.
 * ==================================================================================================== */

/* ====================================================================================================
 * The detector's further inputs (docs/RADAR.md, section 8; roadsense docs/ARCHITECTURE.md, "Detection inputs for
 * a tracker"): what a camera box may carry beyond its edges, and what the tracker does with it. Everything here
 * is behind the scan's `xcfg` (Hyp.cycle's last, optional argument): without it nothing of this runs and a scan
 * is, bit for bit, what it was.
 *
 *   flags        the hidden edges of a box (cut by the view or the frame, under a nearer box) decide its rows (the
 *                tracker's Python side: the detection's kind); here: the centre of a box with a hidden side is the
 *                centre of the visible part (x_det_leaf), and the scan's own boxes hide what stands behind them
 *                (x_box_cover: the miss probability)
 *   low list     boxes below the class threshold: a second kind of detection with its own probability of being
 *                given and its own density of false ones; starts nothing, confirms nothing
 *   appearance   a bounded likelihood ratio of "the same object" against "another one" from the distance between
 *                the box's vector and the leaf's memory of its object's (x_app_lr)
 *   lamps        the pair's colour: which end faces the camera (heading, the birth's stories); its separation: a
 *                ruler on the object (an extra row: x_aux_rows)
 *   width        the box width of a vehicle seen along its axis: a ruler too
 *   grade        the road's vertical curvature in the ground angle's own offset (birth, walk)
 *   scale rate   successive values share their baseline; a lamp-only value has a heavier tail (x_det_leaf)
 *
 * A leaf's memory for these lives beside its row, in the hypotheses' own array (Hyp.X, RXW numbers per row, made
 * when the first scan with an xcfg comes): it is copied where a row is copied.
 * ==================================================================================================== */
#define RXEMB 59                 /* the longest appearance vector (reid32+yuv3x3) */
/* a row's memory: appearance (how often updated, when, of which kind), when it last took a scale-rate row, how often
 * the lamp ruler and the width ruler were used, its run of low-list sightings, what the lamp colours have said of
 * which end is the front (a tally), the last lamp pair (the logarithm of its separation over the cosine of the
 * heading against the line of sight, when, its variance), then the appearance vector's mean (in the units of its
 * kind's distance) */
enum { X_EN, X_ET, X_EK, X_ST, X_NLAMP, X_NWID, X_LOWRUN, X_FRONT, X_LS, X_LST, X_LSR, X_SPARE, X_EMB, RXW = X_EMB + RXEMB };
/* the scan's xcfg */
enum { XC_FLAGS, XC_BOXOCC, XC_BOXOCC_M,
       XC_LOW, XC_LOW_P, XC_LOW_PRUN, XC_LOW_RUN_S, XC_LOW_EXTRA,
       XC_APP, XC_APP_KIND, XC_APP_DARK, XC_APP_BDAY, XC_APP_BNIGHT, XC_APP_GAIN, XC_APP_YDAY, XC_APP_YNIGHT, XC_APP_TAB,
       XC_LCOL = XC_APP_TAB + 16, XC_LCOL_SIG, XC_LSIZE, XC_LSEP_RATIO, XC_LSEP_REL, XC_AUX_EPS, XC_AUX_SPAN, XC_AUX_USES,
       XC_WID, XC_WID_REL, XC_GRADE, XC_SCORR, XC_SROB, RXCFG };

typedef struct RXc {
    double c[RXCFG];
    double *X;                       /* the rows' memory (Hyp.X) */
    const double *A;                 /* the state array */
    const double *E;                 /* the scan's appearance vectors: nd x RXEMB, in distance units */
    double t;                        /* the scan's time */
    double dark;                     /* how dark the scene is: 0 day .. 1 night */
} RXc;


/* ---- scratch memory of one call: blocks freed together ---- */
typedef struct ArBlk { struct ArBlk *next; double align_; } ArBlk;
typedef struct { ArBlk *head; int failed; } Arena;

static void *ar_get(Arena *a, size_t n)
{
    ArBlk *b = (ArBlk *)PyMem_Malloc(sizeof(ArBlk) + (n ? n : 1));
    if (b == NULL) {
        a->failed = 1;
        return NULL;
    }
    b->next = a->head;
    a->head = b;
    return (void *)(b + 1);
}

static void ar_free(Arena *a)
{
    while (a->head) {
        ArBlk *n = a->head->next;
        PyMem_Free(a->head);
        a->head = n;
    }
}

typedef struct {
    int alive, in_map;
    int nleaf, leafcap;
    int32_t *leaf;                   /* its present hypotheses (rows) */
    int32_t primary;                 /* its leaf in the best hypothesis of its cluster (else its likeliest leaf) */
    int32_t cl;                      /* its cluster */
    long long born;                  /* the level of its first node */
    long long freed;                 /* the level at which its handle became free (not taken again in that scan) */
    double kin[4], veh, idnum;       /* what its kind can do; vehicle (1) / person (-1) / not known (0); its id */
    double pe, pp;                   /* probability that it exists; that the primary's story is the true one */
} HTrack;

typedef struct {
    int alive, nt, ng;
    int32_t *tr;                     /* nt objects (handles) */
    int32_t *lf;                     /* ng x nt: each hypothesis' leaf of each object, -1: not in this story */
    double *lw;                      /* ng log weights, normalised, the best first */
} HCl;

typedef struct {
    PyObject_HEAD
    Py_ssize_t cap;                  /* rows of the state array */
    int32_t *parent, *nchild, *owner;    /* per row: its parent row (-1: a root), its children, its object (-1: free) */
    int32_t *freel;
    Py_ssize_t nfree;
    HTrack *tr;
    int ntr;                         /* handles in use (alive or not) */
    int trcap;
    HCl *cl;
    int ncl, clcap;
    long long level;                 /* scans processed */
    /* the knobs (tracker.py CFG `hyp_*`) */
    int n_scan, k_best, max_leaves, keep_steps, max_cluster;
    double prune_ln, keep_s;
    /* counters since the start */
    long long c_scans, c_leaves, c_children, c_hyps, c_murty, c_births, c_pruned;
    /* the smoother's kept gains (r_rts): per row and hypothesis (3 + the free estimate), made when first needed */
    double *sc;
    Py_ssize_t sc_cap;
    double sc_par[RNPAR];
    /* the rows' memory for the detector's further inputs (cap x RXW; NULL until a scan asks for it), and how
     * bright the boxes' images have been lately (the mean of their luma cells, followed over the scans) */
    double *X;
    double dark_y, dark;             /* (dark: 0 day .. 1 night, as the last such scan took it) */
    int dark_has;
} Hyp;

static PyObject *HypType;

/* the row of an object's numbers (tracker.py `_LAY`): hypotheses, free estimate, combination; then the extent
 * (length, width, heading and their covariance), the step that led into this node (for the smoother: the
 * switching matrix, dt, what the car's own movement added), and the node's own record (RMETA numbers) */


static inline double sigm(double l) { return 1.0 / (1.0 + exp(-l)); }

static void hyp_release(Hyp *H)
{
    for (int i = 0; i < H->ntr; i++)
        PyMem_Free(H->tr[i].leaf);
    for (int i = 0; i < H->ncl; i++) {
        PyMem_Free(H->cl[i].tr);
        PyMem_Free(H->cl[i].lf);
        PyMem_Free(H->cl[i].lw);
    }
    PyMem_Free(H->tr);
    PyMem_Free(H->cl);
    PyMem_Free(H->parent);
    PyMem_Free(H->nchild);
    PyMem_Free(H->owner);
    PyMem_Free(H->freel);
    PyMem_Free(H->sc);
    PyMem_Free(H->X);
    H->X = NULL;
    H->dark_y = H->dark = 0.0;
    H->dark_has = 0;
    H->sc = NULL;
    H->sc_cap = 0;
    H->tr = NULL;
    H->cl = NULL;
    H->parent = H->nchild = H->owner = H->freel = NULL;
    H->ntr = H->trcap = H->ncl = H->clcap = 0;
    H->cap = 0;
    H->nfree = 0;
}

static int hyp_rows(Hyp *H, Py_ssize_t cap)
{
    if (cap <= H->cap)
        return 0;
    int32_t *p = (int32_t *)PyMem_Realloc(H->parent, (size_t)cap * sizeof(int32_t));
    if (p)
        H->parent = p;
    int32_t *n = (int32_t *)PyMem_Realloc(H->nchild, (size_t)cap * sizeof(int32_t));
    if (n)
        H->nchild = n;
    int32_t *o = (int32_t *)PyMem_Realloc(H->owner, (size_t)cap * sizeof(int32_t));
    if (o)
        H->owner = o;
    int32_t *f = (int32_t *)PyMem_Realloc(H->freel, (size_t)cap * sizeof(int32_t));
    if (f)
        H->freel = f;
    if (!p || !n || !o || !f) {
        PyErr_NoMemory();
        return -1;
    }
    if (H->X != NULL) {
        double *xn = (double *)PyMem_Realloc(H->X, (size_t)cap * RXW * sizeof(double));
        if (xn == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        memset(xn + (size_t)H->cap * RXW, 0, (size_t)(cap - H->cap) * RXW * sizeof(double));
        H->X = xn;
    }
    for (Py_ssize_t r = cap - 1; r >= H->cap; r--) {    /* the lowest free row is taken first */
        H->parent[r] = -1;
        H->nchild[r] = 0;
        H->owner[r] = -1;
        H->freel[H->nfree++] = (int32_t)r;
    }
    /* keep the free rows in falling order below the new ones: the lowest first */
    if (H->cap > 0 && H->nfree > cap - H->cap) {
        Py_ssize_t old = H->nfree - (cap - H->cap);
        /* the list is [old free rows..., new rows falling]: the new ones go below the old ones */
        int32_t *tmp = (int32_t *)PyMem_Malloc((size_t)H->nfree * sizeof(int32_t));
        if (tmp == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        memcpy(tmp, H->freel + old, (size_t)(cap - H->cap) * sizeof(int32_t));
        memcpy(tmp + (cap - H->cap), H->freel, (size_t)old * sizeof(int32_t));
        memcpy(H->freel, tmp, (size_t)H->nfree * sizeof(int32_t));
        PyMem_Free(tmp);
    }
    H->cap = cap;
    return 0;
}

static PyObject *hyp_new(PyTypeObject *tp, PyObject *args, PyObject *kw)
{
    allocfunc alloc = (allocfunc)PyType_GetSlot(tp, Py_tp_alloc);
    return alloc(tp, 0);
}

/* Hyp(rows, n_scan, k_best, max_leaves, prune_ln, keep_steps, keep_s, max_cluster) */
static int hyp_init(PyObject *self, PyObject *args, PyObject *kw)
{
    Hyp *H = (Hyp *)self;
    Py_ssize_t cap;
    int n_scan, k_best, max_leaves, keep_steps, max_cluster;
    double prune_ln, keep_s;
    if (!PyArg_ParseTuple(args, "niiididi", &cap, &n_scan, &k_best, &max_leaves, &prune_ln, &keep_steps, &keep_s,
                          &max_cluster))
        return -1;
    if (cap < 1 || n_scan < 0 || k_best < 1 || max_leaves < 1 || keep_steps < 1 || max_cluster < 1) {
        PyErr_SetString(PyExc_ValueError, "Hyp: rows, k_best, max_leaves, keep_steps, max_cluster >= 1, n_scan >= 0");
        return -1;
    }
    hyp_release(H);
    H->level = 0;
    H->n_scan = n_scan;
    H->k_best = k_best;
    H->max_leaves = max_leaves;
    H->prune_ln = prune_ln;
    H->keep_steps = keep_steps > n_scan + 1 ? keep_steps : n_scan + 1;
    H->keep_s = keep_s;
    H->max_cluster = max_cluster;
    H->c_scans = H->c_leaves = H->c_children = H->c_hyps = H->c_murty = H->c_births = H->c_pruned = 0;
    return hyp_rows(H, cap);
}

static void hyp_dealloc(PyObject *self)
{
    PyTypeObject *tp = Py_TYPE(self);
    hyp_release((Hyp *)self);
    freefunc f = (freefunc)PyType_GetSlot(tp, Py_tp_free);
    f(self);
    Py_DECREF(tp);
}

/* a free row for object h (parent p, -1: a root), or -1 when the state array is full */
static int32_t h_take(Hyp *H, int h, int32_t p)
{
    if (H->nfree == 0)
        return -1;
    int32_t r = H->freel[--H->nfree];
    H->owner[r] = h;
    H->parent[r] = p;
    H->nchild[r] = 0;
    if (p >= 0)
        H->nchild[p]++;
    return r;
}

static void h_give(Hyp *H, int32_t r)
{
    H->owner[r] = -1;
    H->parent[r] = -1;
    H->nchild[r] = 0;
    H->freel[H->nfree++] = r;
}

/* a leaf is dropped: its row is free, and every ancestor that is left without a child with it */
static void h_drop_up(Hyp *H, int32_t r)
{
    while (r >= 0 && H->nchild[r] == 0) {
        int32_t p = H->parent[r];
        h_give(H, r);
        if (p < 0)
            break;
        H->nchild[p]--;
        r = p;
    }
}

static int h_track_new(Hyp *H)
{
    int h = -1;
    for (int i = 0; i < H->ntr; i++)
        if (!H->tr[i].alive && H->tr[i].leaf == NULL && H->tr[i].freed != H->level) {
            h = i;
            break;
        }
    if (h < 0) {
        if (H->ntr == H->trcap) {
            int nc = H->trcap ? 2 * H->trcap : 32;
            HTrack *t = (HTrack *)PyMem_Realloc(H->tr, (size_t)nc * sizeof(HTrack));
            if (t == NULL)
                return -1;
            H->tr = t;
            H->trcap = nc;
        }
        h = H->ntr++;
    }
    HTrack *t = &H->tr[h];
    memset(t, 0, sizeof(*t));
    t->alive = 1;
    t->freed = -1;
    t->primary = -1;
    t->cl = -1;
    t->leafcap = 4;
    t->leaf = (int32_t *)PyMem_Malloc((size_t)t->leafcap * sizeof(int32_t));
    if (t->leaf == NULL) {
        t->alive = 0;
        return -1;
    }
    return h;
}

static int h_leaf_add(HTrack *t, int32_t r)
{
    if (t->nleaf == t->leafcap) {
        int nc = 2 * t->leafcap;
        int32_t *l = (int32_t *)PyMem_Realloc(t->leaf, (size_t)nc * sizeof(int32_t));
        if (l == NULL)
            return -1;
        t->leaf = l;
        t->leafcap = nc;
    }
    t->leaf[t->nleaf++] = r;
    return 0;
}

static int h_cluster_new(Hyp *H, int nt, int ng)
{
    int c = -1;
    for (int i = 0; i < H->ncl; i++)
        if (!H->cl[i].alive) {
            c = i;
            break;
        }
    if (c < 0) {
        if (H->ncl == H->clcap) {
            int nc = H->clcap ? 2 * H->clcap : 32;
            HCl *q = (HCl *)PyMem_Realloc(H->cl, (size_t)nc * sizeof(HCl));
            if (q == NULL)
                return -1;
            H->cl = q;
            H->clcap = nc;
        }
        c = H->ncl++;
        memset(&H->cl[c], 0, sizeof(HCl));
    }
    HCl *q = &H->cl[c];
    q->tr = (int32_t *)PyMem_Malloc((size_t)(nt ? nt : 1) * sizeof(int32_t));
    q->lf = (int32_t *)PyMem_Malloc((size_t)(nt * ng > 0 ? nt * ng : 1) * sizeof(int32_t));
    q->lw = (double *)PyMem_Malloc((size_t)(ng ? ng : 1) * sizeof(double));
    if (!q->tr || !q->lf || !q->lw) {
        PyMem_Free(q->tr);
        PyMem_Free(q->lf);
        PyMem_Free(q->lw);
        q->tr = q->lf = NULL;
        q->lw = NULL;
        return -1;
    }
    q->alive = 1;
    q->nt = nt;
    q->ng = ng;
    return c;
}

static void h_cluster_free(Hyp *H, int c)
{
    HCl *q = &H->cl[c];
    PyMem_Free(q->tr);
    PyMem_Free(q->lf);
    PyMem_Free(q->lw);
    q->tr = q->lf = NULL;
    q->lw = NULL;
    q->alive = 0;
    q->nt = q->ng = 0;
}

/* an object is gone: its rows are free, its handle too */
static void h_track_free(Hyp *H, int h)
{
    HTrack *t = &H->tr[h];
    for (Py_ssize_t r = 0; r < H->cap; r++)
        if (H->owner[r] == h)
            h_give(H, (int32_t)r);
    PyMem_Free(t->leaf);
    t->leaf = NULL;
    t->nleaf = t->leafcap = 0;
    t->alive = 0;
    t->cl = -1;
    t->primary = -1;
    t->freed = H->level;
}

/* ---- the best assignments in order (Murty) ---- */
#define LAP_BIG 1e17

/* The cheapest assignment of n rows to m >= n columns, every row to one column of its own (the Hungarian method
 * with potentials, O(n^2 m)); c: n x m, LAP_BIG for a pair that may not be. Returns the cost; col[i]. Work: u (n + 1),
 * v, minv (m + 1) doubles; p, way (m + 1) ints; used (m + 1) chars. */
static double lap_solve(const double *c, int n, int m, int *col, double *u, double *v, double *minv, int *p, int *way,
                        char *used)
{
    for (int i = 0; i <= n; i++)
        u[i] = 0.0;
    for (int j = 0; j <= m; j++) {
        v[j] = 0.0;
        p[j] = 0;
    }
    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        for (int j = 0; j <= m; j++) {
            minv[j] = INFINITY;
            used[j] = 0;
        }
        do {
            used[j0] = 1;
            int i0 = p[j0], j1 = 0;
            double delta = INFINITY;
            for (int j = 1; j <= m; j++)
                if (!used[j]) {
                    double cur = c[(i0 - 1) * m + (j - 1)] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            for (int j = 0; j <= m; j++)
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else
                    minv[j] -= delta;
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    double cost = 0.0;
    for (int j = 1; j <= m; j++)
        if (p[j] > 0) {
            col[p[j] - 1] = j - 1;
            cost += c[(p[j] - 1) * m + (j - 1)];
        }
    return cost;
}

typedef struct MNode {
    double cost;
    double *c;                       /* its matrix: the constraints written in */
    int *col;
    int fixed;                       /* rows 0..fixed-1 are held to col[] */
} MNode;

/* The k cheapest assignments of n rows to m columns in order of cost (Murty 1968, the partition of Miller, Stone
 * and Cox: a solution's problem is split by its rows in order, each part holding the rows before and forbidding
 * the row's own pairing). out_col: k x n, out_cost: k. Returns how many there are (<= k), -1: no memory. */
static int murty(Arena *ar, const double *c, int n, int m, int k, int *out_col, double *out_cost)
{
    if (n == 0) {
        out_cost[0] = 0.0;
        return 1;
    }
    size_t wk = (size_t)(n + 1 + 2 * (m + 1)) * sizeof(double) + (size_t)(2 * (m + 1)) * sizeof(int) + (size_t)(m + 1);
    double *u = (double *)ar_get(ar, wk + 16);
    int maxq = k * n + 1, nq = 0;
    MNode *Q = (MNode *)ar_get(ar, (size_t)maxq * sizeof(MNode));
    if (u == NULL || Q == NULL)
        return -1;
    double *v = u + n + 1, *minv = v + m + 1;
    int *p = (int *)(minv + m + 1), *way = p + m + 1;
    char *used = (char *)(way + m + 1);
    MNode first;
    first.c = (double *)ar_get(ar, (size_t)(n * m) * sizeof(double));
    first.col = (int *)ar_get(ar, (size_t)n * sizeof(int));
    if (first.c == NULL || first.col == NULL)
        return -1;
    memcpy(first.c, c, (size_t)(n * m) * sizeof(double));
    first.cost = lap_solve(first.c, n, m, first.col, u, v, minv, p, way, used);
    first.fixed = 0;
    if (first.cost >= LAP_BIG * 0.5)
        return 0;
    Q[nq++] = first;
    int found = 0;
    while (found < k && nq > 0) {
        int b = 0;
        for (int i = 1; i < nq; i++)
            if (Q[i].cost < Q[b].cost)
                b = i;
        MNode nd = Q[b];
        Q[b] = Q[--nq];
        memcpy(out_col + found * n, nd.col, (size_t)n * sizeof(int));
        out_cost[found++] = nd.cost;
        if (found == k)
            break;
        /* its problem split: part i holds rows < i to this solution and forbids row i's pairing */
        for (int i = nd.fixed; i < n; i++) {
            MNode ch;
            ch.c = (double *)ar_get(ar, (size_t)(n * m) * sizeof(double));
            ch.col = (int *)ar_get(ar, (size_t)n * sizeof(int));
            if (ch.c == NULL || ch.col == NULL)
                return -1;
            memcpy(ch.c, nd.c, (size_t)(n * m) * sizeof(double));
            for (int r = nd.fixed; r < i; r++) {        /* held: the row's other columns and the column's other rows */
                int cc = nd.col[r];
                for (int j = 0; j < m; j++)
                    if (j != cc)
                        ch.c[r * m + j] = LAP_BIG;
                for (int r2 = 0; r2 < n; r2++)
                    if (r2 != r)
                        ch.c[r2 * m + cc] = LAP_BIG;
            }
            ch.c[i * m + nd.col[i]] = LAP_BIG;
            ch.fixed = i;
            ch.cost = lap_solve(ch.c, n, m, ch.col, u, v, minv, p, way, used);
            if (ch.cost < LAP_BIG * 0.5 && nq < maxq)
                Q[nq++] = ch;
        }
    }
    return found;
}

/* ---- a scan's detections against a leaf ---- */
#define RDET 31                  /* a detection: kind, sdt, group, vehicle (-1, 0, 1), box width, z[4], R[4], what its
                                    score adds to the log odds of "a real thing", the whole box's aspect (0: not
                                    known), then for a cell of several returns of an extended sensor: their number
                                    and the scatter of their places about the mean (sxx, sxy, syy; sensor axes); 1 if
                                    both sides of the box are in the picture (its width is the object's); then the
                                    detector's further inputs (0 without them): the hidden edges (bits: left 1, top
                                    2, right 4, bottom 8; 16: the bottom is cut by the view or the frame), the share
                                    of the box under nearer boxes, 1 for a box of the low-score list, the lamp
                                    pair's colour (0: no pair, 1 red, 2 white, 3 not known), ln(separation / f) and
                                    its variance, the scale rate's source (0 patch, 1 lamps, 2 both), the angles
                                    below the horizon of the box's top and bottom (NaN: not known), 1 if it has an
                                    appearance vector */
enum { D_KIND, D_SDT, D_GROUP, D_VEH, D_W, D_Z, D_R = 9, D_SCORE = 13, D_ASPECT, D_NRET, D_SXX, D_SXY, D_SYY, D_WOK,
       D_HID, D_OCC, D_LOW, D_LAMPC, D_LSEP, D_LSEPR, D_SSRC, D_TOP, D_BOT, D_HASEMB, D_SPARE };
#define RSCFG 19                 /* gate for 1..4 rows, feat_sigma, feat_same, feat_span, map_bw[3], feat_max_s,
                                    common_min, common_pair, occlusion (0: off), the share hidden from which the
                                    chance of a report falls and at which it is lowest, that lowest chance for a
                                    camera and for a microwave radar; from which spread of its bearing a hypothesis'
                                    bearing is so unsure (rad) that its likelihood is taken as a sum of parts (0: never) */

/* ---- the detector's further inputs: the arithmetic (the overview is above the hypotheses) ---- */
enum { XE_LEN, XE_WID, XE_HDG };
static double r_phi(double x);
static double r_silhouette(const double *e, double los, double *H);
static int r_take_parts(double *x, double *P, const RSens *sen, int kind, double sdt, const double *pose, const double *z,
                        const double *Rf, const double *e, double rel, int cons, double *like);
static inline double *x_of(const RXc *xc, const double *row)
{
    return xc->X + (size_t)((row - xc->A) / RROW) * RXW;
}

static double x_lognpdf(double lnd, double d, double mu, double sg)
{
    double l = (lnd - mu) / sg;
    return exp(-0.5 * l * l) / (d * sg * 2.5066282746310002);
}

/* The appearance likelihood ratio: how much likelier a distance d between a box's vector and an object's remembered
 * one is for a box of that object, `gap` seconds after the memory was last refreshed, than for a box of another
 * object. Log-normal fits to measured distances (c[XC_APP_TAB]: by day and by night, the same object 1, 2 and 4 s
 * apart: mu, sigma each; another object: mu, sigma), the same-object fit taken at the gap (the 1 s fit below 1 s:
 * nothing nearer was measured; between the fits linear in log2 of the gap; the 4 s fit beyond). Bounded to
 * [1 / B, B] (one box must not outvote the geometry; B by day and by night), and monotone in the distance (two
 * log-normals cross twice; the outer crossing is the fits' tails, not evidence).
 * Day and night are mixed in the logarithm by how dark the scene is. */
static double x_app_lr(const double *c, double dark, double d, double gap)
{
    double lr[2];
    d = pymax(d, 0.02);
    double u = log(pymin(pymax(gap, 1.0), 4.0)) / 0.6931471805599453;      /* 0 .. 2 */
    int i = u >= 1.0 ? 1 : 0;
    double w = u - (double)i, lnd0 = log(d);
    for (int q = 0; q < 2; q++) {
        const double *T = c + XC_APP_TAB + 8 * q;
        double mu = T[2 * i] + w * (T[2 * i + 2] - T[2 * i]), sg = T[2 * i + 1] + w * (T[2 * i + 3] - T[2 * i + 1]);
        double B = pymax(c[q ? XC_APP_BNIGHT : XC_APP_BDAY], 1.0), lb = log(B), lnd = lnd0;
        /* the log ratio of two log-normals is a parabola in ln d; its turning point is where the fits' tails take
         * over from the evidence: held there on the far side */
        double a = 1.0 / (T[7] * T[7]) - 1.0 / (sg * sg);
        if (fabs(a) > 1e-9) {
            double ut = (T[6] / (T[7] * T[7]) - mu / (sg * sg)) / a;
            lnd = a > 0.0 ? pymin(lnd, ut) : pymax(lnd, ut);
        }
        double dd = exp(lnd);
        double l = log(pymax(x_lognpdf(lnd, dd, mu, sg), 1e-300) / pymax(x_lognpdf(lnd, dd, T[6], T[7]), 1e-300));
        lr[q] = pymin(lb, pymax(-lb, l));
    }
    return exp((1.0 - dark) * lr[0] + dark * lr[1]);
}

static double x_emb_dist(const double *a, const double *b)
{
    double s = 0.0;
    for (int k = 0; k < RXEMB; k++)
        s += (a[k] - b[k]) * (a[k] - b[k]);
    return sqrt(s);
}

/* E|v| and Var|v| of v ~ N(m, s^2) (the folded normal) */
static void x_folded(double m, double s, double *mean, double *var)
{
    if (!(s > 1e-9)) {
        *mean = fabs(m);
        *var = 0.0;
        return;
    }
    double a = m / s, e = s * 0.7978845608028654 * exp(-0.5 * a * a) + m * (1.0 - 2.0 * r_phi(-a));
    *mean = e;
    *var = pymax(m * m + s * s - e * e, 0.0);
}

/* An extra row of a detection (x_aux_rows). which 0, 1: a size of the object's image against the box height, a
 * measurement of the object's height state: z = lh - ln(c W(lh)) (W(h): a vehicle's width by its height). which 2:
 * the lamp separation's change since the leaf's last pair, a scale rate over sdt. */
typedef struct {
    double z, c, R, near, sdt, span;
    int which;                       /* 0: the lamp pair's separation, 1: the box width, 2: the separation's change */
} XAux;
#define XAUX 3

/* What the scan's own boxes hide of a leaf (x_box_cover), kept so that a box can be left out: in the story "the
 * leaf took box j", box j is the leaf's own and hides nothing of it. Per bearing sample of the leaf's silhouette
 * the most a box hides, which box, and the most another one hides; the cover by tracked objects, the factor for the
 * field's edge, the sensor's own probability, the low list's share. */
typedef struct {
    int on;
    double best[5], second[5];
    int bj[5];
    double cover_obj, pe, psen, klo, incov;
} XBox;

/* A vehicle's width by its height (m): cars 1.8 x 1.5, vans 2.0-2.1 x 2.5, lorries and buses 2.5 x 3.4. Widths vary
 * far less than heights across everything the detector calls a vehicle, and by under a tenth among vehicles of one
 * height: which is what makes a width a ruler without knowing the class. *dw: dW / dh. */
static double x_width_of(double h, double *dw)
{
    double hh = pymin(pymax(h, 1.0), 4.2);
    *dw = (h > 1.0 && h < 4.2) ? 0.35 : 0.0;
    return 1.8 + 0.35 * (hh - 1.5);
}

/* The extra rows detection d gives this leaf (row; xr: its memory), at most three. Rulers: a size of the object
 * that the detection does not know, seen from the object's distance.
 *
 * Seen from one distance, two sizes of an object measure their RATIO, whatever the distance: the box height against
 * the box width, or against the lamp pair's separation. The object's height is a state (lh, with its prior by the
 * kind); the width of a vehicle of that height is known to a tenth, its lamp separation to a fifth. So the ratio is
 * a measurement of the height state itself, free of the range:
 *
 *   the box width   ln(h_px / w_px) = lh - ln(W(lh) s),  s = (L |sin a| + W |cos a|) / W the silhouette over the
 *                   width at the heading believed (a: the heading against the line of sight), for a vehicle whose
 *                   both sides and top and bottom are in the picture and uncovered, seen along its axis (the
 *                   length's share under 0.4 of the width's: within 8 degrees for a car; what the detector calls a
 *                   car is 3.5 to 6 m long, and at a larger angle that is what the box width would measure; the
 *                   silhouette known to a third)
 *   the lamp pair   ln(h_px / sep) = lh - ln(XC_LSEP_RATIO W(lh) |cos a|), within about 50 degrees of the axis
 *
 * and what the height then says of the range it says in every later box, through the box-height row and what the
 * filter knows of how the two hang together: the ruler's information stays with the object as its height does.
 * (As a row on the range the same information is gone with the next seconds' process noise.) Taken by every motion
 * hypothesis, also the moving ones that hold the height against the box's own rows: this row has no range in it.
 *
 * The width's (the separation's) own spread about W(h) is a constant of the object, the same in every box: it is
 * not counted once per box. Each of XC_AUX_USES uses takes that variance times XC_AUX_USES (all uses together: the
 * prior's information once), with the box's own noise beside it, which does average over the uses.
 *
 * What a ruler says beyond that is how the range CHANGES from one box to the next, and that needs no constant: the
 * lamp pair's separation against the leaf's last one is a scale rate over the time between them (a row like the
 * detector's own, to the lamp centroids' fraction of a pixel; not taken where the box's scale rate already comes
 * from the lamp pair). */
static int x_aux_rows(const RXc *xc, const RSens *sen, const double *row, const double *xr, const double *d,
                      const double *pose, XAux *out)
{
    int n = 0;
    const double *c = xc->c;
    if (sen->type != SEN_CAMERA || d[D_LOW] != 0.0 || !(d[D_VEH] > 0.0))
        return 0;
    int whole = ((int)d[D_KIND] & 7) == K_RBH;           /* top and bottom its own: the box height is a row */
    int pair = c[XC_LSIZE] != 0.0 && d[D_LAMPC] > 0.0 && d[D_LSEPR] > 0.0;
    int lamp = pair && whole && xr[X_NLAMP] < c[XC_AUX_USES];
    int wid = c[XC_WID] != 0.0 && whole && d[D_WOK] > 0.0 && d[D_W] > 0.0 && xr[X_NWID] < c[XC_AUX_USES];
    if (!pair && !wid)
        return 0;
    const double *x = row + RO_X, *e = row + RO_E, *Pe = row + RO_PE;
    double dx = x[R_PX] - pose[0], dy = x[R_PY] - pose[1], rc = pymax(hypot(dx, dy), 0.5), los = atan2(dy, dx);
    double near = pymax(rc - r_half_depth(e, los, NULL), 0.5) / rc;
    double a = e[XE_HDG] - los, ca = cos(a), sa = sin(a), va = pymin(pymax(Pe[8], 0.0), 0.5);
    double uses = pymax(c[XC_AUX_USES], 1.0);
    if (pair && fabs(ca) >= 0.6 && xr[X_LSR] > 0.0 && xc->t - xr[X_LST] > 0.05 && xc->t - xr[X_LST] <= 1.2 &&
        !(((int)d[D_KIND] & K_SCALE) && d[D_SSRC] >= 1.0)) {
        XAux *o = &out[n++];
        double dt = xc->t - xr[X_LST];
        o->which = 2;
        o->near = near;
        o->sdt = dt;
        o->span = sen->srspan;
        o->c = 0.0;
        o->z = (d[D_LSEP] - log(fabs(ca)) - xr[X_LS]) / dt;
        o->R = (d[D_LSEPR] + xr[X_LSR]) / (dt * dt);
    }
    if (lamp && fabs(ca) >= 0.6) {
        XAux *o = &out[n++];
        o->which = 0;
        o->near = near;
        o->span = c[XC_AUX_SPAN];
        o->sdt = 0.0;
        o->z = d[D_Z + 2] - (d[D_LSEP] + sen->logf);
        o->c = c[XC_LSEP_RATIO] * fabs(ca);
        o->R = d[D_R + 2] + d[D_LSEPR] + (sa * sa) / (ca * ca) * va + c[XC_LSEP_REL] * c[XC_LSEP_REL] * uses;
    }
    if (wid) {
        double ms, vs, mc, vc, sg = sqrt(va);
        x_folded(sa, fabs(ca) * sg, &ms, &vs);
        x_folded(ca, fabs(sa) * sg, &mc, &vc);
        /* the silhouette AT the heading believed (for a vehicle straight ahead the heading is the line of sight far
         * better than its stated spread says, and the mean over that spread would read every such vehicle a fifth
         * too far); what is not known of the heading is the row's variance: the spread of the folded silhouette and
         * its mean's distance from the value used; what is not known of the length likewise */
        double L = e[XE_LEN], W = pymax(e[XE_WID], 0.3), sil = L * fabs(sa) + W * fabs(ca), dm = L * ms + W * mc - sil;
        double var = L * L * vs + W * W * vc + dm * dm + ms * ms * pymax(Pe[0], 0.0);
        if (sil > 0.2 && var < 0.1225 * sil * sil && L * fabs(sa) < 0.4 * W * fabs(ca)) {
            XAux *o = &out[n++];
            double px = 3.0 / (exp(sen->logf) * d[D_W]);
            o->which = 1;
            o->near = near;
            o->span = c[XC_AUX_SPAN];
            o->sdt = 0.0;
            o->z = d[D_Z + 2] - (log(d[D_W]) + sen->logf);
            o->c = sil / W;
            o->R = d[D_R + 2] + 0.06 * 0.06 + px * px + var / (sil * sil) + c[XC_WID_REL] * c[XC_WID_REL] * uses;
        }
    }
    return n;
}

/* (x, P) takes the extra rows one after the other (they are scalar, their noises taken as independent of the box's
 * other rows: after those rows this is the joint update). Each as a mixture: with XC_AUX_EPS the number is not the
 * object's (the pair found on other blobs, a box edge on a neighbour) and says nothing; taken only where it is
 * likelier the object's than not. cons: the size and the angle offset are held against rows that have the range in
 * them (the scale rate; the rulers measure the height itself and are always taken). Returns the rows' likelihood
 * against "a number anywhere in the row's span"; *used: a bit per row taken. */
static double x_aux_take(const RXc *xc, double *x, double *P, const double *pose, const XAux *aux, int n, int cons,
                         int *used)
{
    double fac = 1.0, eps = xc->c[XC_AUX_EPS];
    for (int q = 0; q < n; q++) {
        const XAux *o = &aux[q];
        double span = o->span;
        double H[4 * RN] = {0}, PH[RN], S = o->R, nu;
        if (o->which == 2)                               /* the image's growth over the time since the last pair */
            nu = o->z - r_scale_row(x, pose, o->sdt, H, o->near);
        else {
            double dw, h = exp(x[R_LH]), w = x_width_of(h, &dw);
            nu = o->z - (x[R_LH] - log(o->c * w));
            H[R_LH] = 1.0 - dw * h / w;
        }
        for (int i = 0; i < RN; i++) {
            double acc = 0.0;
            for (int k = 0; k < RN; k++)
                acc += P[i * RN + k] * H[k];
            PH[i] = acc;
        }
        for (int k = 0; k < RN; k++)
            S += H[k] * PH[k];
        if (!(S > 0.0) || !isfinite(S) || !isfinite(nu))
            continue;
        double inl = (1.0 - eps) * span * exp(-0.5 * pymin(nu * nu / S, 400.0)) / sqrt(2.0 * PI * S);
        fac *= inl + eps;
        if (!(inl > eps))
            continue;
        double K[RN];
        for (int i = 0; i < RN; i++)
            K[i] = PH[i] / S;
        if (cons && o->which == 2)
            K[R_LH] = K[R_DP] = 0.0;
        r_take(x, P, K, H, &o->R, &nu, 1, 1.0);
        if (used)
            *used |= 1 << o->which;
    }
    return fac;
}

/* Detection d as this leaf reads it (dc: a copy, returned where something changes; else d itself):
 *  - a box with ONE side hidden: its centre is the centre of the visible part. The hidden part's width is what the
 *    leaf's silhouette at its range has beyond the box; the centre is moved to the object's by half of that, towards
 *    the hidden side, and half of that is its added spread. Both sides hidden: the spread only.
 *  - successive scale rates of one object share most of their baseline (a value is the mean rate over `sdt`, a new
 *    one comes every 0.2-0.4 s): taken as independent rows they are counted two to four times. The variance times
 *    sdt over the time since this leaf's previous scale row, where that is shorter.
 *  - a scale rate from the lamp pair alone has a heavier tail than its stated spread: Huber's weight with
 *    XC_SROB sigma on the leaf's own innovation (the variance raised until the innovation is XC_SROB sigma). */
static const double *x_det_leaf(const RXc *xc, const RSens *sen, const double *pose, const double *row, const double *d,
                                double *dc)
{
    if (xc == NULL || sen->type != SEN_CAMERA)
        return d;
    const double *c = xc->c;
    int kind = (int)d[D_KIND], base = kind & 7, hid = (int)d[D_HID];
    int side = c[XC_FLAGS] != 0.0 && (hid & 5) != 0 && d[D_W] > 0.0;
    int scale = (kind & K_SCALE) && (c[XC_SCORR] != 0.0 || (c[XC_SROB] > 0.0 && d[D_SSRC] == 1.0));
    if (!side && !scale)
        return d;
    memcpy(dc, d, RDET * sizeof(double));
    const double *x = row + RO_X;
    if (side) {
        int br = (base == K_B || base == K_BT) ? 0 : 1;
        double dx = x[R_PX] - pose[0], dy = x[R_PY] - pose[1], r = pymax(hypot(dx, dy), 1.0);
        double wh = pymax(r_silhouette(row + RO_E, atan2(dy, dx), NULL) / r - d[D_W], 0.0);
        wh = pymin(wh, 0.5);
        if ((hid & 5) == 1)
            dc[D_Z + br] += 0.5 * wh;                    /* the picture's left is the larger bearing */
        else if ((hid & 5) == 4)
            dc[D_Z + br] -= 0.5 * wh;
        dc[D_R + br] += 0.25 * wh * wh;
    }
    if (scale) {
        int ms = base == K_RBH ? 3 : base == K_NONE ? 0 : base == K_B ? 1 : 2;
        const double *xr = x_of(xc, row);
        double R = dc[D_R + ms];
        if (c[XC_SCORR] != 0.0 && xr[X_ST] > 0.0) {
            double since = xc->t - xr[X_ST];
            if (since > 0.0 && since < d[D_SDT])
                R *= d[D_SDT] / pymax(since, 0.05);
        }
        if (c[XC_SROB] > 0.0 && d[D_SSRC] == 1.0) {
            double nu[4], H[4 * RN], S[16];
            r_innov(sen, K_NONE | K_SCALE, d[D_SDT], x, row + RO_P, pose, d + D_Z + ms, &R, nu, H, S, row + RO_E);
            double need = nu[0] * nu[0] / (c[XC_SROB] * c[XC_SROB]);
            if (isfinite(need) && need > S[0])
                R += need - S[0];
        }
        dc[D_R + ms] = R;
    }
    return dc;
}

/* ---- where a measurement is not linear over what is not known of the place ----
 * A bearing is a linear function of the place only over a small part of the range. A hypothesis whose place is
 * unsure by a good share of its range (a new object that "may be driving at any speed", a second after its first
 * box; an object coasted for seconds) is not a Gaussian in the picture: linearised at its mean it claims bearings
 * it cannot reach (driving along the road it does not cross to the other side of the street) and misses ones it
 * can. Such a hypothesis is cut along the direction in which its place is least known into five narrower
 * Gaussians of the same mean and covariance together, each linearised where it is; its likelihood is their sum and
 * its update their moment-matched posterior. */
#define RSPLIT 5
static const double SPLIT_W[RSPLIT] = {0.0763216491, 0.2474417860, 0.3524731300, 0.2474417860, 0.0763216491};
static const double SPLIT_M[RSPLIT] = {-1.6899729111, -0.8009283834, 0.0, 0.8009283834, 1.6899729111};
/* the spread of the parts' means; each part keeps the rest of the variance (the parts together: exactly 1) */
#define SPLIT_VM (2.0 * (0.0763216491 * 1.6899729111 * 1.6899729111 + 0.2474417860 * 0.8009283834 * 0.8009283834))

/* (x, P) as parts: xk (RSPLIT x N) the parts' means, Pk (N x N) their common covariance; returns how many (1: the
 * hypothesis as it is). (ox, oy): where the sensor is; rel: the share of the range from which it is cut. */
static int r_split(const double *x, const double *P, double ox, double oy, double rel, double *xk, double *Pk)
{
    memcpy(xk, x, RN * sizeof(double));
    memcpy(Pk, P, RNN * sizeof(double));
    if (!(rel > 0.0))
        return 1;
    double a = P[0], b = 0.5 * (P[1] + P[RN]), d = P[RN + 1], lam = r_eig2(a, b, d);
    /* cut when the BEARING is unsure by more than `rel` radians: the place's spread across the line of sight */
    double dx = x[R_PX] - ox, dy = x[R_PY] - oy, r = pymax(hypot(dx, dy), 1.0), ux = -dy / r, uy = dx / r;
    double across = ux * ux * a + 2.0 * ux * uy * b + uy * uy * d;
    if (!(lam > 0.0) || !isfinite(lam) || !(sqrt(pymax(across, 0.0)) > rel * r))
        return 1;
    double vx = lam - d, vy = b;
    if (fabs(vx) + fabs(vy) < 1e-12 * lam) {
        vx = b;
        vy = lam - a;
    }
    double vn = hypot(vx, vy);
    if (!(vn > 0.0)) {
        vx = 1.0;
        vy = 0.0;
        vn = 1.0;
    }
    vx /= vn;
    vy /= vn;
    double c[RN], sl = sqrt(lam);
    for (int k = 0; k < RN; k++)
        c[k] = P[k * RN] * vx + P[k * RN + 1] * vy;
    for (int q = 0; q < RSPLIT; q++)
        for (int k = 0; k < RN; k++)
            xk[q * RN + k] = x[k] + c[k] / sl * SPLIT_M[q];
    double f = SPLIT_VM / lam;
    for (int i = 0; i < RN; i++)
        for (int j = i; j < RN; j++)
            Pk[i * RN + j] = Pk[j * RN + i] = 0.5 * (P[i * RN + j] + P[j * RN + i]) - f * c[i] * c[j];
    return RSPLIT;
}

typedef struct {
    const RSens *sen;
    const double *pose, *cfg;
    double *D;
    Py_ssize_t nd, nm;
    int ng;
    const double *mem;
    double *lam0, *loc, *kmat;       /* per detection: the sensor's density, the map's, the map's kernels (nd x nm) */
    double cph, sph;
    Py_ssize_t *J;                   /* work: nd */
    int *ok;
    double *qs, *dets, *lk;
    const RXc *xc;                   /* the detector's further inputs, or NULL */
    const XBox *bx;                  /* what the scan's boxes hide of the leaf at hand, or NULL */
} RAs;

/* what does not depend on the object: each detection's density of "something else" (sensors.py density; the
 * map of where unexplained detections that became objects have been turning up: tracker `_local_density`) */
static void ras_dets(RAs *q, double nscan)
{
    const RSens *sen = q->sen;
    q->ng = 0;
    for (Py_ssize_t j = 0; j < q->nd; j++) {
        const double *d = q->D + j * RDET;
        q->lam0[j] = r_density(sen, (int)d[D_KIND], d + D_Z);
        if (d[D_LOW] != 0.0 && q->xc != NULL)            /* the low list's own false boxes, per scan */
            q->lam0[j] *= q->xc->c[XC_LOW_EXTRA] / sen->extra;
        if ((int)d[D_GROUP] + 1 > q->ng)
            q->ng = (int)d[D_GROUP] + 1;
        q->loc[j] = 0.0;
        for (Py_ssize_t e = 0; e < q->nm; e++)
            q->kmat[j * q->nm + e] = 0.0;
        int base = (int)d[D_KIND] & 7;
        if (sen->type != SEN_CAMERA || !(base == K_RB || base == K_RBH) || q->nm == 0 || d[D_LOW] != 0.0)
            continue;
        int n = base == K_RB ? 2 : 3;
        double norm = pymax(nscan, 1.0);
        for (int a = 0; a < n; a++)
            norm *= q->cfg[7 + a] * 2.5066;
        for (Py_ssize_t e = 0; e < q->nm; e++) {
            const double *me = q->mem + 5 * e;
            if (me[0] < n)
                continue;
            double s = 0.0;
            for (int a = 0; a < n; a++)
                s += pow((d[D_Z + a] - me[1 + a]) / q->cfg[7 + a], 2.0);
            if (s < 25.0) {
                double v = exp(-0.5 * s) / norm;
                q->loc[j] += v;
                q->kmat[j * q->nm + e] = v;
            }
        }
    }
}

static void r_cell_centre(const double *row, const RSens *sen, const double *pose, double *d);

/* A detection's variances as a leaf reads them: what the detection states (R), and for a camera's rows that see
 * the object's near face what is not known of how far that face is in front of the centre (the leaf's extent). */
static void r_reff(const RSens *sen, int kind, const double *row, const double *pose, const double *R, double *Rf)
{
    memcpy(Rf, R, 4 * sizeof(double));
    int base = kind & 7;
    if (sen->type != SEN_CAMERA || base == K_B || base == K_NONE)
        return;
    const double *x = row + RO_X, *e = row + RO_E, *Pe = row + RO_PE;
    double dx = x[R_PX] - pose[0], dy = x[R_PY] - pose[1], rc = pymax(hypot(dx, dy), 0.5), los = atan2(dy, dx), dda;
    double a = e[2] - los, d = r_half_depth(e, los, &dda), rn = pymax(rc - d, 0.5);
    double vd = 0.25 * (cos(a) * cos(a) * Pe[0] + sin(a) * sin(a) * Pe[4]) + dda * dda * pymin(Pe[8], 0.6);
    if (base == K_BT) {
        double dh = sen->hc - exp(x[R_LH]), g = dh / (rn * rn + dh * dh);
        Rf[1] += g * g * vd;
        return;
    }
    double g = sen->hc / (rn * rn + sen->hc * sen->hc);
    Rf[0] += g * g * vd;
    if (base == K_RBH)
        Rf[2] += vd / (rn * rn);
}

/* detection j as this leaf reads it: a cell of several returns is the leaf's centre by the leaf's own rectangle */
static const double *ras_det(const RAs *q, const double *row, Py_ssize_t j, double *dc)
{
    const double *d = q->D + j * RDET;
    if (q->sen->type != SEN_MICROWAVE)
        return q->xc != NULL ? x_det_leaf(q->xc, q->sen, q->pose, row, d, dc) : d;
    memcpy(dc, d, RDET * sizeof(double));
    r_cell_centre(row, q->sen, q->pose, dc);
    return dc;
}

/* One leaf against every detection of the scan. lr[j]: the logarithm of (pd x the likelihood the leaf's hypotheses
 * give detection j x the box feature) over the density of "something else" there, -inf where the pair may not be
 * (outside the gate, a pedestrian's box on a vehicle, a kind of box this place cannot give). pd: the leaf's
 * probability of being reported; (cx, cy): its place in the car frame; ov: vehicle / person / not known; lbw: the
 * width of its last box where the box feature applies (NaN: not); idnum: its object's id (the map's own entries
 * do not count against it). */
static void ras_row(const RAs *q, const double *row, double pd, double pdlo, double cx, double cy, double ov, double lbw,
                    double idnum, double *lr)
{
    const RSens *sen = q->sen;
    const double *pose = q->pose, *cfg = q->cfg, *dt = q->D;
    Py_ssize_t nd = q->nd, *J = q->J;
    int *ok = q->ok;
    double *qs = q->qs, *dets = q->dets, *lk = q->lk;
    for (Py_ssize_t j = 0; j < nd; j++)
        lr[j] = -INFINITY;
    if (!(pd > 0.0))
        return;
    const double *x = row + RO_X, *P = row + RO_P, *mu = row + RO_MU;
    double zp[4], H[4 * RN], HP[4 * RN], S[16], zm[4], Hm[4 * RN], HPm[4 * RN], Sm[16];
    for (int g = 0; g < q->ng; g++) {
        Py_ssize_t nJ = 0;
        for (Py_ssize_t j = 0; j < nd; j++)
            if ((int)dt[j * RDET + D_GROUP] == g)
                J[nJ++] = j;
        if (nJ == 0)
            continue;
        const double *d0 = dt + J[0] * RDET;
        int kind = (int)d0[D_KIND];
        double sdt = d0[D_SDT];
        /* a box cut at the bottom of the view is of a thing nearer than the view's first ground point
         * (CameraBoxes.takes). A box whose bottom is under a nearer box (the detector's flags) is not: it may be of
         * a thing at any range, and says of it a direction only. XC_FLAGS 2: a far leaf may take such a box; 1: it
         * may not (a direction alone does not tell the cars of a row apart: measured, it costs identities) */
        int cutfar = sen->type == SEN_CAMERA && sen->cover && ((kind & 7) == K_B || (kind & 7) == K_BT) &&
                     !(py_hypot(cx, cy) <= sen->near * sen->gmin);
        if (cutfar && q->xc == NULL)
            continue;
        int anyok = 0;
        for (Py_ssize_t a = 0; a < nJ; a++) {
            ok[a] = dt[J[a] * RDET + D_VEH] * ov >= 0.0;    /* a pedestrian's box does not continue a vehicle */
            if (cutfar && (((int)dt[J[a] * RDET + D_HID] & 24) != 8 || q->xc->c[XC_FLAGS] < 2.0))
                ok[a] = 0;
            anyok |= ok[a];
        }
        if (!anyok)
            continue;
        /* the gate: the object as a whole */
        int m = r_h(sen, kind, sdt, x, pose, zp, H, row + RO_E);
        int ar = r_angle_row(sen, kind);
        double gate = (m >= 1 && m <= 4) ? cfg[m - 1] : 16.0;
        r_hph(H, m, P, HP, S);
        int sing = 0;
        anyok = 0;
        for (Py_ssize_t a = 0; a < nJ; a++) {
            double dc[RDET], Rf[4];
            const double *d = ras_det(q, row, J[a], dc);
            double nuj[4], Sj[16], s;
            r_reff(sen, kind, row, pose, d + D_R, Rf);
            for (int r = 0; r < m; r++)
                nuj[r] = r == ar ? wrap_pi(d[D_Z + r] - zp[r]) : d[D_Z + r] - zp[r];
            memcpy(Sj, S, sizeof(double) * (size_t)(m * m));
            for (int r = 0; r < m; r++)
                Sj[r * m + r] = S[r * m + r] + Rf[r];
            if (r_mahal(Sj, m, nuj, &s, NULL) < 0) {
                sing = 1;
                break;
            }
            ok[a] = ok[a] && s <= gate;
            anyok |= ok[a];
        }
        if (sing || !anyok)
            continue;
        /* what each hypothesis, with its own prediction and weight, gives the detection */
        for (Py_ssize_t a = 0; a < nJ; a++)
            lk[a] = 0.0;
        double c2 = pow(2.0 * PI, (double)m);
        for (int mm = 0; mm < 3; mm++) {
            if (mu[mm] < 1e-4)
                continue;
            double xk[RSPLIT * RN], Pk[RNN];
            int np = r_split(row + RO_XS + mm * RN, row + RO_PS + mm * RNN, pose[0], pose[1], cfg[18], xk, Pk);
            for (int part = 0; part < np; part++) {
                double wpart = np == 1 ? 1.0 : SPLIT_W[part];
                r_h(sen, kind, sdt, xk + part * RN, pose, zm, Hm, row + RO_E);
                r_hph(Hm, m, Pk, HPm, Sm);
                int bad = 0;
                for (Py_ssize_t a = 0; a < nJ; a++) {
                    if (!ok[a])
                        continue;
                    double dc[RDET], Rf[4];
                    const double *d = ras_det(q, row, J[a], dc);
                    double nuj[4], Sj[16];
                    r_reff(sen, kind, row, pose, d + D_R, Rf);
                    for (int r = 0; r < m; r++)
                        nuj[r] = r == ar ? wrap_pi(d[D_Z + r] - zm[r]) : d[D_Z + r] - zm[r];
                    memcpy(Sj, Sm, sizeof(double) * (size_t)(m * m));
                    for (int r = 0; r < m; r++)
                        Sj[r * m + r] = Sm[r * m + r] + Rf[r];
                    if (r_mahal(Sj, m, nuj, &qs[a], &dets[a]) < 0) {
                        bad = 1;
                        break;
                    }
                }
                if (bad)
                    continue;
                for (Py_ssize_t a = 0; a < nJ; a++)
                    if (ok[a] && qs[a] < 60.0 && dets[a] > 0.0)
                        lk[a] += mu[mm] * wpart * exp(-0.5 * pymin(qs[a], 60.0)) / sqrt(c2 * dets[a]);
            }
        }
        for (Py_ssize_t a = 0; a < nJ; a++) {
            if (!ok[a] || lk[a] <= 0.0)
                continue;
            const double *d = dt + J[a] * RDET;
            Py_ssize_t j = J[a];
            double F = 1.0;
            if (!isnan(lbw) && d[D_W] > 0.0 && ar >= 0) {     /* the box as a feature */
                double e = fabs(wrap_pi(d[D_Z + ar] - zp[ar])) / np_maximum(0.5 * (d[D_W] + lbw), 1e-4);
                double sg = cfg[4], t = np_minimum(e / sg, 12.0);
                F = cfg[6] * (cfg[5] * 0.7979 / sg * exp(-0.5 * (t * t)) + (1.0 - cfg[5]) / cfg[6]);
            }
            /* the other story's density here: not counting this object's own first detection */
            double own = 0.0;
            for (Py_ssize_t e = 0; e < q->nm; e++)
                if (q->mem[5 * e + 4] == idnum)
                    own += q->kmat[j * q->nm + e];
            double lam = pymax(q->lam0[j], q->loc[j] - own);
            double v;
            if (q->xc != NULL) {
                /* the detector's further inputs: a box of the low list has its own probability of being given; the
                 * appearance vector against the leaf's memory of its object's; the box's extra rows (the lamp
                 * pair's separation, the width of an axial view) given its other rows, on the leaf as a whole */
                const RXc *xc = q->xc;
                const double *xr = x_of(xc, row);
                double pdj = d[D_LOW] != 0.0 ? pdlo : pd, Fx = 1.0;
                if (q->bx != NULL && q->bx->on) {
                    /* the scan's boxes hide the leaf: all of them in the story "not reported" (pd), all but this one
                     * in the story "it took this box" */
                    const XBox *bx = q->bx;
                    double cv = 0.0, c0 = cfg[14], c1 = cfg[15], fl = cfg[16];
                    for (int p = 0; p < 5; p++)
                        cv += 0.2 * (bx->bj[p] == (int)j ? bx->second[p] : bx->best[p]);
                    cv = pymax(cv, bx->cover_obj);
                    double vj = (cv <= c0 ? 1.0 : cv >= c1 ? fl : 1.0 - (1.0 - fl) * (cv - c0) / (c1 - c0)) * bx->pe;
                    double pj = bx->psen * vj;
                    pdj = d[D_LOW] != 0.0 ? pymax(0.0, bx->klo * (bx->incov * vj - pj)) : pj;
                }
                if (xc->c[XC_APP] != 0.0 && d[D_HASEMB] != 0.0 && xr[X_EN] > 0.0 && xr[X_EK] == xc->c[XC_APP_KIND])
                    Fx *= x_app_lr(xc->c, xc->dark, x_emb_dist(xc->E + (size_t)j * RXEMB, xr + X_EMB), xc->t - xr[X_ET]);
                XAux aux[XAUX];
                int na = x_aux_rows(xc, sen, row, xr, d, pose, aux);
                if (na) {
                    double xt[RN], Pt[RNN], Rf[4], l0, dcx[RDET];
                    const double *dl = ras_det(q, row, j, dcx);
                    memcpy(xt, x, sizeof(xt));
                    memcpy(Pt, P, sizeof(Pt));
                    r_reff(sen, kind, row, pose, dl + D_R, Rf);
                    if (r_take_parts(xt, Pt, sen, kind, sdt, pose, dl + D_Z, Rf, row + RO_E, cfg[18], 0, &l0) == 0)
                        Fx *= x_aux_take(xc, xt, Pt, pose, aux, na, 0, NULL);
                }
                v = log(pdj * lk[a] * F * Fx / lam);
            } else
                v = log(pd * lk[a] * F / lam);
            if (isfinite(v))
                lr[j] = v;
        }
    }
}

/* ---- a detection into a leaf: all of it (within a hypothesis the pairing is not in doubt) ---- */
static inline double clamp_dp(double v)          /* min(0.05, max(-0.05, v)) as Python takes it */
{
    return pymin(0.05, pymax(-0.05, v));
}

static int r_take_parts(double *x, double *P, const RSens *sen, int kind, double sdt, const double *pose, const double *z,
                        const double *Rf, const double *e, double rel, int cons, double *like)
{
    double xk[RSPLIT * RN], Pk[RNN], Pq[RSPLIT][RNN], lq[RSPLIT], tot = 0.0;
    int np = r_split(x, P, pose[0], pose[1], rel, xk, Pk);
    for (int q = 0; q < np; q++) {
        double nu[4], H[4 * RN], S[16], Si[16], K[RN * 4], det, t[4], d2 = 0.0;
        int m = r_innov(sen, kind, sdt, xk + q * RN, Pk, pose, z, Rf, nu, H, S, e);
        if (r_inv(S, m, Si, &det) < 0)
            return -1;
        for (int c = 0; c < m; c++) {
            double acc = 0.0;
            for (int r = 0; r < m; r++)
                acc += nu[r] * Si[r * m + c];
            t[c] = acc;
        }
        for (int c = 0; c < m; c++)
            d2 += t[c] * nu[c];
        lq[q] = (np == 1 ? 1.0 : SPLIT_W[q]) * exp(-0.5 * pymin(d2, 400.0)) / sqrt(pymax(pow(2.0 * PI, (double)m) * det, 1e-30));
        tot += lq[q];
        memcpy(Pq[q], Pk, sizeof(Pk));
        r_gain(Pk, H, m, Si, K);
        if (cons)                                        /* the size and the angle offset as consider parameters */
            for (int c = 0; c < m; c++)
                K[R_LH * m + c] = K[R_DP * m + c] = 0.0;
        r_take(xk + q * RN, Pq[q], K, H, Rf, nu, m, 1.0);
    }
    *like = tot;
    if (np == 1 || !(tot > 0.0)) {
        int best = 0;                                    /* (no part fits at all: the middle one's update) */
        if (np > 1)
            best = RSPLIT / 2;
        memcpy(x, xk + best * RN, RN * sizeof(double));
        memcpy(P, Pq[best], RNN * sizeof(double));
        return 0;
    }
    /* the parts' posteriors as one Gaussian again */
    double xn[RN];
    for (int k = 0; k < RN; k++) {
        double acc = 0.0;
        for (int q = 0; q < np; q++)
            acc += lq[q] / tot * xk[q * RN + k];
        xn[k] = acc;
    }
    for (int i = 0; i < RN; i++)
        for (int j = i; j < RN; j++) {
            double acc = 0.0;
            for (int q = 0; q < np; q++)
                acc += lq[q] / tot * (Pq[q][i * RN + j] + (xk[q * RN + i] - xn[i]) * (xk[q * RN + j] - xn[j]));
            P[i * RN + j] = P[j * RN + i] = acc;
        }
    memcpy(x, xn, sizeof(xn));
    return 0;
}

/* The row takes detection d of the sensor at `pose`: every motion hypothesis and the free estimate in full (Joseph
 * form; a hypothesis whose place is too unsure for one linearisation as a sum of parts: r_split), the hypotheses
 * weighed by how well each had predicted it (from `hold` sightings on, unless the sensor measures speed or the
 * detection a scale rate; between the moving ones always; never below `floor`), the combination. *nis: the
 * object's normalised innovation before the update; *dl: what the fit adds to the log odds that the object
 * exists. -1: an innovation covariance is singular (nothing changed). */
static int r_update_full(double *row, const RSens *sen, const double *pose, const double *d, double hold, double floor_,
                         double rel, int cons, double *nis, double *dl, const XAux *aux, int naux, int *used)
{
    double *mu = row + RO_MU;
    int kind = (int)d[D_KIND];
    double sdt = d[D_SDT], hits = row[RO_M + M_HITS];
    double like[3], Rf[4], lf;
    r_reff(sen, kind, row, pose, d + D_R, Rf);
    double nu0[4], H0[4 * RN], S0[16], d20;
    int m0 = r_innov(sen, kind, sdt, row + RO_X, row + RO_P, pose, d + D_Z, Rf, nu0, H0, S0, row + RO_E);
    if (r_mahal(S0, m0, nu0, &d20, NULL) < 0)
        return -1;
    for (int j = 0; j < 3; j++) {
        double *x = row + RO_XS + j * RN, *P = row + RO_PS + j * RNN;
        if (r_take_parts(x, P, sen, kind, sdt, pose, d + D_Z, Rf, row + RO_E, rel,
                         cons && sen->type == SEN_CAMERA && (j != R_STAND || cons >= 2), &like[j]) < 0)
            like[j] = 0.0;
        else if (naux)                                   /* the box's extra rows, after its others (x_aux_take) */
            like[j] *= x_aux_take(sen->xc, x, P, pose, aux, naux, cons && (j != R_STAND || cons >= 2), used);
        if (j == R_STAND)
            for (int k = R_VX; k <= R_AY; k++)
                x[k] = 0.0;
        x[R_DP] = clamp_dp(x[R_DP]);
    }
    if (r_take_parts(row + RO_XF, row + RO_PF, sen, kind, sdt, pose, d + D_Z, Rf, row + RO_E, rel, 0, &lf) == 0) {
        if (naux)
            x_aux_take(sen->xc, row + RO_XF, row + RO_PF, pose, aux, naux, 0, NULL);
        row[RO_XF + R_DP] = clamp_dp(row[RO_XF + R_DP]);
    }
    double tot = (mu[0] * like[0] + mu[1] * like[1]) + mu[2] * like[2];
    double w[3] = {mu[0] * like[0], mu[1] * like[1], mu[2] * like[2]};
    /* (a box of the low-score list does not say how the object moves: its edges are too sloppy to weigh "it
     * stands" against "it drives"; the hypotheses keep their probabilities) */
    if (d[D_LOW] != 0.0)
        tot = 0.0;
    if (tot > 0.0 && (sen->type == SEN_MICROWAVE || hits >= hold || (kind & K_SCALE))) {
        double nm[3], s2 = 0.0, post[3];
        for (int j = 0; j < 3; j++) {
            double v = w[j] / tot;
            post[j] = v;
            nm[j] = (isnan(v) || v >= floor_) ? v : floor_;
            s2 += nm[j];
        }
        /* A hypothesis the detection has all but ruled out is kept at the floor: "it may still turn out so". What
         * is kept is not its old story (a standing hypothesis of a thing that has driven on for seconds stands
         * where the thing was then, and a detection taken from there is nonsense), but the possibility that the
         * thing does so FROM WHERE IT IS: the share that the floor adds is given the other hypotheses' state
         * (for "it stands": their place, no speed). */
        for (int j = 0; j < 3; j++) {
            if (isnan(post[j]) || post[j] >= floor_)
                continue;
            int a = (j + 1) % 3, b = (j + 2) % 3;
            double wa = post[a], wb = post[b], ws = wa + wb, keep = post[j] / floor_;
            if (!(ws > 0.0))
                continue;
            wa /= ws;
            wb /= ws;
            double *xj = row + RO_XS + j * RN, *Pj = row + RO_PS + j * RNN;
            const double *xa = row + RO_XS + a * RN, *xb = row + RO_XS + b * RN;
            const double *Pa = row + RO_PS + a * RNN, *Pb = row + RO_PS + b * RNN;
            double xd[RN], xn[RN], Pn[RNN];
            for (int k = 0; k < RN; k++)
                xd[k] = wa * xa[k] + wb * xb[k];
            if (j == R_STAND)
                for (int k = R_VX; k <= R_AY; k++)
                    xd[k] = 0.0;
            for (int k = 0; k < RN; k++)
                xn[k] = keep * xj[k] + (1.0 - keep) * xd[k];
            for (int r = 0; r < RN; r++)
                for (int c = r; c < RN; c++) {
                    double da = xa[r] - xd[r], db = xb[r] - xd[r], ca = xa[c] - xd[c], cb = xb[c] - xd[c];
                    if (j == R_STAND && (r >= R_VX && r <= R_AY))
                        da = db = 0.0;
                    if (j == R_STAND && (c >= R_VX && c <= R_AY))
                        ca = cb = 0.0;
                    double pd = wa * (Pa[r * RN + c] + da * ca) + wb * (Pb[r * RN + c] + db * cb);
                    if (j == R_STAND && ((r >= R_VX && r <= R_AY) || (c >= R_VX && c <= R_AY)))
                        pd = r == c ? Pj[r * RN + c] : 0.0;      /* its own (small) spread on speed stays */
                    double v = keep * (Pj[r * RN + c] + (xj[r] - xn[r]) * (xj[c] - xn[c])) +
                               (1.0 - keep) * (pd + (xd[r] - xn[r]) * (xd[c] - xn[c]));
                    Pn[r * RN + c] = Pn[c * RN + r] = v;
                }
            memcpy(xj, xn, sizeof(xn));
            memcpy(Pj, Pn, sizeof(Pn));
        }
        for (int j = 0; j < 3; j++)
            mu[j] = nm[j] / s2;
    } else if (tot > 0.0 && w[1] + w[2] > 0.0) {
        double mv = mu[1] + mu[2], ws = w[1] + w[2];
        for (int j = 1; j < 3; j++) {
            double v = mv * w[j] / ws;
            mu[j] = (isnan(v) || v >= floor_) ? v : floor_;
        }
        double s2 = (mu[0] + mu[1]) + mu[2];
        for (int j = 0; j < 3; j++)
            mu[j] = mu[j] / s2;
    }
    double like_all = pymax(tot, 1e-30);
    *dl = pymin(pymax(log(pymax(sen->p_d * like_all / r_density(sen, kind, d + D_Z), 1e-9)), log(1.5)), log(20.0));
    *nis = d20 / m0;
    r_combine(row + RO_XS, row + RO_PS, mu, row + RO_X, row + RO_P);
    return 0;
}

/* ---- an object first seen (sensors.py `start`, tracker.py: what a new object may be doing) ---- */
#define RKB 12                   /* what the detection's kind can do, for an object started by it: a_lon, a_lat, v_max,
                                    v_still, p_stand, v0_std, v0_lat, h, h_rel, width, length (0: no direction of its
                                    own), 1 if the kind is known */
enum { KB_ALON, KB_ALAT, KB_VMAX, KB_VSTILL, KB_PSTAND, KB_V0, KB_V0LAT, KB_H, KB_HREL, KB_WID, KB_LEN, KB_KNOWN };
#define RBCFG 8                  /* new_exist, p_any_side, side_m, with_rel, with_std, p_ca0, mu_floor, births allowed */

/* models.side_on: the probability that a vehicle whose whole box is `aspect` times as wide as tall, seen at
 * `bearing`, points across the car's direction; NaN: the kind or the box does not say */
static double r_side_on(const double *kb, double aspect, double bearing)
{
    if (!(kb[KB_LEN] > 0.0) || !(aspect > 0.0))
        return NAN;
    double w = kb[KB_WID], l = kb[KB_LEN], cb = fabs(cos(bearing)), sb = fabs(sin(bearing));
    double along = (w * cb + l * sb) / kb[KB_H], across = (w * sb + l * cb) / kb[KB_H];
    double la = pow(log(aspect / along) / 0.25, 2.0), lc = pow(log(aspect / across) / 0.25, 2.0);
    double mn = pymin(la, lc);
    double pa = exp(-0.5 * (la - mn)), pc = exp(-0.5 * (lc - mn));
    return pc / (pa + pc);
}

/* U diag(a, b) U^T into the 2 x 2 block (r, r) of P (N x N), U the rotation by the angle of (c, s) */
static void r_rot_block(double *P, int r, double c, double s, double a, double b)
{
    double p00 = c * c * a + s * s * b, p11 = s * s * a + c * c * b, p01 = c * s * (a - b);
    P[r * RN + r] = p00;
    P[(r + 1) * RN + r + 1] = p11;
    P[r * RN + r + 1] = P[(r + 1) * RN + r] = p01;
}

static void r_extent_birth(double *row, const RSens *sen, const double *d, const double *pose, const double *kb, double ps);
static void r_scatter(double L, double W, double th, double from, double *sc, double *mean);
static void r_sensor_at(const RSens *sen, const double *pose, double *sx, double *sy, double *axis);

static inline int wl_front(double ps) { return isnan(ps) || ps <= 0.5; }

/* The row of an object first seen in detection d (kind kb), all of it: the place from the detection, what its kind
 * may be doing as three stories (it stands; it drives with the traffic; it comes the other way, or crosses when
 * its box shows it side-on), the free estimate, the probabilities, the first scale rate, the extent. `t`: the
 * scan's time, `src`: the sensor's number, `level`: the scan's. Returns 1, or 0 when a detection of this kind
 * cannot start an object (nothing written). */
static int r_birth(double *row, const RSens *sen, const double *d, const double *pose, const double *kb,
                   const double *bc, double t, int src, long long level)
{
    int kind = (int)d[D_KIND], base = kind & 7;
    const double *z = d + D_Z, *R = d + D_R;
    double x0[RN], P0[RNN];
    memset(x0, 0, sizeof(x0));
    memset(P0, 0, sizeof(P0));
    /* sensors._prior */
    P0[R_VX * RN + R_VX] = P0[R_VY * RN + R_VY] = kb[KB_V0] * kb[KB_V0];
    P0[R_AX * RN + R_AX] = P0[R_AY * RN + R_AY] = pow(0.6 * kb[KB_ALON], 2.0);
    P0[R_DP * RN + R_DP] = pow(0.6 * (PI / 180.0), 2.0) + pow(0.2 * (PI / 180.0), 2.0);
    P0[R_LH * RN + R_LH] = kb[KB_HREL] * kb[KB_HREL];
    x0[R_LH] = log(kb[KB_H]);
    double bearing = NAN;
    if (sen->type == SEN_CAMERA) {
        double hc = sen->hc, p_dp = sen->bias_var + sen->dp_var;
        P0[R_DP * RN + R_DP] = p_dp;
        if (base == K_BT) {
            /* its feet are out of the picture, its top is not: for a kind whose height is known well and differs
             * enough from the camera's own the top edge says how far it is */
            double ho = kb[KB_H], dh = hc - ho;
            if (kb[KB_KNOWN] == 0.0 || fabs(dh) < sen->top_rel * kb[KB_HREL] * ho)
                return 0;
            double a = z[1] - sen->bias;
            if (a * dh <= 0.0 || fabs(tan(a)) < 1e-4)
                return 0;
            double r = dh / tan(a);
            if (!(1.0 < r && r <= sen->max_range))
                return 0;
            if (sen->grade_k > 0.0)                      /* (the road's vertical curvature: see below) */
                P0[R_DP * RN + R_DP] += pow(0.5 * sen->grade_k * r, 2.0);
            double b = z[0], dr_dh = -ho / tan(a), dr_da = -dh / (sin(a) * sin(a));
            double var_r = dr_dh * dr_dh * kb[KB_HREL] * kb[KB_HREL] + dr_da * dr_da * (R[1] + sen->bias_var);
            if (var_r > pow(sen->top_range * r, 2.0))
                return 0;
            double ang = b + pose[2], ca = cos(ang), sa = sin(ang);
            x0[R_PX] = pose[0] + r * ca;
            x0[R_PY] = pose[1] + r * sa;
            r_rot_block(P0, R_PX, ca, sa, var_r, R[0] * r * r + 0.05);
            double c_lh = dr_dh * kb[KB_HREL] * kb[KB_HREL];
            P0[R_PX * RN + R_LH] = P0[R_LH * RN + R_PX] = ca * c_lh;
            P0[R_PY * RN + R_LH] = P0[R_LH * RN + R_PY] = sa * c_lh;
            bearing = b;
        } else if (base == K_RB || base == K_RBH) {
            double b = z[1];
            /* the road's range: the angle, the known part of its error taken out; never beyond the sensor's reach */
            double ang = pymax(z[0] - sen->bias, atan2(hc, sen->max_range));
            double rg = hc / tan(ang), J = (rg * rg + hc * hc) / hc;
            if (sen->grade_k > 0.0) {
                /* The road ahead is not the plane under the car: with a vertical curvature k the ground at distance
                 * x lies k x^2 / 2 off it, and the ground angle is off by k x / 2. That is not noise from box to
                 * box: it is this object's own offset (the state dp), unknown at birth by sigma_k x / 2 on top of
                 * what every object has. (As row noise it would be averaged away over the object's boxes.) */
                p_dp += pow(0.5 * sen->grade_k * rg, 2.0);
                P0[R_DP * RN + R_DP] = p_dp;
            }
            double var_g = J * J * (R[0] + p_dp);
            double r = rg, var_r = var_g, wg = 1.0, rh = 0.0;
            if (base == K_RBH) {
                /* a second range, independent of the first: the object's own size */
                rh = exp(sen->logf + x0[R_LH] - z[2]) / pymax(cos(b), 0.3);
                double var_h = rh * rh * (kb[KB_HREL] * kb[KB_HREL] + R[2]);
                wg = var_h / (var_g + var_h);
                r = wg * rg + (1.0 - wg) * rh;
                var_r = var_g * var_h / (var_g + var_h);
            }
            double a = b + pose[2], ca = cos(a), sa = sin(a);
            x0[R_PX] = pose[0] + r * ca;
            x0[R_PY] = pose[1] + r * sa;
            r_rot_block(P0, R_PX, ca, sa, var_r, R[1] * r * r + 0.05);
            double c_dp = wg * J * p_dp, c_lh = (1.0 - wg) * rh * kb[KB_HREL] * kb[KB_HREL];
            P0[R_PX * RN + R_DP] = P0[R_DP * RN + R_PX] = ca * c_dp;
            P0[R_PY * RN + R_DP] = P0[R_DP * RN + R_PY] = sa * c_dp;
            P0[R_PX * RN + R_LH] = P0[R_LH * RN + R_PX] = ca * c_lh;
            P0[R_PY * RN + R_LH] = P0[R_LH * RN + R_PY] = sa * c_lh;
            bearing = b;
        } else
            return 0;
    } else if (sen->type == SEN_MICROWAVE) {
        double c = cos(pose[2]), sn = sin(pose[2]);
        double ox = c * sen->mx - sn * sen->my, oy = sn * sen->mx + c * sen->my;
        double sx = pose[0] + ox, sy = pose[1] + oy, svx = pose[3] - pose[5] * oy, svy = pose[4] + pose[5] * ox;
        double r = z[0], a = z[1] + sen->yaw + pose[2], ca = cos(a), sa = sin(a);
        x0[R_PX] = sx + r * ca;
        x0[R_PY] = sy + r * sa;
        double vr = z[2] + ca * svx + sa * svy;         /* the object's own speed along the line of sight */
        x0[R_VX] = vr * ca;
        x0[R_VY] = vr * sa;
        r_rot_block(P0, R_PX, ca, sa, R[0] + 0.1, R[1] * r * r + 0.1);
        /* along the line of sight the speed is measured; across it nothing is known */
        r_rot_block(P0, R_VX, ca, sa, 4.0 * R[2] + 0.5, kb[KB_V0] * kb[KB_V0]);
    } else {
        double c = cos(pose[2]), sn = sin(pose[2]);
        x0[R_PX] = pose[0] + c * z[0] - sn * z[1];
        x0[R_PY] = pose[1] + sn * z[0] + c * z[1];
        r_rot_block(P0, R_PX, c, sn, R[0], R[1]);
    }
    for (int j = 0; j < 3; j++) {
        memcpy(row + RO_XS + j * RN, x0, sizeof(x0));
        memcpy(row + RO_PS + j * RNN, P0, sizeof(P0));
    }
    for (int k = R_VX; k <= R_AY; k++)
        row[RO_XS + R_STAND * RN + k] = 0.0;
    memcpy(row + RO_XF, x0, sizeof(x0));
    memcpy(row + RO_PF, P0, sizeof(P0));
    double *mu = row + RO_MU, ps = NAN;
    double cph = cos(pose[2]), sph = sin(pose[2]);
    if (sen->type == SEN_MICROWAVE) {
        int moving = hypot(x0[R_VX], x0[R_VY]) > 1.0;
        mu[0] = moving ? 0.05 : 0.7;
        mu[1] = moving ? 0.8 : 0.25;
        mu[2] = moving ? 0.15 : 0.05;
    } else {
        double pst = kb[KB_PSTAND], dxo = x0[R_PX] - pose[0], dyo = x0[R_PY] - pose[1];
        double pa = fabs(-sph * dxo + cph * dyo) > bc[2] ? bc[1] : bc[5];
        mu[0] = pst;
        mu[1] = (1.0 - pst) * (1.0 - pa);
        mu[2] = (1.0 - pst) * pa;
        /* What a new object may be doing, before anything is known, is not one blur around "no speed": it stands
         * (its own hypothesis), or it drives with the traffic (the steady hypothesis: about the car's own velocity,
         * along the road the car is on), or it comes the other way (the manoeuvring hypothesis: about the car's
         * velocity reversed, and wider across). The faster the car itself drives, the more that says about the
         * traffic's speed; standing, nothing. A vehicle whose box shows it side-on does neither: it crosses. */
        int wl = kb[KB_LEN] > 0.0;
        double v0 = kb[KB_V0], lat = pow(pymin(kb[KB_V0LAT], v0), 2.0), big = v0 * v0;
        if (d[D_ASPECT] > 0.0 && !isnan(bearing))
            ps = r_side_on(kb, d[D_ASPECT], bearing);
        double ve = wl ? pymin(hypot(pose[3], pose[4]), kb[KB_VMAX]) : 0.0;
        if (isnan(ps) && wl)
            ps = 0.0;
        double *xc = row + RO_XS + R_CV * RN, *xa = row + RO_XS + R_CA * RN;
        if (isnan(ps)) {                                 /* a kind without a direction of its own (a person) */
            xc[R_VX] = xc[R_VY] = 0.0;
            r_rot_block(row + RO_PS + R_CV * RNN, R_VX, cph, sph, big, lat);
            r_rot_block(row + RO_PF, R_VX, cph, sph, big, lat);
        } else {
            double w_ = pow(pymax(bc[3] * ve + bc[4], v0 - ve), 2.0), lat2 = pow(pymin(2.0 * kb[KB_V0LAT], v0), 2.0);
            double m0 = (1.0 - ps) * ve;
            xc[R_VX] = m0 * cph;
            xc[R_VY] = m0 * sph;
            xa[R_VX] = -m0 * cph;
            xa[R_VY] = -m0 * sph;
            if (sen->lcol && d[D_LAMPC] == 2.0) {
                /* white lamps: its front faces the camera. If it drives, it comes the other way: both moving
                 * stories start from the car's speed reversed ("it stands" keeps its share: a car parked facing
                 * the traffic) */
                xc[R_VX] = -m0 * cph;
                xc[R_VY] = -m0 * sph;
            }
            r_rot_block(row + RO_PS + R_CV * RNN, R_VX, cph, sph, (1.0 - ps) * w_ + ps * lat + ps * (1.0 - ps) * ve * ve,
                        (1.0 - ps) * lat + ps * big);
            r_rot_block(row + RO_PS + R_CA * RNN, R_VX, cph, sph, (1.0 - ps) * w_ + ps * lat2 + ps * (1.0 - ps) * ve * ve,
                        (1.0 - ps) * lat2 + ps * big);
            r_rot_block(row + RO_PF, R_VX, cph, sph, big, ps == 0.0 ? lat : (1.0 - ps) * lat + ps * big);
        }
    }
    memset(row + RO_ST, 0, (size_t)(RO_M - RO_ST) * sizeof(double));
    row[RO_ST + 0] = row[RO_ST + 4] = row[RO_ST + 8] = 1.0;
    double *M = row + RO_M;
    memset(M, 0, RMETA * sizeof(double));
    /* its extent; the detection gave the near face (a camera's box) or the sides that face the sensor (a radar's
     * returns): the place of every hypothesis is the centre behind them */
    r_extent_birth(row, sen, d, pose, kb, ps);
    if (sen->lcol && d[D_LAMPC] == 2.0 && wl_front(ps))
        row[RO_E + 2] = wrap_pi(row[RO_E + 2] + PI);     /* the front is the end that shows white lamps */
    {
        double sx, sy, axis, off[2] = {0.0, 0.0};
        r_sensor_at(sen, pose, &sx, &sy, &axis);
        double lx = x0[R_PX] - sx, ly = x0[R_PY] - sy, lr = pymax(hypot(lx, ly), 0.5), los = atan2(ly, lx);
        if (sen->type == SEN_CAMERA) {
            double dd = r_half_depth(row + RO_E, los, NULL);
            off[0] = dd * lx / lr;
            off[1] = dd * ly / lr;
        } else if (sen->type == SEN_MICROWAVE) {
            double sc[3], mm[2];
            r_scatter(row[RO_E], row[RO_E + 1], row[RO_E + 2], los + PI, sc, mm);
            off[0] = -mm[0];
            off[1] = -mm[1];
        }
        for (int j = 0; j < 4; j++) {
            double *xx = j == 3 ? row + RO_XF : row + RO_XS + j * RN;
            xx[R_PX] += off[0];
            xx[R_PY] += off[1];
        }
    }
    M[M_T] = M[M_THIT] = M[M_RANGE_T] = t;
    M[M_L] = log(bc[0] / (1.0 - bc[0])) + d[D_SCORE];
    M[M_HITS] = 1.0;
    M[M_BOXW] = d[D_W] > 0.0 ? d[D_W] : NAN;
    M[M_BOXT] = t;
    M[M_BOXS] = (double)src;
    M[M_NIS] = NAN;
    M[M_DET] = -1.0;
    M[M_LEVEL] = (double)level;
    M[M_VIS] = 1.0;
    M[M_PD] = sen->p_d;
    M[M_SRC] = (double)src;
    if ((kind & K_SCALE) && sen->type == SEN_CAMERA) {
        /* its first box already says how fast it comes nearer: its speed along the line of sight over its
         * distance, before a second box exists. Every hypothesis takes it (for "it stands" it is a range: the
         * car's own speed over the distance), and how well each had expected it weighs them at once. */
        int m = 0;
        if (base == K_RB)
            m = 2;
        else if (base == K_RBH)
            m = 3;
        else if (base == K_BT)
            m = 2;
        else if (base == K_B)
            m = 1;
        double zs = z[m], Rs = R[m], like[3] = {0.0, 0.0, 0.0};
        for (int j = 0; j < 4; j++) {
            double *x = j == 3 ? row + RO_XF : row + RO_XS + j * RN, *P = j == 3 ? row + RO_PF : row + RO_PS + j * RNN;
            double nu[4], H[4 * RN], S[16], K[RN * 4];
            r_innov(sen, K_NONE | K_SCALE, d[D_SDT], x, P, pose, &zs, &Rs, nu, H, S, row + RO_E);
            if (!(S[0] > 0.0) || !isfinite(S[0]))
                continue;
            double Si = 1.0 / S[0];
            r_gain(P, H, 1, &Si, K);
            r_take(x, P, K, H, &Rs, nu, 1, 1.0);
            if (j == 3)
                break;
            like[j] = exp(-0.5 * pymin(nu[0] * nu[0] / S[0], 400.0)) / sqrt(2.0 * PI * S[0]);
            if (j == R_STAND)
                for (int k = R_VX; k <= R_AY; k++)
                    x[k] = 0.0;
        }
        double w[3] = {mu[0] * like[0], mu[1] * like[1], mu[2] * like[2]};
        double ws = (w[0] + w[1]) + w[2];
        if (ws > 0.0) {
            double s2 = 0.0;
            for (int j = 0; j < 3; j++) {
                w[j] = pymax(w[j] / ws, bc[6]);
                s2 += w[j];
            }
            for (int j = 0; j < 3; j++)
                mu[j] = w[j] / s2;
        }
    }
    r_combine(row + RO_XS, row + RO_PS, mu, row + RO_X, row + RO_P);
    return 1;
}

/* ---- the extent of a leaf: length, width (m) and heading (rad, ground frame: where its front points) ----
 * A rectangle on the ground, kept beside the motion state as a Gaussian of its own (the motion state and the
 * shape are taken as independent, as the random-matrix trackers do): born from what the kind is like, turned to
 * the direction of travel when the object moves and held when it stands, measured by a camera box's width (the
 * rectangle's silhouette seen from the camera) and by the scatter of a microwave radar's returns over the sides
 * that face it. docs/RADAR.md, section 4. */
enum { E_LEN, E_WID, E_HDG };

static void r_extent_birth(double *row, const RSens *sen, const double *d, const double *pose, const double *kb, double ps)
{
    double *e = row + RO_E, *Pe = row + RO_PE;
    int wl = kb[KB_LEN] > 0.0, known = kb[KB_KNOWN] != 0.0;
    memset(Pe, 0, 9 * sizeof(double));
    if (wl) {
        e[E_LEN] = kb[KB_LEN];
        e[E_WID] = kb[KB_WID];
        Pe[0] = pow(0.22 * e[E_LEN], 2.0);
        Pe[4] = pow(0.10 * e[E_WID], 2.0);
    } else if (known) {                                  /* a person, a two-wheeler: small, no long axis known */
        e[E_LEN] = e[E_WID] = 0.6;
        Pe[0] = 0.5 * 0.5;
        Pe[4] = 0.15 * 0.15;
    } else {                                             /* not known what it is: as a car, and wide open */
        e[E_LEN] = 4.5;
        e[E_WID] = 1.8;
        Pe[0] = 2.0 * 2.0;
        Pe[4] = 0.5 * 0.5;
    }
    /* a vehicle points along the road the car is on, or across it when its box shows it side-on; how sure that is
     * follows how clearly the box says which (a person has no direction until it walks) */
    double side = (!isnan(ps) && ps > 0.5) ? 0.5 * PI : 0.0;
    e[E_HDG] = wrap_pi(pose[2] + side);
    if (wl && !isnan(ps))
        Pe[8] = pow(0.35 + 4.8 * ps * (1.0 - ps), 2.0);
    else if (wl)
        Pe[8] = 0.6 * 0.6;
    else
        Pe[8] = PI * PI / 3.0;
    row[RO_M + M_SENSE] = 0.0;                           /* which end is the front is not known yet */
}

/* one scalar measurement z = h(e) + noise (variance R), row H, into (e, Pe), Joseph form; the innovation given.
 * Returns the squared normalised innovation (the measurement is not taken beyond `gate`). */
static double r_extent_take(double *e, double *Pe, const double *H, double nu, double R, double gate)
{
    double PH[3], S = R;
    for (int i = 0; i < 3; i++) {
        PH[i] = Pe[i * 3] * H[0] + Pe[i * 3 + 1] * H[1] + Pe[i * 3 + 2] * H[2];
        S += H[i] * PH[i];
    }
    if (!(S > 0.0) || !isfinite(S))
        return INFINITY;
    double d2 = nu * nu / S;
    if (d2 > gate)
        return d2;
    double K[3], A[9], AP[9], Pn[9];
    for (int i = 0; i < 3; i++)
        K[i] = PH[i] / S;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            A[i * 3 + j] = (i == j ? 1.0 : 0.0) - K[i] * H[j];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            AP[i * 3 + j] = A[i * 3] * Pe[j] + A[i * 3 + 1] * Pe[3 + j] + A[i * 3 + 2] * Pe[6 + j];
    for (int i = 0; i < 3; i++)
        for (int j = i; j < 3; j++)
            Pn[i * 3 + j] = Pn[j * 3 + i] = AP[i * 3] * A[j * 3] + AP[i * 3 + 1] * A[j * 3 + 1] + AP[i * 3 + 2] * A[j * 3 + 2] +
                                            K[i] * R * K[j];
    for (int i = 0; i < 3; i++)
        e[i] += K[i] * nu;
    memcpy(Pe, Pn, sizeof(Pn));
    return d2;
}

static void r_extent_bounds(double *e)
{
    e[E_LEN] = pymin(25.0, pymax(0.3, e[E_LEN]));
    e[E_WID] = pymin(3.2, pymax(0.3, e[E_WID]));
    e[E_HDG] = wrap_pi(e[E_HDG]);
}

/* The extent over a step of dt: the size stays; the heading of a thing that moves is its direction of travel (a
 * measurement as good as the velocity's direction is known, weighed by how surely it moves at all; the first
 * time it says which end is the front), the heading of a thing that stands is held. */
static void r_extent_step(double *row, double dt, const double *kin)
{
    double *e = row + RO_E, *Pe = row + RO_PE;
    const double *x = row + RO_X, *P = row + RO_P, *mu = row + RO_MU;
    double adt = fabs(dt), vx = x[R_VX], vy = x[R_VY], v = hypot(vx, vy), pm = 1.0 - mu[R_STAND];
    /* a moving thing may turn (as fast as its kind corners at that speed); a standing one does not */
    double wmax = pm * pymin(kin[1] / pymax(v, 2.0), 0.8);
    Pe[8] += pow(0.01, 2.0) * adt + pow(wmax * adt, 2.0);
    Pe[0] += pow(0.03, 2.0) * adt;                       /* (the size is not held to the last centimetre for ever) */
    Pe[4] += pow(0.01, 2.0) * adt;
    if (v > 2.5 && pm > 0.5) {
        double c = vx / v, s = vy / v;
        double vp = s * s * P[R_VX * RN + R_VX] - 2.0 * c * s * P[R_VX * RN + R_VY] + c * c * P[R_VY * RN + R_VY];
        double R = (pymax(vp, 0.0) / (v * v) + 0.03 * 0.03) / pm;
        double th = atan2(vy, vx), nu = wrap_pi(th - e[E_HDG]);
        if (fabs(nu) > 0.5 * PI && row[RO_M + M_SENSE] == 0.0) {
            e[E_HDG] = wrap_pi(e[E_HDG] + PI);          /* the other end was the front */
            nu = wrap_pi(th - e[E_HDG]);
        }
        double H[3] = {0.0, 0.0, 1.0};
        r_extent_take(e, Pe, H, nu, R, INFINITY);
        if (R < 0.3 * 0.3 && v > 2.5)
            row[RO_M + M_SENSE] = 1.0;
    }
    r_extent_bounds(e);
}

/* the rectangle's silhouette seen from the direction `los` (ground angle from the viewer to its centre): its
 * width across the line of sight, and the derivatives by length, width, heading */
static double r_silhouette(const double *e, double los, double *H)
{
    double a = e[E_HDG] - los, sa = sin(a), ca = cos(a), as = fabs(sa), ac = fabs(ca);
    if (H) {
        H[0] = as;
        H[1] = ac;
        H[2] = e[E_LEN] * (sa >= 0.0 ? ca : -ca) - e[E_WID] * (ca >= 0.0 ? sa : -sa);
    }
    return e[E_LEN] * as + e[E_WID] * ac;
}

/* The scatter of the returns of a rectangle (length L, width W, heading th) seen from the ground direction
 * `from` (the direction from its centre to the sensor): the returns lie on the two sides that face the sensor,
 * each in proportion to how broad it looks from there, evenly along it. Out: the covariance of their places
 * (sxx, syy, sxy; ground axes) and the mean's offset from the centre (mx, my). */
static void r_scatter(double L, double W, double th, double from, double *sc, double *mean)
{
    double a = from - th, dbx = cos(a), dby = sin(a);   /* the sensor's direction in the body's axes */
    double sx = dbx >= 0.0 ? 1.0 : -1.0, sy = dby >= 0.0 ? 1.0 : -1.0;
    /* a side returns as it is long, less when it is seen at a grazing angle (never nothing: wheel arches, sills) */
    double wL = L * (0.4 * fabs(dby) + 0.6), wW = W * (0.4 * fabs(dbx) + 0.6), tot = wL + wW;
    double pc = 0.3, pL = tot > 1e-9 ? (1.0 - pc) * wL / tot : 0.35, pW = 1.0 - pc - pL;
    /* the long side: x even over L at y = sy W / 2; the short side: y even over W at x = sx L / 2; the corner
     * between them, the nearest point, returns most */
    double hx = sx * 0.5 * L, hy = sy * 0.5 * W;
    double mxb = (pW + pc) * hx, myb = (pL + pc) * hy;
    double vx = pL * L * L / 12.0 + (pW + pc) * hx * hx - mxb * mxb;
    double vy = pW * W * W / 12.0 + (pL + pc) * hy * hy - myb * myb;
    double cxy = pc * hx * hy - mxb * myb;
    double c = cos(th), s = sin(th);
    sc[0] = c * c * vx - 2.0 * c * s * cxy + s * s * vy;
    sc[1] = s * s * vx + 2.0 * c * s * cxy + c * c * vy;
    sc[2] = c * s * (vx - vy) + (c * c - s * s) * cxy;
    if (mean) {
        mean[0] = c * mxb - s * myb;
        mean[1] = s * mxb + c * myb;
    }
}

/* the sensor's place on the ground and the angle of its axis */
static void r_sensor_at(const RSens *sen, const double *pose, double *sx, double *sy, double *axis)
{
    double c = cos(pose[2]), sn = sin(pose[2]);
    if (sen->type == SEN_MICROWAVE) {
        *sx = pose[0] + c * sen->mx - sn * sen->my;
        *sy = pose[1] + sn * sen->mx + c * sen->my;
        *axis = pose[2] + sen->yaw;
    } else {
        *sx = pose[0];
        *sy = pose[1];
        *axis = pose[2];
    }
}

/* What a detection says about the extent of the leaf that took it (after the motion state took it). A camera box
 * whose sides are both in the picture: its width as an angle is the silhouette over the range. A cell of several
 * returns of a microwave radar: the scatter of their places is that of the two sides that face the sensor. */
static void r_extent_update(double *row, const RSens *sen, const double *d, const double *pose, const double *kin)
{
    double *e = row + RO_E, *Pe = row + RO_PE;
    const double *x = row + RO_X, *P = row + RO_P;
    double sx, sy, axis;
    r_sensor_at(sen, pose, &sx, &sy, &axis);
    double dx = x[R_PX] - sx, dy = x[R_PY] - sy, r = pymax(hypot(dx, dy), 1.0), los = atan2(dy, dx);
    if (sen->type == SEN_CAMERA && d[D_W] > 0.0 && d[D_WOK] > 0.0) {
        double H[3], wp = r_silhouette(e, los, H) / r;
        for (int i = 0; i < 3; i++)
            H[i] /= r;
        /* how well the range is known (along the line of sight) is how well the box's width says a size */
        double ux = dx / r, uy = dy / r;
        double vr = ux * ux * P[0] + 2.0 * ux * uy * P[1] + uy * uy * P[RN + 1];
        double R = pow(0.06 * d[D_W], 2.0) + pow(3.0 / exp(sen->logf), 2.0) + pow(wp / r, 2.0) * pymax(vr, 0.0);
        r_extent_take(e, Pe, H, d[D_W] - wp, R, 16.0);
    } else if (sen->type == SEN_MICROWAVE && d[D_NRET] >= 3.0) {
        /* the returns' scatter in ground axes (given in the sensor's) */
        double n = d[D_NRET], c = cos(axis), s = sin(axis);
        double zs[3] = {c * c * d[D_SXX] - 2.0 * c * s * d[D_SXY] + s * s * d[D_SYY],
                        s * s * d[D_SXX] + 2.0 * c * s * d[D_SXY] + c * c * d[D_SYY],
                        c * s * (d[D_SXX] - d[D_SYY]) + (c * c - s * s) * d[D_SXY]};
        /* what one return's own error adds: the range's along the line of sight, the azimuth's across it */
        double cl = cos(los), sl = sin(los), ra = d[D_R] * n, rb = d[D_R + 1] * n * r * r;   /* (the cell's R is per mean) */
        double noise[3] = {cl * cl * ra + sl * sl * rb, sl * sl * ra + cl * cl * rb, cl * sl * (ra - rb)};
        for (int pass = 0; pass < 2; pass++) {           /* twice: the model is far from linear in the heading */
            double h0[3], Hn[9], from = los + PI;
            r_scatter(e[E_LEN], e[E_WID], e[E_HDG], from, h0, NULL);
            for (int k = 0; k < 3; k++) {
                double ep[3] = {e[0], e[1], e[2]}, hp[3], step = k == 2 ? 0.02 : 0.05;
                ep[k] += step;
                r_scatter(ep[0], ep[1], ep[2], from, hp, NULL);
                for (int i = 0; i < 3; i++)
                    Hn[i * 3 + k] = (hp[i] - h0[i]) / step;
            }
            /* a sample covariance of n points: its entries' own variances (Wishart) */
            double pxx = h0[0] + noise[0], pyy = h0[1] + noise[1], pxy = h0[2] + noise[2], dof = pymax(n - 1.0, 1.0);
            /* (as the rectangle believed predicts them, so that scatters above and below what is expected weigh
             * alike; never tighter than a rectangle of a metre would give) */
            double qx = pymax(pxx, 0.08), qy = pymax(pyy, 0.08);
            double Rm[3] = {2.0 * qx * qx / dof, 2.0 * qy * qy / dof, (qx * qy + pxy * pxy) / dof};
            double nu[3] = {zs[0] - pxx, zs[1] - pyy, zs[2] - pxy};
            for (int i = 0; i < 3; i++)                  /* the three entries one after the other */
                r_extent_take(e, Pe, Hn + i * 3, nu[i], Rm[i] * (pass ? 1.0 : 4.0), 25.0);
            r_extent_bounds(e);
        }
    }
    r_extent_bounds(e);
}

/* A cell of several returns as a detection of the object's centre: the returns' mean lies on the sides that
 * face the sensor, nearer than the centre by what the leaf's rectangle says; d (a copy) gets the range and the
 * azimuth of the centre, and the variance that the mean of n returns spread over those sides has. */
static void r_cell_centre(const double *row, const RSens *sen, const double *pose, double *d)
{
    if (sen->type != SEN_MICROWAVE)
        return;
    const double *e = row + RO_E, *x = row + RO_X;
    double sx, sy, axis;
    r_sensor_at(sen, pose, &sx, &sy, &axis);
    double dx = x[R_PX] - sx, dy = x[R_PY] - sy, r = pymax(hypot(dx, dy), 1.0), los = atan2(dy, dx);
    double sc[3], m[2], cl = cos(los), sl = sin(los);
    r_scatter(e[E_LEN], e[E_WID], e[E_HDG], los + PI, sc, m);
    double along = cl * m[0] + sl * m[1], across = -sl * m[0] + cl * m[1];
    d[D_Z] -= along;                                     /* (along is negative: the mean is nearer) */
    d[D_Z + 1] -= across / r;
    double va = cl * cl * sc[0] + 2.0 * cl * sl * sc[2] + sl * sl * sc[1];
    double vc = sl * sl * sc[0] - 2.0 * cl * sl * sc[2] + cl * cl * sc[1];
    double n = pymax(d[D_NRET], 1.0);
    d[D_R] += va / n;
    d[D_R + 1] += vc / (n * r * r);
}

/* ---- one scan ---- */
#define HTI 11                   /* a participating object: dt, sa, sp (the car's movement over dt: along, heading),
                                    kin[4], vehicle (-1, 0, 1), id, settled (confirmed or coast: 1), consider (1: its
                                    moving hypotheses hold its size and angle offset against a camera's boxes) */

/* a leaf one scan on: a new row under it with its numbers at the scan's time (the story "not reported"), the
 * step written down for the smoother. -1: no free row. */
static int32_t h_step(Hyp *H, double *A, int32_t r, int h, double dt, double sa, double sp, const double *pose,
                      const double *kin, const RPar *par, double t, const RXc *xc)
{
    int32_t c = h_take(H, h, r);
    if (c < 0)
        return -1;
    double *row = A + (Py_ssize_t)c * RROW;
    const double *src = A + (Py_ssize_t)r * RROW;
    memcpy(row, src, RROW * sizeof(double));
    if (H->X != NULL)                                    /* the row's memory goes with it */
        memcpy(H->X + (size_t)c * RXW, H->X + (size_t)r * RXW, RXW * sizeof(double));
    double *st = row + RO_ST;
    memset(st, 0, (size_t)(RO_M - RO_ST) * sizeof(double));
    st[0] = st[4] = st[8] = 1.0;
    if (fabs(dt) >= 1e-6) {
        double xs1[3 * RN], Ps1[3 * RNN], cc[3], xf1[RN], Pf1[RNN], T[9];
        r_evolve_row(src, dt, kin, par, xs1, Ps1, cc, xf1, Pf1, T);
        if (dt > 0.0) {
            memcpy(row + RO_MU, cc, sizeof(cc));
            memcpy(st, T, sizeof(T));
        }
        st[ST_DT] = dt;
        if (sa > 0.0 || sp > 0.0) {
            /* the car's travel error lies along its heading; its heading error swings the object about the car */
            double ch = cos(pose[2]), sh = sin(pose[2]);
            double dx = src[RO_X + R_PX] - pose[0], dy = src[RO_X + R_PY] - pose[1];
            double a2 = sa * sa, p2 = sp * sp;
            double e00 = a2 * (ch * ch) + p2 * (dy * dy), e01 = a2 * (ch * sh) + p2 * (-dx * dy),
                   e11 = a2 * (sh * sh) + p2 * (dx * dx);
            for (int j = 0; j < 4; j++) {
                double *P = j < 3 ? Ps1 + j * RNN : Pf1;
                P[0] += e00;
                P[1] += e01;
                P[RN] += e01;
                P[RN + 1] += e11;
            }
            st[ST_E] = e00;
            st[ST_E + 1] = e01;
            st[ST_E + 2] = e11;
        }
        if (xc != NULL && xc->c[XC_GRADE] > 0.0) {
            /* the ground angle's offset of an object whose range changes: the road's vertical curvature makes it
             * k x / 2 at distance x, so a change of the range by dr moves it by up to sigma_k dr / 2 */
            double gx = src[RO_X + R_PX] - pose[0], gy = src[RO_X + R_PY] - pose[1], gr = pymax(hypot(gx, gy), 1.0);
            double rate = ((src[RO_X + R_VX] - pose[3]) * gx + (src[RO_X + R_VY] - pose[4]) * gy) / gr;
            double dr = 0.5 * xc->c[XC_GRADE] * rate * dt, q = dr * dr;
            for (int j = 0; j < 3; j++)
                Ps1[j * RNN + R_DP * RN + R_DP] += q;
            Pf1[R_DP * RN + R_DP] += q;
        }
        memcpy(row + RO_XS, xs1, sizeof(xs1));
        memcpy(row + RO_PS, Ps1, sizeof(Ps1));
        memcpy(row + RO_XF, xf1, sizeof(xf1));
        memcpy(row + RO_PF, Pf1, sizeof(Pf1));
        r_combine(row + RO_XS, row + RO_PS, row + RO_MU, row + RO_X, row + RO_P);
        r_extent_step(row, dt, kin);
    }
    double *M = row + RO_M;
    M[M_T] = t;
    M[M_LEVEL] = (double)H->level;
    M[M_DET] = -1.0;
    M[M_NIS] = NAN;
    M[M_DL] = 0.0;
    H->c_leaves++;
    return c;
}

/* the ancestor of row r at or before level `lim` (r itself if it is), -1: its tree does not reach back that far */
static int32_t h_ancestor(const Hyp *H, const double *A, int32_t r, double lim)
{
    while (r >= 0 && A[(Py_ssize_t)r * RROW + RO_M + M_LEVEL] > lim)
        r = H->parent[r];
    return r;
}

static double lse(const double *v, int n)
{
    double mx = -INFINITY, s = 0.0;
    for (int i = 0; i < n; i++)
        if (v[i] > mx)
            mx = v[i];
    if (!isfinite(mx))
        return mx;
    for (int i = 0; i < n; i++)
        s += exp(v[i] - mx);
    return mx + log(s);
}

/* hypotheses k of a cluster table (lf: ng x nt, lw) removed where drop[k]; returns the new count */
static int h_compact(int32_t *lf, double *lw, int ng, int nt, const char *drop)
{
    int n = 0;
    for (int g = 0; g < ng; g++) {
        if (drop[g])
            continue;
        if (n != g) {
            memmove(lf + (size_t)n * nt, lf + (size_t)g * nt, (size_t)nt * sizeof(int32_t));
            lw[n] = lw[g];
        }
        n++;
    }
    return n;
}

/* a table sorted, the most probable hypothesis first (insertion: tables are short; equal weights keep their order) */
static void h_sort(int32_t *lf, double *lw, int ng, int nt, int32_t *tmp)
{
    for (int g = 1; g < ng; g++) {
        double w = lw[g];
        if (!(w > lw[g - 1]))
            continue;
        memcpy(tmp, lf + (size_t)g * nt, (size_t)nt * sizeof(int32_t));
        int k = g;
        while (k > 0 && w > lw[k - 1]) {
            memcpy(lf + (size_t)k * nt, lf + (size_t)(k - 1) * nt, (size_t)nt * sizeof(int32_t));
            lw[k] = lw[k - 1];
            k--;
        }
        memcpy(lf + (size_t)k * nt, tmp, (size_t)nt * sizeof(int32_t));
        lw[k] = w;
    }
}

typedef struct {
    Hyp *H;
    double *A;
    Arena *ar;
    double t;
    PyObject *dead;                  /* handles of objects that ended in this call (a list), or NULL */
    char *mark;                      /* per row, zero between uses */
} Settle;

/* An object's history is cut where it is older than what the smoother and the pruning look back at. */
static void h_trim(Hyp *H, const double *A, HTrack *t, double now)
{
    int32_t r = t->primary, cut = -1;
    if (r < 0)
        return;
    for (int32_t q = r; q >= 0; q = H->parent[q]) {
        const double *M = A + (Py_ssize_t)q * RROW + RO_M;
        if ((double)H->level - M[M_LEVEL] >= (double)H->keep_steps || now - M[M_T] > H->keep_s) {
            cut = q;
            break;
        }
    }
    if (cut < 0 || H->parent[cut] < 0)
        return;
    for (int32_t q = H->parent[cut]; q >= 0; q = H->parent[q])
        if (H->nchild[q] != 1)
            return;                                      /* a branch older than the window: left as it is */
    int32_t q = H->parent[cut];
    H->parent[cut] = -1;
    while (q >= 0) {
        int32_t p = H->parent[q];
        h_give(H, q);
        q = p;
    }
}

/* A cluster's table after it changed (hypotheses made, an object taken out): the trees cut N scans back to the
 * branch of the best hypothesis, the leaves per object bounded, the weights normalised, rows no hypothesis uses
 * freed, every object's leaves, primary and probabilities set, objects that are in no hypothesis ended, objects
 * whose leaf no hypothesis doubts given a cluster of their own. The table must be sorted, the best first. */
static int h_settle(Settle *s, int ci)
{
    Hyp *H = s->H;
    double *A = s->A;
    HCl *c = &H->cl[ci];
    int nt = c->nt, ng = c->ng;
    if (ng == 0 || nt == 0) {
        for (int i = 0; i < nt; i++) {
            int h = c->tr[i];
            if (s->dead) {
                PyObject *o = PyLong_FromLong(h);
                if (o == NULL || PyList_Append(s->dead, o) < 0) {
                    Py_XDECREF(o);
                    return -1;
                }
                Py_DECREF(o);
            }
            h_track_free(H, h);
        }
        h_cluster_free(H, ci);
        return 0;
    }
    char *drop = (char *)ar_get(s->ar, (size_t)ng);
    if (drop == NULL)
        return -1;
    memset(drop, 0, (size_t)ng);
    /* N-scan pruning: what the best hypothesis says about each object N scans back is taken as settled */
    double lim = (double)(H->level - H->n_scan);
    for (int i = 0; i < nt; i++) {
        HTrack *t = &H->tr[c->tr[i]];
        int32_t best = c->lf[i];
        if (best < 0) {
            if ((double)t->born <= lim)
                for (int g = 1; g < ng; g++)
                    if (c->lf[(size_t)g * nt + i] >= 0)
                        drop[g] = 1;
            continue;
        }
        int32_t anc = h_ancestor(H, A, best, lim);
        if (anc < 0)
            continue;
        for (int g = 1; g < ng; g++) {
            int32_t q = c->lf[(size_t)g * nt + i];
            if (q == best || drop[g])
                continue;
            if (q < 0 || h_ancestor(H, A, q, lim) != anc)
                drop[g] = 1;
        }
    }
    {
        int before = ng;
        ng = h_compact(c->lf, c->lw, ng, nt, drop);
        H->c_pruned += before - ng;
    }
    /* no object keeps more than max_leaves leaves: the least probable ones go, with the hypotheses that use them */
    for (int i = 0; i < nt; i++) {
        for (;;) {
            int nleaf = 0;
            int32_t weakest = -1;
            double wmin = INFINITY;
            for (int g = 0; g < ng; g++) {
                int32_t q = c->lf[(size_t)g * nt + i];
                if (q < 0 || s->mark[q])
                    continue;
                s->mark[q] = 1;
                nleaf++;
                double w = 0.0;
                for (int g2 = g; g2 < ng; g2++)
                    if (c->lf[(size_t)g2 * nt + i] == q)
                        w += exp(c->lw[g2] - c->lw[0]);
                if (q != c->lf[i] && w < wmin) {
                    wmin = w;
                    weakest = q;
                }
            }
            for (int g = 0; g < ng; g++)
                if (c->lf[(size_t)g * nt + i] >= 0)
                    s->mark[c->lf[(size_t)g * nt + i]] = 0;
            if (nleaf <= H->max_leaves || weakest < 0)
                break;
            memset(drop, 0, (size_t)ng);
            for (int g = 1; g < ng; g++)
                if (c->lf[(size_t)g * nt + i] == weakest)
                    drop[g] = 1;
            int before = ng;
            ng = h_compact(c->lf, c->lw, ng, nt, drop);
            H->c_pruned += before - ng;
        }
    }
    c->ng = ng;
    double norm = lse(c->lw, ng);
    for (int g = 0; g < ng; g++)
        c->lw[g] -= norm;
    /* rows no hypothesis uses any more; each object's leaves, primary, probabilities */
    for (int i = 0; i < nt; i++) {
        HTrack *t = &H->tr[c->tr[i]];
        for (int g = 0; g < ng; g++) {
            int32_t q = c->lf[(size_t)g * nt + i];
            if (q >= 0)
                s->mark[q] = 1;
        }
        int n = 0;
        for (int k = 0; k < t->nleaf; k++) {
            int32_t q = t->leaf[k];
            if (s->mark[q] == 1) {
                s->mark[q] = 2;                          /* listed once */
                t->leaf[n++] = q;
            } else if (s->mark[q] == 0)
                h_drop_up(H, q);
        }
        t->nleaf = n;
        for (int k = 0; k < n; k++)
            s->mark[t->leaf[k]] = 0;
        t->in_map = c->lf[i] >= 0;
        t->pe = 0.0;
        t->pp = 0.0;
        int32_t prim = c->lf[i];
        if (prim < 0) {                                  /* not in the best story: its likeliest leaf stands for it */
            double wb = -1.0;
            for (int k = 0; k < n; k++) {
                double w = 0.0;
                for (int g = 0; g < ng; g++)
                    if (c->lf[(size_t)g * nt + i] == t->leaf[k])
                        w += exp(c->lw[g]);
                if (w > wb) {
                    wb = w;
                    prim = t->leaf[k];
                }
            }
        }
        for (int g = 0; g < ng; g++) {
            int32_t q = c->lf[(size_t)g * nt + i];
            if (q < 0)
                continue;
            double w = exp(c->lw[g]);
            t->pe += w * sigm(A[(Py_ssize_t)q * RROW + RO_M + M_L]);
            if (q == prim)
                t->pp += w;
        }
        t->primary = prim;
        t->cl = ci;
    }
    /* objects that are in no hypothesis are gone; objects no hypothesis doubts get a cluster of their own */
    int *keep = (int *)ar_get(s->ar, (size_t)(nt + 1) * sizeof(int));
    if (keep == NULL)
        return -1;
    int nk = 0, nconst = 0;
    for (int i = 0; i < nt; i++) {
        int h = c->tr[i];
        HTrack *t = &H->tr[h];
        if (t->nleaf == 0) {
            if (s->dead) {
                PyObject *o = PyLong_FromLong(h);
                if (o == NULL || PyList_Append(s->dead, o) < 0) {
                    Py_XDECREF(o);
                    return -1;
                }
                Py_DECREF(o);
            }
            h_track_free(H, h);
            keep[i] = -1;
            continue;
        }
        int constant = t->nleaf == 1;
        for (int g = 0; constant && g < ng; g++)
            if (c->lf[(size_t)g * nt + i] != t->leaf[0])
                constant = 0;
        keep[i] = constant ? 0 : 1;
        nk += keep[i];
        nconst += constant;
    }
    if (nk == 0 && nconst > 0) {                         /* all settled: the first stays here, alone */
        for (int i = 0; i < nt; i++)
            if (keep[i] == 0) {
                keep[i] = 1;
                nk = 1;
                nconst--;
                break;
            }
    }
    for (int i = 0; i < nt; i++) {
        if (keep[i] != 0)
            continue;
        int h = c->tr[i];
        int c2 = h_cluster_new(H, 1, 1);
        if (c2 < 0) {
            PyErr_NoMemory();
            return -1;
        }
        c = &H->cl[ci];                                  /* the clusters may have moved */
        H->cl[c2].tr[0] = h;
        H->cl[c2].lf[0] = H->tr[h].leaf[0];
        H->cl[c2].lw[0] = 0.0;
        H->tr[h].cl = c2;
    }
    if (nk == 0) {
        h_cluster_free(H, ci);
        return 0;
    }
    if (nk < nt) {
        int32_t *nlf = (int32_t *)ar_get(s->ar, (size_t)(ng * nk + 1) * sizeof(int32_t));
        if (nlf == NULL)
            return -1;
        int n = 0;
        for (int i = 0; i < nt; i++)
            if (keep[i] == 1) {
                c->tr[n] = c->tr[i];
                for (int g = 0; g < ng; g++)
                    nlf[(size_t)g * nk + n] = c->lf[(size_t)g * nt + i];
                n++;
            }
        memcpy(c->lf, nlf, (size_t)(ng * nk) * sizeof(int32_t));
        c->nt = nk;
        /* hypotheses that differed only in what was taken out are one now */
        memset(drop, 0, (size_t)ng);
        for (int g = 0; g < ng; g++) {
            if (drop[g])
                continue;
            for (int g2 = g + 1; g2 < ng; g2++)
                if (!drop[g2] && memcmp(c->lf + (size_t)g * nk, c->lf + (size_t)g2 * nk, (size_t)nk * sizeof(int32_t)) == 0) {
                    double two[2] = {c->lw[g], c->lw[g2]};
                    c->lw[g] = lse(two, 2);
                    drop[g2] = 1;
                }
        }
        c->ng = h_compact(c->lf, c->lw, ng, nk, drop);
        h_sort(c->lf, c->lw, c->ng, nk, nlf);
    }
    for (int i = 0; i < c->nt; i++)
        H->tr[c->tr[i]].cl = ci;
    return 0;
}

/* cluster b joined to cluster a (their objects compete now): every pair of their hypotheses, the K most
 * probable kept. b is freed. */
static int h_merge(Hyp *H, Arena *ar, int a, int b)
{
    HCl *ca = &H->cl[a], *cb = &H->cl[b];
    int nt = ca->nt + cb->nt, na = ca->ng, nb = cb->ng, tot = na * nb;
    int *ord = (int *)ar_get(ar, (size_t)tot * sizeof(int));
    double *w = (double *)ar_get(ar, (size_t)tot * sizeof(double));
    if (ord == NULL || w == NULL)
        return -1;
    for (int i = 0; i < tot; i++) {
        w[i] = ca->lw[i / nb] + cb->lw[i % nb];
        int k = i;
        while (k > 0 && w[i] > w[ord[k - 1]]) {
            ord[k] = ord[k - 1];
            k--;
        }
        ord[k] = i;
    }
    int ng = tot < H->k_best ? tot : H->k_best;
    int32_t *tr = (int32_t *)PyMem_Malloc((size_t)nt * sizeof(int32_t));
    int32_t *lf = (int32_t *)PyMem_Malloc((size_t)(nt * ng) * sizeof(int32_t));
    double *lw = (double *)PyMem_Malloc((size_t)ng * sizeof(double));
    if (!tr || !lf || !lw) {
        PyMem_Free(tr);
        PyMem_Free(lf);
        PyMem_Free(lw);
        return -1;
    }
    memcpy(tr, ca->tr, (size_t)ca->nt * sizeof(int32_t));
    memcpy(tr + ca->nt, cb->tr, (size_t)cb->nt * sizeof(int32_t));
    for (int g = 0; g < ng; g++) {
        int ia = ord[g] / nb, ib = ord[g] % nb;
        memcpy(lf + (size_t)g * nt, ca->lf + (size_t)ia * ca->nt, (size_t)ca->nt * sizeof(int32_t));
        memcpy(lf + (size_t)g * nt + ca->nt, cb->lf + (size_t)ib * cb->nt, (size_t)cb->nt * sizeof(int32_t));
        lw[g] = w[ord[g]];
    }
    double norm = lse(lw, ng);
    for (int g = 0; g < ng; g++)
        lw[g] -= norm;
    for (int i = 0; i < cb->nt; i++)
        H->tr[cb->tr[i]].cl = a;
    PyMem_Free(ca->tr);
    PyMem_Free(ca->lf);
    PyMem_Free(ca->lw);
    ca->tr = tr;
    ca->lf = lf;
    ca->lw = lw;
    ca->nt = nt;
    ca->ng = ng;
    h_cluster_free(H, b);
    return 0;
}

static int uf_find(int *uf, int i)
{
    while (uf[i] != i) {
        uf[i] = uf[uf[i]];
        i = uf[i];
    }
    return i;
}

/* ---- what a sensor can see of an object: other objects in front of it, the edge of its field ----
 * An object behind another one is not reported however surely it is there: its probability of being reported is a
 * function of where the others are. Each settled object's primary is an occluder: an interval of bearings (its
 * silhouette), a range, and for a camera the rows of the picture its box covers. A leaf's own silhouette is
 * sampled across; each sample is hidden as far as a nearer occluder's interval holds it (softly: neither place is
 * known exactly) and, for a camera, as far as the occluder's box reaches up the leaf's box. */
typedef struct {
    double b, hw, r, top, bot, sb, id;
    int k;
} Occ;

typedef struct {
    Hyp *H;
    double *A;
    Arena ar;
    int n;                           /* participating objects */
    int32_t *th;
    double *ti;
    int *pk_of;                      /* by handle: its index among the participating, -1 */
    double t, pose[6], nscan;
    RSens sen;
    int src;
    RPar par;
    double *D, *KB;
    Py_ssize_t nd;
    double acfg[RSCFG], ucfg[2], bcfg[RBCFG];
    RAs as;
    /* the leaves after the step */
    int nl;
    int32_t *lrow, *lold;
    int *lk;                         /* its object's index among the participating */
    double *lpd, *lcov, *lmiss, *lnr;    /* p_detect, coverage, ln(1 - r pd), ln r */
    double *lr;                      /* nl x nd: ln(pd lk F / lam), -inf: may not be */
    int32_t *missidx;                /* by row: the old leaf's index among the leaves, -1 */
    int32_t *detchild;               /* nl x nd: the row of leaf a having taken detection j, -1 */
    int *det_cl;                     /* per detection: its cluster, -1 */
    int32_t *birth_h;                /* per detection: the object it started (handle), -1; -2: it cannot start one */
    int births_left;
    PyObject *dead, *births;
    char *mark;
    Occ *occ;                        /* the scan's occluders */
    int nocc;
    RXc *xc;                         /* the detector's further inputs, or NULL */
    double *lpdlo;                   /* per leaf: the probability of a box in the low-score list */
    XBox *xb;                        /* per leaf: what the scan's own boxes hide of it (NULL: not asked) */
} Scan;

static double r_phi(double x)
{
    return 0.5 * (1.0 + erf(x * 0.7071067811865476));
}

/* a leaf's silhouette as the sensor sees it: bearing in the sensor's axes, half width, range, the angles below
 * the horizon of its top and bottom (camera), how well its bearing is known */
static void r_outline(const Scan *s, const double *row, Occ *o)
{
    const double *x = row + RO_X, *P = row + RO_P, *e = row + RO_E;
    double sx, sy, axis;
    r_sensor_at(&s->sen, s->pose, &sx, &sy, &axis);
    double dx = x[R_PX] - sx, dy = x[R_PY] - sy, r = pymax(hypot(dx, dy), 0.5), los = atan2(dy, dx);
    o->r = r;
    o->b = wrap_pi(los - axis);
    o->hw = 0.5 * r_silhouette(e, los, NULL) / r;
    double ux = -dy / r, uy = dx / r;
    o->sb = sqrt(pymax(ux * ux * P[0] + 2.0 * ux * uy * P[1] + uy * uy * P[RN + 1], 0.0)) / r;
    if (s->sen.type == SEN_CAMERA) {
        o->bot = atan2(s->sen.hc, r);
        o->top = atan2(s->sen.hc - exp(x[R_LH]), r);
    } else {
        o->bot = 1.0;
        o->top = -1.0;
    }
}

/* How much of leaf a's box the scan's own boxes hide, tracked or not, first list or low (a camera): the boxes
 * that are nearer. Nearer as the detector orders two boxes: the bottom edge lower in the picture, and a bottom cut by
 * the view or the frame lower than it shows. The leaf's own bottom edge is known only as well as its range: a box
 * counts as nearer as far as its bottom lies below the leaf's by more than XC_BOXOCC_M of what is not known of the
 * difference (the leaf's own box, a little lower than predicted, must not make the leaf's miss free). The cover:
 * the leaf's silhouette at five bearings, each hidden as far as a nearer box holds it across and up the leaf's box. */
static double x_box_cover(const Scan *s, int a, const Occ *me, XBox *bx)
{
    const double *row = s->A + (Py_ssize_t)s->lrow[a] * RROW, *x = row + RO_X, *P = row + RO_P;
    double dx = x[R_PX] - s->pose[0], dy = x[R_PY] - s->pose[1], rc = pymax(hypot(dx, dy), 0.5), los = atan2(dy, dx);
    double rn = pymax(rc - r_half_depth(row + RO_E, los, NULL), 0.5), hc = s->sen.hc, g = hc / (rn * rn + hc * hc);
    double ux = dx / rc, uy = dy / rc, vr = ux * ux * P[0] + 2.0 * ux * uy * P[1] + uy * uy * P[RN + 1];
    double pb = atan2(hc, rn) + s->sen.bias + x[R_DP], vpb = g * g * pymax(vr, 0.0) + P[R_DP * RN + R_DP];
    double h = exp(x[R_LH]) / rn, pt = pb - h, m = s->xc->c[XC_BOXOCC_M], cover = 0.0;
    for (int p = 0; p < 5; p++) {
        bx->best[p] = bx->second[p] = 0.0;
        bx->bj[p] = -1;
    }
    if (!(h > 1e-6) || !(me->hw > 0.0))
        return 0.0;
    for (int p = 0; p < 5; p++) {
        double beta = me->b + me->hw * (-0.8 + 0.4 * p);
        for (Py_ssize_t j = 0; j < s->nd; j++) {
            const double *d = s->D + (size_t)j * RDET;
            if (!(d[D_W] > 0.0) || isnan(d[D_TOP]) || isnan(d[D_BOT]))
                continue;
            int base = (int)d[D_KIND] & 7, br = (base == K_B || base == K_BT) ? 0 : 1;
            if (base == K_NONE)
                continue;
            double hw = 0.5 * d[D_W], sg = pymin(sqrt(d[D_R + br] + me->sb * me->sb), 0.35 * hw) + 1e-4;
            if (fabs(d[D_Z + br] - beta) > hw + 3.5 * sg)
                continue;
            double in = r_phi((d[D_Z + br] + hw - beta) / sg) - r_phi((d[D_Z + br] - hw - beta) / sg);
            if (!(in > 1e-3))
                continue;
            double bot = d[D_BOT], near = 1.0;
            if (((int)d[D_HID] & 16) == 0)
                near = r_phi((bot - pb) / sqrt(vpb + (br == 1 ? d[D_R] : 1e-5)) - m);
            else
                bot = pymax(bot, pb);
            double up = pymin(1.0, pymax(0.0, (pymin(pb, bot) - pymax(pt, d[D_TOP])) / h));
            double v = in * up * near;
            if (v > bx->best[p]) {
                bx->second[p] = bx->best[p];
                bx->best[p] = v;
                bx->bj[p] = (int)j;
            } else if (v > bx->second[p])
                bx->second[p] = v;
        }
        cover += 0.2 * bx->best[p];
    }
    return cover;
}

/* How much of leaf a the sensor can see (1: all of it), written into its record with what hides it. */
static double r_visible(Scan *s, int a, double cx, double cy)
{
    double *M = s->A + (Py_ssize_t)s->lrow[a] * RROW + RO_M;
    M[M_VIS] = 1.0;
    M[M_OCC] = 0.0;
    M[M_OCCBY] = 0.0;
    if (s->acfg[13] == 0.0)
        return 1.0;
    Occ me;
    r_outline(s, s->A + (Py_ssize_t)s->lrow[a] * RROW, &me);
    double c0 = s->acfg[14], c1 = s->acfg[15], fl = s->sen.type == SEN_CAMERA ? s->acfg[16] : s->acfg[17];
    /* behind other objects */
    double cover = 0.0, by = 0.0, bybest = 0.0;
    if (s->nocc > 0) {
        double contrib[64];
        int no = s->nocc < 64 ? s->nocc : 64;
        for (int q = 0; q < no; q++)
            contrib[q] = 0.0;
        for (int p = 0; p < 5; p++) {
            double beta = me.b + me.hw * (-0.8 + 0.4 * p), hid = 0.0;
            for (int q = 0; q < no; q++) {
                const Occ *o = &s->occ[q];
                if (o->k == s->lk[a] || !(o->r < me.r - 1.0))
                    continue;
                /* (how far the hidden one may have strayed while nobody saw it does not open the view: that it
                 * is not reported is itself the evidence that it is still behind) */
                double sg = pymin(sqrt(o->sb * o->sb + me.sb * me.sb), 0.35 * o->hw) + 1e-4;
                double in = r_phi((o->b + o->hw - beta) / sg) - r_phi((o->b - o->hw - beta) / sg);
                double up = 1.0;
                if (s->sen.type == SEN_CAMERA) {         /* the rows of the leaf's box that the occluder's box holds */
                    double h = me.bot - me.top;
                    up = h > 1e-6 ? pymin(1.0, pymax(0.0, (pymin(me.bot, o->bot) - pymax(me.top, o->top)) / h)) : 1.0;
                }
                double v = in * up;
                contrib[q] += v;
                if (v > hid)
                    hid = v;
            }
            cover += 0.2 * hid;
        }
        for (int q = 0; q < no; q++)
            if (contrib[q] > bybest) {
                bybest = contrib[q];
                by = s->occ[q].id;
            }
    }
    XBox *bx = NULL;
    if (s->xc != NULL && s->xc->c[XC_BOXOCC] != 0.0 && s->sen.type == SEN_CAMERA && s->nd > 0 && s->xb != NULL) {
        bx = &s->xb[a];
        bx->cover_obj = cover;
        double cb = x_box_cover(s, a, &me, bx);          /* behind a box of this scan (an object or not yet one) */
        if (cb > cover) {
            cover = cb;
            by = 0.0;
        }
    }
    double pv = cover <= c0 ? 1.0 : cover >= c1 ? fl : 1.0 - (1.0 - fl) * (cover - c0) / (c1 - c0);
    /* cut by the edge of the field */
    double lo = 0.0, hi = 0.0, pe = 1.0;
    int edged = 0;
    if (s->sen.type == SEN_CAMERA && s->sen.cover) {
        lo = s->sen.bmin;
        hi = s->sen.bmax;
        edged = 1;
    } else if (s->sen.type == SEN_MICROWAVE) {
        lo = -s->sen.fov;
        hi = s->sen.fov;
        edged = 1;
    }
    if (edged && me.hw > 1e-6) {
        double in = pymin(me.b + me.hw, hi) - pymax(me.b - me.hw, lo);
        double f = pymin(1.0, pymax(0.0, in / (2.0 * me.hw)));
        pe = f >= 0.9 ? 1.0 : f <= 0.4 ? fl : fl + (1.0 - fl) * (f - 0.4) / 0.5;
    }
    double vis = pv * pe;
    if (bx != NULL) {
        bx->pe = pe;
        bx->on = 1;
    }
    M[M_VIS] = vis;
    if (pv < 0.5 && pv <= pe) {
        M[M_OCC] = 1.0;
        M[M_OCCBY] = by;
    } else if (pe < 0.8)
        M[M_OCC] = 2.0;
    return vis;
}

/* the occluders of a scan: the primaries of the settled objects, as they are at the scan's time */
static int h_occluders(Scan *s)
{
    s->nocc = 0;
    if (s->acfg[13] == 0.0 || s->n == 0)
        return 0;
    s->occ = (Occ *)ar_get(&s->ar, (size_t)s->n * sizeof(Occ));
    if (s->occ == NULL)
        return -1;
    for (int a = 0; a < s->nl; a++) {
        int k = s->lk[a];
        const double *ti = s->ti + (size_t)k * HTI;
        if (ti[9] == 0.0 || s->lold[a] != s->H->tr[s->th[k]].primary)
            continue;
        Occ *o = &s->occ[s->nocc++];
        r_outline(s, s->A + (Py_ssize_t)s->lrow[a] * RROW, o);
        o->k = k;
        o->id = ti[8];
    }
    return 0;
}

/* every leaf against the scan's detections (as they are in s->D, with the sensor's bias as it is in s->sen) */
static void h_ratios(Scan *s)
{
    Hyp *H = s->H;
    double cph = cos(s->pose[2]), sph = sin(s->pose[2]);
    ras_dets(&s->as, s->nscan);
    for (int a = 0; a < s->nl; a++) {
        const double *row = s->A + (Py_ssize_t)s->lrow[a] * RROW, *x = row + RO_X, *M = row + RO_M;
        const double *ti = s->ti + (size_t)s->lk[a] * HTI;
        double dx = x[R_PX] - s->pose[0], dy = x[R_PY] - s->pose[1];
        double cx = cph * dx + sph * dy, cy = -sph * dx + cph * dy;
        double vis = r_visible(s, a, cx, cy);
        double psen = r_p_detect(&s->sen, cx, cy, x), pd = psen * vis;
        s->lpd[a] = pd;
        s->lcov[a] = r_coverage(&s->sen, cx, cy);
        if (s->lcov[a] <= 0.0)
            s->A[(Py_ssize_t)s->lrow[a] * RROW + RO_M + M_OCC] = 3.0;    /* outside the sensor's field */
        double r = sigm(M[M_L]);
        s->lnr[a] = log(pymax(r, 1e-300));
        s->lmiss[a] = log(pymax(1.0 - r * pd, 1e-12));
        double pdlo = 0.0;
        if (s->xc != NULL && s->xc->c[XC_LOW] != 0.0 && s->sen.type == SEN_CAMERA) {
            /* A visible object is in the first list (pd), in the low-score list (pdlo) or in neither. Of what the
             * first list leaves the low list holds a share: more inside a run of misses of an object the first list
             * had a moment ago (the detector misses in runs, and the object is then mostly just under the threshold)
             * than for an object taken at random */
            const double *c = s->xc->c;
            double q = s->lcov[a] > 0.0 ? vis : 0.0;
            double k = (M[M_HITS] >= 2.0 && s->t - M[M_THIT] <= c[XC_LOW_RUN_S]) ? c[XC_LOW_PRUN] : c[XC_LOW_P];
            pdlo = pymax(0.0, k * (q - pd));
            s->lmiss[a] = log(pymax(1.0 - r * (pd + pdlo), 1e-12));
            if (s->xb != NULL)
                s->xb[a].klo = k;
        }
        s->lpdlo[a] = pdlo;
        s->as.bx = NULL;
        if (s->xb != NULL && s->xb[a].on) {
            s->xb[a].psen = psen;
            s->xb[a].incov = s->lcov[a] > 0.0 ? 1.0 : 0.0;
            s->as.bx = &s->xb[a];
        }
        double lbw = NAN;
        if (M[M_HITS] >= 2.0 && (int)M[M_BOXS] == s->src && s->t - M[M_BOXT] <= s->acfg[10] && !isnan(M[M_BOXW]))
            lbw = M[M_BOXW];
        ras_row(&s->as, row, pd, pdlo, cx, cy, ti[7], lbw, ti[8], s->lr + (size_t)a * s->nd);
        (void)H;
    }
}

/* What all objects of one scan share: the camera body's pitch at that moment (tracker.py `_common`). The settled
 * objects' primaries, paired as the scan alone would pair them and sure of it, say what it was (the middle one of
 * their angle innovations, by weight); the part the sensor's slow pitch error explains goes into that error
 * (bias: [value, variance], updated), the rest is this scan's own and is taken out of the scan's angles.
 * Returns 1 when it was taken (the detections and the sensor's bias changed), 0 when not, -1: error. */
static int h_common(Scan *s, double *bias)
{
    Hyp *H = s->H;
    int n = s->n, nd = (int)s->nd, cmin = (int)s->acfg[11];
    if (n < cmin || nd < cmin)
        return 0;
    /* the primaries' leaves */
    int *pa = (int *)ar_get(&s->ar, (size_t)n * sizeof(int));
    double *rat = (double *)ar_get(&s->ar, (size_t)(3 * n * nd + 1) * sizeof(double));
    Py_ssize_t *pi = (Py_ssize_t *)ar_get(&s->ar, (size_t)(2 * (n + nd) + 2) * sizeof(Py_ssize_t));
    if (pa == NULL || rat == NULL || pi == NULL)
        return -1;
    double *gain = rat + n * nd, *pair = gain + n * nd;
    Py_ssize_t *pj = pi + n + nd + 1;
    for (int k = 0; k < n; k++)
        pa[k] = -1;
    for (int a = 0; a < s->nl; a++)
        if (s->lold[a] == H->tr[s->th[s->lk[a]]].primary)
            pa[s->lk[a]] = a;
    for (int k = 0; k < n; k++)
        for (int j = 0; j < nd; j++) {
            double g = -INFINITY, r = 0.0;
            if (pa[k] >= 0) {
                int a = pa[k];
                double v = s->lr[(size_t)a * nd + j];
                if (isfinite(v)) {
                    v -= log(pymax(1.0 - s->lpd[a], 1e-12));     /* against "missed": the object taken as existing */
                    r = exp(v);
                    if (v > 0.0)
                        g = v;
                }
            }
            rat[k * nd + j] = r;
            gain[k * nd + j] = g;
        }
    if (r_marginals(rat, n, nd, pair) < 0)
        return -1;
    Py_ssize_t np_ = r_assign(gain, n, nd, pi, pj);
    if (np_ < 0)
        return -1;
    double *v = (double *)ar_get(&s->ar, (size_t)(3 * np_ + 1) * sizeof(double));
    Py_ssize_t *ko = (Py_ssize_t *)ar_get(&s->ar, (size_t)(np_ + 1) * sizeof(Py_ssize_t));
    if (v == NULL || ko == NULL)
        return -1;
    double *w = v + np_, *cw = w + np_;
    Py_ssize_t ns = 0;
    for (Py_ssize_t q = 0; q < np_; q++) {
        int k = (int)pi[q], j = (int)pj[q], a = pa[k];
        const double *d = s->D + (size_t)j * RDET, *row = s->A + (Py_ssize_t)s->lrow[a] * RROW;
        int base = (int)d[D_KIND] & 7;
        if (!(base == K_RB || base == K_RBH) || s->ti[(size_t)k * HTI + 9] == 0.0 || row[RO_M + M_HITS] < 4.0 ||
            !(pair[k * nd + j] >= s->acfg[12]) || d[D_LOW] != 0.0)
            continue;
        double z[4], Hm[4 * RN];
        const double *P = row + RO_P;
        r_h(&s->sen, (int)d[D_KIND], d[D_SDT], row + RO_X, s->pose, z, Hm, row + RO_E);
        double s00 = 0.0;
        for (int c = 0; c < RN; c++) {
            double tt = 0.0;
            for (int i = 0; i < RN; i++)
                tt += Hm[i] * P[i * RN + c];
            s00 += tt * Hm[c];
        }
        s00 += d[D_R];
        v[ns] = d[D_Z] - z[0];
        w[ns] = 1.0 / s00;
        ns++;
    }
    if (ns < cmin)
        return 0;
    double m = r_wmedian(v, w, ns, ko, cw);
    double var_m = 1.57 / cw[ns - 1];                    /* the variance of a weighted median */
    /* the fast part's variance: what the pitch's own movement left unknown of it at this scan (bias[3]) */
    double jit = bias[3], tot = bias[1] + jit + var_m;
    double slow = m * bias[1] / tot, fast = m * jit / tot;
    bias[0] += slow;
    bias[1] = pymax(bias[1] * (1.0 - bias[1] / tot), s->sen.bias_floor2);
    s->sen.bias = bias[0];
    s->sen.bias_var = bias[1];
    double left = jit * (1.0 - jit / tot);              /* what is still not known of this scan's own pitch */
    bias[2] += fast;
    bias[3] = left;
    for (int j = 0; j < nd; j++) {
        double *d = s->D + (size_t)j * RDET;
        int base = (int)d[D_KIND] & 7, r = base == K_BT ? 1 : 0;
        if (base == K_RB || base == K_RBH || base == K_BT) {
            d[D_Z + r] -= fast;
            d[D_R + r] = pymax(d[D_R + r] - jit + left, 0.25 * d[D_R + r]);
        }
    }
    return 1;
}

/* the row of leaf a having taken detection j (made when first asked for) */
static int32_t h_detchild(Scan *s, int a, int j)
{
    Hyp *H = s->H;
    int32_t *slot = &s->detchild[(size_t)a * s->nd + j];
    if (*slot >= 0)
        return *slot;
    int h = s->th[s->lk[a]];
    int32_t r = h_take(H, h, s->lold[a]);
    if (r < 0)
        return -1;
    double *row = s->A + (Py_ssize_t)r * RROW, *M = row + RO_M;
    const double *d = s->D + (size_t)j * RDET;
    memcpy(row, s->A + (Py_ssize_t)s->lrow[a] * RROW, RROW * sizeof(double));
    double nis = NAN, dl = 0.0, dc[RDET];
    int base = (int)d[D_KIND] & 7, low = 0;
    memcpy(dc, d, sizeof(dc));
    r_cell_centre(row, &s->sen, s->pose, dc);
    if (s->xc == NULL) {
        if (H->X != NULL)
            memcpy(H->X + (size_t)r * RXW, H->X + (size_t)s->lrow[a] * RXW, RXW * sizeof(double));
        if (r_update_full(row, &s->sen, s->pose, dc, s->ucfg[0], s->ucfg[1], s->acfg[18],
                          (int)s->ti[(size_t)s->lk[a] * HTI + 10], &nis, &dl, NULL, 0, NULL) == 0)
            r_extent_update(row, &s->sen, d, s->pose, s->ti + (size_t)s->lk[a] * HTI + 3);
    } else {
        /* with the detector's further inputs: the detection as this leaf reads it, its extra rows, and what the
         * leaf remembers for the scans to come */
        const RXc *xc = s->xc;
        const double *c = xc->c;
        double *xr = x_of(xc, row);
        XAux aux[XAUX];
        int used = 0, na;
        memcpy(xr, x_of(xc, s->A + (Py_ssize_t)s->lrow[a] * RROW), RXW * sizeof(double));
        x_det_leaf(xc, &s->sen, s->pose, row, d, dc);
        na = x_aux_rows(xc, &s->sen, row, xr, d, s->pose, aux);
        low = d[D_LOW] != 0.0;
        int ok = r_update_full(row, &s->sen, s->pose, dc, s->ucfg[0], s->ucfg[1], s->acfg[18],
                               (int)s->ti[(size_t)s->lk[a] * HTI + 10], &nis, &dl, aux, na, &used) == 0;
        if (ok && !low)                                  /* (a low box's edges teach no size) */
            r_extent_update(row, &s->sen, d, s->pose, s->ti + (size_t)s->lk[a] * HTI + 3);
        if (used & 1)
            xr[X_NLAMP] += 1.0;
        if (used & 2)
            xr[X_NWID] += 1.0;
        if ((int)d[D_KIND] & K_SCALE)
            xr[X_ST] = s->t;
        if (c[XC_LSIZE] != 0.0 && d[D_LAMPC] > 0.0 && d[D_LSEPR] > 0.0) {
            /* the lamp pair as it was seen now, for the next one's change (the heading against the line of sight
             * taken out) */
            const double *x = row + RO_X;
            double ca = fabs(cos(row[RO_E + 2] - atan2(x[R_PY] - s->pose[1], x[R_PX] - s->pose[0])));
            if (ca >= 0.6) {
                xr[X_LS] = d[D_LSEP] - log(ca);
                xr[X_LST] = s->t;
                xr[X_LSR] = d[D_LSEPR];
            }
        }
        xr[X_LOWRUN] = low ? xr[X_LOWRUN] + 1.0 : 0.0;
        if (c[XC_APP] != 0.0 && d[D_HASEMB] != 0.0 && !low && ((int)d[D_HID] & 15) == 0 && d[D_OCC] < 0.2) {
            /* the object's appearance as remembered: the mean of its whole, uncovered first-list boxes' vectors,
             * following them with a gain (a vector of another kind starts it anew) */
            const double *e = xc->E + (size_t)j * RXEMB;
            if (xr[X_EN] > 0.0 && xr[X_EK] == c[XC_APP_KIND]) {
                for (int k = 0; k < RXEMB; k++)
                    xr[X_EMB + k] += c[XC_APP_GAIN] * (e[k] - xr[X_EMB + k]);
                xr[X_EN] += 1.0;
            } else {
                memcpy(xr + X_EMB, e, RXEMB * sizeof(double));
                xr[X_EN] = 1.0;
            }
            xr[X_EK] = c[XC_APP_KIND];
            xr[X_ET] = s->t;
        }
        if (ok && c[XC_LCOL] != 0.0 && (d[D_LAMPC] == 1.0 || d[D_LAMPC] == 2.0) && d[D_VEH] > 0.0) {
            /* The lamp pair's colour: red, the rear faces the camera; white, the front. Which end is the front,
             * where the object's motion has not shown it: turned when two more boxes say the other end than say
             * this one (a colour is wrong now and then). And a measurement of the heading, along the line of sight,
             * to XC_LCOL_SIG and no better however often it comes (a three-quarter view shows both lamps too). */
            double *e = row + RO_E, *Pe = row + RO_PE;
            const double *x = row + RO_X;
            double los = atan2(x[R_PY] - s->pose[1], x[R_PX] - s->pose[0]);
            double tg = d[D_LAMPC] == 2.0 ? los + PI : los, dd = wrap_pi(tg - e[2]);
            if (fabs(dd) > 0.5 * PI) {
                xr[X_FRONT] -= 1.0;
                if (xr[X_FRONT] <= -2.0 && row[RO_M + M_SENSE] == 0.0) {
                    e[2] = wrap_pi(e[2] + PI);
                    dd = wrap_pi(tg - e[2]);
                    xr[X_FRONT] = 1.0;
                }
            } else
                xr[X_FRONT] = pymin(xr[X_FRONT] + 1.0, 3.0);
            double R = c[XC_LCOL_SIG] * c[XC_LCOL_SIG];
            if (fabs(dd) <= 0.5 * PI && Pe[8] > 1.0001 * R) {
                double Hh[3] = {0.0, 0.0, 1.0};
                r_extent_take(e, Pe, Hh, dd, Pe[8] * R / (Pe[8] - R), INFINITY);
                r_extent_bounds(e);
            }
        }
    }
    if (low) {
        /* a box of the low-score list: it takes back the miss and adds the least a detection adds (it must not
         * confirm anything: no sighting counted, no score, no box remembered as a feature) */
        M[M_L] = pymax(-12.0, pymin(12.0, M[M_L] + pymin(dl, log(1.5))));
    } else {
        M[M_L] = pymax(-12.0, pymin(12.0, M[M_L] + dl + d[D_SCORE]));
        M[M_HITS] += 1.0;
    }
    M[M_THIT] = s->t;
    if (!(s->sen.type == SEN_CAMERA && (base == K_B || base == K_BT)))
        M[M_RANGE_T] = s->t;                             /* a box cut at the bottom gives a direction only */
    if (d[D_W] > 0.0 && !low) {
        M[M_BOXW] = d[D_W];
        M[M_BOXT] = s->t;
        M[M_BOXS] = (double)s->src;
    }
    M[M_NIS] = nis;
    M[M_DL] = dl;
    if (s->sen.type == SEN_MICROWAVE && s->sen.extended != 0.0)
        M[M_HOLD] = 2.0;                                 /* seen by a sensor that returns the parts of a thing */
    M[M_DET] = (double)j;
    M[M_SRC] = (double)s->src;
    M[M_PD] = s->lpd[a];
    if (h_leaf_add(&H->tr[h], r) < 0)
        return -1;
    *slot = r;
    H->c_children++;
    return r;
}

/* the object detection j starts (made when first asked for): its handle, or -1 when it starts none */
static int h_birth(Scan *s, int j)
{
    Hyp *H = s->H;
    if (s->birth_h[j] != -1)
        return s->birth_h[j] >= 0 ? s->birth_h[j] : -1;
    s->birth_h[j] = -2;
    if (s->births_left <= 0 || s->KB == NULL)
        return -1;
    if (s->D[(size_t)j * RDET + D_LOW] != 0.0)           /* a box of the low-score list starts nothing */
        return -1;
    int h = h_track_new(H);
    if (h < 0)
        return -1;
    int32_t r = h_take(H, h, -1);
    if (r < 0) {
        h_track_free(H, h);
        return -1;
    }
    double *row = s->A + (Py_ssize_t)r * RROW;
    memset(row, 0, RROW * sizeof(double));
    const double *kb = s->KB + (size_t)j * RKB;
    if (!r_birth(row, &s->sen, s->D + (size_t)j * RDET, s->pose, kb, s->bcfg, s->t, s->src, H->level)) {
        h_track_free(H, h);
        return -1;
    }
    row[RO_M + M_DET] = (double)j;
    if (s->xc != NULL) {                                 /* its memory for the detector's further inputs */
        const double *d = s->D + (size_t)j * RDET;
        double *xr = x_of(s->xc, row);
        memset(xr, 0, RXW * sizeof(double));
        if (s->xc->c[XC_APP] != 0.0 && d[D_HASEMB] != 0.0 && ((int)d[D_HID] & 15) == 0 && d[D_OCC] < 0.2) {
            memcpy(xr + X_EMB, s->xc->E + (size_t)j * RXEMB, RXEMB * sizeof(double));
            xr[X_EN] = 1.0;
            xr[X_EK] = s->xc->c[XC_APP_KIND];
            xr[X_ET] = s->t;
        }
        if ((int)d[D_KIND] & K_SCALE)
            xr[X_ST] = s->t;
        if (d[D_LAMPC] == 1.0 || d[D_LAMPC] == 2.0)
            xr[X_FRONT] = 1.0;
        if (s->xc->c[XC_LSIZE] != 0.0 && d[D_LAMPC] > 0.0 && d[D_LSEPR] > 0.0) {
            const double *x = row + RO_X;
            double ca = fabs(cos(row[RO_E + 2] - atan2(x[R_PY] - s->pose[1], x[R_PX] - s->pose[0])));
            if (ca >= 0.6) {
                xr[X_LS] = d[D_LSEP] - log(ca);
                xr[X_LST] = s->t;
                xr[X_LSR] = d[D_LSEPR];
            }
        }
    } else if (H->X != NULL)
        memset(H->X + (size_t)r * RXW, 0, RXW * sizeof(double));
    HTrack *t = &H->tr[h];
    t->born = H->level;
    memcpy(t->kin, kb, 4 * sizeof(double));
    t->veh = s->D[(size_t)j * RDET + D_VEH];
    t->idnum = -1.0;
    t->leaf[0] = r;
    t->nleaf = 1;
    t->primary = r;
    s->birth_h[j] = h;
    s->births_left--;
    H->c_births++;
    return h;
}

/* One cluster's hypotheses after the scan: for every hypothesis it held, the best assignments of the cluster's
 * detections to its leaves (each leaf: one detection or none; each detection: one leaf, or "something else",
 * which starts an object where a detection can), the most probable of them all kept. */
static int h_generate(Scan *s, int ci)
{
    Hyp *H = s->H;
    HCl *c = &H->cl[ci];
    int nt = c->nt, ng = c->ng, nd = (int)s->nd, K = H->k_best;
    int *cdet = (int *)ar_get(&s->ar, (size_t)(nd + 1) * sizeof(int));
    int *pk = (int *)ar_get(&s->ar, (size_t)(nt + 1) * sizeof(int));
    if (cdet == NULL || pk == NULL)
        return -1;
    int ndc = 0, any = 0;
    for (int j = 0; j < nd; j++)
        if (s->det_cl[j] == ci)
            cdet[ndc++] = j;
    for (int i = 0; i < nt; i++) {
        pk[i] = s->pk_of[c->tr[i]];
        any |= pk[i] >= 0;
    }
    if (!any)
        return 0;
    /* how many assignments each hypothesis may give: by its weight */
    int *kg = (int *)ar_get(&s->ar, (size_t)ng * sizeof(int));
    if (kg == NULL)
        return -1;
    int maxc = 0;
    for (int g = 0; g < ng; g++) {
        int k = (int)ceil(K * exp(c->lw[g]));
        kg[g] = k < 1 ? 1 : (k > K ? K : k);
        if (nt > H->max_cluster)
            kg[g] = 1;                                   /* a cluster too large to enumerate: the best one only */
        maxc += kg[g];
    }
    int *cg = (int *)ar_get(&s->ar, (size_t)maxc * sizeof(int));
    double *clw = (double *)ar_get(&s->ar, (size_t)maxc * sizeof(double));
    int *cas = (int *)ar_get(&s->ar, (size_t)(maxc * nt + 1) * sizeof(int));
    int *rti = (int *)ar_get(&s->ar, (size_t)(nt + 1) * sizeof(int));
    int *ra = (int *)ar_get(&s->ar, (size_t)(nt + 1) * sizeof(int));
    double *C = (double *)ar_get(&s->ar, (size_t)(nt * (ndc + nt) + 1) * sizeof(double));
    int *cols = (int *)ar_get(&s->ar, (size_t)(K * nt + 1) * sizeof(int));
    double *costs = (double *)ar_get(&s->ar, (size_t)(K + 1) * sizeof(double));
    if (!cg || !clw || !cas || !rti || !ra || !C || !cols || !costs)
        return -1;
    int nc = 0;
    for (int g = 0; g < ng; g++) {
        int n = 0;
        double base = c->lw[g];
        for (int i = 0; i < nt; i++) {
            int32_t q = c->lf[(size_t)g * nt + i];
            if (q < 0 || pk[i] < 0)
                continue;
            int a = s->missidx[q];
            rti[n] = i;
            ra[n] = a;
            base += s->lmiss[a];
            n++;
        }
        int m = ndc + n;
        for (int r = 0; r < n; r++) {
            int a = ra[r];
            for (int jd = 0; jd < ndc; jd++) {
                double v = s->lr[(size_t)a * nd + cdet[jd]];
                /* "it took detection j" against "it was missed and j is something else" */
                C[r * m + jd] = isfinite(v) ? -(s->lnr[a] + v - s->lmiss[a]) : LAP_BIG;
            }
            for (int r2 = 0; r2 < n; r2++)
                C[r * m + ndc + r2] = r2 == r ? 0.0 : LAP_BIG;
        }
        Arena tmp = {NULL, 0};
        int ns = murty(&tmp, C, n, m, kg[g], cols, costs);
        ar_free(&tmp);
        if (ns < 0)
            return -1;
        H->c_murty += ns;
        for (int q = 0; q < ns; q++) {
            cg[nc] = g;
            clw[nc] = base - costs[q];
            int *as = cas + (size_t)nc * nt;
            for (int i = 0; i < nt; i++)
                as[i] = -1;
            for (int r = 0; r < n; r++)
                if (cols[q * n + r] < ndc)
                    as[rti[r]] = cdet[cols[q * n + r]];
            nc++;
        }
    }
    if (nc == 0)
        return 0;
    /* the most probable of all of them, at most K, none less probable than the best by more than prune_ln */
    int *ord = (int *)ar_get(&s->ar, (size_t)nc * sizeof(int));
    if (ord == NULL)
        return -1;
    for (int i = 0; i < nc; i++) {
        int k = i;
        while (k > 0 && clw[i] > clw[ord[k - 1]]) {
            ord[k] = ord[k - 1];
            k--;
        }
        ord[k] = i;
    }
    int ng2 = nc < K ? nc : K;
    while (ng2 > 1 && clw[ord[ng2 - 1]] < clw[ord[0]] - H->prune_ln)
        ng2--;
    /* the objects started in them: a detection of the cluster that a kept hypothesis leaves over */
    int nb = 0;
    int *bj = (int *)ar_get(&s->ar, (size_t)(ndc + 1) * sizeof(int));
    int *bh = (int *)ar_get(&s->ar, (size_t)(ndc + 1) * sizeof(int));
    char *usedj = (char *)ar_get(&s->ar, (size_t)(nd + 1));
    if (!bj || !bh || !usedj)
        return -1;
    for (int jd = 0; jd < ndc; jd++) {
        int j = cdet[jd], left = 0;
        for (int q = 0; q < ng2 && !left; q++) {
            const int *as = cas + (size_t)ord[q] * nt;
            int taken = 0;
            for (int i = 0; i < nt; i++)
                if (as[i] == j) {
                    taken = 1;
                    break;
                }
            left = !taken;
        }
        if (!left)
            continue;
        int h = h_birth(s, j);
        if (h >= 0) {
            bj[nb] = j;
            bh[nb] = h;
            nb++;
        }
    }
    int nt2 = nt + nb;
    int32_t *tr2 = (int32_t *)PyMem_Malloc((size_t)nt2 * sizeof(int32_t));
    int32_t *lf2 = (int32_t *)PyMem_Malloc((size_t)(nt2 * ng2) * sizeof(int32_t));
    double *lw2 = (double *)PyMem_Malloc((size_t)ng2 * sizeof(double));
    if (!tr2 || !lf2 || !lw2) {
        PyMem_Free(tr2);
        PyMem_Free(lf2);
        PyMem_Free(lw2);
        return -1;
    }
    c = &H->cl[ci];
    memcpy(tr2, c->tr, (size_t)nt * sizeof(int32_t));
    for (int b = 0; b < nb; b++)
        tr2[nt + b] = bh[b];
    int fail = 0;
    for (int q = 0; q < ng2 && !fail; q++) {
        int g = cg[ord[q]];
        const int *as = cas + (size_t)ord[q] * nt;
        int32_t *o = lf2 + (size_t)q * nt2;
        memset(usedj, 0, (size_t)nd);
        for (int i = 0; i < nt; i++) {
            int32_t old = c->lf[(size_t)g * nt + i];
            if (old < 0 || pk[i] < 0) {
                o[i] = old;
                continue;
            }
            int a = s->missidx[old];
            if (as[i] < 0) {
                o[i] = s->lrow[a];
                s->mark[s->lrow[a]] = 3;                 /* the story "not reported" is in use */
            } else {
                o[i] = h_detchild(s, a, as[i]);
                usedj[as[i]] = 1;
                if (o[i] < 0)
                    fail = 1;
            }
        }
        for (int b = 0; b < nb; b++)
            o[nt + b] = usedj[bj[b]] ? -1 : H->tr[bh[b]].leaf[0];
        lw2[q] = clw[ord[q]];
    }
    if (fail) {
        PyMem_Free(tr2);
        PyMem_Free(lf2);
        PyMem_Free(lw2);
        PyErr_SetString(PyExc_MemoryError, "radar: the state array is full");
        return -1;
    }
    PyMem_Free(c->tr);
    PyMem_Free(c->lf);
    PyMem_Free(c->lw);
    c->tr = tr2;
    c->lf = lf2;
    c->lw = lw2;
    c->nt = nt2;
    c->ng = ng2;
    for (int b = 0; b < nb; b++)
        H->tr[bh[b]].cl = ci;
    H->c_hyps += ng2;
    return 0;
}

/* One scan: the core of Hyp.cycle. sc: its time, sensor, pose, constants, the state array (A) and the
 * numbers of objects (n) and detections (nd), set by the caller; th: the objects' handles; ti_in (n, HTI), D_in
 * (nd, RDET), KB_in (nd, RKB) | NULL, mem_in (nm, 5); bias: the sensor's five numbers where it has them (changed in
 * place). 0: done (sc->dead: the handles that ended, a list; sc->birth_h: per detection the object it started; the
 * caller frees sc->ar and sc->dead); 1: *need_out rows must be free first (nothing was changed); -1: an error is set. */
static int scan_core(Hyp *H, Scan *sc, Py_ssize_t cap, const Py_ssize_t *th, Py_ssize_t n, const double *ti_in,
                     const double *D_in, const double *KB_in, Py_ssize_t nm, const double *mem_in, int has_bias,
                     double *bias, Py_ssize_t *need_out)
{
    int took = 0;
    double t = sc->t;
    int nd = (int)sc->nd;
    if (cap < H->cap) {
        PyErr_SetString(PyExc_ValueError, "Hyp.cycle: the state array is smaller than the rows in use");
        return -1;
    }
    if (hyp_rows(H, cap) < 0)
        return -1;
    /* the rows this scan may need */
    {
        Py_ssize_t L = 0;
        for (Py_ssize_t k = 0; k < n; k++) {
            if (th[k] >= H->ntr || !H->tr[th[k]].alive) {
                PyErr_SetString(PyExc_ValueError, "Hyp.cycle: an object that does not exist");
                return -1;
            }
            for (Py_ssize_t k2 = 0; k2 < k; k2++)
                if (th[k2] == th[k]) {
                    PyErr_SetString(PyExc_ValueError, "Hyp.cycle: an object named twice");
                    return -1;
                }
            L += H->tr[th[k]].nleaf;
        }
        Py_ssize_t kids = L * nd, lim = (Py_ssize_t)H->k_best * (n > 0 ? n : 1);
        Py_ssize_t need = L + (kids < lim ? kids : lim) + nd;
        if (H->nfree < need) {
            *need_out = need;
            return 1;
        }
        sc->nl = (int)L;
    }
    H->level++;
    H->c_scans++;
    {
        int nl = sc->nl;
        size_t nlz = (size_t)(nl ? nl : 1), ndz = (size_t)(nd ? nd : 1);
        sc->th = (int32_t *)ar_get(&sc->ar, (size_t)(n + 1) * sizeof(int32_t));
        sc->ti = (double *)ar_get(&sc->ar, (size_t)(n + 1) * HTI * sizeof(double));
        sc->pk_of = (int *)ar_get(&sc->ar, (size_t)(H->ntr + nd + 1) * sizeof(int));
        sc->D = (double *)ar_get(&sc->ar, ndz * RDET * sizeof(double));
        sc->KB = KB_in == NULL ? NULL : (double *)ar_get(&sc->ar, ndz * RKB * sizeof(double));
        double *mem = (double *)ar_get(&sc->ar, (size_t)(5 * nm + 1) * sizeof(double));
        sc->lrow = (int32_t *)ar_get(&sc->ar, 2 * nlz * sizeof(int32_t));
        sc->lk = (int *)ar_get(&sc->ar, nlz * sizeof(int));
        sc->lpd = (double *)ar_get(&sc->ar, 5 * nlz * sizeof(double));
        sc->lr = (double *)ar_get(&sc->ar, nlz * ndz * sizeof(double));
        sc->missidx = (int32_t *)ar_get(&sc->ar, (size_t)cap * sizeof(int32_t));
        sc->detchild = (int32_t *)ar_get(&sc->ar, nlz * ndz * sizeof(int32_t));
        sc->det_cl = (int *)ar_get(&sc->ar, ndz * sizeof(int));
        sc->birth_h = (int32_t *)ar_get(&sc->ar, ndz * sizeof(int32_t));
        sc->mark = (char *)ar_get(&sc->ar, (size_t)cap);
        sc->as.lam0 = (double *)ar_get(&sc->ar, (5 * ndz + ndz * (size_t)(nm ? nm : 1)) * sizeof(double));
        sc->as.J = (Py_ssize_t *)ar_get(&sc->ar, ndz * sizeof(Py_ssize_t));
        sc->as.ok = (int *)ar_get(&sc->ar, ndz * sizeof(int));
        if (sc->ar.failed) {
            PyErr_NoMemory();
            return -1;
        }
        sc->lold = sc->lrow + nlz;
        sc->lcov = sc->lpd + nlz;
        sc->lmiss = sc->lcov + nlz;
        sc->lnr = sc->lmiss + nlz;
        sc->lpdlo = sc->lnr + nlz;
        sc->as.xc = sc->xc;
        if (sc->xc != NULL) {
            if (H->X == NULL && (H->X = (double *)PyMem_Calloc((size_t)H->cap * RXW, sizeof(double))) == NULL) {
                PyErr_NoMemory();
                return -1;
            }
            sc->xc->X = H->X;
            sc->xc->A = sc->A;
            sc->xc->t = t;
            if (sc->xc->c[XC_BOXOCC] != 0.0) {
                sc->xb = (XBox *)ar_get(&sc->ar, nlz * sizeof(XBox));
                if (sc->xb == NULL) {
                    PyErr_NoMemory();
                    return -1;
                }
                memset(sc->xb, 0, nlz * sizeof(XBox));
            }
        }
        sc->as.loc = sc->as.lam0 + ndz;
        sc->as.qs = sc->as.loc + ndz;
        sc->as.dets = sc->as.qs + ndz;
        sc->as.lk = sc->as.dets + ndz;
        sc->as.kmat = sc->as.lk + ndz;
        sc->as.sen = &sc->sen;
        sc->as.pose = sc->pose;
        sc->as.cfg = sc->acfg;
        sc->as.D = sc->D;
        sc->as.nd = nd;
        sc->as.nm = nm;
        sc->as.mem = mem;
        memset(sc->mark, 0, (size_t)cap);
        for (Py_ssize_t r = 0; r < cap; r++)
            sc->missidx[r] = -1;
        for (int i = 0; i < nl * nd; i++)
            sc->detchild[i] = -1;
        for (int j = 0; j < nd; j++) {
            sc->det_cl[j] = -1;
            sc->birth_h[j] = -1;
        }
        for (int i = 0; i < H->ntr + nd + 1; i++)
            sc->pk_of[i] = -1;
        if (n)
            memcpy(sc->ti, ti_in, (size_t)n * HTI * sizeof(double));
        if (nd)
            memcpy(sc->D, D_in, (size_t)nd * RDET * sizeof(double));
        if (nd && sc->KB)
            memcpy(sc->KB, KB_in, (size_t)nd * RKB * sizeof(double));
        if (nm)
            memcpy(mem, mem_in, (size_t)nm * 5 * sizeof(double));
        sc->births_left = (int)sc->bcfg[7];
        if ((sc->dead = PyList_New(0)) == NULL)
            return -1;
        /* 1. every leaf of the objects that take part, one scan on */
        int a = 0;
        for (Py_ssize_t k = 0; k < n; k++) {
            int h = (int)th[k];
            HTrack *tr = &H->tr[h];
            const double *ti = sc->ti + (size_t)k * HTI;
            sc->th[k] = h;
            sc->pk_of[h] = (int)k;
            memcpy(tr->kin, ti + 3, 4 * sizeof(double));
            tr->veh = ti[7];
            tr->idnum = ti[8];
            for (int q = 0; q < tr->nleaf; q++) {
                int32_t old = tr->leaf[q];
                int32_t c = h_step(H, sc->A, old, h, ti[0], ti[1], ti[2], sc->pose, tr->kin, &sc->par, t, sc->xc);
                if (c < 0) {
                    PyErr_SetString(PyExc_MemoryError, "radar: the state array is full");
                    return -1;
                }
                sc->lrow[a] = c;
                sc->lold[a] = old;
                sc->lk[a] = (int)k;
                sc->missidx[old] = a;
                tr->leaf[q] = c;
                a++;
            }
        }
        /* 2. the leaves against the detections; the scan's common angle taken out, and once more without it */
        if (h_occluders(sc) < 0) {
            PyErr_NoMemory();
            return -1;
        }
        if (has_bias && sc->sen.type == SEN_CAMERA) {
            /* The camera body's pitch against the road moves while the car drives (bumps, braking, the road's own
             * waves): a Gauss-Markov process of spread sqrt(gm_var) that forgets within gm_tau. Its value at the
             * last scan (bias[2], known to bias[3]) says less and less about this scan's; what it still says is
             * taken out of the scan's ground angles, what it no longer says is their variance. (Before: a white
             * 0.1 degrees per frame, an eighth of what a driving car's camera does; a standing car a pitch bump
             * away then looked like one that drives.) */
            double dtb = t - bias[4];
            double rho = (dtb >= 0.0 && dtb < 1e6 && sc->sen.gm_tau > 0.0) ? exp(-dtb / sc->sen.gm_tau) : 0.0;
            double pp = rho * bias[2], pvp = rho * rho * bias[3] + sc->sen.gm_var * (1.0 - rho * rho);
            for (int j = 0; j < nd; j++) {
                double *d = sc->D + (size_t)j * RDET;
                int base = (int)d[D_KIND] & 7, r = base == K_BT ? 1 : 0;
                if (base == K_RB || base == K_RBH || base == K_BT) {
                    d[D_Z + r] -= pp;
                    d[D_R + r] += pvp;
                }
            }
            if (dtb >= 0.0) {
                bias[2] = pp;
                bias[3] = pvp;
                bias[4] = t;
            }
        }
        h_ratios(sc);
        if (has_bias && nd) {
            took = h_common(sc, bias);
            if (took < 0) {
                if (!PyErr_Occurred())
                    PyErr_NoMemory();
                return -1;
            }
            if (took)
                h_ratios(sc);
        }
        for (a = 0; a < nl; a++) {
            double *M = sc->A + (Py_ssize_t)sc->lrow[a] * RROW + RO_M;
            M[M_PD] = sc->lpd[a];
            M[M_COV] = sc->lcov[a];
        }
        /* 3. objects that share a detection compete: one cluster */
        {
            int *uf = (int *)ar_get(&sc->ar, (size_t)(H->ntr + 1) * sizeof(int));
            int *rep = (int *)ar_get(&sc->ar, (size_t)(H->ntr + 1) * sizeof(int));
            if (uf == NULL || rep == NULL) {
                PyErr_NoMemory();
                return -1;
            }
            for (int i = 0; i < H->ntr; i++) {
                uf[i] = i;
                rep[i] = -1;
            }
            for (int c = 0; c < H->ncl; c++)
                if (H->cl[c].alive)
                    for (int i = 1; i < H->cl[c].nt; i++)
                        uf[uf_find(uf, H->cl[c].tr[i])] = uf_find(uf, H->cl[c].tr[0]);
            for (int j = 0; j < nd; j++) {
                int first = -1;
                for (a = 0; a < nl; a++)
                    if (isfinite(sc->lr[(size_t)a * nd + j])) {
                        int h = sc->th[sc->lk[a]];
                        if (first < 0)
                            first = h;
                        else
                            uf[uf_find(uf, h)] = uf_find(uf, first);
                    }
            }
            int ncl0 = H->ncl;
            for (int c = 0; c < ncl0; c++) {
                if (!H->cl[c].alive || H->cl[c].nt == 0)
                    continue;
                int r = uf_find(uf, H->cl[c].tr[0]);
                if (rep[r] < 0)
                    rep[r] = c;
                else if (h_merge(H, &sc->ar, rep[r], c) < 0) {
                    PyErr_NoMemory();
                    return -1;
                }
            }
            for (int j = 0; j < nd; j++)
                for (a = 0; a < nl; a++)
                    if (isfinite(sc->lr[(size_t)a * nd + j])) {
                        sc->det_cl[j] = H->tr[sc->th[sc->lk[a]]].cl;
                        break;
                    }
        }
        /* 4. each cluster's hypotheses; then its trees cut and its objects' primaries set */
        Settle st = {H, sc->A, &sc->ar, t, sc->dead, sc->mark};
        int ncl0 = H->ncl;
        for (int c = 0; c < ncl0; c++) {
            if (!H->cl[c].alive)
                continue;
            int part = 0;
            for (int i = 0; i < H->cl[c].nt; i++)
                part |= sc->pk_of[H->cl[c].tr[i]] >= 0;
            if (!part)
                continue;
            if (h_generate(sc, c) < 0) {
                if (!PyErr_Occurred())
                    PyErr_NoMemory();
                return -1;
            }
        }
        /* the stories "not reported" that are in use: a miss tells against the object */
        for (a = 0; a < nl; a++) {
            if (sc->mark[sc->lrow[a]] != 3)
                continue;
            sc->mark[sc->lrow[a]] = 0;
            if (sc->lpd[a] > 0.0)
                sc->A[(Py_ssize_t)sc->lrow[a] * RROW + RO_M + M_L] += log(pymax(1.0 - (sc->lpd[a] + sc->lpdlo[a]), 1e-3));
        }
        /* a detection no object could have given starts one by itself */
        for (int j = 0; j < nd; j++) {
            if (sc->det_cl[j] >= 0)
                continue;
            int h = h_birth(sc, j);
            if (h < 0)
                continue;
            int c2 = h_cluster_new(H, 1, 1);
            if (c2 < 0) {
                PyErr_NoMemory();
                return -1;
            }
            H->cl[c2].tr[0] = h;
            H->cl[c2].lf[0] = H->tr[h].leaf[0];
            H->cl[c2].lw[0] = 0.0;
            H->tr[h].cl = c2;
            H->tr[h].in_map = 1;
            H->tr[h].pp = 1.0;
            H->tr[h].pe = sigm(sc->A[(Py_ssize_t)H->tr[h].leaf[0] * RROW + RO_M + M_L]);
        }
        for (int c = 0; c < ncl0; c++) {
            if (!H->cl[c].alive)
                continue;
            int part = 0;
            for (int i = 0; i < H->cl[c].nt; i++) {
                int h = H->cl[c].tr[i];
                part |= (h < H->ntr && sc->pk_of[h] >= 0) || H->tr[h].born == H->level;
            }
            if (!part)
                continue;
            if (h_settle(&st, c) < 0) {
                if (!PyErr_Occurred())
                    PyErr_NoMemory();
                return -1;
            }
        }
        for (Py_ssize_t k = 0; k < n; k++)
            if (H->tr[th[k]].alive)
                h_trim(H, sc->A, &H->tr[th[k]], t);
    }
    return 0;
}

/* an object's handle checked */
static HTrack *h_get(Hyp *H, long h)
{
    if (h < 0 || h >= H->ntr || !H->tr[h].alive) {
        PyErr_SetString(PyExc_ValueError, "Hyp: an object that does not exist");
        return NULL;
    }
    return &H->tr[h];
}

/* Hyp.drop(S, handle) -> [handles ended]: the object is taken out (the upkeep dropped it, or it was merged
 * into another); its cluster's hypotheses that differed only in it become one. */
static PyObject *hyp_drop(PyObject *self, PyObject *args)
{
    Hyp *H = (Hyp *)self;
    PyObject *oS;
    long h;
    if (!PyArg_ParseTuple(args, "Ol", &oS, &h))
        return NULL;
    HTrack *t = h_get(H, h);
    if (t == NULL)
        return NULL;
    View vS;
    Py_ssize_t cap;
    if (store_get(oS, &vS, &cap) < 0)
        return NULL;
    PyObject *dead = PyList_New(0);
    Arena ar = {NULL, 0};
    char *mark = (char *)ar_get(&ar, (size_t)H->cap);
    int ci = t->cl;
    if (dead == NULL || mark == NULL || cap < H->cap) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_ValueError, "Hyp.drop: no memory, or a state array that is too small");
        goto fail;
    }
    memset(mark, 0, (size_t)H->cap);
    if (ci >= 0) {
        HCl *c = &H->cl[ci];
        int nt = c->nt;
        for (int i = 0; i < nt; i++)
            if (c->tr[i] == h)
                for (int g = 0; g < c->ng; g++)
                    c->lf[(size_t)g * nt + i] = -1;     /* in no story any more: h_settle ends it */
        /* hypotheses that differed only in this object are one now */
        char *drop = (char *)ar_get(&ar, (size_t)c->ng + 1);
        int32_t *tmp = (int32_t *)ar_get(&ar, (size_t)(nt + 1) * sizeof(int32_t));
        if (drop == NULL || tmp == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        memset(drop, 0, (size_t)c->ng);
        for (int g = 0; g < c->ng; g++) {
            if (drop[g])
                continue;
            for (int g2 = g + 1; g2 < c->ng; g2++)
                if (!drop[g2] && memcmp(c->lf + (size_t)g * nt, c->lf + (size_t)g2 * nt, (size_t)nt * sizeof(int32_t)) == 0) {
                    double two[2] = {c->lw[g], c->lw[g2]};
                    c->lw[g] = lse(two, 2);
                    drop[g2] = 1;
                }
        }
        c->ng = h_compact(c->lf, c->lw, c->ng, nt, drop);
        h_sort(c->lf, c->lw, c->ng, nt, tmp);
        Settle st = {H, (double *)vS.b.buf, &ar, 0.0, dead, mark};
        if (h_settle(&st, ci) < 0) {
            if (!PyErr_Occurred())
                PyErr_NoMemory();
            goto fail;
        }
    } else
        h_track_free(H, (int)h);
    ar_free(&ar);
    view_drop(&vS);
    return dead;
fail:
    Py_XDECREF(dead);
    ar_free(&ar);
    view_drop(&vS);
    return NULL;
}

/* Hyp.rows(handle) -> every row of the object (its whole tree), the primary first */
static PyObject *hyp_rows_of(PyObject *self, PyObject *arg)
{
    Hyp *H = (Hyp *)self;
    long h = PyLong_AsLong(arg);
    if (h == -1 && PyErr_Occurred())
        return NULL;
    HTrack *t = h_get(H, h);
    if (t == NULL)
        return NULL;
    PyObject *l = PyList_New(0);
    if (l == NULL)
        return NULL;
    for (int pass = 0; pass < 2; pass++)
        for (Py_ssize_t r = 0; r < H->cap; r++) {
            if (H->owner[r] != h || (pass == 0) != (r == t->primary))
                continue;
            PyObject *o = PyLong_FromSsize_t(r);
            if (o == NULL || PyList_Append(l, o) < 0) {
                Py_XDECREF(o);
                Py_DECREF(l);
                return NULL;
            }
            Py_DECREF(o);
        }
    return l;
}

/* Hyp.chain(handle) -> the primary's story: its rows from now back to the oldest kept */
static PyObject *hyp_chain(PyObject *self, PyObject *arg)
{
    Hyp *H = (Hyp *)self;
    long h = PyLong_AsLong(arg);
    if (h == -1 && PyErr_Occurred())
        return NULL;
    HTrack *t = h_get(H, h);
    if (t == NULL)
        return NULL;
    PyObject *l = PyList_New(0);
    if (l == NULL)
        return NULL;
    for (int32_t r = t->primary; r >= 0; r = H->parent[r]) {
        PyObject *o = PyLong_FromLong(r);
        if (o == NULL || PyList_Append(l, o) < 0) {
            Py_XDECREF(o);
            Py_DECREF(l);
            return NULL;
        }
        Py_DECREF(o);
    }
    return l;
}

/* Hyp.leaves(handle) -> [(row, probability)] the object's hypotheses with their probabilities in its cluster */
static PyObject *hyp_leaves(PyObject *self, PyObject *arg)
{
    Hyp *H = (Hyp *)self;
    long h = PyLong_AsLong(arg);
    if (h == -1 && PyErr_Occurred())
        return NULL;
    HTrack *t = h_get(H, h);
    if (t == NULL)
        return NULL;
    PyObject *l = PyList_New(t->nleaf);
    if (l == NULL)
        return NULL;
    const HCl *c = t->cl >= 0 ? &H->cl[t->cl] : NULL;
    int col = -1;
    for (int i = 0; c && i < c->nt; i++)
        if (c->tr[i] == h)
            col = i;
    for (int k = 0; k < t->nleaf; k++) {
        double w = 0.0;
        for (int g = 0; c && col >= 0 && g < c->ng; g++)
            if (c->lf[(size_t)g * c->nt + col] == t->leaf[k])
                w += exp(c->lw[g]);
        PyObject *o = Py_BuildValue("(id)", (int)t->leaf[k], w);
        if (o == NULL) {
            Py_DECREF(l);
            return NULL;
        }
        PyList_SetItem(l, k, o);
    }
    return l;
}

/* Hyp.start(S, row_values, kin, vehicle, id) -> (handle, row): an object put in from outside (tests, tools): one
 * leaf with the given row (RROW numbers), a cluster of its own */
static PyObject *hyp_start(PyObject *self, PyObject *args)
{
    Hyp *H = (Hyp *)self;
    PyObject *oS, *orow, *okin;
    double veh, idnum;
    if (!PyArg_ParseTuple(args, "OOOdd", &oS, &orow, &okin, &veh, &idnum))
        return NULL;
    View vS;
    Py_ssize_t cap;
    if (store_get(oS, &vS, &cap) < 0)
        return NULL;
    PyObject *res = NULL;
    double kin[4];
    if (cap < H->cap || hyp_rows(H, cap) < 0 || rnums_get(okin, 4, kin, "kin") < 0) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_ValueError, "Hyp.start: a state array that is too small");
        goto done;
    }
    int h = h_track_new(H);
    int32_t r = h >= 0 ? h_take(H, h, -1) : -1;
    int c = r >= 0 ? h_cluster_new(H, 1, 1) : -1;
    if (c < 0) {
        if (h >= 0)
            h_track_free(H, h);
        PyErr_SetString(PyExc_MemoryError, "Hyp.start: no free row");
        goto done;
    }
    double *row = (double *)vS.b.buf + (Py_ssize_t)r * RROW;
    if (H->X != NULL)
        memset(H->X + (size_t)r * RXW, 0, RXW * sizeof(double));
    if (rnums_get(orow, RROW, row, "row") < 0) {
        h_track_free(H, h);
        h_cluster_free(H, c);
        goto done;
    }
    HTrack *t = &H->tr[h];
    memcpy(t->kin, kin, sizeof(kin));
    t->veh = veh;
    t->idnum = idnum;
    t->born = H->level;
    t->leaf[0] = r;
    t->nleaf = 1;
    t->primary = r;
    t->cl = c;
    t->in_map = 1;
    t->pp = 1.0;
    t->pe = sigm(row[RO_M + M_L]);
    row[RO_M + M_LEVEL] = (double)H->level;
    H->cl[c].tr[0] = h;
    H->cl[c].lf[0] = r;
    H->cl[c].lw[0] = 0.0;
    res = Py_BuildValue("(ii)", h, (int)r);
done:
    view_drop(&vS);
    return res;
}

/* Hyp.info() -> the knobs, what is in use, the counters */
static PyObject *hyp_info(PyObject *self, PyObject *unused)
{
    Hyp *H = (Hyp *)self;
    int nt = 0, nc = 0, nl = 0, ng = 0, gmax = 0, tmax = 0;
    for (int i = 0; i < H->ntr; i++)
        if (H->tr[i].alive) {
            nt++;
            nl += H->tr[i].nleaf;
        }
    for (int i = 0; i < H->ncl; i++)
        if (H->cl[i].alive) {
            nc++;
            ng += H->cl[i].ng;
            if (H->cl[i].ng > gmax)
                gmax = H->cl[i].ng;
            if (H->cl[i].nt > tmax)
                tmax = H->cl[i].nt;
        }
    return Py_BuildValue("{s:n,s:n,s:i,s:i,s:i,s:i,s:i,s:i,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:i,s:i,s:i,s:d,s:i,s:d}",
                         "rows", H->cap, "rows_free", H->nfree, "objects", nt, "leaves", nl, "clusters", nc,
                         "hypotheses", ng, "hypotheses_max", gmax, "cluster_max", tmax, "level", H->level,
                         "scans", H->c_scans, "steps", H->c_leaves, "updates", H->c_children, "made", H->c_hyps,
                         "assignments", H->c_murty, "births", H->c_births, "pruned", H->c_pruned,
                         "n_scan", H->n_scan, "k_best", H->k_best, "max_leaves", H->max_leaves, "prune_ln", H->prune_ln,
                         "keep_steps", H->keep_steps, "keep_s", H->keep_s);
}

/* Hyp.clusters() -> [(objects, [(log weight, leaves)])] for tools and tests */
static PyObject *h_clusters(Hyp *H, int all)
{
    PyObject *l = PyList_New(0);
    if (l == NULL)
        return NULL;
    for (int ci = 0; ci < H->ncl; ci++) {
        const HCl *c = &H->cl[ci];
        if (!c->alive && !all)
            continue;
        PyObject *tr = PyList_New(c->nt), *gs = PyList_New(c->ng), *item = NULL;
        if (tr == NULL || gs == NULL)
            goto bad;
        for (int i = 0; i < c->nt; i++)
            PyList_SetItem(tr, i, PyLong_FromLong(c->tr[i]));
        for (int g = 0; g < c->ng; g++) {
            PyObject *lf = PyList_New(c->nt);
            if (lf == NULL)
                goto bad;
            for (int i = 0; i < c->nt; i++)
                PyList_SetItem(lf, i, PyLong_FromLong(c->lf[(size_t)g * c->nt + i]));
            PyObject *e = Py_BuildValue("(dN)", c->lw[g], lf);
            if (e == NULL)
                goto bad;
            PyList_SetItem(gs, g, e);
        }
        item = PyTuple_Pack(2, tr, gs);
        if (item == NULL || PyList_Append(l, item) < 0)
            goto bad;
        Py_DECREF(item);
        Py_DECREF(tr);
        Py_DECREF(gs);
        continue;
    bad:
        Py_XDECREF(tr);
        Py_XDECREF(gs);
        Py_XDECREF(item);
        Py_DECREF(l);
        return NULL;
    }
    return l;
}

static PyObject *hyp_clusters(PyObject *self, PyObject *unused)
{
    return h_clusters((Hyp *)self, 0);
}

/* Hyp.dump() -> everything it holds as plain lists (tests record it; Hyp.load restores it) */
static PyObject *hyp_dump(PyObject *self, PyObject *unused)
{
    Hyp *H = (Hyp *)self;
    PyObject *rows = PyList_New(0), *trs = PyList_New(0), *cls = h_clusters(H, 1), *fr = PyList_New(H->nfree);
    PyObject *res = NULL;
    if (!rows || !trs || !cls || !fr)
        goto done;
    for (Py_ssize_t i = 0; i < H->nfree; i++)
        PyList_SetItem(fr, i, PyLong_FromLong(H->freel[i]));
    for (Py_ssize_t r = 0; r < H->cap; r++) {
        if (H->owner[r] < 0)
            continue;
        PyObject *o = Py_BuildValue("(niii)", r, (int)H->parent[r], (int)H->nchild[r], (int)H->owner[r]);
        if (o == NULL || PyList_Append(rows, o) < 0) {
            Py_XDECREF(o);
            goto done;
        }
        Py_DECREF(o);
    }
    for (int i = 0; i < H->ntr; i++) {
        const HTrack *t = &H->tr[i];
        PyObject *lv = PyList_New(t->alive ? t->nleaf : 0);
        if (lv == NULL)
            goto done;
        for (int k = 0; t->alive && k < t->nleaf; k++)
            PyList_SetItem(lv, k, PyLong_FromLong(t->leaf[k]));
        PyObject *o = Py_BuildValue("(iiiiLddddddddN)", t->alive, t->in_map, (int)t->primary, (int)t->cl, t->alive ? t->born : -1LL, t->kin[0],
                                    t->kin[1], t->kin[2], t->kin[3], t->veh, t->idnum, t->pe, t->pp, lv);
        if (o == NULL || PyList_Append(trs, o) < 0) {
            Py_XDECREF(o);
            goto done;
        }
        Py_DECREF(o);
    }
    res = Py_BuildValue("{s:n,s:L,s:(iiididi),s:O,s:O,s:O,s:O}", "cap", H->cap, "level", H->level, "knobs", H->n_scan,
                        H->k_best, H->max_leaves, H->prune_ln, H->keep_steps, H->keep_s, H->max_cluster, "rows", rows,
                        "objects", trs, "clusters", cls, "free", fr);
    if (res != NULL && H->X != NULL) {
        /* the rows' memory for the detector's further inputs (only where a scan has asked for it): per row in use
         * [row, its RXW numbers]; the brightness followed */
        PyObject *xl = PyList_New(0);
        int bad = xl == NULL;
        for (Py_ssize_t r = 0; !bad && r < H->cap; r++) {
            if (H->owner[r] < 0)
                continue;
            PyObject *o = PyList_New(RXW + 1);
            if (o == NULL) {
                bad = 1;
                break;
            }
            PyList_SetItem(o, 0, PyLong_FromSsize_t(r));
            for (int k = 0; k < RXW; k++)
                PyList_SetItem(o, k + 1, PyFloat_FromDouble(H->X[(size_t)r * RXW + k]));
            bad = PyList_Append(xl, o) < 0;
            Py_DECREF(o);
        }
        PyObject *dk = bad ? NULL : Py_BuildValue("(id)", H->dark_has, H->dark_y);
        if (bad || dk == NULL || PyDict_SetItemString(res, "x", xl) < 0 || PyDict_SetItemString(res, "dark", dk) < 0)
            Py_CLEAR(res);
        Py_XDECREF(xl);
        Py_XDECREF(dk);
    }
done:
    Py_XDECREF(rows);
    Py_XDECREF(trs);
    Py_XDECREF(cls);
    Py_XDECREF(fr);
    return res;
}

/* Hyp.load(dump) -> None: this object made what `dump` recorded (it must be new: nothing in use) */
static PyObject *hyp_load(PyObject *self, PyObject *d)
{
    Hyp *H = (Hyp *)self;
    PyObject *rows, *trs, *cls, *fr, *knobs;
    Py_ssize_t cap;
    long long level;
    if (!PyDict_Check(d) || !(rows = PyDict_GetItemString(d, "rows")) || !(trs = PyDict_GetItemString(d, "objects")) ||
        !(cls = PyDict_GetItemString(d, "clusters")) || !(fr = PyDict_GetItemString(d, "free")) ||
        !(knobs = PyDict_GetItemString(d, "knobs")) || !PyDict_GetItemString(d, "cap") || !PyDict_GetItemString(d, "level")) {
        PyErr_SetString(PyExc_TypeError, "Hyp.load: what Hyp.dump gave is needed");
        return NULL;
    }
    cap = PyLong_AsSsize_t(PyDict_GetItemString(d, "cap"));
    level = PyLong_AsLongLong(PyDict_GetItemString(d, "level"));
    int n_scan, k_best, max_leaves, keep_steps, max_cluster;
    double prune_ln, keep_s;
    if (PyErr_Occurred() || !PyArg_ParseTuple(knobs, "iiididi", &n_scan, &k_best, &max_leaves, &prune_ln, &keep_steps, &keep_s,
                                              &max_cluster))
        return NULL;
    hyp_release(H);
    H->n_scan = n_scan;
    H->k_best = k_best;
    H->max_leaves = max_leaves;
    H->prune_ln = prune_ln;
    H->keep_steps = keep_steps;
    H->keep_s = keep_s;
    H->max_cluster = max_cluster;
    H->level = level;
    if (hyp_rows(H, cap) < 0)
        return NULL;
    H->nfree = 0;
    for (Py_ssize_t i = 0; i < PyList_Size(fr); i++)
        H->freel[H->nfree++] = (int32_t)PyLong_AsLong(PyList_GetItem(fr, i));
    for (Py_ssize_t i = 0; i < PyList_Size(rows); i++) {
        Py_ssize_t r;
        int p, nc, ow;
        if (!PyArg_ParseTuple(PyList_GetItem(rows, i), "niii", &r, &p, &nc, &ow) || r < 0 || r >= cap) {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_ValueError, "Hyp.load: a row outside the state array");
            return NULL;
        }
        H->parent[r] = p;
        H->nchild[r] = nc;
        H->owner[r] = ow;
    }
    for (Py_ssize_t i = 0; i < PyList_Size(trs); i++) {
        int h = h_track_new(H);
        if (h < 0)
            return PyErr_NoMemory();
        HTrack *t = &H->tr[h];
        int alive, in_map, prim, cl;
        PyObject *lv;
        if (!PyArg_ParseTuple(PyList_GetItem(trs, i), "iiiiLddddddddO", &alive, &in_map, &prim, &cl, &t->born, &t->kin[0],
                              &t->kin[1], &t->kin[2], &t->kin[3], &t->veh, &t->idnum, &t->pe, &t->pp, &lv))
            return NULL;
        t->in_map = in_map;
        t->primary = prim;
        t->cl = -1;
        for (Py_ssize_t k = 0; k < PyList_Size(lv); k++)
            if (h_leaf_add(t, (int32_t)PyLong_AsLong(PyList_GetItem(lv, k))) < 0)
                return PyErr_NoMemory();
        if (!alive) {                                    /* a handle that was free: free here too, its number kept */
            t->alive = 0;
            t->born = -1;
        }
    }
    for (int i = 0; i < H->ntr; i++)
        if (!H->tr[i].alive) {
            PyMem_Free(H->tr[i].leaf);
            H->tr[i].leaf = NULL;
            H->tr[i].nleaf = H->tr[i].leafcap = 0;
        }
    for (Py_ssize_t i = 0; i < PyList_Size(cls); i++) {
        PyObject *tr, *gs;
        if (!PyArg_ParseTuple(PyList_GetItem(cls, i), "OO", &tr, &gs))
            return NULL;
        int nt = (int)PyList_Size(tr), ng = (int)PyList_Size(gs);
        int c = h_cluster_new(H, nt, ng);
        if (c < 0)
            return PyErr_NoMemory();
        for (int k = 0; k < nt; k++) {
            H->cl[c].tr[k] = (int32_t)PyLong_AsLong(PyList_GetItem(tr, k));
            if (H->cl[c].tr[k] >= 0 && H->cl[c].tr[k] < H->ntr)
                H->tr[H->cl[c].tr[k]].cl = c;
        }
        for (int g = 0; g < ng; g++) {
            PyObject *lf;
            if (!PyArg_ParseTuple(PyList_GetItem(gs, g), "dO", &H->cl[c].lw[g], &lf))
                return NULL;
            for (int k = 0; k < nt; k++)
                H->cl[c].lf[(size_t)g * nt + k] = (int32_t)PyLong_AsLong(PyList_GetItem(lf, k));
        }        if (nt == 0)                                     /* a slot that was free: its number kept */
            H->cl[c].alive = 2;
    }
    for (int c = 0; c < H->ncl; c++)
        if (H->cl[c].alive == 2)
            h_cluster_free(H, c);
    {
        PyObject *xl = PyDict_GetItemString(d, "x"), *dk = PyDict_GetItemString(d, "dark");
        if (xl != NULL && PyList_Check(xl)) {
            if ((H->X = (double *)PyMem_Calloc((size_t)H->cap * RXW, sizeof(double))) == NULL)
                return PyErr_NoMemory();
            for (Py_ssize_t i = 0; i < PyList_Size(xl); i++) {
                PyObject *o = PyList_GetItem(xl, i);
                if (!PyList_Check(o) || PyList_Size(o) != RXW + 1)
                    continue;
                Py_ssize_t r = PyLong_AsSsize_t(PyList_GetItem(o, 0));
                if (r < 0 || r >= H->cap)
                    continue;
                for (int k = 0; k < RXW; k++)
                    H->X[(size_t)r * RXW + k] = PyFloat_AsDouble(PyList_GetItem(o, k + 1));
            }
            if (dk != NULL && !PyArg_ParseTuple(dk, "id", &H->dark_has, &H->dark_y))
                return NULL;
        }
    }
    if (PyErr_Occurred())
        return NULL;
    Py_RETURN_NONE;
}


/* ---- the past of a story, smoothed (fixed lag) ----
 * The chain of nodes from an object's primary back is one story's history: at each node the motion hypotheses
 * as they were after that scan (filtered), and the step that led into it (the switching matrix, dt, what the
 * car's own movement added). A backward pass over the chain gives, for every node, the estimate from ALL the
 * detections of the story, the later ones included: a Rauch-Tung-Striebel pass per motion hypothesis with the
 * hypotheses mixed backwards (the smoother of the interacting multiple model filter: Nadarajah, Tharmarasa,
 * McDonald, Kirubarajan 2012), and a plain RTS pass for the free estimate. docs/RADAR.md, section 6. */

/* the transition of one hypothesis over dt, dense (r_predict's F) */
static void r_trans(int model, double dt, const RPar *p, double *F)
{
    memset(F, 0, RNN * sizeof(double));
    for (int i = 0; i < RN; i++)
        F[i * RN + i] = 1.0;
    if (model == R_STAND) {
        for (int i = R_VX; i <= R_AY; i++)
            F[i * RN + i] = 0.0;
    } else if (model == R_CV) {
        F[R_PX * RN + R_VX] = F[R_PY * RN + R_VY] = dt;
        F[R_AX * RN + R_AX] = F[R_AY * RN + R_AY] = 0.0;
    } else {
        double rho = exp(-fabs(dt) / p->tau);
        F[R_PX * RN + R_VX] = F[R_PY * RN + R_VY] = dt;
        F[R_PX * RN + R_AX] = F[R_PY * RN + R_AY] = 0.5 * dt * dt;
        F[R_VX * RN + R_AX] = F[R_VY * RN + R_AY] = dt;
        F[R_AX * RN + R_AX] = F[R_AY * RN + R_AY] = rho;
    }
}

/* the Cholesky factor of a symmetric positive matrix A (N x N): L lower; -1 where A is not positive */
static int r_chol8(const double *A, double *L)
{
    memset(L, 0, RNN * sizeof(double));
    for (int i = 0; i < RN; i++)
        for (int j = 0; j <= i; j++) {
            double acc = A[i * RN + j];
            for (int k = 0; k < j; k++)
                acc -= L[i * RN + k] * L[j * RN + k];
            if (i == j) {
                if (!(acc > 0.0) || !isfinite(acc))
                    return -1;
                L[i * RN + i] = sqrt(acc);
            } else
                L[i * RN + j] = acc / L[j * RN + j];
        }
    return 0;
}

/* The gain of one step back over a prediction (r_rts): the prediction (xp, Pp) of the filtered (x, P) over dt with
 * what the car's own movement added (e), and G = P F^T Pp^-1. 0: the prediction is not positive. */
static int r_rts_gain(int model, const double *x, const double *P, double dt, const double *kin, const RPar *par,
                      const double *e, double *xp, double *Pp, double *G)
{
    double F[RNN], C[RNN], L[RNN];
    r_predict(model, x, P, dt, kin[0], kin[1], kin[2], par, xp, Pp);
    Pp[0] += e[0];
    Pp[1] += e[1];
    Pp[RN] += e[1];
    Pp[RN + 1] += e[2];
    r_trans(model, dt, par, F);
    if (all_finite(P, RNN)) {
        /* C = P F^T over the entries of F that are not zero (at most three per row): with a finite P the terms left
         * out are zeros, and a sum that starts at +0 is the same float with or without them */
        int nz[RN][RN], nn[RN];
        for (int j = 0; j < RN; j++) {
            nn[j] = 0;
            for (int k = 0; k < RN; k++)
                if (F[j * RN + k] != 0.0)
                    nz[j][nn[j]++] = k;
        }
        for (int i = 0; i < RN; i++)
            for (int j = 0; j < RN; j++) {
                double acc = 0.0;
                for (int q = 0; q < nn[j]; q++)
                    acc += P[i * RN + nz[j][q]] * F[j * RN + nz[j][q]];
                C[i * RN + j] = acc;
            }
    } else
        for (int i = 0; i < RN; i++)                     /* C = P F^T */
            for (int j = 0; j < RN; j++) {
                double acc = 0.0;
                for (int k = 0; k < RN; k++)
                    acc += P[i * RN + k] * F[j * RN + k];
                C[i * RN + j] = acc;
            }
    if (r_chol8(Pp, L) < 0)
        return 0;                                        /* (a prediction that is not positive: the node stays filtered) */
    for (int i = 0; i < RN; i++) {                       /* G = C Pp^-1: each row solved against L L^T */
        double y[RN];
        for (int j = 0; j < RN; j++) {
            double acc = C[i * RN + j];
            for (int k = 0; k < j; k++)
                acc -= L[j * RN + k] * y[k];
            y[j] = acc / L[j * RN + j];
        }
        for (int j = RN - 1; j >= 0; j--) {
            double acc = y[j];
            for (int k = j + 1; k < RN; k++)
                acc -= L[k * RN + j] * G[i * RN + k];
            G[i * RN + j] = acc / L[j * RN + j];
        }
    }
    return 1;
}

/* What r_rts_gain gave for a step is kept with the step's child row, per hypothesis (and the free estimate): the
 * inputs it was computed from (x, P, dt, e, kin) and the results. A later pass over the same story finds most steps
 * unchanged (a sighting adds one node at the front); where the kept inputs are the same bits, the kept results are
 * what the computation would give again. Nothing has to be told when rows are rewritten (the height prior, the
 * sensor's registration, the opened uncertainty, a group's support, the heading prior, a row taken anew): the inputs
 * then differ and the gain is computed. The angle offset and the height (x[dp], x[lh]) are left out of what is
 * compared: the prediction carries each of them over as it is (r_predict: its coefficient is 1, its noise is in P)
 * and nothing else of the prediction, its covariance or the gain reads them, so the kept results hold with the
 * present two values put in. (The sensor's registration moves every row's angle offset at every message: with them
 * compared, nothing kept of a camera's objects would ever be used.) The model's constants are kept once (Hyp.sc_par): other constants, nothing
 * kept is used. */
#define RSC_IN (RN + RNN + 1 + 3 + 3)
#define RSC_W (RSC_IN + 1 + RN + RNN + RNN)

/* One step back over a prediction: from the filtered (x, P) at a node and the smoothed (xs, Ps) at its child, the
 * smoothed (x, P) at the node (in place). model, dt, kin, par: the step; e: what the car's own movement added to
 * the place at the child (e00, e01, e11). keep: RSC_W numbers kept with the step (NULL: nothing is kept). */
static void r_rts(int model, double *x, double *P, const double *xs, const double *Ps, double dt, const double *kin,
                  const RPar *par, const double *e, double *keep)
{
    double in[RSC_IN], loc[1 + RN + RNN + RNN], *out = loc;
    if (keep) {
        memcpy(in, x, RN * sizeof(double));
        in[R_DP] = in[R_LH] = 0.0;                       /* (not inputs of the gain: see above) */
        memcpy(in + RN, P, RNN * sizeof(double));
        in[RN + RNN] = dt;
        memcpy(in + RN + RNN + 1, e, 3 * sizeof(double));
        memcpy(in + RN + RNN + 4, kin, 3 * sizeof(double));
        out = keep + RSC_IN;
        if (out[0] == 0.0 || memcmp(keep, in, sizeof(in)) != 0) {
            out[0] = 0.0;                                /* (not valid while it is written) */
            int ok = r_rts_gain(model, x, P, dt, kin, par, e, out + 1, out + 1 + RN, out + 1 + RN + RNN);
            memcpy(keep, in, sizeof(in));
            out[0] = ok ? 1.0 : 2.0;
        }
        out[1 + R_DP] = x[R_DP];                         /* the prediction of the angle offset and of the height: */
        out[1 + R_LH] = x[R_LH];                         /* themselves */
    } else
        out[0] = r_rts_gain(model, x, P, dt, kin, par, e, out + 1, out + 1 + RN, out + 1 + RN + RNN) ? 1.0 : 2.0;
    if (out[0] != 1.0)
        return;
    const double *xp = out + 1, *Pp = xp + RN, *G = Pp + RNN;
    double dx[RN], D[RNN], GD[RNN];
    for (int i = 0; i < RN; i++)
        dx[i] = xs[i] - xp[i];
    for (int i = 0; i < RNN; i++)
        D[i] = Ps[i] - Pp[i];
    for (int i = 0; i < RN; i++) {
        double acc = 0.0;
        for (int k = 0; k < RN; k++)
            acc += G[i * RN + k] * dx[k];
        x[i] += acc;
    }
    for (int i = 0; i < RN; i++)
        for (int j = 0; j < RN; j++) {
            double acc = 0.0;
            for (int k = 0; k < RN; k++)
                acc += G[i * RN + k] * D[k * RN + j];
            GD[i * RN + j] = acc;
        }
    for (int i = 0; i < RN; i++)
        for (int j = i; j < RN; j++) {
            double acc = 0.0;
            for (int k = 0; k < RN; k++)
                acc += GD[i * RN + k] * G[j * RN + k];
            double v = 0.5 * (P[i * RN + j] + P[j * RN + i]) + acc;
            P[i * RN + j] = P[j * RN + i] = v;
        }
    for (int i = 0; i < RN; i++)                         /* (a variance does not go below nothing by rounding) */
        if (P[i * RN + i] < 0.0)
            P[i * RN + i] = 0.0;
}

/* The primary's story of object t smoothed backwards over at most max_steps nodes and max_s seconds (the core of
 * Hyp.after and Hyp.history). 0: the story has one node (nothing given). 1: hist (NULL: not wanted; room for
 * max_steps + 1 rows of 6): (t, x, y, vx, vy, p_stand) at every node, the newest first, *nh_out rows; lag_out[5],
 * fl_out[5], *pl_out: the smoothed velocity, the free estimate's and the probability of "it stands" at the lag. */
static int smooth_core(Hyp *H, const double *A, HTrack *t, const RPar *par, int max_steps, double max_s, double lag_s,
                       double *hist, int *nh_out, double *lag_out, double *fl_out, double *pl_out)
{
    int32_t c = t->primary;
    if (c < 0 || H->parent[c] < 0)
        return 0;
    /* the kept gains: for the rows there are now (more rows, or other constants of the model: started anew; no
     * memory for them: the pass computes every gain) */
    if (H->sc == NULL || H->sc_cap != H->cap || memcmp(H->sc_par, par, sizeof(H->sc_par)) != 0) {
        PyMem_Free(H->sc);
        H->sc = (double *)PyMem_Calloc((size_t)H->cap * 4 * RSC_W, sizeof(double));
        H->sc_cap = H->sc ? H->cap : 0;
        memcpy(H->sc_par, par, sizeof(H->sc_par));
    }
    double *sc = H->sc;
    {
        const double *row = A + (Py_ssize_t)c * RROW;
        double xs[3 * RN], Ps[3 * RNN], mu[3], xf[RN], Pf[RNN], t0 = row[RO_M + M_T];
        memcpy(xs, row + RO_XS, sizeof(xs));
        memcpy(Ps, row + RO_PS, sizeof(Ps));
        memcpy(mu, row + RO_MU, sizeof(mu));
        memcpy(xf, row + RO_XF, sizeof(xf));
        memcpy(Pf, row + RO_PF, sizeof(Pf));
        int nh = 0;
        double lag[5] = {row[RO_X + R_VX], row[RO_X + R_VY], sqrt(pymax(row[RO_P + R_VX * RN + R_VX], 0.0)),
                         sqrt(pymax(row[RO_P + R_VY * RN + R_VY], 0.0)), 0.0};
        double fl[5] = {xf[R_VX], xf[R_VY], Pf[R_VX * RN + R_VX], Pf[R_VX * RN + R_VY], Pf[R_VY * RN + R_VY]}, pl = mu[R_STAND];
        int have_lag = 0;
        if (hist) {
            double *o = hist + 6 * nh;
            o[0] = t0;
            o[1] = row[RO_X + R_PX];
            o[2] = row[RO_X + R_PY];
            o[3] = row[RO_X + R_VX];
            o[4] = row[RO_X + R_VY];
            o[5] = mu[R_STAND];
        }
        nh++;
        for (int step = 0; step < max_steps; step++) {
            int32_t p = H->parent[c];
            if (p < 0)
                break;
            const double *rc = A + (Py_ssize_t)c * RROW, *rp = A + (Py_ssize_t)p * RROW;
            if (t0 - rp[RO_M + M_T] > max_s)
                break;
            const double *T = rc + RO_ST, *e = rc + RO_ST + ST_E, *kin = t->kin;
            double dt = rc[RO_ST + ST_DT];
            const double *mp = rp + RO_MU;
            /* the hypotheses mixed backwards: what the node's hypothesis j became at the child */
            double cc[3], d[3], xn[3 * RN], Pn[3 * RNN], mn[3];
            for (int i = 0; i < 3; i++)
                cc[i] = T[i] * mp[0] + T[3 + i] * mp[1] + T[6 + i] * mp[2];
            for (int j = 0; j < 3; j++) {
                double b[3], xb[RN], Pb[RNN];
                d[j] = 0.0;
                for (int i = 0; i < 3; i++) {
                    b[i] = cc[i] > 1e-12 ? T[3 * j + i] * mu[i] / cc[i] : 0.0;
                    d[j] += b[i];
                }
                memcpy(xn + j * RN, rp + RO_XS + j * RN, RN * sizeof(double));
                memcpy(Pn + j * RNN, rp + RO_PS + j * RNN, RNN * sizeof(double));
                if (!(d[j] > 1e-12))
                    continue;                            /* (nothing of it went on: it stays as it was filtered) */
                for (int k = 0; k < RN; k++)
                    xb[k] = (b[0] * xs[k] + b[1] * xs[RN + k] + b[2] * xs[2 * RN + k]) / d[j];
                for (int r = 0; r < RN; r++)
                    for (int q = r; q < RN; q++) {
                        double acc = 0.0;
                        for (int i = 0; i < 3; i++)
                            acc += b[i] * (Ps[i * RNN + r * RN + q] + (xs[i * RN + r] - xb[r]) * (xs[i * RN + q] - xb[q]));
                        Pb[r * RN + q] = Pb[q * RN + r] = acc / d[j];
                    }
                r_rts(j, xn + j * RN, Pn + j * RNN, xb, Pb, dt, kin, par, e, sc ? sc + (4 * (size_t)c + j) * RSC_W : NULL);
                if (j == R_STAND)
                    for (int k = R_VX; k <= R_AY; k++)
                        xn[j * RN + k] = 0.0;
            }
            double ms = 0.0;
            for (int j = 0; j < 3; j++) {
                mn[j] = mp[j] * d[j];
                ms += mn[j];
            }
            if (ms > 0.0)
                for (int j = 0; j < 3; j++)
                    mn[j] /= ms;
            else
                memcpy(mn, mp, sizeof(mn));
            /* the free estimate: one steady-velocity filter, a plain pass */
            double xfn[RN], Pfn[RNN];
            memcpy(xfn, rp + RO_XF, sizeof(xfn));
            memcpy(Pfn, rp + RO_PF, sizeof(Pfn));
            r_rts(R_CV, xfn, Pfn, xf, Pf, dt, kin, par, e, sc ? sc + (4 * (size_t)c + 3) * RSC_W : NULL);
            memcpy(xs, xn, sizeof(xs));
            memcpy(Ps, Pn, sizeof(Ps));
            memcpy(mu, mn, sizeof(mu));
            memcpy(xf, xfn, sizeof(xf));
            memcpy(Pf, Pfn, sizeof(Pf));
            double x[RN], P[RNN], tp = rp[RO_M + M_T];
            r_combine(xs, Ps, mu, x, P);
            if (hist) {
                double *o = hist + 6 * nh;
                o[0] = tp;
                o[1] = x[R_PX];
                o[2] = x[R_PY];
                o[3] = x[R_VX];
                o[4] = x[R_VY];
                o[5] = mu[R_STAND];
            }
            nh++;
            if (!have_lag) {
                lag[0] = x[R_VX];
                lag[1] = x[R_VY];
                lag[2] = sqrt(pymax(P[R_VX * RN + R_VX], 0.0));
                lag[3] = sqrt(pymax(P[R_VY * RN + R_VY], 0.0));
                lag[4] = t0 - tp;
                fl[0] = xf[R_VX];
                fl[1] = xf[R_VY];
                fl[2] = Pf[R_VX * RN + R_VX];
                fl[3] = Pf[R_VX * RN + R_VY];
                fl[4] = Pf[R_VY * RN + R_VY];
                pl = mu[R_STAND];
                have_lag = t0 - tp >= lag_s;
            }
            c = p;
            if (hist == NULL && have_lag)
                break;                                   /* the lag is all that was asked for: nothing further back changes it */
        }
        memcpy(lag_out, lag, sizeof(lag));
        memcpy(fl_out, fl, sizeof(fl));
        *pl_out = pl;
        *nh_out = nh;
    }
    return 1;
}

/* ====================================================================================================
 * One scan's whole cycle (radar/tracker.py `Tracker.process`): the objects that take part and what the scan needs
 * to know of each, the detections as rows, the scan, what it did to every object, and after the tracker's own
 * bookkeeping of the sightings (class vote, score, sensors: Python) the checks of earlier predictions, the
 * trajectories, the predictions written down and the smoothed past.
 *
 * Beside the state array the tracker holds a table of its objects (one row per handle, tracker.py `OT_*`) and
 * their trajectories (one row per handle: the count, then (t, x, y, vx, vy) per point, the oldest first).
 * ==================================================================================================== */
#define ROT_MAXCHK 12             /* predictions an object has written down and not yet checked, at most (8 + horizons) */
enum { ROT_SLOT, ROT_PE, ROT_PP, ROT_NLEAF, ROT_INMAP, ROT_COVER, ROT_HIDDEN, ROT_DP0, ROT_ID, ROT_STATE, ROT_KIN,
       ROT_LABEL = ROT_KIN + 4, ROT_RANGE, ROT_CONS, ROT_PERR, ROT_NCHECKS, ROT_UNPRED, ROT_HIST_T, ROT_HASVS, ROT_VS,
       ROT_FS = ROT_VS + 5, ROT_PSLAG = ROT_FS + 5, ROT_NCHK, ROT_CHK, ROTW = ROT_CHK + 7 * ROT_MAXCHK };

/* a writable float64 array of two dimensions; width > 0: of that many columns */
static int tab_get(PyObject *o, View *v, Py_ssize_t width, Py_ssize_t *rows, Py_ssize_t *cols, const char *what)
{
    if (view_get(o, v, 1, 'd', 2, what) < 0)
        return -1;
    if (width > 0 && v->b.shape[1] != width) {
        view_drop(v);
        PyErr_Format(PyExc_TypeError, "%s: rows of %zd numbers are needed", what, width);
        return -1;
    }
    *rows = v->b.shape[0];
    if (cols)
        *cols = v->b.shape[1];
    return 0;
}

/* the keys of a detection (tracker.py `process`: dets), interned once */
enum { RDK_KIND, RDK_ATTRS, RDK_CLASS, RDK_CONF, RDK_PROB, RDK_SDT, RDK_W, RDK_Z, RDK_R, RDK_ASPECT, RDK_CELL, RDK_WOK,
       RDK_X, RDK_EMB, RDK_N };
static PyObject *dkey[RDK_N + 1];
static const char *const DKEYS[RDK_N + 1] = {"kind", "attrs", "class", "class_conf", "prob", "sdt", "w", "z", "R",
                                            "aspect", "cell", "wok", "x", "emb", NULL};

static int dkeys_init(void)
{
    for (int k = 0; DKEYS[k]; k++)
        if (dkey[k] == NULL && (dkey[k] = PyUnicode_InternFromString(DKEYS[k])) == NULL)
            return -1;
    return 0;
}

/* d.get(key) (borrowed; NULL without an error: not there); *err set on an error */
static PyObject *dget(PyObject *d, int key, int *err)
{
    PyObject *v = PyDict_GetItemWithError(d, dkey[key]);
    if (v == NULL && PyErr_Occurred())
        *err = 1;
    return v;
}

/* at most `most` numbers of a float64 array or of a sequence into out; how many, -1: an error is set */
static int det_nums(PyObject *o, double *out, int most, const char *what)
{
    PyObject *l = NULL;
    int is_list = PyList_Check(o);
    if (!(is_list || PyTuple_Check(o))) {
        Py_buffer b;
        if (PyObject_GetBuffer(o, &b, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
            const char *f = b.format ? b.format : "B";
            if (*f == '@' || *f == '=' || *f == '<')
                f++;
            if (f[0] == 'd' && f[1] == 0 && b.itemsize == 8 && b.ndim == 1) {
                Py_ssize_t n = b.shape[0];
                if (n > most) {
                    PyBuffer_Release(&b);
                    PyErr_Format(PyExc_ValueError, "%s: more than %d numbers", what, most);
                    return -1;
                }
                memcpy(out, b.buf, (size_t)n * sizeof(double));
                PyBuffer_Release(&b);
                return (int)n;
            }
            PyBuffer_Release(&b);
        } else
            PyErr_Clear();
        l = PySequence_List(o);
        if (l == NULL)
            return -1;
        o = l;
        is_list = 1;
    }
    Py_ssize_t n = is_list ? PyList_Size(o) : PyTuple_Size(o);
    if (n > most) {
        Py_XDECREF(l);
        PyErr_Format(PyExc_ValueError, "%s: more than %d numbers", what, most);
        return -1;
    }
    for (Py_ssize_t k = 0; k < n; k++) {
        double v = PyFloat_AsDouble(seq_item(o, is_list, k));
        if (v == -1.0 && PyErr_Occurred()) {
            Py_XDECREF(l);
            return -1;
        }
        out[k] = v;
    }
    Py_XDECREF(l);
    return (int)n;
}

/* float(v or 0.0) of d.get(key); -1: an error is set */
static int det_float(PyObject *d, int key, double *out)
{
    int err = 0;
    PyObject *v = dget(d, key, &err);
    *out = 0.0;
    if (err)
        return -1;
    if (v == NULL)
        return 0;
    int tr = PyObject_IsTrue(v);
    if (tr < 0)
        return -1;
    if (tr) {
        *out = PyFloat_AsDouble(v);
        if (*out == -1.0 && PyErr_Occurred())
            return -1;
    }
    return 0;
}

#define RDET_BINS 64
/* The scan's detections as rows (RDET) and what each one's kind can do for an object started by it (RKB): a
 * detection is a dict {"z", "R", "kind", "sdt", "w", "wok", "cell", "attrs": {"class", "class_conf" | "prob",
 * "aspect"}}. dcfg: (kinds: {kind: its code}, kind_fn(kind) -> code, labels: {class as given: (vehicle 1 / person -1
 * / not known 0, the RKB numbers)}, label_fn(class) -> that tuple, odds_fn(bin) -> what a score of that bin adds
 * to the log odds of a real thing, the number of score bins). The dicts are the tracker's memory of what the
 * functions said; a function is asked where its dict does not know. -1: an error is set. */
static int det_rows(PyObject *dets, Py_ssize_t nd, PyObject *dcfg, double *D, double *KB, const RXc *xc, double *E)
{
    PyObject *okinds, *okfn, *olabs, *olfn, *oofn, *groups = NULL;
    int bins, rc = -1;
    double odds[RDET_BINS];
    char have[RDET_BINS];
    if (!PyTuple_Check(dcfg) || !PyArg_ParseTuple(dcfg, "OOOOOi", &okinds, &okfn, &olabs, &olfn, &oofn, &bins))
        return -1;
    if (!PyDict_Check(okinds) || !PyDict_Check(olabs) || bins < 1 || bins > RDET_BINS) {
        PyErr_SetString(PyExc_TypeError, "radar: the detections' tables");
        return -1;
    }
    if (dkeys_init() < 0 || (groups = PyDict_New()) == NULL)
        return -1;
    memset(have, 0, sizeof(have));
    for (Py_ssize_t j = 0; j < nd; j++) {
        PyObject *d = PyList_GetItem(dets, j), *at, *v, *k, *sdt;
        double *row = D + (size_t)j * RDET, *kb = KB + (size_t)j * RKB;
        int err = 0;
        if (d == NULL)
            goto done;
        if (!PyDict_Check(d)) {
            PyErr_SetString(PyExc_TypeError, "radar: a detection must be a dict");
            goto done;
        }
        memset(row, 0, RDET * sizeof(double));
        /* its kind, and its group: the detections of one kind and scale-rate baseline are predicted alike */
        k = dget(d, RDK_KIND, &err);
        sdt = dget(d, RDK_SDT, &err);
        if (err)
            goto done;
        if (k == NULL)
            k = Py_None;
        if (sdt == NULL)
            sdt = Py_None;
        v = PyDict_GetItemWithError(okinds, k);
        if (v == NULL) {
            if (PyErr_Occurred())
                goto done;
            PyObject *r = PyObject_CallFunctionObjArgs(okfn, k, NULL);
            if (r == NULL)
                goto done;
            row[D_KIND] = PyFloat_AsDouble(r);
            Py_DECREF(r);
        } else
            row[D_KIND] = PyFloat_AsDouble(v);
        if (row[D_KIND] == -1.0 && PyErr_Occurred())
            goto done;
        if (det_float(d, RDK_SDT, &row[D_SDT]) < 0)
            goto done;
        {
            PyObject *key = PyTuple_Pack(2, k, sdt);
            if (key == NULL)
                goto done;
            v = PyDict_GetItemWithError(groups, key);
            if (v == NULL) {
                PyObject *num = PyErr_Occurred() ? NULL : PyLong_FromSsize_t(PyDict_Size(groups));
                if (num == NULL || PyDict_SetItem(groups, key, num) < 0) {
                    Py_XDECREF(num);
                    Py_DECREF(key);
                    goto done;
                }
                row[D_GROUP] = PyLong_AsDouble(num);
                Py_DECREF(num);
            } else
                row[D_GROUP] = PyLong_AsDouble(v);
            Py_DECREF(key);
        }
        /* its attributes: the class (what its kind can do), the detector's score, the box's aspect */
        at = dget(d, RDK_ATTRS, &err);
        if (err)
            goto done;
        if (at != NULL) {
            int tr = PyObject_IsTrue(at);
            if (tr < 0)
                goto done;
            if (!tr)
                at = NULL;
            else if (!PyDict_Check(at)) {
                PyErr_SetString(PyExc_TypeError, "radar: a detection's attrs must be a dict");
                goto done;
            }
        }
        {
            PyObject *cls = at ? dget(at, RDK_CLASS, &err) : NULL, *lr = NULL, *got;
            if (err)
                goto done;
            if (cls == NULL)
                cls = Py_None;
            got = PyDict_GetItemWithError(olabs, cls);
            if (got == NULL) {
                PyErr_Clear();                           /* (a class that cannot be a key: asked every time) */
                lr = got = PyObject_CallFunctionObjArgs(olfn, cls, NULL);
                if (got == NULL)
                    goto done;
            }
            if (!PyTuple_Check(got) || PyTuple_Size(got) != RKB + 1) {
                Py_XDECREF(lr);
                PyErr_SetString(PyExc_TypeError, "radar: a label's numbers");
                goto done;
            }
            row[D_VEH] = PyFloat_AsDouble(PyTuple_GetItem(got, 0));
            for (int q = 0; q < RKB; q++)
                kb[q] = PyFloat_AsDouble(PyTuple_GetItem(got, q + 1));
            Py_XDECREF(lr);
            if (PyErr_Occurred())
                goto done;
        }
        if (det_float(d, RDK_W, &row[D_W]) < 0)
            goto done;
        {
            PyObject *z = dget(d, RDK_Z, &err), *R = dget(d, RDK_R, &err);
            if (err)
                goto done;
            if (z == NULL || R == NULL) {
                PyErr_SetString(PyExc_KeyError, z == NULL ? "z" : "R");
                goto done;
            }
            if (det_nums(z, row + D_Z, 4, "z") < 0 || det_nums(R, row + D_R, 4, "R") < 0)
                goto done;
        }
        if (at != NULL) {
            /* the score: a number in 0..1 (not a truth value), `class_conf` where the key is there, else `prob` */
            v = dget(at, RDK_CONF, &err);
            if (v == NULL && !err)
                v = dget(at, RDK_PROB, &err);
            if (err)
                goto done;
            if (v != NULL && !PyBool_Check(v) && (PyFloat_Check(v) || PyLong_Check(v))) {
                double sc = PyFloat_AsDouble(v);
                if (sc == -1.0 && PyErr_Occurred())
                    PyErr_Clear();                       /* (an int beyond a float: not in 0..1) */
                else if (0.0 <= sc && sc <= 1.0) {
                    int b = (int)(sc * bins);
                    if (b > bins - 1)
                        b = bins - 1;
                    if (!have[b]) {
                        PyObject *r = PyObject_CallFunction(oofn, "i", b);
                        if (r == NULL)
                            goto done;
                        odds[b] = PyFloat_AsDouble(r);
                        Py_DECREF(r);
                        if (odds[b] == -1.0 && PyErr_Occurred())
                            goto done;
                        have[b] = 1;
                    }
                    row[D_SCORE] = odds[b];
                }
            }
            if (det_float(at, RDK_ASPECT, &row[D_ASPECT]) < 0)
                goto done;
        }
        /* a cell of several returns: their number and scatter; one detection: (1, 0, 0, 0) */
        row[D_NRET] = 1.0;
        v = dget(d, RDK_CELL, &err);
        if (err)
            goto done;
        if (v != NULL) {
            int tr = PyObject_IsTrue(v);
            if (tr < 0)
                goto done;
            if (tr) {
                double c4[4];
                int nc = det_nums(v, c4, 4, "cell");
                if (nc < 0)
                    goto done;
                if (nc != 4) {
                    PyErr_SetString(PyExc_ValueError, "radar: a cell is four numbers");
                    goto done;
                }
                memcpy(row + D_NRET, c4, sizeof(c4));
            }
        }
        v = dget(d, RDK_WOK, &err);
        if (err)
            goto done;
        if (v != NULL) {
            int tr = PyObject_IsTrue(v);
            if (tr < 0)
                goto done;
            row[D_WOK] = tr ? 1.0 : 0.0;
        }
        if (xc != NULL) {
            /* the detector's further inputs: "x": (hidden edges, covered share, low list, lamp colour,
             * ln(separation / f), its variance, the scale rate's source, the box's top and bottom as angles below
             * the horizon, 1 if the bottom is cut by the view or the frame); "emb": the appearance vector's int8
             * values, of the scan's kind (27 for yuv3x3: value / 32; 59 for reid32+yuv3x3: 32 learnt / 127, then
             * the 27 at half weight), here in the units of the kind's distance */
            row[D_TOP] = row[D_BOT] = NAN;
            v = dget(d, RDK_X, &err);
            if (err)
                goto done;
            if (v != NULL) {
                double xv[10];
                int nx = det_nums(v, xv, 10, "x");
                if (nx < 0)
                    goto done;
                if (nx == 10) {
                    row[D_HID] = xv[0] + (xv[9] != 0.0 ? 16.0 : 0.0);
                    row[D_OCC] = xv[1];
                    row[D_LOW] = xv[2];
                    row[D_LAMPC] = xv[3];
                    row[D_LSEP] = xv[4];
                    row[D_LSEPR] = xv[5];
                    row[D_SSRC] = xv[6];
                    row[D_TOP] = xv[7];
                    row[D_BOT] = xv[8];
                }
            }
            v = dget(d, RDK_EMB, &err);
            if (err)
                goto done;
            int kind = (int)xc->c[XC_APP_KIND], need = kind == 1 ? 27 : kind == 2 ? 59 : 0;
            if (v != NULL && need && PyBytes_Check(v) && PyBytes_Size(v) == need) {
                const signed char *b = (const signed char *)PyBytes_AsString(v);
                double *e = E + (size_t)j * RXEMB;
                if (b == NULL)
                    goto done;
                for (int q = 0; q < need; q++)
                    e[q] = kind == 1 ? b[q] / 32.0 : q < 32 ? b[q] / 127.0 : 0.5 * b[q] / 32.0;
                row[D_HASEMB] = 1.0;
            }
        }
    }
    rc = 0;
done:
    Py_DECREF(groups);
    return rc;
}

/* Hyp.cycle(S, table, hist, objects, t, noise, pose, sensor, src, dets, dcfg, mem | None, nscan, acfg, par, ucfg,
 *           bcfg, bias | None, ccfg) -> (late, sightings, births, ended, bias | None)   or   an int
 * One scan (Tracker.process): `objects` are the handles of all the tracker's objects, in its order. An object whose
 * time is more than ccfg[0] (oosm_max_s) ahead of the scan's does not take part (`late` counts them). For the others
 * the scan's info is made here from their rows of the table and of the state array: dt, what the car's own movement
 * over it is not sure about (noise(the object's time, t) -> (along, heading), asked once per time), what the kind
 * can do, vehicle / person, id, settled, and whether its moving hypotheses hold its height and angle offset against
 * a camera's boxes (ccfg[1]: consider_moving; from ucfg[0] sightings on; not while a sensor that measures range has
 * seen it within ccfg[2] seconds; ccfg[3]: the standing hypothesis too). dets: the detections as the tracker gets
 * them (det_rows). pose: the car's (X, Y, psi, vx, vy, w) at t; src: the sensor's number; mem: (nm, 5) the entries
 * of the map of unexplained detections; nscan: the scans the map's memory holds; bias: the sensor's angle error
 * [slow value, variance, fast value, variance, time] where it has one.
 * Written into the table: of every object that took part and lives its primary row, the probabilities that it
 * exists and that the primary's story is the true one, its leaves, whether it is in the best story, when a scan
 * last covered its place and when it was last hidden; a new object's row and an empty trajectory.
 * Returns the count of late objects, the sightings [(the object's place in `objects`, detection)], the births
 * [(handle, detection)], the handles of objects that ended, the sensor's bias after the scan. An int > 0: the
 * state array needs that many free rows first; an int < 0: the table and the trajectories need -int rows first
 * (nothing was changed in either case). */
static PyObject *hyp_cycle(PyObject *self, PyObject *args)
{
    Hyp *H = (Hyp *)self;
    PyObject *oS, *oT, *oHi, *otr, *onoise, *opose, *osen, *odets, *odcfg, *omem, *oacfg, *opar, *oucfg, *obcfg, *obias,
        *occfg, *oxcfg = Py_None;
    double t, nscan;
    int src;
    if (!PyArg_ParseTuple(args, "OOOOdOOOiOOOdOOOOOO|O", &oS, &oT, &oHi, &otr, &t, &onoise, &opose, &osen, &src, &odets,
                          &odcfg, &omem, &nscan, &oacfg, &opar, &oucfg, &obcfg, &obias, &occfg, &oxcfg))
        return NULL;
    RXc xcs;
    double *emb = NULL;
    memset(&xcs, 0, sizeof(xcs));
    PyObject *res = NULL, *hits = NULL, *births = NULL, *bias_o = NULL;
    View vS, vT, vH;
    vS.held = vT.held = vH.held = 0;
    Scan s;
    memset(&s, 0, sizeof(s));
    s.H = H;
    s.t = t;
    s.nscan = nscan;
    s.src = src;
    Py_ssize_t cap, na = 0, *th = NULL, *part = NULL, nm = 0, need = 0, trows, hrows, hcols, n = 0, late = 0;
    double bias[5] = {0.0, 0.0, 0.0, 0.0, -1e18}, ccfg[4], *in = NULL;
    int has_bias = obias != Py_None;
    if (store_get(oS, &vS, &cap) < 0 || tab_get(oT, &vT, ROTW, &trows, NULL, "table") < 0 ||
        tab_get(oHi, &vH, 0, &hrows, &hcols, "hist") < 0 || (th = idx_get(otr, 1 << 30, &na)) == NULL ||
        sens_get(osen, &s.sen) < 0 || rnums_get(opose, 6, s.pose, "pose") < 0 || rnums_get(oacfg, RSCFG, s.acfg, "acfg") < 0 ||
        rpar_get(opar, &s.par) < 0 || rnums_get(oucfg, 2, s.ucfg, "ucfg") < 0 || rnums_get(obcfg, RBCFG, s.bcfg, "bcfg") < 0 ||
        (has_bias && rnums_get(obias, 5, bias, "bias") < 0) || rnums_get(occfg, 4, ccfg, "ccfg") < 0)
        goto done;
    if (oxcfg != Py_None && s.sen.type == SEN_CAMERA) {  /* the detector's further inputs (a camera's boxes) */
        if (rnums_get(oxcfg, RXCFG, xcs.c, "xcfg") < 0)
            goto done;
        s.xc = &xcs;
        s.sen.xc = &xcs;
        s.sen.grade_k = xcs.c[XC_GRADE];
        s.sen.lcol = xcs.c[XC_LCOL] != 0.0;
    }
    if (!PyList_Check(odets)) {
        PyErr_SetString(PyExc_TypeError, "Hyp.cycle: the detections must be a list");
        goto done;
    }
    if (hcols < 6 || (hcols - 1) % 5 != 0) {
        PyErr_SetString(PyExc_TypeError, "Hyp.cycle: the trajectories: a count and points of five numbers");
        goto done;
    }
    Py_ssize_t nd = PyList_Size(odets);
    s.A = (double *)vS.b.buf;
    s.nd = nd;
    if (omem != Py_None && (nm = batch_len(omem)) < 0)
        goto done;
    if (trows < H->ntr + nd || hrows < H->ntr + nd) {    /* every detection may start an object */
        res = PyLong_FromSsize_t(-(Py_ssize_t)(H->ntr + nd));
        goto done;
    }
    double *tab = (double *)vT.b.buf, *hist = (double *)vH.b.buf;
    {
        size_t n_ti = (size_t)na * HTI, n_d = (size_t)nd * RDET, n_kb = (size_t)nd * RKB;
        in = (double *)PyMem_Malloc((n_ti + n_d + n_kb + (size_t)nm * 5 + 3 * (size_t)na + 1) * sizeof(double));
        part = (Py_ssize_t *)PyMem_Malloc((size_t)(2 * na + 1) * sizeof(Py_ssize_t));
        if (in == NULL || part == NULL) {
            PyErr_NoMemory();
            goto done;
        }
        double *ti = in, *D = ti + n_ti, *KB = D + n_d, *mem = KB + n_kb, *nt = mem + (size_t)nm * 5, *nv = nt + na;
        Py_ssize_t *pos = part + na;
        int nn = 0;
        /* the objects that take part, and what the scan needs to know of each */
        for (Py_ssize_t i = 0; i < na; i++) {
            Py_ssize_t h = th[i];
            if (h >= trows) {
                PyErr_SetString(PyExc_IndexError, "Hyp.cycle: an object outside the table");
                goto done;
            }
            const double *T = tab + h * ROTW;
            Py_ssize_t slot = (Py_ssize_t)T[ROT_SLOT];
            if (!(T[ROT_SLOT] >= 0.0) || slot >= cap) {
                PyErr_SetString(PyExc_IndexError, "radar: a row outside the state array");
                goto done;
            }
            const double *M = s.A + slot * RROW + RO_M;
            double ot = M[M_T];
            if (t < ot - ccfg[0]) {
                late++;
                continue;
            }
            int q = 0;
            while (q < nn && nt[q] != ot)
                q++;
            if (q == nn) {
                PyObject *r = PyObject_CallFunction(onoise, "dd", ot, t);
                if (r == NULL)
                    goto done;
                int ok = PyTuple_Check(r) && PyArg_ParseTuple(r, "dd", &nv[2 * nn], &nv[2 * nn + 1]);
                Py_DECREF(r);
                if (!ok) {
                    if (!PyErr_Occurred())
                        PyErr_SetString(PyExc_TypeError, "Hyp.cycle: noise() must give two numbers");
                    goto done;
                }
                nt[nn++] = ot;
            }
            double *o = ti + (size_t)n * HTI;
            o[0] = t - ot;
            o[1] = nv[2 * q];
            o[2] = nv[2 * q + 1];
            memcpy(o + 3, T + ROT_KIN, 4 * sizeof(double));
            o[7] = T[ROT_LABEL];
            o[8] = T[ROT_ID];
            o[9] = (T[ROT_STATE] == 1.0 || T[ROT_STATE] == 2.0) ? 1.0 : 0.0;
            if (ccfg[1] == 0.0 || (double)(long long)M[M_HITS] < s.ucfg[0] || t - T[ROT_RANGE] <= ccfg[2])
                o[10] = 0.0;
            else
                o[10] = ccfg[3] != 0.0 ? 2.0 : 1.0;
            part[n] = h;
            pos[n] = i;
            n++;
        }
        s.n = (int)n;
        if (s.xc != NULL) {
            emb = (double *)PyMem_Calloc((size_t)(nd ? nd : 1) * RXEMB, sizeof(double));
            if (emb == NULL) {
                PyErr_NoMemory();
                goto done;
            }
            xcs.E = emb;
        }
        if (det_rows(odets, nd, odcfg, D, KB, s.xc, emb) < 0 || (nm && rows_get(omem, nm, 5, mem, "mem") < 0))
            goto done;
        if (s.xc != NULL) {
            /* how dark the scene is, from the boxes' own images (the vectors' nine luma cells: (Y - 110) / 50),
             * followed over the scans; XC_APP_DARK >= 0: told */
            int kind = (int)xcs.c[XC_APP_KIND], off = kind == 2 ? 32 : 0, ne = 0;
            double ys = 0.0;
            for (Py_ssize_t j = 0; kind && j < nd; j++)
                if (D[(size_t)j * RDET + D_HASEMB] != 0.0) {
                    for (int q = 0; q < 9; q++)
                        ys += emb[(size_t)j * RXEMB + off + q] * (kind == 2 ? 2.0 : 1.0);
                    ne++;
                }
            if (ne) {
                ys /= 9.0 * ne;
                H->dark_y = H->dark_has ? H->dark_y + 0.2 * (ys - H->dark_y) : ys;
                H->dark_has = 1;
            }
            if (xcs.c[XC_APP_DARK] >= 0.0)
                xcs.dark = pymin(xcs.c[XC_APP_DARK], 1.0);
            else if (H->dark_has && xcs.c[XC_APP_YDAY] > xcs.c[XC_APP_YNIGHT])
                xcs.dark = pymin(1.0, pymax(0.0, (xcs.c[XC_APP_YDAY] - H->dark_y) / (xcs.c[XC_APP_YDAY] - xcs.c[XC_APP_YNIGHT])));
            H->dark = xcs.dark;
        }
        int r = scan_core(H, &s, cap, part, n, ti, D, KB, nm, mem, has_bias, bias, &need);
        if (r < 0)
            goto done;
        if (r == 1) {
            res = PyLong_FromSsize_t(need);
            goto done;
        }
        /* what the scan did to every object */
        if ((hits = PyList_New(0)) == NULL || (births = PyList_New(0)) == NULL)
            goto done;
        for (Py_ssize_t k = 0; k < n; k++) {
            HTrack *tr = &H->tr[part[k]];
            if (!tr->alive)
                continue;
            double *T = tab + part[k] * ROTW;
            const double *M = s.A + (Py_ssize_t)tr->primary * RROW + RO_M;
            T[ROT_SLOT] = (double)tr->primary;
            T[ROT_PE] = tr->pe;
            T[ROT_PP] = tr->pp;
            T[ROT_NLEAF] = (double)tr->nleaf;
            T[ROT_INMAP] = tr->in_map ? 1.0 : 0.0;
            if (M[M_COV] > 0.0)
                T[ROT_COVER] = t;
            if (M[M_OCC] == 1.0 || M[M_OCC] == 2.0)      /* hidden by an object or cut by the edge in this scan */
                T[ROT_HIDDEN] = t;
            int j = (int)M[M_DET];
            if (j >= 0) {
                T[ROT_COVER] = t;
                PyObject *o = Py_BuildValue("(ni)", pos[k], j);
                if (o == NULL || PyList_Append(hits, o) < 0) {
                    Py_XDECREF(o);
                    goto done;
                }
                Py_DECREF(o);
            }
        }
        for (Py_ssize_t j = 0; j < nd; j++) {
            int h = s.birth_h[j];
            if (h < 0 || !H->tr[h].alive || H->tr[h].born != H->level)
                continue;
            HTrack *tr = &H->tr[h];
            double *T = tab + (Py_ssize_t)h * ROTW;
            memset(T, 0, ROTW * sizeof(double));
            T[ROT_SLOT] = (double)tr->primary;
            T[ROT_PE] = tr->pe;
            T[ROT_PP] = tr->pp;
            T[ROT_NLEAF] = 1.0;
            T[ROT_INMAP] = tr->in_map ? 1.0 : 0.0;
            T[ROT_COVER] = t;
            T[ROT_HIDDEN] = -1e18;
            T[ROT_HIST_T] = -1e18;
            T[ROT_RANGE] = -1e18;
            T[ROT_DP0] = s.A[(Py_ssize_t)tr->primary * RROW + RO_P + R_DP * RN + R_DP];
            hist[(Py_ssize_t)h * hcols] = 0.0;
            PyObject *o = Py_BuildValue("(in)", h, j);
            if (o == NULL || PyList_Append(births, o) < 0) {
                Py_XDECREF(o);
                goto done;
            }
            Py_DECREF(o);
        }
        if (has_bias)
            bias_o = Py_BuildValue("(ddddd)", bias[0], bias[1], bias[2], bias[3], bias[4]);
        else {
            bias_o = Py_None;
            Py_INCREF(bias_o);
        }
        if (bias_o == NULL)
            goto done;
        res = Py_BuildValue("(nOOOO)", late, hits, births, s.dead, bias_o);
    }
done:
    Py_XDECREF(hits);
    Py_XDECREF(births);
    Py_XDECREF(bias_o);
    Py_XDECREF(s.dead);
    ar_free(&s.ar);
    PyMem_Free(th);
    PyMem_Free(part);
    PyMem_Free(in);
    PyMem_Free(emb);
    view_drop(&vS);
    view_drop(&vT);
    view_drop(&vH);
    return res;
}

/* Hyp.xrow(row) -> the row's memory for the detector's further inputs (RXW numbers: X_*), or None before any scan
 * asked for it */
static PyObject *hyp_xrow(PyObject *self, PyObject *arg)
{
    Hyp *H = (Hyp *)self;
    Py_ssize_t r = PyLong_AsSsize_t(arg);
    if (r == -1 && PyErr_Occurred())
        return NULL;
    if (H->X == NULL)
        Py_RETURN_NONE;
    if (r < 0 || r >= H->cap) {
        PyErr_SetString(PyExc_IndexError, "Hyp.xrow: a row outside the state array");
        return NULL;
    }
    PyObject *l = PyList_New(RXW);
    for (int k = 0; l && k < RXW; k++)
        PyList_SetItem(l, k, PyFloat_FromDouble(H->X[(size_t)r * RXW + k]));
    return l;
}

/* Hyp.xstate() -> (1 if the rows have a memory for the detector's further inputs, the boxes' mean luma level as
 * followed | None, how dark the last scan took the scene to be: 0 day .. 1 night) */
static PyObject *hyp_xstate(PyObject *self, PyObject *unused)
{
    Hyp *H = (Hyp *)self;
    if (H->dark_has)
        return Py_BuildValue("(idd)", H->X != NULL, H->dark_y, H->dark);
    return Py_BuildValue("(iOd)", H->X != NULL, Py_None, H->dark);
}

/* radar_app_lr(xcfg, dark, xa, xb, gap) -> the appearance likelihood ratio (x_app_lr) between two rows' remembered
 * vectors (Hyp.xrow), `gap` seconds apart, in a scene `dark` (0 day .. 1 night); 1.0 where either has none or
 * they are of different kinds */
static PyObject *n_radar_app_lr(PyObject *self, PyObject *args)
{
    PyObject *oc, *oa, *ob;
    double dark, gap, c[RXCFG], a[RXW], b[RXW];
    if (!PyArg_ParseTuple(args, "OdOOd", &oc, &dark, &oa, &ob, &gap))
        return NULL;
    if (rnums_get(oc, RXCFG, c, "xcfg") < 0 || rnums_get(oa, RXW, a, "xa") < 0 || rnums_get(ob, RXW, b, "xb") < 0)
        return NULL;
    if (!(a[X_EN] > 0.0) || !(b[X_EN] > 0.0) || a[X_EK] != b[X_EK])
        return PyFloat_FromDouble(1.0);
    return PyFloat_FromDouble(x_app_lr(c, pymin(1.0, pymax(0.0, dark)), x_emb_dist(a + X_EMB, b + X_EMB), gap));
}

/* a point at the end of a trajectory of at most hn points (the oldest goes where it is full) */
static void hist_push(double *Hh, Py_ssize_t hn, double t, double x, double y, double vx, double vy)
{
    Py_ssize_t n = (Py_ssize_t)Hh[0];
    if (n >= hn) {
        memmove(Hh + 1, Hh + 6, (size_t)(hn - 1) * 5 * sizeof(double));
        n = hn - 1;
    }
    double *p = Hh + 1 + 5 * n;
    p[0] = t;
    p[1] = x;
    p[2] = y;
    p[3] = vx;
    p[4] = vy;
    Hh[0] = (double)(n + 1);
}

enum { RAFT_WINDOW, RAFT_INCONS, RAFT_INFLATE, RAFT_HIST_S, RAFT_NHZ, RAFT_HZ, RAFT_SMOOTH = RAFT_HZ + 4, RAFT_STEPS, RAFT_SMOOTH_S, RAFT_SHORT_S,
       RAFT_LAG_S, RAFT_WHOLE_S, RAFT_N };

/* Hyp.after(S, table, hist, sighted, kept, born, t, par, cfg) -> None
 * What follows a scan's sightings once the tracker has done its own bookkeeping of them (Tracker.process).
 * sighted: the handles of the objects that took a detection. Each one's expectations whose time has passed are
 * compared with where it is now believed to have been then (how well its predictions hold: consistency, the
 * error; one that keeps being elsewhere has the uncertainty of all its stories opened), and its place and velocity
 * go to the end of its trajectory, which is kept cfg[RAFT_HIST_S] long.
 * kept: those of them that still live. Each writes down where it expects to be at the horizons ahead, and its
 * past is smoothed (smooth_core): the velocity a lag ago after every sighting, the whole trajectory as far as
 * the smoother reaches once per cfg[RAFT_WHOLE_S].
 * born: the objects this scan started: the first point of their trajectories.
 * cfg: check_window_s, inconsistent, inflate, history_s, the number of horizons (at most 4) and the horizons,
 * smooth (0: not), the smoother's steps, how far back the whole pass and the short pass reach, the lag, how often
 * the whole pass runs. */
static PyObject *hyp_after(PyObject *self, PyObject *args)
{
    Hyp *H = (Hyp *)self;
    PyObject *oS, *oT, *oHi, *ohit, *okept, *oborn, *opar, *ocfg;
    double t;
    if (!PyArg_ParseTuple(args, "OOOOOOdOO", &oS, &oT, &oHi, &ohit, &okept, &oborn, &t, &opar, &ocfg))
        return NULL;
    PyObject *res = NULL;
    View vS, vT, vH;
    vS.held = vT.held = vH.held = 0;
    Py_ssize_t cap, trows, hrows, hcols, nh = 0, nk = 0, nb = 0, *hit = NULL, *kept = NULL, *born = NULL;
    double cfg[RAFT_N];
    RPar par;
    if (store_get(oS, &vS, &cap) < 0 || tab_get(oT, &vT, ROTW, &trows, NULL, "table") < 0 ||
        tab_get(oHi, &vH, 0, &hrows, &hcols, "hist") < 0 || rpar_get(opar, &par) < 0 || rnums_get(ocfg, RAFT_N, cfg, "cfg") < 0)
        goto done;
    Py_ssize_t lim = trows < hrows ? trows : hrows;
    if (lim > H->ntr)
        lim = H->ntr;
    if ((hit = idx_get(ohit, lim, &nh)) == NULL || (kept = idx_get(okept, lim, &nk)) == NULL ||
        (born = idx_get(oborn, lim, &nb)) == NULL)
        goto done;
    int nhz = (int)cfg[RAFT_NHZ], steps = (int)cfg[RAFT_STEPS];
    if (cap < H->cap || hcols < 6 || (hcols - 1) % 5 != 0 || nhz < 0 || nhz > 4 || steps < 0) {
        PyErr_SetString(PyExc_ValueError, "Hyp.after: the arrays or the constants do not fit");
        goto done;
    }
    Py_ssize_t hn = (hcols - 1) / 5;
    double *A = (double *)vS.b.buf, *tab = (double *)vT.b.buf, *hist = (double *)vH.b.buf;
    for (Py_ssize_t i = 0; i < nh + nk + nb; i++) {
        Py_ssize_t h = i < nh ? hit[i] : i < nh + nk ? kept[i - nh] : born[i - nh - nk];
        double v = tab[h * ROTW + ROT_SLOT];
        if (!(v >= 0.0) || (Py_ssize_t)v >= cap) {
            PyErr_SetString(PyExc_IndexError, "radar: a row outside the state array");
            goto done;
        }
    }
    for (Py_ssize_t i = 0; i < nh; i++) {
        Py_ssize_t h = hit[i];
        double *T = tab + h * ROTW, *C = T + ROT_CHK, *Hh = hist + h * hcols;
        double *row = A + (Py_ssize_t)T[ROT_SLOT] * RROW;
        /* expectations whose time has passed against where the object is now believed to have been then */
        int nc = (int)T[ROT_NCHK], keep = 0;
        if (nc > ROT_MAXCHK)
            nc = ROT_MAXCHK;
        for (int k = 0; k < nc; k++) {
            const double *c = C + 7 * k;
            if (c[0] > t) {
                if (keep != k)
                    memmove(C + 7 * keep, c, 7 * sizeof(double));
                keep++;
                continue;
            }
            if (t - c[0] > cfg[RAFT_WINDOW])
                continue;                                /* no update near that time: nothing to check it with */
            double m2, e;
            if (r_check(row, t - c[0], c[1], c[2], c + 3, &m2, &e) < 0)
                continue;
            if (T[ROT_NCHECKS] == 0.0) {
                T[ROT_CONS] = m2;
                T[ROT_PERR] = e;
            } else {
                T[ROT_CONS] = T[ROT_CONS] + 0.25 * (m2 - T[ROT_CONS]);
                T[ROT_PERR] = T[ROT_PERR] + 0.25 * (e - T[ROT_PERR]);
            }
            T[ROT_NCHECKS] += 1.0;
        }
        if (keep > 8) {
            memmove(C, C + 7 * (keep - 8), 8 * 7 * sizeof(double));
            keep = 8;
        }
        T[ROT_NCHK] = (double)keep;
        T[ROT_UNPRED] = 0.0;
        if (T[ROT_NCHECKS] >= 4.0 && T[ROT_CONS] > cfg[RAFT_INCONS]) {
            T[ROT_UNPRED] = 1.0;                          /* it keeps being somewhere else than predicted: believe less */
            for (Py_ssize_t r = 0; r < H->cap; r++) {    /* every story of it */
                if (H->owner[r] != (int32_t)h)
                    continue;
                double *q = A + r * RROW;
                for (int k = 0; k < 3 * RNN; k++)
                    q[RO_PS + k] *= cfg[RAFT_INFLATE];
                for (int k = 0; k < RNN; k++)
                    q[RO_PF + k] *= cfg[RAFT_INFLATE];
                r_combine(q + RO_XS, q + RO_PS, q + RO_MU, q + RO_X, q + RO_P);
            }
            T[ROT_CONS] = 0.5 * (T[ROT_CONS] + 1.0);
        }
        /* its trajectory */
        hist_push(Hh, hn, t, row[RO_X + R_PX], row[RO_X + R_PY], row[RO_X + R_VX], row[RO_X + R_VY]);
        Py_ssize_t n = (Py_ssize_t)Hh[0], old = 0;
        while (old < n && Hh[1 + 5 * old] < t - cfg[RAFT_HIST_S])
            old++;
        if (old) {
            memmove(Hh + 1, Hh + 1 + 5 * old, (size_t)(n - old) * 5 * sizeof(double));
            Hh[0] = (double)(n - old);
        }
    }
    for (Py_ssize_t i = 0; i < nk; i++) {                /* written down now, checked when the time has come */
        double *T = tab + kept[i] * ROTW;
        const double *row = A + (Py_ssize_t)T[ROT_SLOT] * RROW;
        double ot = row[RO_M + M_T];
        for (int k = 0; k < nhz; k++) {
            int nc = (int)T[ROT_NCHK];
            if (nc >= ROT_MAXCHK)
                break;
            double *c = T + ROT_CHK + 7 * nc, tf = t + cfg[RAFT_HZ + k], dt = tf - ot;
            c[0] = tf;
            if (fabs(dt) < 1e-6) {
                c[1] = row[RO_X + R_PX];
                c[2] = row[RO_X + R_PY];
                c[3] = row[RO_P];
                c[4] = row[RO_P + 1];
                c[5] = row[RO_P + RN];
                c[6] = row[RO_P + RN + 1];
            } else
                r_peek_pos(row, dt, T + ROT_KIN, &par, c + 1);
            T[ROT_NCHK] = (double)(nc + 1);
        }
    }
    if (cfg[RAFT_SMOOTH] != 0.0)
        for (Py_ssize_t i = 0; i < nk; i++) {
            Py_ssize_t h = kept[i];
            HTrack *tr = &H->tr[h];
            if (!tr->alive) {
                PyErr_SetString(PyExc_ValueError, "Hyp: an object that does not exist");
                goto done;
            }
            double *T = tab + h * ROTW;
            double ot = A[(Py_ssize_t)T[ROT_SLOT] * RROW + RO_M + M_T], lag[5], fl[5], pl;
            /* back to the lag: the first node at least the lag back within cfg[RAFT_SHORT_S], and once per
             * cfg[RAFT_WHOLE_S] within cfg[RAFT_SMOOTH_S] (sightings far apart: the nearer reach finds none). The
             * trajectory itself is smoothed when it is asked for (Hyp.history). */
            int whole = ot - T[ROT_HIST_T] >= cfg[RAFT_WHOLE_S], np = 0;
            if (!smooth_core(H, A, tr, &par, steps, whole ? cfg[RAFT_SMOOTH_S] : cfg[RAFT_SHORT_S], cfg[RAFT_LAG_S], NULL,
                             &np, lag, fl, &pl))
                continue;
            memcpy(T + ROT_VS, lag, sizeof(lag));
            memcpy(T + ROT_FS, fl, sizeof(fl));
            T[ROT_PSLAG] = pl;
            T[ROT_HASVS] = 1.0;
            if (whole)
                T[ROT_HIST_T] = ot;
        }
    for (Py_ssize_t i = 0; i < nb; i++) {
        const double *row = A + (Py_ssize_t)tab[born[i] * ROTW + ROT_SLOT] * RROW;
        hist_push(hist + born[i] * hcols, hn, t, row[RO_X + R_PX], row[RO_X + R_PY], 0.0, 0.0);
    }
    res = Py_None;
    Py_INCREF(res);
done:
    PyMem_Free(hit);
    PyMem_Free(kept);
    PyMem_Free(born);
    view_drop(&vS);
    view_drop(&vT);
    view_drop(&vH);
    return res;
}

/* Hyp.history(S, hist, handle, par, max_steps, max_s) -> [(t, x, y, vx, vy)], the oldest first
 * An object's trajectory on the ground, as it is asked for: the points written at its sightings (hist: the filtered
 * place and velocity of each), and in their place, as far back as the smoother reaches from the primary now
 * (smooth_core: at most max_steps nodes and max_s seconds), the smoothed place and velocity of every node of the
 * primary's story. Nothing is written. An object that has ended, or whose story has one node: the points as written. */
static PyObject *hyp_history(PyObject *self, PyObject *args)
{
    Hyp *H = (Hyp *)self;
    PyObject *oS, *oHi, *opar, *res = NULL;
    Py_ssize_t h;
    int max_steps;
    double max_s;
    if (!PyArg_ParseTuple(args, "OOnOid", &oS, &oHi, &h, &opar, &max_steps, &max_s))
        return NULL;
    View vS, vH;
    vS.held = vH.held = 0;
    Py_ssize_t cap, hrows, hcols;
    RPar par;
    double *hb = NULL;
    if (store_get(oS, &vS, &cap) < 0 || tab_get(oHi, &vH, 0, &hrows, &hcols, "hist") < 0 || rpar_get(opar, &par) < 0)
        goto done;
    if (h < 0 || h >= hrows || hcols < 6 || (hcols - 1) % 5 != 0 || max_steps < 0 || cap < H->cap) {
        PyErr_SetString(PyExc_ValueError, "Hyp.history: the arrays or the object do not fit");
        goto done;
    }
    {
        const double *Hh = (const double *)vH.b.buf + h * hcols;
        Py_ssize_t n = (Py_ssize_t)Hh[0], hn = (hcols - 1) / 5;
        double lag[5], fl[5], pl, t_old = INFINITY;
        int np = 0;
        hb = (double *)PyMem_Malloc((size_t)(max_steps + 1) * 6 * sizeof(double));
        if (hb == NULL) {
            PyErr_NoMemory();
            goto done;
        }
        if (h < H->ntr && H->tr[h].alive &&
            smooth_core(H, (const double *)vS.b.buf, &H->tr[h], &par, max_steps, max_s, INFINITY, hb, &np, lag, fl, &pl))
            t_old = hb[6 * (np - 1)] - 1e-6;             /* older than the smoother reaches: as it was written */
        else
            np = 0;
        if ((res = PyList_New(0)) == NULL)
            goto done;
        Py_ssize_t keep = 0;
        for (Py_ssize_t k = 0; k < n; k++)
            keep += Hh[1 + 5 * k] < t_old;
        Py_ssize_t skip = keep + np > hn ? keep + np - hn : 0;   /* (at most hn points: the oldest go) */
        for (Py_ssize_t k = 0; k < n + np; k++) {
            const double *p = k < n ? Hh + 1 + 5 * k : hb + 6 * (np - 1 - (k - n));
            if (k < n && !(p[0] < t_old))
                continue;
            if (skip > 0) {
                skip--;
                continue;
            }
            PyObject *o = Py_BuildValue("(ddddd)", p[0], p[1], p[2], p[3], p[4]);
            if (o == NULL || PyList_Append(res, o) < 0) {
                Py_XDECREF(o);
                Py_CLEAR(res);
                goto done;
            }
            Py_DECREF(o);
        }
    }
done:
    PyMem_Free(hb);
    view_drop(&vS);
    view_drop(&vH);
    return res;
}

/* Hyp.sync(table, objects) -> None: what the hypotheses say of each object now, into its row of the table: its
 * primary row, the probabilities that it exists and that the primary's story is the true one, its leaves, whether it
 * is in the best story. A scan writes these for the objects that took part; an object taken out (Hyp.drop) settles
 * its cluster anew, which may name another leaf of a cluster-mate its primary: then the table follows. */
static PyObject *hyp_sync(PyObject *self, PyObject *args)
{
    Hyp *H = (Hyp *)self;
    PyObject *oT, *otr, *res = NULL;
    if (!PyArg_ParseTuple(args, "OO", &oT, &otr))
        return NULL;
    View vT;
    vT.held = 0;
    Py_ssize_t trows, n = 0, *th = NULL;
    if (tab_get(oT, &vT, ROTW, &trows, NULL, "table") < 0 || (th = idx_get(otr, trows, &n)) == NULL)
        goto done;
    for (Py_ssize_t k = 0; k < n; k++) {
        if (th[k] >= H->ntr || !H->tr[th[k]].alive)
            continue;
        const HTrack *tr = &H->tr[th[k]];
        double *T = (double *)vT.b.buf + th[k] * ROTW;
        T[ROT_SLOT] = (double)tr->primary;
        T[ROT_PE] = tr->pe;
        T[ROT_PP] = tr->pp;
        T[ROT_NLEAF] = (double)tr->nleaf;
        T[ROT_INMAP] = tr->in_map ? 1.0 : 0.0;
    }
    res = Py_None;
    Py_INCREF(res);
done:
    PyMem_Free(th);
    view_drop(&vT);
    return res;
}

static PyMethodDef hyp_methods[] = {
    {"sync", hyp_sync, METH_VARARGS, "the objects' primaries and probabilities as the hypotheses hold them, into the table"},
    {"history", hyp_history, METH_VARARGS, "an object's trajectory, smoothed as far as the smoother reaches"},
    {"cycle", hyp_cycle, METH_VARARGS, "one scan of the tracker: its objects in, what the scan did to them out"},
    {"after", hyp_after, METH_VARARGS, "after a scan's sightings: predictions checked and written down, trajectories, the smoothed past"},
    {"drop", hyp_drop, METH_VARARGS, "an object taken out"},
    {"rows", hyp_rows_of, METH_O, "every row of an object, the primary first"},
    {"chain", hyp_chain, METH_O, "the primary's rows from now back"},
    {"leaves", hyp_leaves, METH_O, "an object's hypotheses and their probabilities"},
    {"start", hyp_start, METH_VARARGS, "an object put in from outside"},
    {"info", hyp_info, METH_NOARGS, "knobs, sizes, counters"},
    {"clusters", hyp_clusters, METH_NOARGS, "the clusters' tables"},
    {"dump", hyp_dump, METH_NOARGS, "everything it holds, as plain lists"},
    {"load", hyp_load, METH_O, "what dump recorded, restored"},
    {"xrow", hyp_xrow, METH_O, "a row's memory for the detector's further inputs"},
    {"xstate", hyp_xstate, METH_NOARGS, "whether the rows have that memory, and the scene's brightness as followed"},
    {NULL, NULL, 0, NULL}};
static PyType_Slot hyp_slots[] = {
    {Py_tp_new, hyp_new}, {Py_tp_init, hyp_init}, {Py_tp_dealloc, hyp_dealloc}, {Py_tp_methods, hyp_methods}, {0, NULL}};
static PyType_Spec hyp_spec = {"chris_integrator.native._chris_native.Hyp", sizeof(Hyp), 0, Py_TPFLAGS_DEFAULT, hyp_slots};

/* radar_cells(returns (n, 3): range, azimuth (rad), range rate; eps_m, eps_v, S | None, rows, dts, sensor, pose,
 *             margin) -> [(range, azimuth, range rate, n, sxx, sxy, syy)]
 * The returns of one scan of a sensor that gives several per object, grouped into cells: first the returns that
 * lie on a known object (rows: the objects' rows, each taken dt on at its velocity; inside its rectangle widened by
 * `margin` and by what is not known of its place, with a range rate that fits its velocity) are that object's
 * cell, a return that fits two going to the nearer; the rest by single linkage (two returns are one cell's when
 * they are nearer than eps_m and their range rates differ by less than eps_v). A cell: the mean of its returns'
 * places (sensor axes, given back as range and azimuth), the mean range rate, the number of returns and the
 * scatter of their places about the mean. The objects' cells first, in the objects' order. */
static PyObject *n_radar_cells(PyObject *self, PyObject *args)
{
    PyObject *oz, *oS = Py_None, *orows = Py_None, *odt = Py_None, *osen = Py_None, *opose = Py_None;
    double eps_m, eps_v, margin = 0.0;
    if (!PyArg_ParseTuple(args, "Odd|OOOOOd", &oz, &eps_m, &eps_v, &oS, &orows, &odt, &osen, &opose, &margin))
        return NULL;
    Py_ssize_t n = batch_len(oz);
    if (n < 0)
        return NULL;
    double *z = (double *)PyMem_Malloc((size_t)(5 * n + 1) * sizeof(double));
    int *lab = (int *)PyMem_Malloc((size_t)(2 * n + 1) * sizeof(int));
    PyObject *res = NULL;
    View vS;
    vS.held = 0;
    Py_ssize_t cap, B = 0, *rows = NULL;
    double *dts = NULL, pose[6];
    RSens sen;
    if (z == NULL || lab == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    if (n && rows_get(oz, n, 3, z, "returns") < 0)
        goto done;
    double *xy = z + 3 * n;
    int *own = lab + n;
    for (Py_ssize_t i = 0; i < n; i++) {
        xy[2 * i] = z[3 * i] * cos(z[3 * i + 1]);
        xy[2 * i + 1] = z[3 * i] * sin(z[3 * i + 1]);
        lab[i] = (int)i;
        own[i] = -1;
    }
    if (oS != Py_None && n) {
        if (store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &B)) == NULL || sens_get(osen, &sen) < 0 ||
            rnums_get(opose, 6, pose, "pose") < 0)
            goto done;
        dts = (double *)PyMem_Malloc((size_t)(B + 1) * sizeof(double));
        if (dts == NULL) {
            PyErr_NoMemory();
            goto done;
        }
        if (B && rows_get(odt, B, 1, dts, "dts") < 0)
            goto done;
        double sx, sy, axis;
        r_sensor_at(&sen, pose, &sx, &sy, &axis);
        double ca = cos(axis), sa = sin(axis), cps = cos(pose[2]), sps = sin(pose[2]);
        double ox = cps * sen.mx - sps * sen.my, oy = sps * sen.mx + cps * sen.my;
        double svx = pose[3] - pose[5] * oy, svy = pose[4] + pose[5] * ox;
        for (Py_ssize_t i = 0; i < n; i++) {
            double gx = sx + ca * xy[2 * i] - sa * xy[2 * i + 1], gy = sy + sa * xy[2 * i] + ca * xy[2 * i + 1];
            double best = INFINITY;
            for (Py_ssize_t b = 0; b < B; b++) {
                const double *row = (const double *)vS.b.buf + rows[b] * RROW, *x = row + RO_X, *P = row + RO_P, *e = row + RO_E;
                double cx = x[R_PX] + x[R_VX] * dts[b], cy = x[R_PY] + x[R_VY] * dts[b];
                double c = cos(e[E_HDG]), sn = sin(e[E_HDG]);
                double bx = c * (gx - cx) + sn * (gy - cy), by = -sn * (gx - cx) + c * (gy - cy);
                double m = margin + pymin(2.0 * sqrt(pymax(r_eig2(P[0], P[RN], P[RN + 1]), 0.0)), 2.5);
                double hx = 0.5 * e[E_LEN] + m, hy = 0.5 * e[E_WID] + m;
                if (fabs(bx) > hx || fabs(by) > hy)
                    continue;
                double lx = gx - sx, ly = gy - sy, lr = pymax(hypot(lx, ly), 0.5);
                double rr = ((x[R_VX] - svx) * lx + (x[R_VY] - svy) * ly) / lr;
                double sv = sqrt(pymax(P[R_VX * RN + R_VX] + P[R_VY * RN + R_VY], 0.0));
                if (fabs(z[3 * i + 2] - rr) > 2.0 * eps_v + 2.0 * sv)
                    continue;
                double q = pymax(fabs(bx) / hx, fabs(by) / hy);
                if (q < best) {
                    best = q;
                    own[i] = (int)b;
                }
            }
        }
        for (Py_ssize_t i = 0; i < n; i++)               /* an object's returns are one cell */
            for (Py_ssize_t j = i + 1; j < n; j++)
                if (own[i] >= 0 && own[j] == own[i]) {
                    int a = uf_find(lab, (int)i), b = uf_find(lab, (int)j);
                    if (a != b)
                        lab[a > b ? a : b] = a > b ? b : a;
                }
    }
    for (Py_ssize_t i = 0; i < n; i++)
        for (Py_ssize_t j = i + 1; j < n; j++)
            if (own[i] < 0 && own[j] < 0 && hypot(xy[2 * i] - xy[2 * j], xy[2 * i + 1] - xy[2 * j + 1]) <= eps_m &&
                fabs(z[3 * i + 2] - z[3 * j + 2]) <= eps_v) {
                int a = uf_find(lab, (int)i), b = uf_find(lab, (int)j);
                if (a != b)
                    lab[a > b ? a : b] = a > b ? b : a;
            }
    res = PyList_New(0);
    for (int pass = 0; res && pass < 2; pass++)
        for (Py_ssize_t i = 0; res && i < n; i++) {
            if (uf_find(lab, (int)i) != (int)i || (own[i] >= 0) != (pass == 0))
                continue;
            double mx = 0.0, my = 0.0, mr = 0.0, sxx = 0.0, sxy = 0.0, syy = 0.0;
            int m = 0;
            for (Py_ssize_t j = i; j < n; j++)
                if (uf_find(lab, (int)j) == (int)i) {
                    mx += xy[2 * j];
                    my += xy[2 * j + 1];
                    mr += z[3 * j + 2];
                    m++;
                }
            mx /= m;
            my /= m;
            mr /= m;
            for (Py_ssize_t j = i; j < n; j++)
                if (uf_find(lab, (int)j) == (int)i) {
                    double ax = xy[2 * j] - mx, ay = xy[2 * j + 1] - my;
                    sxx += ax * ax;
                    sxy += ax * ay;
                    syy += ay * ay;
                }
            double dof = m > 1 ? m - 1.0 : 1.0;
            PyObject *t = Py_BuildValue("(dddiddd)", hypot(mx, my), atan2(my, mx), mr, m, sxx / dof, sxy / dof, syy / dof);
            if (t == NULL || PyList_Append(res, t) < 0) {
                Py_XDECREF(t);
                Py_CLEAR(res);
                break;
            }
            Py_DECREF(t);
        }
done:
    PyMem_Free(z);
    PyMem_Free(lab);
    PyMem_Free(rows);
    PyMem_Free(dts);
    view_drop(&vS);
    return res;
}


/* ---- structure: objects that stand in a row, wait in a queue, drive as a platoon ----
 * radar_groups(S, rows, info (n, 2): motion class (1 stands, 2 drives, 0 neither), 1 if it may found a group;
 *              pose (X, Y, psi), cfg: gap_stand, gap_move, dv, line_m, line_rel, lane_m, min_n)
 *   -> [(kind (1 row, 2 queue, 3 platoon), [members], (px, py, ux, uy), across spread)]
 * Two objects are linked when both stand, or both drive at like velocity, and the free room between their
 * rectangles along the line that joins them is under the gap. A linked set of at least min_n whose centres lie on
 * one line (the worst one taken out while they do not, within line_m + line_rel x its length) is a group: its
 * line (a point and a direction, ground frame) and how far the members lie off it (rms). Standing on a line that
 * begins in the car's own lane ahead (its nearest member within lane_m of the car's axis): a queue (two are
 * enough); standing beside: a row; driving: a platoon. */
#define RGCFG 7
static PyObject *n_radar_groups(PyObject *self, PyObject *args)
{
    PyObject *oS, *orows, *oinfo, *opose, *ocfg;
    if (!PyArg_ParseTuple(args, "OOOOO", &oS, &orows, &oinfo, &opose, &ocfg))
        return NULL;
    View vS;
    vS.held = 0;
    PyObject *res = NULL;
    Py_ssize_t cap, n = 0, *rows = NULL;
    double pose[3], cfg[RGCFG], *info = NULL;
    int *lab = NULL;
    if (store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &n)) == NULL || rnums_get(opose, 3, pose, "pose") < 0 ||
        rnums_get(ocfg, RGCFG, cfg, "cfg") < 0)
        goto done;
    info = (double *)PyMem_Malloc((size_t)(2 * n + 1) * sizeof(double));
    lab = (int *)PyMem_Malloc((size_t)(3 * n + 1) * sizeof(int));
    if (info == NULL || lab == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    if (n && rows_get(oinfo, n, 2, info, "info") < 0)
        goto done;
    const double *A = (const double *)vS.b.buf;
    int *mem = lab + n, *in = mem + n;
    for (Py_ssize_t i = 0; i < n; i++)
        lab[i] = (int)i;
    for (Py_ssize_t i = 0; i < n; i++) {
        const double *ri = A + rows[i] * RROW, *xi = ri + RO_X, *ei = ri + RO_E;
        int ci = (int)info[2 * i];
        if (ci == 0)
            continue;
        for (Py_ssize_t j = i + 1; j < n; j++) {
            const double *rj = A + rows[j] * RROW, *xj = rj + RO_X, *ej = rj + RO_E;
            if ((int)info[2 * j] != ci)
                continue;
            double dx = xj[R_PX] - xi[R_PX], dy = xj[R_PY] - xi[R_PY], dist = hypot(dx, dy), los = atan2(dy, dx);
            double room = dist - r_half_depth(ei, los, NULL) - r_half_depth(ej, los, NULL);
            if (ci == 2 && hypot(xj[R_VX] - xi[R_VX], xj[R_VY] - xi[R_VY]) > cfg[2])
                continue;
            if (room > (ci == 1 ? cfg[0] : cfg[1]))
                continue;
            int a = uf_find(lab, (int)i), b = uf_find(lab, (int)j);
            if (a != b)
                lab[a > b ? a : b] = a > b ? b : a;
        }
    }
    res = PyList_New(0);
    for (Py_ssize_t r0 = 0; res && r0 < n; r0++) {
        if (uf_find(lab, (int)r0) != (int)r0 || (int)info[2 * r0] == 0)
            continue;
        int m = 0, cls = (int)info[2 * r0];
        for (Py_ssize_t j = r0; j < n; j++)
            if (uf_find(lab, (int)j) == (int)r0)
                mem[m++] = (int)j;
        double px = 0.0, py = 0.0, ux = 1.0, uy = 0.0, rms = 0.0, len = 0.0;
        while (m >= 2) {
            px = py = 0.0;
            for (int k = 0; k < m; k++) {
                px += A[rows[mem[k]] * RROW + RO_X + R_PX];
                py += A[rows[mem[k]] * RROW + RO_X + R_PY];
            }
            px /= m;
            py /= m;
            double sxx = 0.0, sxy = 0.0, syy = 0.0;
            for (int k = 0; k < m; k++) {
                double ax = A[rows[mem[k]] * RROW + RO_X + R_PX] - px, ay = A[rows[mem[k]] * RROW + RO_X + R_PY] - py;
                sxx += ax * ax;
                sxy += ax * ay;
                syy += ay * ay;
            }
            double th = 0.5 * atan2(2.0 * sxy, sxx - syy);
            ux = cos(th);
            uy = sin(th);
            double worst = -1.0, lo = INFINITY, hi = -INFINITY, ss = 0.0;
            int wk = -1;
            for (int k = 0; k < m; k++) {
                double ax = A[rows[mem[k]] * RROW + RO_X + R_PX] - px, ay = A[rows[mem[k]] * RROW + RO_X + R_PY] - py;
                double al = ux * ax + uy * ay, ac = fabs(-uy * ax + ux * ay);
                lo = pymin(lo, al);
                hi = pymax(hi, al);
                ss += ac * ac;
                if (ac > worst) {
                    worst = ac;
                    wk = k;
                }
            }
            len = hi - lo;
            rms = sqrt(ss / m);
            if (worst <= cfg[3] + cfg[4] * len)
                break;
            mem[wk] = mem[--m];                          /* the one farthest off the line is not of it */
        }
        /* the line against the car: does it pass through the car's own lane? */
        /* in the car's own lane? the member nearest the car is ahead of it and within lane_m of its axis, and the
         * line runs on from there along the car's direction (a row beside the road starts beside the car's axis) */
        double best = INFINITY, ny = 0.0, nx = 0.0, cp = cos(pose[2]), sp_ = sin(pose[2]);
        for (int k = 0; k < m; k++) {
            double ax = A[rows[mem[k]] * RROW + RO_X + R_PX] - pose[0], ay = A[rows[mem[k]] * RROW + RO_X + R_PY] - pose[1];
            if (hypot(ax, ay) < best) {
                best = hypot(ax, ay);
                nx = cp * ax + sp_ * ay;
                ny = -sp_ * ax + cp * ay;
            }
        }
        double along_car = fabs(ux * cp + uy * sp_);
        int kind = cls == 2 ? 3 : (nx > 0.0 && fabs(ny) <= cfg[5] && along_car > 0.8) ? 2 : 1;
        int need = kind == 2 ? 2 : (int)cfg[6], founders = 0;
        for (int k = 0; k < m; k++)
            founders += info[2 * mem[k] + 1] != 0.0;
        if (m < need || founders < 2)
            continue;
        /* the members in their order along the line */
        for (int k = 1; k < m; k++) {
            int v = mem[k], q = k;
            double av = ux * (A[rows[v] * RROW + RO_X + R_PX] - px) + uy * (A[rows[v] * RROW + RO_X + R_PY] - py);
            while (q > 0 && ux * (A[rows[mem[q - 1]] * RROW + RO_X + R_PX] - px) + uy * (A[rows[mem[q - 1]] * RROW + RO_X + R_PY] - py) > av) {
                mem[q] = mem[q - 1];
                q--;
            }
            mem[q] = v;
        }
        PyObject *ml = PyList_New(m);
        if (ml == NULL) {
            Py_CLEAR(res);
            break;
        }
        for (int k = 0; k < m; k++)
            PyList_SetItem(ml, k, PyLong_FromLong(mem[k]));
        PyObject *g = Py_BuildValue("(iN(dddd)d)", kind, ml, px, py, ux, uy, rms);
        if (g == NULL || PyList_Append(res, g) < 0) {
            Py_XDECREF(g);
            Py_CLEAR(res);
            break;
        }
        Py_DECREF(g);
    }
    (void)in;
done:
    PyMem_Free(rows);
    PyMem_Free(info);
    PyMem_Free(lab);
    view_drop(&vS);
    return res;
}

/* radar_group_support(S, rows, line (px, py, ux, uy), sigma, gain, floor) -> None
 * What a row of standing things says about one of its members (rows: every leaf of that member; the line fitted to
 * the OTHER members): the member stands on the line (a measurement of its place across the line, 1 sigma `sigma`,
 * taken by its standing hypothesis), and standing is likelier for it than for a thing by itself (the standing
 * hypothesis' probability times `gain`, the others never below `floor`: a member that pulls out is still found by
 * its own detections, which weigh far more). */
static PyObject *n_radar_group_support(PyObject *self, PyObject *args)
{
    PyObject *oS, *orows, *oline;
    double sigma, gain, floor_;
    if (!PyArg_ParseTuple(args, "OOOddd", &oS, &orows, &oline, &sigma, &gain, &floor_))
        return NULL;
    View vS;
    vS.held = 0;
    Py_ssize_t cap, n = 0, *rows = NULL;
    double ln[4];
    PyObject *res = NULL;
    if (store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &n)) == NULL || rnums_get(oline, 4, ln, "line") < 0)
        goto done;
    for (Py_ssize_t k = 0; k < n; k++) {
        double *row = (double *)vS.b.buf + rows[k] * RROW, *x = row + RO_XS, *P = row + RO_PS, *mu = row + RO_MU;
        double H[RN] = {0}, R = sigma * sigma;
        H[R_PX] = -ln[3];
        H[R_PY] = ln[2];
        double nu = -(H[R_PX] * (x[R_PX] - ln[0]) + H[R_PY] * (x[R_PY] - ln[1]));
        /* the row says where the member is across the line to `sigma`, and no better however often it is asked
         * (every group_s, from the same few neighbours): the member's own variance across the line is brought down
         * to sigma^2 at most (the measurement's variance p R / (p - R)); one that knows its place better by its own
         * detections takes nothing from the row. (Taken as an independent measurement at every call, the row made
         * its members' places overconfident: docs/RADAR.md, section 7.) */
        double p = H[R_PX] * (H[R_PX] * P[0] + H[R_PY] * P[RN]) + H[R_PY] * (H[R_PX] * P[1] + H[R_PY] * P[RN + 1]);
        int take = p > 1.0001 * R;
        if (take)
            R = p * R / (p - R);
        double S = p + R;
        if (take && S > 0.0 && isfinite(S) && nu * nu <= 9.0 * S) {
            double K[RN], Si = 1.0 / S, Hm[4 * RN] = {0}, nv[4] = {nu, 0.0, 0.0, 0.0};
            memcpy(Hm, H, sizeof(H));
            r_gain(P, Hm, 1, &Si, K);
            r_take(x, P, K, Hm, &R, nv, 1, 1.0);
            for (int q = R_VX; q <= R_AY; q++)
                x[q] = 0.0;
        }
        double w[3] = {mu[0] * gain, mu[1], mu[2]}, ws = w[0] + w[1] + w[2], s2 = 0.0;
        for (int j = 0; j < 3; j++) {
            w[j] = pymax(w[j] / ws, floor_);
            s2 += w[j];
        }
        for (int j = 0; j < 3; j++)
            mu[j] = w[j] / s2;
        r_combine(row + RO_XS, row + RO_PS, mu, row + RO_X, row + RO_P);
    }
    res = Py_None;
    Py_INCREF(res);
done:
    PyMem_Free(rows);
    view_drop(&vS);
    return res;
}

/* One Gaussian (x, P) of a row: the prior of lh exchanged, in information form: Lambda += a e e^T, eta += b e (e the lh
 * axis). Sherman-Morrison: P' = P - a P e e^T P / (1 + a p), x' = x + P e (b - a x_lh) / (1 + a p), p = P[lh, lh].
 * a < 0 takes information out (a wider prior); it is held so that lh's variance grows at most fivefold (1 + a p
 * >= 0.2, a and b scaled together: the same target mean). */
static void r_lh_swap(double *x, double *P, double a, double b)
{
    double p = P[R_LH * RN + R_LH], d = 1.0 + a * p;
    if (!(p > 0.0) || !isfinite(d))
        return;
    if (d < 0.2) {
        double f = -0.8 / (a * p);
        a *= f;
        b *= f;
        d = 0.2;
    }
    double Pe[RN], g = (b - a * x[R_LH]) / d;
    for (int i = 0; i < RN; i++)
        Pe[i] = P[i * RN + R_LH];
    for (int i = 0; i < RN; i++) {
        x[i] += Pe[i] * g;
        for (int j = 0; j < RN; j++)
            P[i * RN + j] -= a * Pe[i] * Pe[j] / d;
    }
}

/* radar_lh_prior(S, rows, a, b) -> None
 * The prior of an object's height (lh) its rows hold, exchanged for another (tracker.py `_size`): in every motion
 * hypothesis and the free estimate of every row, the old prior's information out and the new one's in (a =
 * 1/v1 - 1/v0, b = m1/v1 - m0/v0); the combination after. A standing hypothesis keeps no speed. */
static PyObject *n_radar_lh_prior(PyObject *self, PyObject *args)
{
    PyObject *oS, *orows;
    double a, b;
    if (!PyArg_ParseTuple(args, "OOdd", &oS, &orows, &a, &b))
        return NULL;
    View vS;
    vS.held = 0;
    Py_ssize_t cap, n = 0, *rows = NULL;
    PyObject *res = NULL;
    if (store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &n)) == NULL)
        goto done;
    for (Py_ssize_t k = 0; k < n; k++) {
        double *row = (double *)vS.b.buf + rows[k] * RROW;
        for (int j = 0; j < 3; j++)
            r_lh_swap(row + RO_XS + j * RN, row + RO_PS + j * RNN, a, b);
        for (int q = R_VX; q <= R_AY; q++)
            row[RO_XS + R_STAND * RN + q] = 0.0;
        r_lh_swap(row + RO_XF, row + RO_PF, a, b);
        r_combine(row + RO_XS, row + RO_PS, row + RO_MU, row + RO_X, row + RO_P);
    }
    res = Py_None;
    Py_INCREF(res);
done:
    PyMem_Free(rows);
    view_drop(&vS);
    return res;
}

/* radar_heading_prior(S, rows, axis, los, sigma, snap, mirror) -> number of rows changed
 * The road's axis at a standing vehicle's place as a prior of its heading (tracker.py `road_prior`), in every row
 * of it whose motion has not shown which end is the front. A camera's box width measures |sin a| (a: the heading
 * against the line of sight `los`), not on which side of it the vehicle points: the heading and its mirror about
 * the line of sight (2 los - heading) explain the boxes alike, and the road decides between them: when the mirror
 * lies within `mirror` of the axis (modulo a half turn) and nearer it than the heading, the vehicle is turned to it
 * (the reflection changes the sign of the heading's covariances; a car parked at an angle to the road, whose two
 * readings are both off the axis, is not turned). Then the
 * axis as a measurement of the heading, within `snap` of it (a car parked across the road keeps its own), to
 * `sigma` and no better however often it is asked: the heading's variance is brought down to sigma^2 at most. */
static PyObject *n_radar_heading_prior(PyObject *self, PyObject *args)
{
    PyObject *oS, *orows;
    double axis, los, sigma, snap, mirror;
    if (!PyArg_ParseTuple(args, "OOddddd", &oS, &orows, &axis, &los, &sigma, &snap, &mirror))
        return NULL;
    View vS;
    vS.held = 0;
    Py_ssize_t cap, n = 0, *rows = NULL;
    PyObject *res = NULL;
    long changed = 0;
    if (store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &n)) == NULL)
        goto done;
    double R = sigma * sigma;
    for (Py_ssize_t k = 0; k < n; k++) {
        double *row = (double *)vS.b.buf + rows[k] * RROW, *e = row + RO_E, *Pe = row + RO_PE;
        /* not where the motion has shown the front, nor where a sensor that returns the parts of a thing (an
         * extended radar's cells) measures the rectangle's orientation itself */
        if (row[RO_M + M_SENSE] != 0.0 || row[RO_M + M_HOLD] >= 2.0)
            continue;
        double h = e[E_HDG], hm = 2.0 * los - h;
        double d = wrap_pi(2.0 * (axis - h)) / 2.0, dm = wrap_pi(2.0 * (axis - hm)) / 2.0;
        int did = 0;
        if (fabs(dm) + 0.05 < fabs(d) && fabs(dm) <= mirror) {
            e[E_HDG] = wrap_pi(hm);
            Pe[2] = -Pe[2];
            Pe[6] = -Pe[6];
            Pe[5] = -Pe[5];
            Pe[7] = -Pe[7];
            d = dm;
            did = 1;
        }
        if (fabs(d) <= snap && Pe[8] > 1.0001 * R) {
            double H[3] = {0.0, 0.0, 1.0}, Rr = Pe[8] * R / (Pe[8] - R);
            r_extent_take(e, Pe, H, d, Rr, INFINITY);
            did = 1;
        }
        if (did) {
            r_extent_bounds(e);
            changed++;
        }
    }
    res = PyLong_FromLong(changed);
done:
    PyMem_Free(rows);
    view_drop(&vS);
    return res;
}

/* radar_kbest(cost (n, m), k) -> [(cost, [column of each row])]: the k cheapest assignments of n rows to m >= n
 * columns in order (the hypotheses' ranked assignment, for tests and tools); a pair that may not be: >= 1e17 */
static PyObject *n_radar_kbest(PyObject *self, PyObject *args)
{
    PyObject *oc;
    int k;
    if (!PyArg_ParseTuple(args, "Oi", &oc, &k))
        return NULL;
    View v;
    if (view_get(oc, &v, 0, 'd', 2, "cost") < 0)
        return NULL;
    int n = (int)v.b.shape[0], m = (int)v.b.shape[1];
    PyObject *res = NULL;
    Arena ar = {NULL, 0};
    if (n > m || k < 1 || n < 1) {
        PyErr_SetString(PyExc_ValueError, "radar_kbest: 1 <= rows <= columns, k >= 1");
        goto done;
    }
    int *cols = (int *)ar_get(&ar, (size_t)(k * n) * sizeof(int));
    double *costs = (double *)ar_get(&ar, (size_t)k * sizeof(double));
    int ns = cols && costs ? murty(&ar, DATA(v), n, m, k, cols, costs) : -1;
    if (ns < 0) {
        PyErr_NoMemory();
        goto done;
    }
    res = PyList_New(ns);
    for (int q = 0; res && q < ns; q++) {
        PyObject *l = PyList_New(n);
        if (l == NULL) {
            Py_CLEAR(res);
            break;
        }
        for (int i = 0; i < n; i++)
            PyList_SetItem(l, i, PyLong_FromLong(cols[q * n + i]));
        PyObject *t = Py_BuildValue("(dN)", costs[q], l);
        if (t == NULL) {
            Py_CLEAR(res);
            break;
        }
        PyList_SetItem(res, q, t);
    }
done:
    ar_free(&ar);
    view_drop(&v);
    return res;
}

#include "predict.c"                 /* the `predict` part (chris_integrator/predict): its functions are PREDICT_METHODS */
#include "decide.c"                  /* the `decide` part (chris_integrator/decide): its functions are DECIDE_METHODS */

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
    {"ekf_run", n_ekf_run, METH_VARARGS, "Filter._run: the pass over the queued entries and their rows"},
    {"lanes_measure", n_lanes_measure, METH_VARARGS, "the lane tracker's measurement arrays from a road message's lines"},
    {"lanes_predict", n_lanes_predict, METH_VARARGS, "the lane tracker's process noise, in place"},
    {"lanes_fit", n_lanes_fit, METH_VARARGS, "measurement noise, gate, innovation and fit of every line"},
    {"lanes_apply", n_lanes_apply, METH_VARARGS, "the scalar-gain update of the fitted lines, in place"},
    {"round_lists", n_round_lists, METH_VARARGS, "np.round(a, decimals).tolist(), optionally of np.sqrt(a)"},
    {"geom_centre", n_geom_centre, METH_VARARGS, "the road geometry's centre line lists (RoadGeometryFusion._centre)"},
    {"geom_reach", n_geom_reach, METH_VARARGS, "how far the camera's centre line reaches (RoadGeometryFusion._reach)"},
    {"radar_peek", n_radar_peek, METH_VARARGS, "Tracker.peek: where objects will be (x, P, mu), into given arrays"},
    {"radar_h", n_radar_h, METH_VARARGS, "a sensor's measurement function (sensors.py h) into given arrays"},
    {"radar_combine_rows", n_radar_combine_rows, METH_VARARGS, "the objects' combinations from their hypotheses, in their rows"},
    {"radar_predict1", n_radar_predict1, METH_VARARGS, "one hypothesis forward by dt (models.predict)"},
    {"radar_still1", n_radar_still1, METH_VARARGS, "how still one hypothesis is (models.still)"},
    {"radar_transition1", n_radar_transition1, METH_VARARGS, "the switching probabilities over dt (models.transition)"},
    {"radar_marginals", n_radar_marginals, METH_VARARGS, "tracker.marginals into a given array"},
    {"radar_assign", n_radar_assign, METH_O, "tracker.assign"},
    {"radar_peek_models", n_radar_peek_models, METH_VARARGS, "one row's motion hypotheses after dt, each by itself"},
    {"radar_cells", n_radar_cells, METH_VARARGS, "an extended sensor's returns grouped into cells"},
    {"radar_groups", n_radar_groups, METH_VARARGS, "rows, queues and platoons among the objects"},
    {"radar_group_support", n_radar_group_support, METH_VARARGS, "what a standing row says about one member"},
    {"radar_lh_prior", n_radar_lh_prior, METH_VARARGS, "an object's height prior exchanged in all its rows"},
    {"radar_heading_prior", n_radar_heading_prior, METH_VARARGS, "the road's axis as a standing vehicle's heading prior"},
    {"radar_app_lr", n_radar_app_lr, METH_VARARGS, "the appearance likelihood ratio between two rows' remembered vectors"},
    {"radar_kbest", n_radar_kbest, METH_VARARGS, "the k cheapest assignments in order (Murty)"},
    {"radar_frame", n_radar_frame, METH_VARARGS, "the output's per-object numbers (box.py Radar.frame)"},
    {"radar_life", n_radar_life, METH_VARARGS, "Tracker.manage's numbers: stillness, finiteness, out-of-view place"},
    {"radar_merge", n_radar_merge, METH_VARARGS, "Tracker._merge's geometry: the near pairs and their test"},
    {"radar_register", n_radar_register, METH_VARARGS, "Tracker._register for one sensor: the shared offset, taken out"},
    PREDICT_METHODS
    EGOMOTION_METHODS
    DECIDE_METHODS
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
    if (PyModule_AddObject(m, "Fallback", Fallback) < 0 || PyModule_AddIntConstant(m, "VERSION", NATIVE_VERSION) < 0 ||
        PyModule_AddIntConstant(m, "RADAR_ROW", RROW) < 0 || PyModule_AddIntConstant(m, "RADAR_SENS", RSENS) < 0)
        goto fail;
    if ((NisWinType = PyType_FromSpec(&niswin_spec)) == NULL || (TrendType = PyType_FromSpec(&trend_spec)) == NULL)
        goto fail;
    Py_INCREF(NisWinType);
    Py_INCREF(TrendType);
    if (PyModule_AddObject(m, "NisWindow", NisWinType) < 0 || PyModule_AddObject(m, "Trend", TrendType) < 0)
        goto fail;
    if ((SelfWinType = PyType_FromSpec(&selfwin_spec)) == NULL || (StreamType = PyType_FromSpec(&stream_spec)) == NULL)
        goto fail;
    Py_INCREF(SelfWinType);
    Py_INCREF(StreamType);
    if (PyModule_AddObject(m, "SelfWindow", SelfWinType) < 0 || PyModule_AddObject(m, "StreamState", StreamType) < 0)
        goto fail;
    if (egomotion_types(m) < 0)
        goto fail;
    if ((HypType = PyType_FromSpec(&hyp_spec)) == NULL)
        goto fail;
    Py_INCREF(HypType);
    if (PyModule_AddObject(m, "Hyp", HypType) < 0)
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
