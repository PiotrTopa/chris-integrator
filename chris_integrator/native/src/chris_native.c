/* chris-integrator's native core: the arithmetic and the text work of the per-message hot path.
 *
 * One C99 file, the CPython limited API (one binary for every CPython >= 3.11), no numpy headers (arrays are
 * read and written through the buffer protocol), no other dependency than libm. Every function here has a
 * pure-Python counterpart with the same contract (chris_integrator/native/__init__.py says which is in use);
 * tests/test_native.py runs both and compares them.
 *
 *   dumps(obj)                 json.dumps(obj, separators=(",", ":")): the same text, byte for byte
 *   loads(bytes | str)         json.loads: the same objects
 *   round(x, n)                builtins.round: the same result
 *   interp_all(xs, ys, qs)     trackers.view._interp_all
 *   lanes_measure, lanes_predict, lanes_fit, lanes_apply, round_lists     the lane tracker's array work
 *                              (trackers/lanes.py): the same bits as the numpy expressions
 *   ekf_predict, ekf_project, ekf_update, ekf_sym, ekf_rts     the estimator's linear algebra (core/ekf.py,
 *                              core/ego.py); elementwise steps give numpy's bits, sums of products are taken
 *                              in index order (numpy hands those to BLAS, whose order is its own)
 *   radar_evolve, radar_combine, radar_p_still     the radar's motion model (radar/models.py, tracker.py):
 *                              interaction, prediction, combination of the hypotheses; agree to rounding
 *
 * What is not certain to give the reference's result is never guessed: the function raises `Fallback` (or
 * calls the reference itself, `round`), and the caller takes the pure-Python path for that one call.
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

#define NATIVE_VERSION 4

static PyObject *Fallback;          /* raised where the pure-Python path must decide */
static PyObject *builtin_round;
static PyObject *rounded_type;      /* a tuple subclass (values, digits): a list written rounded, see dumps */

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
                uint64_t k = (uint64_t)r;
                if (v < 0.0 && buf_ch(b, '-') < 0)
                    return -1;
                if (n == 0) {
                    if (put_u64(b, k) < 0)
                        return -1;
                    return buf_put(b, ".0", 2);
                }
                uint64_t p = 1;
                for (int i = 0; i < n; i++)
                    p *= 10;
                if (put_u64(b, k / p) < 0 || buf_ch(b, '.') < 0)
                    return -1;
                char t[24];
                uint64_t f = k % p;
                for (int i = n - 1; i >= 0; i--) {
                    t[i] = (char)('0' + f % 10);
                    f /= 10;
                }
                return buf_put(b, t, (size_t)n);
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

static int put_rounded(Buf *b, PyObject *o, int depth)
{
    PyObject *vals = PyTuple_GetItem(o, 0), *nd = PyTuple_GetItem(o, 1);
    if (vals == NULL || nd == NULL)
        return -1;
    long n = PyLong_AsLong(nd);
    if (n == -1 && PyErr_Occurred())
        return -1;
    if (!PyList_CheckExact(vals) || n < 0 || n > 22) {
        PyErr_SetNone(Fallback);
        return -1;
    }
    Py_ssize_t len = PyList_Size(vals);
    if (buf_ch(b, '[') < 0)
        return -1;
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject *v = PyList_GetItem(vals, i);          /* borrowed; no Python code runs while it is in use */
        if (v == NULL)
            return -1;
        if (i && buf_ch(b, ',') < 0)
            return -1;
        if (PyFloat_CheckExact(v)) {
            int ok;
            double r = round_fast(PyFloat_AsDouble(v), (int)n, &ok);
            if (ok) {
                if (put_float(b, r) < 0)
                    return -1;
                continue;
            }
        } else if (v == Py_None) {
            if (buf_put(b, "null", 4) < 0)
                return -1;
            continue;
        } else if (PyLong_CheckExact(v)) {              /* round(int, n >= 0) is the int itself */
            if (put_int(b, v) < 0)
                return -1;
            continue;
        }
        if (!PyFloat_CheckExact(v)) {                   /* any other type: the caller's pure path (its __round__ */
            PyErr_SetNone(Fallback);                    /* would run Python code in the middle of this walk) */
            return -1;
        }
        /* a float next to a decimal tie: float.__round__ itself (C code, no callbacks) */
        PyObject *r = PyObject_CallFunction(builtin_round, "Ol", v, n);
        if (r == NULL)
            return -1;
        int rc = PyFloat_CheckExact(r) ? put_double(b, PyFloat_AsDouble(r)) : -2;
        Py_DECREF(r);
        if (rc == -2)
            PyErr_SetNone(Fallback);
        if (rc < 0)
            return -1;
    }
    return buf_ch(b, ']');
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
    if (rounded_type != NULL && (PyObject *)Py_TYPE(o) == rounded_type)
        return put_rounded(b, o, depth);
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

static PyObject *n_set_rounded_type(PyObject *self, PyObject *t)
{
    if (!PyType_Check(t) || !PyType_IsSubtype((PyTypeObject *)t, &PyTuple_Type)) {
        PyErr_SetString(PyExc_TypeError, "a tuple subclass is needed");
        return NULL;
    }
    Py_INCREF(t);
    Py_XDECREF(rounded_type);
    rounded_type = t;
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
 * The estimator's linear algebra (core/ekf.py, core/ego.py are the reference; names follow them)
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
            PyErr_SetString(PyExc_ValueError, "ekf_project: index");
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
 * interp_all(xs, ys, qs): trackers/view.py `_interp_all` for lists or tuples of floats and small ints
 * (anything else: Fallback)
 * ==================================================================================================== */
/* a plain float, or a plain int small enough to be exact as a double (Python's mixed arithmetic then gives
 * what the double arithmetic gives) */
static int get_num(PyObject *o, double *v)
{
    if (PyFloat_CheckExact(o)) {
        *v = PyFloat_AsDouble(o);
        return 1;
    }
    if (PyLong_CheckExact(o)) {
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

static PyObject *n_interp_all(PyObject *self, PyObject *args)
{
    PyObject *xs, *ys, *qs;
    if (!PyArg_ParseTuple(args, "OOO", &xs, &ys, &qs))
        return NULL;
    int lx = PyList_CheckExact(xs), ly = PyList_CheckExact(ys), lq = PyList_CheckExact(qs);
    if ((!lx && !PyTuple_CheckExact(xs)) || (!ly && !PyTuple_CheckExact(ys)) || (!lq && !PyTuple_CheckExact(qs))) {
        PyErr_SetNone(Fallback);
        return NULL;
    }
    Py_ssize_t nx = lx ? PyList_Size(xs) : PyTuple_Size(xs), ny = ly ? PyList_Size(ys) : PyTuple_Size(ys);
    Py_ssize_t nq = lq ? PyList_Size(qs) : PyTuple_Size(qs);
    Py_ssize_t n = nx < ny ? nx : ny;
    if (n < 1 || n > 4096) {                        /* an empty table: the reference's own answer */
        PyErr_SetNone(Fallback);
        return NULL;
    }
    double sx[64], sy[64], *X = sx, *Y = sy;
    char sn[64], *isnone = sn;                      /* ys may hold None */
    PyObject *out = NULL;
    void *heap = NULL;
    if (n > 64) {
        heap = PyMem_Malloc((size_t)n * (2 * sizeof(double) + 1));
        if (heap == NULL)
            return PyErr_NoMemory();
        X = (double *)heap;
        Y = X + n;
        isnone = (char *)(Y + n);
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *y = seq_item(ys, ly, i);
        isnone[i] = (y == Py_None);
        Y[i] = 0.0;
        if (!get_num(seq_item(xs, lx, i), &X[i]) || (!isnone[i] && !get_num(y, &Y[i]))) {
            PyErr_SetNone(Fallback);
            goto done;
        }
    }
    out = PyList_New(nq);
    if (out == NULL)
        goto done;
    Py_ssize_t i = 1;
    int have_prev = 0;
    double prev = 0.0;
    PyObject *y0 = seq_item(ys, ly, 0);
    for (Py_ssize_t q = 0; q < nq; q++) {
        double x;
        PyObject *item;
        if (!get_num(seq_item(qs, lq, q), &x)) {
            Py_DECREF(out);
            out = NULL;
            PyErr_SetNone(Fallback);
            goto done;
        }
        if (x <= X[0]) {
            item = y0;
            Py_INCREF(item);
        } else {
            if (have_prev && !(x >= prev))
                i = 1;
            have_prev = 1;
            prev = x;
            while (i < n && !(x <= X[i]))
                i++;
            if (i >= n || isnone[i - 1] || isnone[i]) {
                item = Py_None;
                Py_INCREF(item);
            } else {
                double a = Y[i - 1], b = Y[i];
                double den = X[i] - X[i - 1];
                if (den == 0.0)                       /* Python: (xs[i] - xs[i - 1]) or 1.0 */
                    den = 1.0;
                double t = (x - X[i - 1]) / den;
                item = PyFloat_FromDouble(a + (b - a) * t);
                if (item == NULL) {
                    Py_DECREF(out);
                    out = NULL;
                    goto done;
                }
            }
        }
        PyList_SetItem(out, q, item);
    }
done:
    PyMem_Free(heap);
    return out;
}

/* ====================================================================================================
 * The lane tracker's array work (trackers/lanes.py is the reference; names follow it). Elementwise
 * steps as numpy does them; `mean` over the gate points adds in numpy's order (seq_sum below).
 * ==================================================================================================== */
#define N_LANE 4
#define N_LINES 6

/* The mean over the gate points, as numpy takes it in trackers/lanes.py: `a[:, gate].mean(axis=1)`. The
 * selection a[:, gate] comes out in column order, and numpy then adds each line's values one after the
 * other, in order (not pairwise). tests/test_native.py holds this against numpy; a numpy that adds otherwise
 * fails that test, and the tracker's two fit measures then agree with the reference to rounding. */
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

/* a list of n plain floats or ints into `out`; 0 = not such a list (the reference decides), else 1 and
 * *finite = every value is finite */
static int list_row(PyObject *lst, Py_ssize_t n, double *out, int *finite)
{
    if (lst == NULL || !PyList_CheckExact(lst) || PyList_Size(lst) != n)
        return 0;
    int fin = 1;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *v = PyList_GetItem(lst, i);
        double d;
        if (PyFloat_CheckExact(v))
            d = PyFloat_AsDouble(v);
        else if (PyLong_CheckExact(v)) {
            d = PyLong_AsDouble(v);
            if (d == -1.0 && PyErr_Occurred()) {
                PyErr_Clear();
                return 0;
            }
        } else
            return 0;
        if (!isfinite(d))
            fin = 0;
        out[i] = d;
    }
    *finite = fin;
    return 1;
}

/* lanes_measure(srcs, x, std_x0, std_x1, std_bad, std_good, y, s, prob, ok)
 * RoadGeometryTracker._measure after the distances are known, for the usual frame: six dicts whose "y" and
 * "y_std" are lists of len(x) numbers. Fills y[6,n], s[6,n], prob[6], ok[6]. Anything else: Fallback. */
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
        PyErr_SetNone(Fallback);
        return NULL;
    }
    if (view_get(ox, &x, 0, 'd', 1, "x") < 0 || view_get(oy, &y, 1, 'd', 2, "y") < 0 ||
        view_get(os, &sv, 1, 'd', 2, "s") < 0 || view_get(oprob, &prob, 1, 'd', 1, "prob") < 0 ||
        view_get(ook, &ok, 1, 'b', 1, "ok") < 0)
        goto done;
    Py_ssize_t n = x.b.shape[0];
    if (n < 1 || n > 1024 || y.b.shape[0] != N_LINES || y.b.shape[1] != n || sv.b.shape[0] != N_LINES ||
        sv.b.shape[1] != n || prob.b.shape[0] != N_LINES || ok.b.shape[0] != N_LINES) {
        PyErr_SetString(PyExc_ValueError, "lanes_measure: shapes");
        goto done;
    }
    rows = (double *)PyMem_Malloc((size_t)(2 * N_LINES + 1) * (size_t)n * sizeof(double));
    if (rows == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    int fin[2 * N_LINES];
    double pv[N_LANE];
    PyObject *ky = PyUnicode_FromString("y"), *ks = PyUnicode_FromString("y_std"), *kp = PyUnicode_FromString("prob");
    int good_shape = (ky && ks && kp);
    for (int i = 0; good_shape && i < N_LINES; i++) {
        PyObject *d = PyList_GetItem(srcs, i);
        if (d == NULL || !PyDict_CheckExact(d)) {
            good_shape = 0;
            break;
        }
        PyObject *ly = PyDict_GetItemWithError(d, ky), *ls = PyDict_GetItemWithError(d, ks);
        if (!list_row(ly, n, rows + (Py_ssize_t)i * n, &fin[i]) ||
            !list_row(ls, n, rows + (Py_ssize_t)(N_LINES + i) * n, &fin[N_LINES + i])) {
            good_shape = 0;
            break;
        }
        if (i < N_LANE) {
            /* prob: min(1, max(0, float(p))) if p is a finite int / float (a bool is an int), else 0 */
            PyObject *p = PyDict_GetItemWithError(d, kp);
            double v = 0.0;
            if (p == NULL || p == Py_None)
                v = 0.0;
            else if (PyFloat_CheckExact(p)) {
                v = PyFloat_AsDouble(p);
                v = isfinite(v) ? clamp01(v) : 0.0;
            } else if (PyLong_CheckExact(p) || p == Py_True || p == Py_False) {
                v = PyLong_AsDouble(p);
                if (v == -1.0 && PyErr_Occurred()) {
                    PyErr_Clear();
                    good_shape = 0;
                    break;
                }
                v = clamp01(v);
            } else {
                good_shape = 0;                     /* any other type: the reference decides */
                break;
            }
            pv[i] = v;
        }
    }
    Py_XDECREF(ky); Py_XDECREF(ks); Py_XDECREF(kp);
    if (PyErr_Occurred())
        goto done;
    if (!good_shape) {
        PyErr_SetNone(Fallback);
        goto done;
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
        if (!fin[i]) {                               /* offsets not finite: the line is not there this frame */
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
        int have_s = fin[N_LINES + i];
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

/* ---- the radar, phase 2: the objects' rows, the sensors' measurement functions, one scan's association
 * (gate, likelihoods of the hypotheses, the box as a feature, the pairing's probabilities, the assignment),
 * the update of every pair of a scan, and the output's per-object numbers. radar/tracker.py, sensors.py and
 * box.py are the reference; the same formulas, sums of products in index order, small systems solved by
 * Gaussian elimination with partial pivoting (numpy hands them to LAPACK): results agree to rounding. ---- */

/* an object's row in the tracker's state array (tracker.py `_LAY`) */
enum { RO_XS = 0, RO_PS = 24, RO_MU = 216, RO_XF = 219, RO_PF = 227, RO_X = 291, RO_P = 299, RROW = 363 };

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
static int nums_get(PyObject *o, Py_ssize_t n, double *out, const char *what)
{
    return rows_get(o, n, 1, out, what);
}

/* ---- the sensors (sensors.py): what a scan measures of a state ---- */
enum { SEN_CAMERA = 0, SEN_MICROWAVE = 1 };
enum { K_B = 0, K_RB = 1, K_RBH = 2, K_BT = 3, K_NONE = 4, K_SCALE = 8 };
#define RSENS 24

typedef struct {
    int type;
    /* camera */
    double hc, logf, bias;
    int cover;
    double bmin, bmax, gmin, max_range, p_d, p_cut, h_min, h_full, p_small, near, extra, vol[4], srspan, lhspan,
        born, born_s;
    /* microwave */
    double mx, my, yaw, fov, clutter;
} RSens;

/* sensors.py `Sensor.native()`: 24 numbers */
static int sens_get(PyObject *o, RSens *s)
{
    double v[RSENS];
    if (nums_get(o, RSENS, v, "sensor") < 0)
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
    } else if (s->type == SEN_MICROWAVE) {
        s->mx = v[1]; s->my = v[2]; s->yaw = v[3]; s->fov = v[4]; s->max_range = v[5]; s->p_d = v[6]; s->clutter = v[7];
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
    int b = kind & 7;
    return (b == K_B || b == K_BT) ? 0 : b == K_NONE ? -1 : 1;
}

/* CameraBoxes._scale_row */
static double r_scale_row(const double *x, const double *pose, double sdt, double *row)
{
    double dx = x[R_PX] - pose[0], dy = x[R_PY] - pose[1];
    double c = cos(pose[2]), s = sin(pose[2]);
    double depth = pymax(c * dx + s * dy, 0.5);
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

/* sensor.h(x, pose, det): the predicted measurement z (m) and its Jacobian H (m x N); returns m */
static int r_h(const RSens *s, int kind, double sdt, const double *x, const double *pose, double *z, double *H)
{
    memset(H, 0, 4 * RN * sizeof(double));
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
    if (base != K_NONE) {
        double dx = x[R_PX] - pose[0], dy = x[R_PY] - pose[1];
        double r = pymax(py_hypot(dx, dy), 0.5);
        double brow[RN] = {0};
        brow[R_PX] = -dy / (r * r);
        brow[R_PY] = dx / (r * r);
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
            H[RN + R_PX] = -dh / q * dx / r;
            H[RN + R_PY] = -dh / q * dy / r;
            H[RN + R_LH] = -ho * r / q;
            z[1] = atan2(dh, r) + s->bias;
            m = 2;
        } else {
            double g = -hc / (r * r + hc * hc);
            H[R_PX] = g * dx / r;
            H[R_PY] = g * dy / r;
            H[R_DP] = 1.0;
            z[0] = atan2(hc, r) + s->bias + x[R_DP];
            z[1] = b;
            memcpy(H + RN, brow, sizeof(brow));
            m = 2;
            if (base != K_RB) {
                double c = cos(pose[2]), sn = sin(pose[2]);
                double depth = pymax(c * dx + sn * dy, 0.5);
                H[2 * RN + R_PX] = -c / depth;
                H[2 * RN + R_PY] = -sn / depth;
                H[2 * RN + R_LH] = 1.0;
                z[2] = s->logf + x[R_LH] - log(depth);
                m = 3;
            }
        }
    }
    if (kind & K_SCALE) {
        z[m] = r_scale_row(x, pose, sdt, H + m * RN);
        m++;
    }
    return m;
}

/* CameraBoxes.coverage / MicrowaveRadar.coverage at a place of the car frame */
static double r_coverage(const RSens *s, double xc, double yc)
{
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
    if (s->type == SEN_MICROWAVE)
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
    if (s->type == SEN_MICROWAVE)
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

/* ---- one scan's association (Tracker.process) ---- */
#define RDET 13                  /* a detection: kind, sdt, group, vehicle (-1, 0, 1), box width, z[4], R[4] */
#define RSCFG 10                 /* gate for 1..4 rows, feat_sigma, feat_same, feat_span, map_bw[3] */

/* radar_scan(S, rows, sensor, pose, objs, dets, mem, nscan, cfg, pd_out, cov_out, ratio_out|None,
 *            gain_out|None, feat_out|None) -> [(i, j, gain, pair, lam)]
 *   objs: (nt, 3) [kind of the object's label (-1 a person or bike, 0 unknown, 1 a vehicle), width of its last
 *         box if the box feature applies (NaN if not), id];  dets: (nd, RDET);  mem: (nm, 5) the map's entries that
 *         count [rows, z0, z1, z2, object id] or None;  nscan: scans in the map's memory.
 * Every object's probability of being reported (pd_out) and the sensor's coverage at its place (cov_out); the
 * pairs of the scan's assignment with their gain, their probability in the scan as a whole (marginals) and the
 * other story's density over the box feature (tracker `feat`). */
static PyObject *n_radar_scan(PyObject *self, PyObject *args)
{
    PyObject *oS, *orows, *osen, *opose, *oobj, *odet, *omem, *ocfg, *opd, *ocov, *orat, *ogain, *ofeat;
    double nscan;
    if (!PyArg_ParseTuple(args, "OOOOOOOdOOOOOO", &oS, &orows, &osen, &opose, &oobj, &odet, &omem, &nscan, &ocfg,
                          &opd, &ocov, &orat, &ogain, &ofeat))
        return NULL;
    PyObject *res = NULL;
    View vS, vpd, vcov, vrat, vgain, vfeat;
    vS.held = vpd.held = vcov.held = vrat.held = vgain.held = vfeat.held = 0;
    Py_ssize_t cap, nt = 0, nd, nm = 0, *rows = NULL, *pi = NULL, *pj = NULL;
    double *ob = NULL, *dt = NULL, *mem = NULL, *wk = NULL;
    double pose[6], cfg[RSCFG], *pd, *cov, *rat, *gain, *feat;
    RSens sen;
    if (store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &nt)) == NULL || sens_get(osen, &sen) < 0 ||
        nums_get(opose, 6, pose, "pose") < 0 || nums_get(ocfg, RSCFG, cfg, "cfg") < 0)
        goto done;
    nd = batch_len(odet);
    if (nd < 0)
        goto done;
    if (omem != Py_None) {
        nm = batch_len(omem);
        if (nm < 0)
            goto done;
    }
    ob = (double *)PyMem_Malloc((size_t)(3 * nt + RDET * nd + 5 * nm + 1) * sizeof(double));
    /* work: ratio, gain, feat, own (nt x nd each) if not given; lam0, loc (nd); car (2 nt) */
    wk = (double *)PyMem_Malloc((size_t)(4 * nt * nd + 2 * nd + 2 * nt + 1) * sizeof(double));
    pi = (Py_ssize_t *)PyMem_Malloc((size_t)(2 * (nt + nd) + 2) * sizeof(Py_ssize_t));
    if (ob == NULL || wk == NULL || pi == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    pj = pi + nt + nd + 1;
    dt = ob + 3 * nt;
    mem = dt + RDET * nd;
    if ((nt && rows_get(oobj, nt, 3, ob, "objs") < 0) || (nd && rows_get(odet, nd, RDET, dt, "dets") < 0) ||
        (nm && rows_get(omem, nm, 5, mem, "mem") < 0) ||
        out_get(opd, &vpd, nt, 0, &pd, "pd_out") < 0 || out_get(ocov, &vcov, nt, 0, &cov, "cov_out") < 0 ||
        out_get(orat, &vrat, nt * nd, 1, &rat, "ratio_out") < 0 || out_get(ogain, &vgain, nt * nd, 1, &gain, "gain_out") < 0 ||
        out_get(ofeat, &vfeat, nt * nd, 1, &feat, "feat_out") < 0)
        goto done;
    {
    double *own = wk + 3 * nt * nd, *lam0 = wk + 4 * nt * nd, *loc = lam0 + nd, *car = loc + nd;
    if (rat == NULL)
        rat = wk;
    if (gain == NULL)
        gain = wk + nt * nd;
    if (feat == NULL)
        feat = wk + 2 * nt * nd;
    double *A = (double *)vS.b.buf;
    double cph = cos(pose[2]), sph = sin(pose[2]);
    for (Py_ssize_t i = 0; i < nt; i++) {
        const double *x = A + rows[i] * RROW + RO_X;
        double dx = x[R_PX] - pose[0], dy = x[R_PY] - pose[1];
        car[2 * i] = cph * dx + sph * dy;
        car[2 * i + 1] = -sph * dx + cph * dy;
        pd[i] = r_p_detect(&sen, car[2 * i], car[2 * i + 1], x);
        cov[i] = r_coverage(&sen, car[2 * i], car[2 * i + 1]);
    }
    for (Py_ssize_t k = 0; k < nt * nd; k++) {
        rat[k] = 0.0;
        gain[k] = -INFINITY;
        feat[k] = 0.0;
        own[k] = 0.0;
    }
    if (nd && nt) {
        int ng = 0;
        for (Py_ssize_t j = 0; j < nd; j++) {
            const double *d = dt + j * RDET;
            lam0[j] = r_density(&sen, (int)d[0], d + 5);
            if ((int)d[2] + 1 > ng)
                ng = (int)d[2] + 1;
            /* the map: unexplained detections that became objects, near this one (tracker `_local_density`) */
            loc[j] = 0.0;
            int base = (int)d[0] & 7;
            if (sen.type != SEN_CAMERA || !(base == K_RB || base == K_RBH) || nm == 0)
                continue;
            int n = base == K_RB ? 2 : 3;
            double norm = pymax(nscan, 1.0);
            for (int a = 0; a < n; a++)
                norm *= cfg[7 + a] * 2.5066;
            for (Py_ssize_t e = 0; e < nm; e++) {
                const double *me = mem + 5 * e;
                if (me[0] < n)
                    continue;
                double q = 0.0;
                for (int a = 0; a < n; a++)
                    q += pow((d[5 + a] - me[1 + a]) / cfg[7 + a], 2.0);
                if (q < 25.0) {
                    double v = exp(-0.5 * q) / norm;
                    loc[j] += v;
                    for (Py_ssize_t i = 0; i < nt; i++)
                        if (ob[3 * i + 2] == me[4])
                            own[i * nd + j] += v;
                }
            }
        }
        Py_ssize_t *J = pi;                              /* a group's detections (pi is free until the end) */
        int *ok = (int *)PyMem_Malloc((size_t)nd * sizeof(int));
        double *qs = (double *)PyMem_Malloc((size_t)(3 * nd) * sizeof(double));
        if (ok == NULL || qs == NULL) {
            PyMem_Free(ok);
            PyMem_Free(qs);
            PyErr_NoMemory();
            goto done;
        }
        double *dets = qs + nd, *lk = dets + nd;
        double zp[4], H[4 * RN], HP[4 * RN], S[16], zm[4], Hm[4 * RN], HPm[4 * RN], Sm[16];
        for (Py_ssize_t i = 0; i < nt; i++) {
            if (pd[i] <= 0.0)
                continue;
            const double *row = A + rows[i] * RROW;
            const double *x = row + RO_X, *P = row + RO_P, *mu = row + RO_MU;
            double ov = ob[3 * i], lbw = ob[3 * i + 1];
            for (int g = 0; g < ng; g++) {
                Py_ssize_t nJ = 0;
                for (Py_ssize_t j = 0; j < nd; j++)
                    if ((int)dt[j * RDET + 2] == g)
                        J[nJ++] = j;
                if (nJ == 0)
                    continue;
                const double *d0 = dt + J[0] * RDET;
                int kind = (int)d0[0];
                double sdt = d0[1];
                if (sen.type == SEN_CAMERA && sen.cover && ((kind & 7) == K_B || (kind & 7) == K_BT) &&
                    !(py_hypot(car[2 * i], car[2 * i + 1]) <= sen.near * sen.gmin))
                    continue;                            /* CameraBoxes.takes */
                int anyok = 0;
                for (Py_ssize_t a = 0; a < nJ; a++) {
                    ok[a] = dt[J[a] * RDET + 3] * ov >= 0.0;    /* a pedestrian's box does not continue a vehicle */
                    anyok |= ok[a];
                }
                if (!anyok)
                    continue;
                /* the gate: the object as a whole */
                int m = r_h(&sen, kind, sdt, x, pose, zp, H);
                int ar = r_angle_row(&sen, kind);
                double gate = (m >= 1 && m <= 4) ? cfg[m - 1] : 16.0;
                r_hph(H, m, P, HP, S);
                int sing = 0;
                anyok = 0;
                for (Py_ssize_t a = 0; a < nJ; a++) {
                    const double *d = dt + J[a] * RDET;
                    double nuj[4], Sj[16], q;
                    for (int r = 0; r < m; r++)
                        nuj[r] = r == ar ? wrap_pi(d[5 + r] - zp[r]) : d[5 + r] - zp[r];
                    memcpy(Sj, S, sizeof(double) * (size_t)(m * m));
                    for (int r = 0; r < m; r++)
                        Sj[r * m + r] = S[r * m + r] + d[9 + r];
                    if (r_mahal(Sj, m, nuj, &q, NULL) < 0) {
                        sing = 1;                        /* numpy: LinAlgError for the whole group */
                        break;
                    }
                    ok[a] = ok[a] && q <= gate;
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
                    r_h(&sen, kind, sdt, row + RO_XS + mm * RN, pose, zm, Hm);
                    r_hph(Hm, m, row + RO_PS + mm * RNN, HPm, Sm);
                    int bad = 0;
                    for (Py_ssize_t a = 0; a < nJ; a++) {
                        if (!ok[a])
                            continue;
                        const double *d = dt + J[a] * RDET;
                        double nuj[4], Sj[16];
                        for (int r = 0; r < m; r++)
                            nuj[r] = r == ar ? wrap_pi(d[5 + r] - zm[r]) : d[5 + r] - zm[r];
                        memcpy(Sj, Sm, sizeof(double) * (size_t)(m * m));
                        for (int r = 0; r < m; r++)
                            Sj[r * m + r] = Sm[r * m + r] + d[9 + r];
                        if (r_mahal(Sj, m, nuj, &qs[a], &dets[a]) < 0) {
                            bad = 1;
                            break;
                        }
                    }
                    if (bad)
                        continue;
                    for (Py_ssize_t a = 0; a < nJ; a++)
                        if (ok[a] && qs[a] < 60.0 && dets[a] > 0.0)
                            lk[a] += mu[mm] * exp(-0.5 * pymin(qs[a], 60.0)) / sqrt(c2 * dets[a]);
                }
                for (Py_ssize_t a = 0; a < nJ; a++) {
                    if (!ok[a] || lk[a] <= 0.0)
                        continue;
                    const double *d = dt + J[a] * RDET;
                    Py_ssize_t j = J[a];
                    double F = 1.0;
                    if (!isnan(lbw) && d[4] > 0.0 && ar >= 0) {      /* the box as a feature */
                        double e = fabs(wrap_pi(d[5 + ar] - zp[ar])) / np_maximum(0.5 * (d[4] + lbw), 1e-4);
                        double sg = cfg[4], t = np_minimum(e / sg, 12.0);
                        F = cfg[6] * (cfg[5] * 0.7979 / sg * exp(-0.5 * (t * t)) + (1.0 - cfg[5]) / cfg[6]);
                    }
                    /* the other story's density here: not counting this object's own first detection */
                    double lam = pymax(lam0[j], loc[j] - own[i * nd + j]);
                    double r = pd[i] * lk[a] * F / ((1.0 - pd[i]) * lam);
                    rat[i * nd + j] = r;
                    double gg = log(r);
                    if (gg > 0.0) {
                        gain[i * nd + j] = gg;
                        feat[i * nd + j] = lam / F;
                    }
                }
            }
        }
        PyMem_Free(ok);
        PyMem_Free(qs);
    }
    /* the pairing's probabilities in the scan as a whole, and the assignment */
    double *pair = own;                                  /* free now */
    if (r_marginals(rat, nt, nd, pair) < 0)
        goto done;
    Py_ssize_t np_ = r_assign(gain, nt, nd, pi, pj);
    if (np_ < 0)
        goto done;
    res = PyList_New(np_);
    if (res == NULL)
        goto done;
    for (Py_ssize_t q = 0; q < np_; q++) {
        Py_ssize_t k = pi[q] * nd + pj[q];
        PyObject *t = Py_BuildValue("(nnddd)", pi[q], pj[q], gain[k], pair[k], feat[k]);
        if (t == NULL) {
            Py_CLEAR(res);
            goto done;
        }
        PyList_SetItem(res, q, t);
    }
    }
done:
    PyMem_Free(rows);
    PyMem_Free(ob);
    PyMem_Free(wk);
    PyMem_Free(pi);
    view_drop(&vS); view_drop(&vpd); view_drop(&vcov); view_drop(&vrat); view_drop(&vgain); view_drop(&vfeat);
    return res;
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

/* one hypothesis (or the free estimate) takes the detection with weight beta: x, P in place */
static void r_take(double *x, double *P, const double *K, const double *S, const double *nu, int m, double beta)
{
    double dx[RN], KS[RN * 4], Pn[RNN];
    for (int i = 0; i < RN; i++) {
        double acc = 0.0;
        for (int k = 0; k < m; k++)
            acc = k ? acc + K[i * m + k] * nu[k] : K[i * m + k] * nu[k];
        dx[i] = acc;
    }
    for (int i = 0; i < RN; i++)
        for (int l = 0; l < m; l++) {
            double acc = 0.0;
            for (int k = 0; k < m; k++)
                acc = k ? acc + K[i * m + k] * S[k * m + l] : K[i * m + k] * S[k * m + l];
            KS[i * m + l] = acc;
        }
    double bb = beta * (1.0 - beta);
    for (int i = 0; i < RN; i++)
        for (int j = 0; j < RN; j++) {
            double acc = 0.0;
            for (int k = 0; k < m; k++)
                acc = k ? acc + KS[i * m + k] * K[j * m + k] : KS[i * m + k] * K[j * m + k];
            Pn[i * RN + j] = P[i * RN + j] - beta * acc + bb * (dx[i] * dx[j]);
        }
    for (int i = 0; i < RN; i++)
        x[i] = x[i] + beta * dx[i];
    for (int i = 0; i < RN; i++)
        for (int j = 0; j < RN; j++)
            P[i * RN + j] = 0.5 * (Pn[i * RN + j] + Pn[j * RN + i]);
}

static inline double clamp_dp(double v)          /* min(0.05, max(-0.05, v)) as Python takes it */
{
    return pymin(0.05, pymax(-0.05, v));
}

/* nu = z - h(x) wrapped, S = H P H^T + diag(R); returns m */
static int r_innov(const RSens *s, int kind, double sdt, const double *x, const double *P, const double *pose,
                   const double *z, const double *R, double *nu, double *H, double *S)
{
    double zp[4], HP[4 * RN];
    int m = r_h(s, kind, sdt, x, pose, zp, H);
    int ar = r_angle_row(s, kind);
    for (int r = 0; r < m; r++)
        nu[r] = r == ar ? wrap_pi(z[r] - zp[r]) : z[r] - zp[r];
    r_hph(H, m, P, HP, S);
    for (int r = 0; r < m; r++)
        S[r * m + r] = S[r * m + r] + R[r];
    return m;
}

#define RPAIR 4                  /* a pair: p_detect, lam (feat), pair probability (NaN: none), the object's hits */

/* radar_update(S, rows, sensor, pose, dets, pairs, cfg) -> [(nis, dlogodds)]
 *   rows: the objects' rows (one per pair);  dets: (np, RDET) each pair's detection (as `_common` left it);
 *   pairs: (np, RPAIR);  cfg: (mu_hold_hits, mu_floor).
 * Every pair's detection into its object: the hypotheses, the free estimate, the probabilities of the hypotheses
 * and the combination are written into the object's row; returned per pair: the normalized innovation of the
 * object as a whole (before the update) and what the detection adds to the object's log odds of existing. */
static PyObject *n_radar_update(PyObject *self, PyObject *args)
{
    PyObject *oS, *orows, *osen, *opose, *odet, *opair, *ocfg;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &oS, &orows, &osen, &opose, &odet, &opair, &ocfg))
        return NULL;
    PyObject *res = NULL;
    View vS;
    vS.held = 0;
    Py_ssize_t cap, np_ = 0, *rows = NULL;
    double *buf = NULL, pose[6], cfg[2];
    RSens sen;
    if (store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &np_)) == NULL || sens_get(osen, &sen) < 0 ||
        nums_get(opose, 6, pose, "pose") < 0 || nums_get(ocfg, 2, cfg, "cfg") < 0)
        goto done;
    buf = (double *)PyMem_Malloc((size_t)((RDET + RPAIR) * np_ + 1) * sizeof(double));
    if (buf == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    if (np_ && (rows_get(odet, np_, RDET, buf, "dets") < 0 || rows_get(opair, np_, RPAIR, buf + RDET * np_, "pairs") < 0))
        goto done;
    for (Py_ssize_t q = 0; q < np_; q++)
        for (Py_ssize_t q2 = q + 1; q2 < np_; q2++)
            if (rows[q] == rows[q2]) {
                PyErr_SetString(PyExc_ValueError, "radar_update: one object in two pairs");
                goto done;
            }
    res = PyList_New(np_);
    if (res == NULL)
        goto done;
    double *A = (double *)vS.b.buf;
    for (Py_ssize_t q = 0; q < np_; q++) {
        const double *d = buf + q * RDET, *pp = buf + RDET * np_ + q * RPAIR;
        double *row = A + rows[q] * RROW;
        double *mu = row + RO_MU;
        int kind = (int)d[0];
        double sdt = d[1], pd = pp[0], lam = pp[1], pair = pp[2], hits = pp[3];
        double other = pymax((1.0 - pd) * lam, 1e-300);
        double like[3], Hs[3][4 * RN], nus[3][4], Ss[3][16], Sis[3][16];
        int m = 0;
        for (int j = 0; j < 3; j++) {
            double det;
            m = r_innov(&sen, kind, sdt, row + RO_XS + j * RN, row + RO_PS + j * RNN, pose, d + 5, d + 9, nus[j], Hs[j], Ss[j]);
            if (r_inv(Ss[j], m, Sis[j], &det) < 0) {
                PyErr_SetString(PyExc_ValueError, "radar_update: a singular innovation covariance");   /* numpy: LinAlgError */
                Py_CLEAR(res);
                goto done;
            }
            double t[4], d2 = 0.0;
            for (int c = 0; c < m; c++) {
                double acc = 0.0;
                for (int r = 0; r < m; r++)
                    acc = r ? acc + nus[j][r] * Sis[j][r * m + c] : nus[j][r] * Sis[j][r * m + c];
                t[c] = acc;
            }
            for (int c = 0; c < m; c++)
                d2 = c ? d2 + t[c] * nus[j][c] : t[c] * nus[j][c];
            like[j] = exp(-0.5 * pymin(d2, 400.0)) / sqrt(pymax(pow(2.0 * PI, (double)m) * det, 1e-30));
        }
        /* whose the detection is: one question for the whole object, shrunk by the pairing's probability */
        double lo = (mu[0] * like[0] + mu[1] * like[1]) + mu[2] * like[2];
        double beta_o = pd * lo / (pd * lo + other);
        double comp = (isnan(pair) || beta_o <= 0.0) ? 1.0 : pymin(1.0, pair / beta_o);
        beta_o *= comp;
        /* the object as a whole, before the update (its NIS) */
        double nu0[4], H0[4 * RN], S0[16], d20;
        int m0 = r_innov(&sen, kind, sdt, row + RO_X, row + RO_P, pose, d + 5, d + 9, nu0, H0, S0);
        if (r_mahal(S0, m0, nu0, &d20, NULL) < 0) {
            PyErr_SetString(PyExc_ValueError, "radar_update: a singular innovation covariance");
            Py_CLEAR(res);
            goto done;
        }
        for (int j = 0; j < 3; j++) {
            double K[RN * 4];
            double *x = row + RO_XS + j * RN, *P = row + RO_PS + j * RNN;
            double beta = pymax(comp * pd * like[j] / (pd * like[j] + other), beta_o);
            r_gain(P, Hs[j], m, Sis[j], K);
            r_take(x, P, K, Ss[j], nus[j], m, beta);
            if (j == R_STAND)
                for (int k = R_VX; k <= R_AY; k++)
                    x[k] = 0.0;
            x[R_DP] = clamp_dp(x[R_DP]);
        }
        /* the free estimate takes it the same way */
        {
            double nu[4], H[4 * RN], Sf[16], Si[16], det, K[RN * 4];
            double *x = row + RO_XF, *P = row + RO_PF;
            int mf = r_innov(&sen, kind, sdt, x, P, pose, d + 5, d + 9, nu, H, Sf);
            if (r_inv(Sf, mf, Si, &det) < 0) {
                PyErr_SetString(PyExc_ValueError, "radar_update: a singular innovation covariance");
                Py_CLEAR(res);
                goto done;
            }
            double t[4], d2 = 0.0;
            for (int c = 0; c < mf; c++) {
                double acc = 0.0;
                for (int r = 0; r < mf; r++)
                    acc = r ? acc + nu[r] * Si[r * mf + c] : nu[r] * Si[r * mf + c];
                t[c] = acc;
            }
            for (int c = 0; c < mf; c++)
                d2 = c ? d2 + t[c] * nu[c] : t[c] * nu[c];
            double lf = exp(-0.5 * pymin(d2, 400.0)) / sqrt(pymax(pow(2.0 * PI, (double)mf) * det, 1e-30));
            double beta = pymax(comp * pd * lf / (pd * lf + other), beta_o);
            r_gain(P, H, mf, Si, K);
            r_take(x, P, K, Sf, nu, mf, beta);
            x[R_DP] = clamp_dp(x[R_DP]);
        }
        /* the hypotheses weighed by how well each predicted the detection */
        double tot = (mu[0] * like[0] + mu[1] * like[1]) + mu[2] * like[2];
        double w[3];
        for (int j = 0; j < 3; j++)
            w[j] = mu[j] * (pd * like[j] + other);
        if (tot > 0.0 && (sen.type == SEN_MICROWAVE || hits >= cfg[0] || (kind & K_SCALE))) {
            double ws = (w[0] + w[1]) + w[2], nm[3];
            for (int j = 0; j < 3; j++) {
                double v = w[j] / ws;
                nm[j] = (isnan(v) || v >= cfg[1]) ? v : cfg[1];
            }
            double s2 = (nm[0] + nm[1]) + nm[2];
            for (int j = 0; j < 3; j++)
                mu[j] = nm[j] / s2;
        } else if (tot > 0.0 && w[1] + w[2] > 0.0) {
            double mv = mu[1] + mu[2], ws = w[1] + w[2];
            for (int j = 1; j < 3; j++) {
                double v = mv * w[j] / ws;
                mu[j] = (isnan(v) || v >= cfg[1]) ? v : cfg[1];
            }
            double s2 = (mu[0] + mu[1]) + mu[2];
            for (int j = 0; j < 3; j++)
                mu[j] = mu[j] / s2;
        }
        double like_all = pymax(tot, 1e-30);
        double dl = pymin(pymax(log(pymax(sen.p_d * like_all / r_density(&sen, kind, d + 5), 1e-9)), log(1.5)), log(20.0));
        r_combine(row + RO_XS, row + RO_PS, mu, row + RO_X, row + RO_P);
        PyObject *t = Py_BuildValue("(dd)", d20 / m0, dl);
        if (t == NULL) {
            Py_CLEAR(res);
            goto done;
        }
        PyList_SetItem(res, q, t);
    }
done:
    PyMem_Free(rows);
    PyMem_Free(buf);
    view_drop(&vS);
    return res;
}

/* ---- the motion model on the objects' rows ---- */
/* one object's hypotheses and free estimate after dt (Tracker._evolve): into xs1 (3N), Ps1 (3NN), c (3), and
 * the free estimate into xf1, Pf1 when given */
static void r_evolve_row(const double *row, double dt, const double *K, const RPar *par, double *xs1, double *Ps1,
                         double *c, double *xf1, double *Pf1)
{
    double T[9], xm[3 * RN], Pm[3 * RNN];
    const double *X = row + RO_XS, *P = row + RO_PS;
    double q0 = r_still(X + RN, P + RNN, K[3]), q1 = r_still(X + 2 * RN, P + 2 * RNN, K[3]);
    r_transition(dt, q0, q1, par, T);
    r_mix(X, P, row + RO_MU, T, row + RO_XF, row + RO_PF, xm, Pm, c);
    for (int j = 0; j < 3; j++)
        r_predict(j, xm + j * RN, Pm + j * RNN, dt, K[0], K[1], K[2], par, xs1 + j * RN, Ps1 + j * RNN);
    if (xf1)
        r_predict(R_CV, row + RO_XF, row + RO_PF, dt, K[0], K[1], K[2], par, xf1, Pf1);
}

/* radar_advance(S, rows, dt, kin, par, noise, pose) -> None. Tracker._advance_all for B objects, committed in
 * their rows: interaction, prediction (and the free estimate's), the probabilities only where dt > 0, what the
 * car's own movement is not sure about added to where each is (noise: (sa, sp) per object, pose (X, Y, psi) at
 * the target time; the object's place before the step), the combination. */
static PyObject *n_radar_advance(PyObject *self, PyObject *args)
{
    PyObject *oS, *orows, *odt, *okin, *opar, *onoise, *opose;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &oS, &orows, &odt, &okin, &opar, &onoise, &opose))
        return NULL;
    PyObject *res = NULL;
    View vS;
    vS.held = 0;
    Py_ssize_t cap, B = 0, *rows = NULL;
    double *dk = NULL, pose[3];
    RPar par;
    if (store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &B)) == NULL || rpar_get(opar, &par) < 0 ||
        nums_get(opose, 3, pose, "pose") < 0)
        goto done;
    dk = (double *)PyMem_Malloc((size_t)(B ? B : 1) * 7 * sizeof(double));
    if (dk == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    if (rows_get(odt, B, 1, dk, "dt") < 0 || rows_get(okin, B, 4, dk + B, "kin") < 0 ||
        rows_get(onoise, B, 2, dk + 5 * B, "noise") < 0)
        goto done;
    double *A = (double *)vS.b.buf;
    double ch = cos(pose[2]), sh = sin(pose[2]);
    for (Py_ssize_t b = 0; b < B; b++) {
        double *row = A + rows[b] * RROW;
        double dt = dk[b], sa = dk[5 * B + 2 * b], sp = dk[5 * B + 2 * b + 1];
        double xs1[3 * RN], Ps1[3 * RNN], c[3], xf1[RN], Pf1[RNN];
        r_evolve_row(row, dt, dk + B + 4 * b, &par, xs1, Ps1, c, xf1, Pf1);
        if (dt > 0.0)
            memcpy(row + RO_MU, c, sizeof(c));
        if (sa > 0.0 || sp > 0.0) {
            /* the car's travel error lies along its heading; its heading error swings the object about the car */
            double dx = row[RO_X + R_PX] - pose[0], dy = row[RO_X + R_PY] - pose[1];
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
        }
        memcpy(row + RO_XS, xs1, sizeof(xs1));
        memcpy(row + RO_PS, Ps1, sizeof(Ps1));
        memcpy(row + RO_XF, xf1, sizeof(xf1));
        memcpy(row + RO_PF, Pf1, sizeof(Pf1));
        r_combine(row + RO_XS, row + RO_PS, row + RO_MU, row + RO_X, row + RO_P);
    }
    res = Py_None;
    Py_INCREF(res);
done:
    PyMem_Free(rows);
    PyMem_Free(dk);
    view_drop(&vS);
    return res;
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
        r_evolve_row(row, dt, dk + B + 4 * b, &par, xs1, Ps1, muo + 3 * b, NULL, NULL);
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

/* radar_still(S, rows, v_still) -> [p_still] (Tracker.p_still_many on the objects' rows) */
static PyObject *n_radar_still(PyObject *self, PyObject *args)
{
    PyObject *oS, *orows, *ov;
    if (!PyArg_ParseTuple(args, "OOO", &oS, &orows, &ov))
        return NULL;
    PyObject *res = NULL;
    View vS;
    vS.held = 0;
    Py_ssize_t cap, B = 0, *rows = NULL;
    double *v0 = NULL;
    if (store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &B)) == NULL)
        goto done;
    v0 = (double *)PyMem_Malloc((size_t)(B ? B : 1) * sizeof(double));
    if (v0 == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    if (rows_get(ov, B, 1, v0, "v_still") < 0)
        goto done;
    res = PyList_New(B);
    if (res == NULL)
        goto done;
    double *A = (double *)vS.b.buf;
    for (Py_ssize_t b = 0; b < B; b++) {
        const double *row = A + rows[b] * RROW;
        const double *X = row + RO_XS, *P = row + RO_PS, *m = row + RO_MU;
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
    PyMem_Free(rows);
    PyMem_Free(v0);
    view_drop(&vS);
    return res;
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
    if (sens_get(osen, &sen) < 0 || nums_get(ox, RN, x, "x") < 0 || nums_get(opose, 6, pose, "pose") < 0 ||
        out_get(oz, &vz, 4, 0, &z, "z_out") < 0 || out_get(oH, &vH, 4 * RN, 0, &H, "H_out") < 0)
        goto done;
    res = PyLong_FromLong(r_h(&sen, kind, sdt, x, pose, z, H));
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
       RK_MOTORBIKE, RK_N };
static PyObject *rkey[RK_N + 1];
static const char *const RKEYS[RK_N + 1] = {
    "t_s", "x_m", "y_m", "std_m", "stand", "cv", "ca", "id", "state", "exist", "age_s", "hits", "sources", "seen",
    "vx_mps", "vy_mps", "ax_mps2", "ay_mps2", "speed_mps", "heading_rad", "v_rel_mps", "std", "motion",
    "p_stationary", "models", "class", "flags", "path_offset_m", "ttc_s", "p_threat", "consistency", "pred_err_m",
    "angle_offset_deg", "predict", "predict_path", "label", "confidence", "source", "votes", "on_path", "in_path",
    "threat", "parked", "with_traffic", "stationary", "human", "bike", "motorbike", NULL};

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
    double *buf = NULL, pose[3], ego[2], cfg[RFCFG], hz[8];
    RPar par;
    RPath path = {NULL, NULL, 0, 0.0, 0.0};
    if (rkeys_init() < 0 || store_get(oS, &vS, &cap) < 0 || (rows = idx_get(orows, cap, &B)) == NULL ||
        rpar_get(opar, &par) < 0 || nums_get(opose, 3, pose, "pose") < 0 || nums_get(oego, 2, ego, "ego") < 0 ||
        nums_get(ocfg, RFCFG, cfg, "cfg") < 0)
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
    if (nums_get(ohz, H, hz, "horizons") < 0)
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
        if (nums_get(ofc, RFLAGS, fc, "flag cfg") < 0)
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
            if (fabs(dt) < 1e-6) {
                memcpy(xo, row + RO_X, sizeof(double) * RN);
                memcpy(Po, row + RO_P, sizeof(double) * RNN);
                memcpy(mm, row + RO_MU, sizeof(mm));
            } else {
                double xs1[3 * RN], Ps1[3 * RNN];
                r_evolve_row(row, dt, kin + 4 * b, &par, xs1, Ps1, mm, NULL, NULL);
                r_combine(xs1, Ps1, mm, xo, Po);
            }
            if (h == 0)
                memcpy(mu, mm, sizeof(mm));
        }
        /* ground -> car axes */
        double dx = x[R_PX] - X, dy = x[R_PY] - Y;
        double pc0 = ch * dx + sh * dy, pc1 = -sh * dx + ch * dy;
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
            double q0 = ch * qx + sh * qy, q1 = -sh * qx + ch * qy;
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
        (has_pose && nums_get(opose, 2, pose, "pose") < 0))
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
                r_evolve_row(row, dt, kin + 4 * b, &par, xs1, Ps1, mm, NULL, NULL);
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

#define RMCFG 3                  /* merge_m, merge_rel, merge_dv */

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
        nums_get(oego, 2, ego, "ego") < 0 || nums_get(ocfg, RMCFG, cfg, "cfg") < 0)
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
                r_evolve_row(rb, tm[i] - tm[j], kin + 4 * j, &par, xs1, Ps1, mm, NULL, NULL);
                r_combine(xs1, Ps1, mm, xb, Pp);
            }
            double r = py_hypot(xa[R_PX] - here[2 * i], xa[R_PY] - here[2 * i + 1]);
            double lim = cfg[0] + cfg[1] * r;
            double dv = hypot(xb[R_VX] - xa[R_VX], xb[R_VY] - xa[R_VY]);
            double sv = sqrt(pymax((Pa[R_VX * RN + R_VX] + Pa[R_VY * RN + R_VY]) + (Pb[R_VX * RN + R_VX] + Pb[R_VY * RN + R_VY]), 0.0));
            int same = !(py_hypot(xb[R_PX] - xa[R_PX], xb[R_PY] - xa[R_PY]) > lim) && !(dv > cfg[2] + 2.0 * sv);
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

/* radar_check(S, row, back, px, py, Pp) -> (m2, e) | None. Tracker._check for one expectation: the object's
 * present estimate taken back by `back` seconds against where it said it would be then (px, py; Pp: the 2 x 2
 * covariance it gave): half the squared distance in both uncertainties (+ 0.05 m2 each way), and the distance.
 * None: the sum is singular. */
static PyObject *n_radar_check(PyObject *self, PyObject *args)
{
    PyObject *oS, *oPp;
    Py_ssize_t slot;
    double back, px, py, Pp[4];
    if (!PyArg_ParseTuple(args, "OndddO", &oS, &slot, &back, &px, &py, &oPp))
        return NULL;
    View vS;
    Py_ssize_t cap;
    if (store_get(oS, &vS, &cap) < 0)
        return NULL;
    PyObject *res = NULL;
    if (slot < 0 || slot >= cap) {
        PyErr_SetString(PyExc_IndexError, "radar: a row outside the state array");
        goto done;
    }
    if (nums_get(oPp, 4, Pp, "Pp") < 0)
        goto done;
    {
        const double *row = (const double *)vS.b.buf + slot * RROW, *x = row + RO_X, *P = row + RO_P;
        double ex = x[R_PX] - x[R_VX] * back - px, ey = x[R_PY] - x[R_VY] * back - py;
        double S[4] = {Pp[0] + P[0] + 0.05, Pp[1] + P[1] + 0.0, Pp[2] + P[RN] + 0.0, Pp[3] + P[RN + 1] + 0.05};
        double nu[2] = {ex, ey}, d2;
        if (r_mahal(S, 2, nu, &d2, NULL) < 0) {
            res = Py_None;
            Py_INCREF(res);
        } else
            res = Py_BuildValue("(dd)", d2 / 2.0, py_hypot(ex, ey));
    }
done:
    view_drop(&vS);
    return res;
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
            for (Py_ssize_t i = 0; i < n; i++) {         /* argsort, stable (numpy's for small arrays) */
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
            while (idx < n && cw[idx] < half)            /* np.searchsorted(cw, half): the first cw >= half */
                idx++;
            if (idx >= n)
                idx = n - 1;
            m = v[k[idx]];
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

/* ==================================================================================================== */
static PyMethodDef methods[] = {
    {"dumps", n_dumps, METH_O, "json.dumps(obj, separators=(',', ':')); raises Fallback where json must decide"},
    {"loads", n_loads, METH_O, "json.loads(bytes | str); raises Fallback where json must decide"},
    {"set_rounded_type", n_set_rounded_type, METH_O, "the tuple subclass (values, digits) dumps writes rounded"},
    {"round", n_round, METH_VARARGS, "builtins.round"},
    {"round_list", n_round_list, METH_VARARGS, "[v if v is None else round(v, n) for v in values]"},
    {"interp_all", n_interp_all, METH_VARARGS, "trackers.view._interp_all for lists of floats"},
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
    {"radar_evolve", n_radar_evolve, METH_VARARGS, "Tracker._evolve (+ combine_b, + the free estimate's prediction) for B objects"},
    {"radar_combine", n_radar_combine, METH_VARARGS, "combine_b for B objects into given arrays"},
    {"radar_p_still", n_radar_p_still, METH_VARARGS, "Tracker.p_still_many: a list of floats"},
    {"radar_advance", n_radar_advance, METH_VARARGS, "Tracker._advance_all on the objects' rows"},
    {"radar_peek", n_radar_peek, METH_VARARGS, "Tracker.peek_many on the objects' rows into given arrays"},
    {"radar_still", n_radar_still, METH_VARARGS, "Tracker.p_still_many on the objects' rows"},
    {"radar_h", n_radar_h, METH_VARARGS, "a sensor's measurement function (sensors.py h) into given arrays"},
    {"radar_marginals", n_radar_marginals, METH_VARARGS, "tracker.marginals into a given array"},
    {"radar_assign", n_radar_assign, METH_O, "tracker.assign"},
    {"radar_scan", n_radar_scan, METH_VARARGS, "one scan's association (Tracker.process): pairs, p_detect, coverage"},
    {"radar_update", n_radar_update, METH_VARARGS, "the update of a scan's pairs (Tracker._update's arithmetic)"},
    {"radar_frame", n_radar_frame, METH_VARARGS, "the output's per-object numbers (box.py Radar.frame)"},
    {"radar_life", n_radar_life, METH_VARARGS, "Tracker.manage's numbers: stillness, finiteness, out-of-view place"},
    {"radar_merge", n_radar_merge, METH_VARARGS, "Tracker._merge's geometry: the near pairs and their test"},
    {"radar_check", n_radar_check, METH_VARARGS, "Tracker._check for one expectation"},
    {"radar_register", n_radar_register, METH_VARARGS, "Tracker._register for one sensor: the shared offset, taken out"},
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
