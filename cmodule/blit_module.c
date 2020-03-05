#include <Python.h>
#include <numpy/arrayobject.h>
#include <pthread.h>

#define CHECK_PYARRAY(...)                           \
do {                                                 \
    void * _ptr[] = {__VA_ARGS__};                   \
    size_t _i;                                       \
    size_t _size = sizeof(_ptr) / sizeof(*_ptr);     \
    if (_size > 1 && _ptr[0] == NULL) {              \
        for (_i = 1; _i < _size; ++_i)               \
            Py_XDECREF(_ptr[_i]);                    \
        return NULL;                                 \
    }                                                \
} while(0);                                                                                        

typedef struct ThreadData {
    int32_t mask_size;
    uint8_t *im1_data, *im2_data, *mask_data, *out_data;
} ThreadData;

// Low Precision Linear Interp Op
inline void blit(const int32_t mask_size,
                 const uint8_t * const restrict im1_data,
                 const uint8_t * const restrict im2_data,
                 const uint8_t * const restrict mask_data,
                 uint8_t * const restrict out_data) {

    for (int32_t i = 0; i < mask_size; ++i) {
        int32_t ptr = 3 * i;
        if (mask_data[i] == 0) {
            memcpy(out_data + ptr, im2_data + ptr, 3);
        } else if (mask_data[i] == 255) {
            memcpy(out_data + ptr, im1_data + ptr, 3);
        } else {
            for (int32_t j = 0; j < 3; ++j) {
                int32_t ptr_c = ptr + j;
                uint16_t i0 = (uint16_t)mask_data[i] * im1_data[ptr_c];
                uint16_t i1 = (uint16_t)(255 - mask_data[i]) * im2_data[ptr_c];
                out_data[ptr_c] = (uint8_t)((i0 + i1) >> 8);
            }
        }
    }
}

void* blit_slice(void * args) {

    ThreadData* td = args;
    
    int32_t mask_size  = td->mask_size;
    uint8_t *im1_data  = td->im1_data;
    uint8_t *im2_data  = td->im2_data;
    uint8_t *mask_data = td->mask_data;
    uint8_t *out_data  = td->out_data;

    blit(mask_size, im1_data, im2_data, mask_data, out_data);

    return NULL;
}

// Single Thread Implementation
static PyObject* fast_blit(PyObject* self, PyObject* args) {
    
    PyObject      *im1;      
    PyObject      *im2;
    PyObject      *mask;
    PyObject      *out;
    PyArrayObject *im1_array;
    PyArrayObject *im2_array;
    PyArrayObject *mask_array;

    if (!PyArg_ParseTuple(args, "OOO", &im1, &im2, &mask))
        return NULL;

    out = NULL;

    im1_array  = (PyArrayObject *)PyArray_FROM_OTF(im1, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    CHECK_PYARRAY(im1_array)
    im2_array  = (PyArrayObject *)PyArray_FROM_OTF(im2, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    CHECK_PYARRAY(im2_array, im1_array)
    mask_array = (PyArrayObject *)PyArray_FROM_OTF(mask, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    CHECK_PYARRAY(mask_array, im1_array, im2_array)

    npy_intp *shape = PyArray_DIMS(im1_array);
    int32_t ndim = PyArray_NDIM(im1_array);
    int32_t size = (int32_t) PyArray_SIZE(im1_array);
    int32_t mask_size = (int32_t) PyArray_SIZE(mask);

    if (size != mask_size * 3)
        return NULL;

    out = PyArray_SimpleNew(ndim, shape, NPY_UINT8);
    if (out == NULL)
        return NULL;

    uint8_t *im1_data  = (uint8_t *) PyArray_DATA(im1_array);
    uint8_t *im2_data  = (uint8_t *) PyArray_DATA(im2_array);
    uint8_t *mask_data = (uint8_t *) PyArray_DATA(mask_array);
    uint8_t *out_data  = (uint8_t *) PyArray_DATA((PyArrayObject *) out);   

    Py_BEGIN_ALLOW_THREADS

    blit(mask_size, im1_data, im2_data, mask_data, out_data);

    Py_END_ALLOW_THREADS

    Py_INCREF(out);
    return out;

    fail:
        Py_XDECREF(out);
        return NULL;
}

// Dual Threads Implementation (WIP)
static PyObject* fast_blit2(PyObject* self, PyObject* args) {
    
    PyObject      *im1;      
    PyObject      *im2;
    PyObject      *mask;
    PyObject      *out;
    PyArrayObject *im1_array;
    PyArrayObject *im2_array;
    PyArrayObject *mask_array;

    if (!PyArg_ParseTuple(args, "OOO", &im1, &im2, &mask))
        return NULL;


    out = NULL;
    pthread_t thread0, thread1;

    im1_array  = (PyArrayObject *)PyArray_FROM_OTF(im1, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    CHECK_PYARRAY(im1_array)
    im2_array  = (PyArrayObject *)PyArray_FROM_OTF(im2, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    CHECK_PYARRAY(im2_array, im1_array)
    mask_array = (PyArrayObject *)PyArray_FROM_OTF(mask, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    CHECK_PYARRAY(mask_array, im1_array, im2_array)

    npy_intp *shape = PyArray_DIMS(im1_array);
    int32_t ndim = PyArray_NDIM(im1_array);
    int32_t size = (int32_t) PyArray_SIZE(im1_array);
    int32_t mask_size = (int32_t) PyArray_SIZE(mask);

    if (size != mask_size * 3)
        return NULL;

    out = PyArray_SimpleNew(ndim, shape, NPY_UINT8);

    Py_BEGIN_ALLOW_THREADS

    ThreadData td0;
    td0.im1_data  = (uint8_t *) PyArray_DATA(im1_array);
    td0.im2_data  = (uint8_t *) PyArray_DATA(im2_array);
    td0.mask_data = (uint8_t *) PyArray_DATA(mask_array);
    td0.out_data  = (uint8_t *) PyArray_DATA((PyArrayObject *) out);    
    td0.mask_size = mask_size >> 1;

    pthread_create(&thread0, NULL, blit_slice, (void*)&td0);

    ThreadData td1;
    td1.im1_data  = td0.im1_data + (size >> 1);
    td1.im2_data  = td0.im2_data + (size >> 1);
    td1.mask_data = td0.mask_data + (mask_size >> 1);
    td1.out_data  = td0.out_data + (size >> 1);
    td1.mask_size = mask_size - (mask_size >> 1);

    pthread_create(&thread1, NULL, blit_slice, (void*)&td1);

    pthread_join(thread0, NULL);
    pthread_join(thread1, NULL);

    Py_END_ALLOW_THREADS

    Py_INCREF(out);
    return out;

    fail:
        Py_XDECREF(out);
        return NULL;
}

static PyMethodDef FastBlit[] = {
    {"fast_blit", fast_blit, METH_VARARGS, "fast blit"},
    {"fast_blit2", fast_blit2, METH_VARARGS, "fast blit2"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem = {
    PyModuleDef_HEAD_INIT,
    "fast_blit", "Low Precision Blit Operator",
    -1,
    FastBlit,
};

PyMODINIT_FUNC PyInit_blit_module(void) {
    PyObject *module;
    module = PyModule_Create(&cModPyDem);
    if (module == NULL)
        return NULL;

    import_array();
    if (PyErr_Occurred())
        return NULL;

    return module;
}
