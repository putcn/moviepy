// Dev

#include <Python.h>
#include <numpy/arrayobject.h>

#define NULL_CHECK(val) if (val == NULL) return NULL;

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
	im2_array  = (PyArrayObject *)PyArray_FROM_OTF(im2, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
	mask_array = (PyArrayObject *)PyArray_FROM_OTF(mask, NPY_UINT8, NPY_ARRAY_IN_ARRAY);

    npy_intp *shape = PyArray_DIMS(im1_array);
    int32_t ndim = PyArray_NDIM(im1_array);
    int32_t size = (int32_t) PyArray_SIZE(im1_array);
    int32_t mask_size = (int32_t) PyArray_SIZE(mask);

    if (size != mask_size * 3)
		goto fail;

	out = PyArray_SimpleNew(ndim, shape, NPY_UINT8);
	NULL_CHECK(out)

	uint8_t *im1_data  = (uint8_t *) PyArray_DATA(im1_array);
	uint8_t *im2_data  = (uint8_t *) PyArray_DATA(im2_array);
	uint8_t *mask_data = (uint8_t *) PyArray_DATA(mask_array);
	uint8_t *out_data  = (uint8_t *) PyArray_DATA((PyArrayObject *) out);	

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

	Py_INCREF(out);
	return out;

	fail:
		Py_XDECREF(out);
		return NULL;
}

static PyMethodDef FastBlit[] = {
	{"fast_blit", fast_blit, METH_VARARGS, "fast blit"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem = {
	PyModuleDef_HEAD_INIT,
	"fast_blit", "docs...",
	-1,
	FastBlit
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
