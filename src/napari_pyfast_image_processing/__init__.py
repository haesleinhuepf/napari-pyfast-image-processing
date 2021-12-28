

__version__ = "0.0.1"

import napari
import fast
import numpy as np
from toolz import curry
from functools import wraps
from typing import Callable
import inspect
from napari_tools_menu import register_function
from napari_time_slicer import time_slicer
from napari_plugin_engine import napari_hook_implementation


@napari_hook_implementation
def napari_experimental_provide_function():
    return [gaussian_blur, image_gradient, laplacian_of_gaussian_2D, image_sharpening_2D, dilation, erosion]


def _fast_to_numpy(fast_image):
    return np.asarray(fast_image)


def _numpy_to_fast(numpy_image):
    return fast.Image.createFromArray(numpy_image)


@curry
def plugin_function(
        function: Callable,
        convert_input_to_float: bool = False,
        convert_input_to_uint8: bool = False
) -> Callable:
    # copied from https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/pyclesperanto_prototype/_tier0/_plugin_function.py
    @wraps(function)
    def worker_function(*args, **kwargs):
        sig = inspect.signature(function)
        # create mapping from position and keyword arguments to parameters
        # will raise a TypeError if the provided arguments do not match the signature
        # https://docs.python.org/3/library/inspect.html#inspect.Signature.bind
        bound = sig.bind(*args, **kwargs)
        # set default values for missing arguments
        # https://docs.python.org/3/library/inspect.html#inspect.BoundArguments.apply_defaults
        bound.apply_defaults()

        input_shape = None

        # copy images to pyFAST-types, and create output array if necessary
        for key, value in bound.arguments.items():
            np_value = None
            if isinstance(value, np.ndarray):
                np_value = value
                input_shape = np_value.shape

            elif 'pyclesperanto_prototype._tier0._pycl.OCLArray' in str(type(value)) or \
                    'dask.array.core.Array' in str(type(value)):
                # compatibility with pyclesperanto and dask
                np_value = np.asarray(value)

            if convert_input_to_float and np_value is not None:
                np_value = np_value.astype(float)
            if convert_input_to_uint8 and np_value is not None:
                np_value = np_value.astype(np.uint8)

            if np_value is not None:
                if np_value.dtype == bool:
                    np_value = np_value * 1
                bound.arguments[key] = _numpy_to_fast(np_value)

        # call the decorated function
        result = function(*bound.args, **bound.kwargs)

        if isinstance(result, fast.fast.Image):
            result = _fast_to_numpy(result)

        if input_shape is not None and result is not None:
            if len(input_shape) < len(result.shape):
                result = result[..., 0]

        return result

    worker_function.__module__ = "napari_pyfast_image_processing"

    return worker_function

@register_function(menu="Filtering / noise removal > Gaussian blur (npyFAST)")
@time_slicer
@plugin_function
def gaussian_blur(image: napari.types.ImageData, standard_deviation: float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    filter_obj = fast.GaussianSmoothing.create()
    filter_obj.setInputData(image)
    filter_obj.setStandardDeviation(float(standard_deviation))
    return filter_obj.runAndGetOutputData()


@register_function(menu="Filtering / edge enhancement > Image gradient (npyFAST)")
@time_slicer
@plugin_function
def image_gradient(image: napari.types.ImageData, viewer: napari.Viewer = None) -> napari.types.ImageData:
    filter_obj = fast.ImageGradient.create()
    filter_obj.setInputData(image)
    return filter_obj.runAndGetOutputData()


@register_function(menu="Filtering / edge enhancement > Laplacian of Gaussian (2D only, npyFAST)")
@time_slicer
@plugin_function
def laplacian_of_gaussian_2D(image: napari.types.ImageData, standard_deviation:float = 2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    filter_obj = fast.LaplacianOfGaussian.create()
    filter_obj.setInputData(image)
    filter_obj.setStandardDeviation(float(standard_deviation))
    return filter_obj.runAndGetOutputData()


@register_function(menu="Filtering / edge enhancement > Unsharp mask (2D only, npyFAST)")
@time_slicer
@plugin_function
def image_sharpening_2D(image: napari.types.ImageData, gain:float=2, viewer: napari.Viewer = None) -> napari.types.ImageData:
    filter_obj = fast.ImageSharpening.create()
    filter_obj.setInputData(image)
    filter_obj.setGain(float(gain))
    return filter_obj.runAndGetOutputData()


@register_function(menu="Segmentation post-processing > Binary dilation (npyFAST)")
@time_slicer
@plugin_function(convert_input_to_uint8=True)
def dilation(image: napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    filter_obj = fast.Dilation.create()
    filter_obj.setInputData(image)
    return filter_obj.runAndGetOutputData()


@register_function(menu="Segmentation post-processing > Binary erosion (npyFAST)")
@time_slicer
@plugin_function(convert_input_to_uint8=True)
def erosion(image: napari.types.LabelsData, viewer: napari.Viewer = None) -> napari.types.LabelsData:
    filter_obj = fast.Erosion.create()
    filter_obj.setInputData(image)
    return filter_obj.runAndGetOutputData()





