from functools import wraps
import inspect

import numpy
import cupy

from cupy.testing import product  # NOQA


def parameterize(keys, args, _head=False):
    def f(klass):
        param_names = keys
        params = [
            [a[name] if name in a else None for a in args]
            for name in param_names]

        orig_param_names = getattr(klass, 'param_names', [])
        orig_params = getattr(klass, 'params', [])

        if orig_params:
            if not isinstance(orig_params[0], (tuple, list)):
                orig_params = [orig_params]
                if len(orig_param_names) == 0:
                    orig_param_names = ['param']
                assert len(orig_param_names) == 1
        else:
            assert len(orig_param_names) == 0

        if _head:
            params += orig_params
            param_names += orig_param_names
        else:
            params = orig_params + params
            param_names = orig_param_names + param_names

        assert len(params) == len(param_names)

        klass.params = params
        klass.param_names = param_names

        return klass

    return f


def gpu(func):
    """Annotation to perform synchronization after running benchmark."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        stream = cupy.cuda.stream.get_current_stream()
        func(self, *args, **kwargs)
        stream.synchronize()

    return wrapper


def xp(klass):
    """Class annotation to compare performance between NumPy/CuPy."""

    # Append parameter axis.
    klass = parameterize(
        ['xp'],
        product({'xp': ['numpy', 'cupy']}),
        _head=True,
    )(klass)

    # Overwrite parameter values ('numpy', 'cupy') to actual module refernece.
    members = inspect.getmembers(
        klass,
        predicate=lambda _: inspect.ismethod(_) or inspect.isfunction(_))

    for (name, method) in members:
        if not (name == 'setup' or name.startswith('time_')): continue

        def _wrap_method(method):
            @wraps(method)
            def wrapper(self, xp, *args, **kwargs):
                if xp == 'numpy':
                    xp = numpy
                    stream = None
                elif xp == 'cupy':
                    xp = cupy
                    stream = cupy.cuda.stream.get_current_stream()
                else:
                    raise AssertionError()

                method(self, xp, *args, **kwargs)

                if stream:
                    stream.synchronize()

            return wrapper

        setattr(klass, name, _wrap_method(method))

    return klass
