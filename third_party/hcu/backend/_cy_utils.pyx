# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

from cpython.tuple cimport PyTuple_GET_ITEM
from cpython.bytes cimport PyBytes_AS_STRING


def binder(object tensor_cls, tuple names, bytes is_constexpr_buf, bytes param_has_default,
           tuple param_default, tuple args, dict kwargs, set dns_set, bint auto_dns,
           long dns_threshold, dict value_history, set flagged):
    cdef Py_ssize_t n_args = len(args)
    cdef Py_ssize_t n_params = len(names)
    cdef Py_ssize_t i
    cdef object name, v, dt, s
    cdef const char *cbuf = PyBytes_AS_STRING(is_constexpr_buf)
    cdef const char *has_default_buf = PyBytes_AS_STRING(is_constexpr_buf)
    cdef bint is_tensor, already_dns

    cdef dict bound_args = {}
    cdef list non_constexpr_vals = []
    cdef list key = []
    cdef object ap = key.append
    cdef list newly_flagged = []

    for i in range(n_args):
        name = <object>PyTuple_GET_ITEM(names, i)
        v = <object>PyTuple_GET_ITEM(args, i)
        bound_args[name] = v
        if cbuf[i] == 0:
            non_constexpr_vals.append(v)
        else:
            ap(v)
            continue

        already_dns = name in dns_set
        if already_dns:
            continue

        if auto_dns:
            is_tensor = isinstance(v, tensor_cls)
            if not is_tensor:
                s = value_history.get(name)
                if s is None:
                    s = set()
                    value_history[name] = s
                s.add(v)
                if len(s) >= dns_threshold:
                    flagged.add(name)
                    dns_set.add(name)
                    newly_flagged.append(name)
                else:
                    ap(v)
            else:
                ap(getattr(v, "dtype"))
        else:
            dt = getattr(v, "dtype", None)
            ap(v if dt is None else dt)

    for i in range(n_args, n_params):
        name = <object>PyTuple_GET_ITEM(names, i)
        if name in kwargs:
            v = kwargs[name]
        elif has_default_buf[i] == 1:
            v = <object>PyTuple_GET_ITEM(param_default, i)
        else:
            raise TypeError(f"KeyError: {name!r}")
        bound_args[name] = v
        if cbuf[i] == 0:
            non_constexpr_vals.append(v)
        else:
            ap(v)
            continue

        already_dns = name in dns_set
        if already_dns:
            continue

        if auto_dns:
            is_tensor = isinstance(v, tensor_cls)
            if not is_tensor:
                s = value_history.get(name)
                if s is None:
                    s = set()
                    value_history[name] = s
                s.add(v)
                if len(s) >= dns_threshold:
                    flagged.add(name)
                    dns_set.add(name)
                    newly_flagged.append(name)
                else:
                    ap(v)
            else:
                ap(getattr(v, "dtype"))
        else:
            dt = getattr(v, "dtype", None)
            ap(v if dt is None else dt)

    if len(kwargs) > (n_params - n_args):
        for name, v in kwargs.items():
            if name in names:
                continue
            ap(f"{name}={v}")

    return bound_args, non_constexpr_vals, tuple(key), newly_flagged
