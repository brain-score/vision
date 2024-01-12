import inspect

from result_caching import is_iterable


def fullname(obj):
    module = obj.__module__
    name = obj.__name__ if inspect.isfunction(obj) else obj.__class__.__name__
    return module + "." + name


def make_list(element):
    if not is_iterable(element) or isinstance(element, (str, bytes)):
        return [element]
    return element
