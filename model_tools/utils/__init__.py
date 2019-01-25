import inspect


def fullname(obj):
    module = obj.__module__
    name = obj.__name__ if inspect.isfunction(obj) else obj.__class__.__name__
    return module + "." + name
