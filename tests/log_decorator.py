import logging
import functools

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def log_function_call(func):
    """Decorator to log function calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_args = ', '.join([str(a) for a in args] + [f"{k}={v}" for k, v in kwargs.items()])
        logging.debug(f"Calling {func.__name__}({func_args})")
        result = func(*args, **kwargs)
        logging.debug(f"{func.__name__} returned {result}")
        return result
    return wrapper

def decorate_all_functions(module):
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and not name.startswith("__"):
            setattr(module, name, log_function_call(obj))
