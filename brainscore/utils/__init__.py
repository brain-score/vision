"""
Provide generic helper classes.
"""

import copy


def fullname(obj):
    """ Resolve the full module-qualified name of an object. Typically used for logger naming. """
    return obj.__module__ + "." + obj.__class__.__name__


def map_fields(obj, func):
    for field_name, field_value in vars(obj).items():
        field_value = func(field_value)
        setattr(obj, field_name, field_value)


def combine_fields(objs, func):
    if len(objs) == 0:
        return objs
    fields = list(vars(objs[0]).keys())
    field_values = {field_name: [] for field_name in fields}
    for obj in objs:
        for field_name in fields:
            field_value = getattr(obj, field_name)
            field_values[field_name].append(field_value)
    field_values = dict(map(
        lambda field_name_values: (field_name_values[0], func(field_name_values[1])), field_values.items()))
    ctr = objs[0].__class__
    return ctr(**field_values)


def recursive_dict_merge(dict1, dict2):
    """
    Merges dictionaries (of dictionaries).
    Preference is given to the second dict, i.e. if a key occurs in both dicts, the value from `dict2` is used.
    """
    result = copy.deepcopy(dict1)
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            result[key] = recursive_dict_merge(dict1[key], dict2[key])
        else:
            result[key] = dict2[key]
    return result


class LazyLoad:
    def __init__(self, load_fnc):
        self.load_fnc = load_fnc
        self.content = None

    def __getattr__(self, name):
        self._ensure_loaded()
        return getattr(self.content, name)

    def __setattr__(self, key, value):
        if key not in ['content', 'load_fnc']:
            self._ensure_loaded()
            return setattr(self.content, key, value)
        return super(LazyLoad, self).__setattr__(key, value)

    def __getitem__(self, item):
        self._ensure_loaded()
        return self.content.__getitem__(item)

    def __setitem__(self, key, value):
        self._ensure_loaded()
        return self.content.__setitem__(key, value)

    def _ensure_loaded(self):
        if self.content is None:
            self.content = self.load_fnc()

    def reload(self):
        self.content = self.load_fnc()

    def __call__(self, *args, **kwargs):
        self._ensure_loaded()
        return self.content(*args, **kwargs)

    def __len__(self):
        self._ensure_loaded()
        return len(self.content)

    @property
    def __class__(self):
        self._ensure_loaded()
        return self.content.__class__
