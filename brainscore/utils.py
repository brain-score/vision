import copy


def fullname(obj):
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
