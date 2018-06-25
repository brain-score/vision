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
