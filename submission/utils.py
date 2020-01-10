class UniqueKeyDict(dict):
    def __setitem__(self, key, *args, **kwargs):
        if key in self:
            raise KeyError("Key '{}' already exists with value '{}'.".format(key, self[key]))
        super(UniqueKeyDict, self).__setitem__(key, *args, **kwargs)
