import logging

_logger = logging.getLogger(__name__)


class UniqueKeyDict(dict):
    def __init__(self, reload=False, **kwargs):
        super().__init__(**kwargs)
        self.reload = reload

    def __setitem__(self, key, *args, **kwargs):
        if key in self:
            raise KeyError("Key '{}' already exists with value '{}'.".format(key, self[key]))
        super(UniqueKeyDict, self).__setitem__(key, *args, **kwargs)

    def __getitem__(self, item):
        value = super(UniqueKeyDict, self).__getitem__(item)
        if self.reload:
            _logger.warning(f'{item} is accessed again and reloaded')
            value.reload()
        return value




