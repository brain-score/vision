import os
import peewee


pwdb = peewee.SqliteDatabase(os.path.join(os.path.dirname(__file__), "lookup.db"))


class Lookup(object):
    """A Lookup gives an abstraction layer for finding out where a DataAssembly can be fetched from.  """
    def __init__(self):
        pass

    def lookup_assembly(self, name):
        pass


class SQLiteLookup(Lookup):
    """A Lookup that uses a local SQLite file.  """
    def __init__(self, db_file=None):
        super(SQLiteLookup, self).__init__()

    def lookup_assembly(self, name):
        pwdb.connect(reuse_if_open=True)
        return AssemblyModel.get(AssemblyModel.name == name)


class PostgreSQLLookup(Lookup):
    """A Lookup that uses a Postgres database.  """
    def __init__(self):
        super(PostgreSQLLookup, self).__init__()


class WebServiceLookup(Lookup):
    """A Lookup that uses a web service.  """
    def __init__(self):
        super(WebServiceLookup, self).__init__()


class AssemblyModel(peewee.Model):
    """An AssemblyModel stores information about the canonical location where the data
    for a DataAssembly is stored.  """
    name = peewee.CharField()
    assembly_class = peewee.CharField()

    class Meta:
        database = pwdb


class AssemblyStoreModel(peewee.Model):
    """An AssemblyStoreModel stores the location of a DataAssembly data file.  """
    assembly_type = peewee.CharField()
    location_type = peewee.CharField()
    location = peewee.CharField()

    class Meta:
        database = pwdb


class AssemblyStoreMap(peewee.Model):
    """An AssemblyStoreMap links an AssemblyRecord to an AssemblyStore.  """
    assembly_model = peewee.ForeignKeyField(AssemblyModel, backref="assembly_store_maps")
    assembly_store_model = peewee.ForeignKeyField(AssemblyStoreModel, backref="assembly_store_maps")

    class Meta:
        database = pwdb


class AssemblyLookupError(Exception):
    pass


_lookup_types = {
    "SQLite": SQLiteLookup,
    "PostgreSQL": PostgreSQLLookup,
    "WebService": WebServiceLookup
}


def get_lookup(type="SQLite"):
    return _lookup_types[type]()
