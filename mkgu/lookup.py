import os
import peewee


pwdb = peewee.SqliteDatabase(os.path.join(os.path.dirname(__file__), "lookup.db"))
