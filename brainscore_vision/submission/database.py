import json
import logging

from peewee import PostgresqlDatabase, SqliteDatabase

from brainscore_vision.submission.models import database
from brainscore_vision.submission.utils import get_secret


def connect_db(db_secret):
    if 'sqlite3' not in db_secret:
        secret = get_secret(db_secret)
        db_configs = json.loads(secret)
        postgres = PostgresqlDatabase(db_configs['dbInstanceIdentifier'],
                                      **{'host': db_configs['host'], 'port': 5432,
                                         'user': db_configs['username'], 'password': db_configs['password']})
        database.initialize(postgres)
    else:
        sqlite = SqliteDatabase(db_secret)
        database.initialize(sqlite)
