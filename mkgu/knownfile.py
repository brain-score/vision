import datetime
import os
import hashlib
import peewee


_db = None


def get_db():
    global _db
    if _db is None:
        db_file = os.path.join(os.path.dirname(__file__), "knownfile.db")
        _db = peewee.SqliteDatabase(db_file)
    return _db


def hash_a_file(path, buffer_size=64*2**10):
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        buffer = f.read(buffer_size)
        while len(buffer) > 0:
            sha1.update(buffer)
            buffer = f.read(buffer_size)
    return sha1.hexdigest()



class FileRecord(peewee.Model):
    sha1 = peewee.CharField(unique=True)

    class Meta:
        database = get_db()


class Sighting(peewee.Model):
    location = peewee.CharField()
    file_record = peewee.ForeignKeyField(FileRecord, backref="sightings")
    stamp = peewee.DateTimeField(default=datetime.datetime.now)

    class Meta:
        database = get_db()


class KnownFile(object):
    def __init__(self, path):
        self.path = path
        self.exists = os.path.exists(path)
        if self.exists:
            self.realpath = os.path.realpath(path)
            self.isfile = os.path.isfile(path)
            if self.isfile:
                self.sha1 = hash_a_file(path)
                self.file_record, fr_created = FileRecord.get_or_create(sha1=self.sha1)
                self.sighting, s_created = Sighting.get_or_create(location=self.path, file_record=self.file_record)

            # self.sighting.save()


get_db().create_tables([FileRecord, Sighting])
