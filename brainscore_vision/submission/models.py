from peewee import Model as PeeweeModel, Proxy, CharField, ForeignKeyField, IntegerField, BooleanField, DateTimeField, \
    FloatField, TextField, PrimaryKeyField

database = Proxy()


class PeeweeBase(PeeweeModel):
    class Meta:
        database = database


class Reference(PeeweeBase):
    author = CharField()
    bibtex = TextField()
    url = CharField()
    year = IntegerField()

    class Meta:
        table_name = 'brainscore_reference'
        schema = 'public'


class BenchmarkType(PeeweeBase):
    identifier = CharField(primary_key=True)
    reference = ForeignKeyField(column_name='reference_id', field='id', model=Reference)
    order = IntegerField()
    parent = ForeignKeyField(column_name='parent_id', field='identifier', model='self', null=True)
    visible = BooleanField(default=False, null=False)
    domain = CharField(max_length=200, default="vision")

    class Meta:
        table_name = 'brainscore_benchmarktype'
        schema = 'public'


class BenchmarkMeta(PeeweeBase):
    number_of_images = IntegerField(null=True)
    number_of_recording_sites = IntegerField(null=True)
    recording_sites = CharField(max_length=100, null=True)
    behavioral_task = CharField(max_length=100, null=True)

    class Meta:
        table_name = 'brainscore_benchmarkmeta'
        schema = 'public'


class BenchmarkInstance(PeeweeBase):
    benchmark = ForeignKeyField(column_name='benchmark_type_id', field='identifier', model=BenchmarkType)
    ceiling = FloatField(null=True)
    ceiling_error = FloatField(null=True)
    version = IntegerField(null=True)
    meta = ForeignKeyField(model=BenchmarkMeta)

    class Meta:
        table_name = 'brainscore_benchmarkinstance'
        schema = 'public'


class User(PeeweeBase):
    email = CharField(index=True, null=True)
    is_active = BooleanField()
    is_staff = BooleanField()
    is_superuser = BooleanField()
    last_login = DateTimeField(null=True)
    password = CharField()

    class Meta:
        table_name = 'brainscore_user'
        schema = 'public'


class Submission(PeeweeBase):
    id = PrimaryKeyField()
    # IDs will not be incremental when resubmitting models
    submitter = ForeignKeyField(column_name='submitter_id', field='id', model=User)
    timestamp = DateTimeField(null=True)
    model_type = CharField()
    status = CharField()

    # equivalent to ID until language changes were added: (ID 6756)
    jenkins_id = IntegerField()

    class Meta:
        table_name = 'brainscore_submission'
        schema = 'public'


class Model(PeeweeBase):
    name = CharField()
    owner = ForeignKeyField(column_name='owner_id', field='id', model=User)
    reference = ForeignKeyField(column_name='reference_id', field='id', model=Reference)
    submission = ForeignKeyField(column_name='submission_id', field='id', model=Submission)
    visual_degrees = IntegerField(null=True)  # null during creation of new model without having model object loaded
    public = BooleanField()
    competition = CharField(max_length=200, default=None, null=True)
    domain = CharField(max_length=200, default="vision")

    class Meta:
        table_name = 'brainscore_model'
        schema = 'public'


class Score(PeeweeBase):
    benchmark = ForeignKeyField(column_name='benchmark_id', field='id', model=BenchmarkInstance)
    end_timestamp = DateTimeField(null=True)
    error = FloatField(null=True)
    model = ForeignKeyField(column_name='model_id', field='id', model=Model)
    score_ceiled = FloatField(null=True)
    score_raw = FloatField(null=True)
    start_timestamp = DateTimeField(null=True)
    comment = CharField(null=True)

    class Meta:
        table_name = 'brainscore_score'
        schema = 'public'
