from peewee import Model as PeeweeModel, Proxy, CharField, ForeignKeyField, IntegerField, BooleanField, DateTimeField, \
    FloatField, TextField, PrimaryKeyField

database = Proxy()


class BaseModel(PeeweeModel):
    class Meta:
        database = database


class Reference(BaseModel):
    author = CharField()
    bibtex = TextField()
    url = CharField()
    year = IntegerField()

    class Meta:
        table_name = 'brainscore_reference'
        schema = 'public'


class BenchmarkType(BaseModel):
    identifier = CharField(primary_key=True)
    reference = ForeignKeyField(column_name='reference_id', field='id', model=Reference)
    order = IntegerField()
    parent = ForeignKeyField(column_name='parent_id', field='identifier', model='self', null=True)
    visible = BooleanField(default=False, null=False)

    class Meta:
        table_name = 'brainscore_benchmarktype'
        schema = 'public'


class BenchmarkInstance(BaseModel):
    benchmark = ForeignKeyField(column_name='benchmark_type_id', field='identifier', model=BenchmarkType)
    ceiling = FloatField(null=True)
    ceiling_error = FloatField(null=True)
    version = IntegerField(null=True)

    class Meta:
        table_name = 'brainscore_benchmarkinstance'
        schema = 'public'


class User(BaseModel):
    email = CharField(index=True, null=True)
    is_active = BooleanField()
    is_staff = BooleanField()
    is_superuser = BooleanField()
    last_login = DateTimeField(null=True)
    password = CharField()

    class Meta:
        table_name = 'brainscore_user'
        schema = 'public'


class Submission(BaseModel):
    id = PrimaryKeyField()  # We use jenkins id as id for the submission.
    # IDs will not be incremental when resubmitting models
    submitter = ForeignKeyField(column_name='submitter_id', field='id', model=User)
    timestamp = DateTimeField(null=True)
    model_type = CharField()
    status = CharField()

    class Meta:
        table_name = 'brainscore_submission'
        schema = 'public'


class Model(BaseModel):
    name = CharField()
    owner = ForeignKeyField(column_name='owner_id', field='id', model=User)
    reference = ForeignKeyField(column_name='reference_id', field='id', model=Reference)
    submission = ForeignKeyField(column_name='submission_id', field='id', model=Submission)
    visual_degrees = IntegerField(null=True)  # null during creation of new model without having model object loaded
    public = BooleanField()

    class Meta:
        table_name = 'brainscore_model'
        schema = 'public'


class Score(BaseModel):
    benchmark = ForeignKeyField(column_name='benchmark_id', field='id', model=BenchmarkInstance)
    end_timestamp = DateTimeField(null=True)
    error = FloatField(null=True)
    model = ForeignKeyField(column_name='model_id', field='id', model=Model)
    score_ceiled = FloatField(null=True)
    score_raw = FloatField(null=True)
    start_timestamp = DateTimeField(null=True)
    comment = CharField(null=True)

    # jenkins_id = IntegerField(null=True)

    class Meta:
        table_name = 'brainscore_score'
        schema = 'public'
