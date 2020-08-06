import datetime
import distutils

from brainscore.submission.models import Model, Submission


class BaseConfig:
    """
    Base configuration class, containing properties needed for every submission.
    """

    def __init__(self, work_dir, jenkins_id, db_secret, config_path):
        self.work_dir = work_dir
        self.jenkins_id = jenkins_id
        self.config_path = config_path
        self.db_secret = db_secret


class MultiConfig(BaseConfig):
    """
        Configuration class for rerunning models.
        A user can submit only a list of model id's and a benchmark to run new benchmarks or updated versions.
        The configuration preselects the required submissions based on database entries.
    """

    def __init__(self, model_ids, **kwargs):
        super(MultiConfig, self).__init__(**kwargs)
        self.models = []
        self.submission_entries = {}
        for id in model_ids:
            model = Model.get(id=id)
            submission: Submission = model.submission
            self.models.append(model)
            if submission.id not in self.submission_entries:
                self.submission_entries[submission.id] = submission


class SubmissionConfig(BaseConfig):
    """
    Configuration properties for newly submitted models, which also have a submission entry in database.
    """

    def __init__(self, model_type, user_id, jenkins_id, public, **kwargs):
        super(SubmissionConfig, self).__init__(jenkins_id=jenkins_id, **kwargs)
        self.submission = Submission.create(id=jenkins_id, submitter=user_id, timestamp=datetime.datetime.now(),
                                            model_type=model_type, status='running')
        self.public = public


def object_decoder(config, work_dir, config_path, db_secret, jenkins_id):
    """
    This method takes a bunch of input configurations from console flags and
    configuration json and bundles them depending on the time of submission in a class object.
    This should help to better understand which properties are available and reformats properly.
    """
    if 'model_ids' in config:
        return MultiConfig(model_ids=config['model_ids'], work_dir=work_dir, config_path=config_path,
                           jenkins_id=jenkins_id, db_secret=db_secret)
    else:
        return SubmissionConfig(model_type=config['model_type'], user_id=config['user_id'], work_dir=work_dir,
                                config_path=config_path,
                                jenkins_id=jenkins_id, db_secret=db_secret,
                                public=config['public'] == True)
