import datetime
import distutils

from brainscore.submission.models import Model, Submission


class BaseConfig:
    def __init__(self, work_dir, jenkins_id, db_secret, config_path):
        self.work_dir = work_dir
        self.jenkins_id = jenkins_id
        self.config_path = config_path
        self.db_secret = db_secret
        return


class MultiConfig(BaseConfig):
    def __init__(self, model_ids, **kwargs):
        BaseConfig.__init__(self, **kwargs)
        self.models = []
        self.submissions = {}
        for id in model_ids:
            model = Model.get(id=id)
            submission: Submission = model.submission
            self.models.append(model)
            if submission.id not in self.submissions:
                self.submissions[submission.id] = submission


class SubmissionConfig(BaseConfig):
    def __init__(self, model_type, user_id, jenkins_id, public, **kwargs):
        BaseConfig.__init__(self, jenkins_id=jenkins_id, **kwargs)
        self.submission = Submission.create(id=jenkins_id, submitter=user_id, timestamp=datetime.datetime.now(),
                                            model_type=model_type, status='running')
        self.public = public


def object_decoder(config, work_dir, config_path, db_secret, jenkins_id):
    if 'model_ids' in config:
        return MultiConfig(model_ids=config['model_ids'], work_dir=work_dir, config_path=config_path,
                           jenkins_id=jenkins_id, db_secret=db_secret)
    else:
        return SubmissionConfig(model_type=config['model_type'], user_id=config['user_id'], work_dir=work_dir,
                                config_path=config_path,
                                jenkins_id=jenkins_id, db_secret=db_secret, public=bool(distutils.util.strtobool(config['public'])))
