from typing import List, Union, Dict

from brainscore_core import Score, Benchmark
from brainscore_core.submission import RunScoringEndpoint, DomainPlugins
from brainscore_core.submission.endpoints import make_argparser, retrieve_models_and_benchmarks, get_user_id, \
    send_email_to_submitter as send_email_to_submitter_core
from brainscore_vision import load_model, load_benchmark, score
from brainscore_vision.submission import config


class VisionPlugins(DomainPlugins):
    def load_model(self, model_identifier: str):
        return load_model(model_identifier)

    def load_benchmark(self, benchmark_identifier: str) -> Benchmark:
        return load_benchmark(benchmark_identifier)

    def score(self, model_identifier: str, benchmark_identifier: str) -> Score:
        return score(model_identifier, benchmark_identifier)


vision_plugins = VisionPlugins()
run_scoring_endpoint = RunScoringEndpoint(vision_plugins, db_secret=config.get_database_secret())


def run_scoring(args_dict: Dict[str, Union[str, List]]):
    benchmarks, models = retrieve_models_and_benchmarks(args_dict)

    run_scoring_endpoint(domain="vision", jenkins_id=args_dict["jenkins_id"],
                         models=models, benchmarks=benchmarks, user_id=args_dict["user_id"],
                         model_type="brainmodel", public=args_dict["public"],
                         competition=args_dict["competition"])


def send_email_to_submitter(uid: int, domain: str, pr_number: str,
                            mail_username: str, mail_password: str):
    send_email_to_submitter_core(uid=uid, domain=domain, pr_number=pr_number,
                                 db_secret=config.get_database_secret(),
                                 mail_username=mail_username, mail_password=mail_password)


if __name__ == '__main__':
    parser = make_argparser()
    args, remaining_args = parser.parse_known_args()
    args_dict = vars(args)

    if 'user_id' not in args_dict or args_dict['user_id'] is None:
        user_id = get_user_id(args_dict['author_email'], db_secret=config.get_database_secret())
        args_dict['user_id'] = user_id

    run_scoring(args_dict)
