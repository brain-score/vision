from typing import List, Union, Dict

from brainscore_core import Score, Benchmark
from brainscore_core.submission import RunScoringEndpoint, DomainPlugins
from brainscore_core.submission.endpoints import make_argparser, resolve_models_benchmarks, get_user_id, \
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
    model_ids, benchmark_ids = resolve_models_benchmarks(domain="vision", args_dict=args_dict)

    for benchmark in benchmark_ids:
        for model in model_ids:
            run_scoring_endpoint(domain="vision", jenkins_id=args_dict["jenkins_id"],
                                 model_identifier=model, benchmark_identifier=benchmark,
                                 user_id=args_dict["user_id"], model_type="brainmodel",
                                 public=args_dict["public"], competition=args_dict["competition"])


def send_email_to_submitter(uid: int, domain: str, pr_number: str,
                            mail_username: str, mail_password: str):
    send_email_to_submitter_core(uid=uid, domain=domain, pr_number=pr_number,
                                 db_secret=config.get_database_secret(),
                                 mail_username=mail_username, mail_password=mail_password)


def call_jenkins_vision_gated(plugin_info: Union[str, Dict[str, Union[List[str], str]]]):
    """Trigger the ``core/job/gated_score_plugins`` Jenkins job for vision submissions.

    Mirrors :func:`brainscore_core.submission.endpoints.call_jenkins` but routes
    to the gated orchestrator (`brainscore-scoring` CLI) instead of the legacy
    ``dev_score_plugins`` bash pipeline. Owning the trigger here lets vision
    cut over independently of language, which keeps its own
    :func:`call_jenkins_language` override in brain-score/language.
    """
    import json
    import os
    import requests
    from requests.auth import HTTPBasicAuth

    jenkins_base = "http://www.brain-score-jenkins.com:8080"
    jenkins_user = os.environ['JENKINS_USER']
    jenkins_token = os.environ['JENKINS_TOKEN']
    jenkins_trigger = os.environ['JENKINS_TRIGGER']
    jenkins_job = "core/job/gated_score_plugins"

    url = f'{jenkins_base}/job/{jenkins_job}/buildWithParameters?token={jenkins_trigger}'

    if isinstance(plugin_info, str):
        plugin_info = json.loads(plugin_info)

    payload = {k: v for k, v in plugin_info.items() if plugin_info[k]}
    try:
        auth_basic = HTTPBasicAuth(username=jenkins_user, password=jenkins_token)
        requests.get(url, params=payload, auth=auth_basic)
    except Exception as e:
        print(f'Could not initiate Jenkins job because of {e}')


if __name__ == '__main__':
    # Wire up resource-usage tracking ONLY in the scoring-container entrypoint.
    # Keeping this out of brainscore_vision/__init__.py so regular imports
    # (tests, notebooks, REPL) don't register an atexit stdout-emitter hook.
    try:
        from brainscore_resource_tracking import enable_tracking as _enable_resource_tracking
        _enable_resource_tracking()
    except ImportError:
        pass

    parser = make_argparser()
    args, remaining_args = parser.parse_known_args()
    args_dict = vars(args)

    if args.fn == 'run_scoring':
        if 'user_id' not in args_dict or args_dict['user_id'] is None:
            user_id = get_user_id(args_dict['author_email'], db_secret=config.get_database_secret())
            args_dict['user_id'] = user_id
        run_scoring(args_dict)
    elif args.fn == 'resolve_models_benchmarks':
        resolve_models_benchmarks(domain="vision", args_dict=args_dict)
    else:
        raise ValueError(f'Invalid method: {args.fn}')
