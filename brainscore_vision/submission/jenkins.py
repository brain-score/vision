"""Standalone Jenkins-trigger entrypoint for vision submissions.

Lives apart from :mod:`brainscore_vision.submission.endpoints` on purpose:
that module instantiates ``RunScoringEndpoint`` at import time, which connects
to the production DB and therefore requires AWS credentials. The GitHub
Actions runner that fires the post-merge scoring trigger has no AWS access
(by design — its only job is to POST to Jenkins), so importing the heavyweight
module from CI fails with ``NoCredentialsError`` before the trigger function
even runs.

Mirrors the pattern used by ``brainscore_language.submission.endpoints`` where
``call_jenkins_language`` lives alongside lightweight code that doesn't touch
the DB at import.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Union

import requests
from requests.auth import HTTPBasicAuth


def call_jenkins_vision_gated(plugin_info: Union[str, Dict[str, Union[List[str], str]]]) -> None:
    """Trigger the ``core/job/gated_score_plugins`` Jenkins job for vision submissions.

    Mirrors :func:`brainscore_core.submission.endpoints.call_jenkins` but routes
    to the gated orchestrator (``brainscore-scoring`` CLI) instead of the legacy
    ``dev_score_plugins`` bash pipeline. Owning the trigger inside vision lets
    this domain cut over independently of language, which keeps its own
    :func:`call_jenkins_language` override in brain-score/language.
    """
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
