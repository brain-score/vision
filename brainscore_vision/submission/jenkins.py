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


# The GHA workflow's ``plugin_info`` payload uses the lowercase parameter
# names the legacy ``dev_score_plugins`` freestyle job exposed (``email``,
# ``new_models`` etc). The gated pipeline's Jenkinsfile (``scoring/Jenkinsfile``
# in brain-score/infrastructure) declares uppercase string parameters
# (``AUTHOR_EMAIL``, ``NEW_MODELS`` etc), so the lowercase payload keys would
# get silently dropped and the Validate stage trips on the empty AUTHOR_EMAIL.
# Translate at the trigger boundary — the Jenkinsfile's contract is the
# stable one and shouldn't be loosened just because the legacy GHA payload
# spells things differently.
_PLUGIN_INFO_TO_JENKINS_PARAM = {
    "email": "AUTHOR_EMAIL",
    "new_models": "NEW_MODELS",
    "new_benchmarks": "NEW_BENCHMARKS",
    "user_id": "USER_ID",
    "public": "PUBLIC",
    "domain": "DOMAIN",
    "competition": "COMPETITION",
    "specified_only": "SPECIFIED_ONLY",
    "disable_gates": "DISABLE_GATES",
}


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

    # Drop empty values (matches legacy behaviour) and rename to the
    # Jenkinsfile-declared parameter names; pass any unrecognised keys
    # through unchanged so future Jenkinsfile additions don't require a
    # parallel update here.
    payload = {}
    for key, value in plugin_info.items():
        if not value:
            continue
        payload[_PLUGIN_INFO_TO_JENKINS_PARAM.get(key, key)] = value

    try:
        auth_basic = HTTPBasicAuth(username=jenkins_user, password=jenkins_token)
        requests.get(url, params=payload, auth=auth_basic)
    except Exception as e:
        print(f'Could not initiate Jenkins job because of {e}')
