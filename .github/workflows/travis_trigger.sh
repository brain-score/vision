#!/bin/bash

GH_WORKFLOW_TRIGGER=$1
TRAVIS_PULL_REQUEST_SHA=$2

curl -L -X POST \
-H "Authorization: token $GH_WORKFLOW_TRIGGER" \
-d '{"state": "success", "description": "Run automerge workflow for plugin-only PR", 
  "context": "continuous-integration/travis"}' \
  "https://api.github.com/repos/brain-score/brain-score/statuses/$TRAVIS_PULL_REQUEST_SHA"
