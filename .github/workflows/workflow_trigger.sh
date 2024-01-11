#!/bin/bash

GH_WORKFLOW_TRIGGER=$1
PULL_REQUEST_SHA=$2
STATUS_DESCRIPTION=$3
CONTEXT=$4

curl -L -X POST \
-H "Authorization: token $GH_WORKFLOW_TRIGGER" \
-d $'{"state": "success", "description": "'"$STATUS_DESCRIPTION"'", 
  "context": "'"$CONTEXT"'"}' \
  "https://api.github.com/repos/brain-score/brain-score/statuses/$PULL_REQUEST_SHA"
