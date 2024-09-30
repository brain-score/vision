#!/bin/bash

REPO_DIR="/Users/caroljiang/Downloads/vision" 
BRANCH_NAME="test-pr-workflow" 
# PR_TITLE="test automatic pr"
# PR_BODY="test pr"
# TARGET_BRANCH="cj-data" 

# echo "$GITHUB_TOKEN"
# exit 1

cd "$REPO_DIR" || { echo "Repository directory not found"; exit 1; }

git checkout -b "$BRANCH_NAME"
git add brainscore_vision/data/data_test_pr/
git commit -m "test automatic add data"
git push origin "$BRANCH_NAME"

# response=$(curl -s \
#   -X POST \
#   -H "Accept: application/vnd.github+json" \
#   -H "Authorization: Bearer $GITHUB_TOKEN" \
#   -H "X-GitHub-Api-Version: 2022-11-28" \
#   https://api.github.com/repos/vision/actions/workflows/create-pr.yml/dispatches \
#   -d '{
#     "title": "'"$PR_TITLE"'",
#     "body": "'"$PR_BODY"'",
#     "head": "'"$BRANCH_NAME"'",
#     "base": "'"$TARGET_BRANCH"'"
#   }')

# pr_num=$(echo $response | python -c "import sys, json; print(json.load(sys.stdin)['number'])")
# echo "PR #${pr_num} created"
