REPO_DIR="/Users/caroljiang/Downloads/vision"  # Update this to your local repository path
BRANCH_NAME="test-pr" # Name of the new branch
PR_TITLE="test pr"
PR_BODY="test pr"
TARGET_BRANCH="cj-NWB-conversion" # Base branch for the PR (usually 'main' or 'master')

# echo "Cloning ${repo_url}"
# rm -rf ${devspace}/${domain}
# git clone --depth=50 $repo_url.git ${devspace}/${domain}
echo "$GITHUB_TOKEN"
# exit 1

cd "$REPO_DIR" || { echo "Repository directory not found"; exit 1; }
git checkout -b "$BRANCH_NAME"

set +e
git add brainscore_vision/data/data_test_pr/
git commit -m "add data"
git push origin "$BRANCH_NAME"
set -e

echo "Creating pull request"
response=$(curl \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN"\
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/brain-score/vision/pulls \
  -d '{
        "title": "'"$PR_TITLE"'",
        "body": "'"$PR_BODY"'",
        "head": "'"$BRANCH_NAME"'",
        "base": "'"$TARGET_BRANCH"'"
    }')

pr_num=$(echo $response | python -c "import sys, json; print(json.load(sys.stdin)['number'])")
echo "PR #${pr_num} created"
