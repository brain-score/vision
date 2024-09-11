#!/bin/bash

REPO_DIR="/Users/caroljiang/Downloads/vision"  # Update this to your local repository path
BRANCH_NAME="test-pr" # Name of the new branch
PR_TITLE="test pr"
PR_BODY="test pr"
TARGET_BRANCH="cj-NWB-conversion" # Base branch for the PR (usually 'main' or 'master')

# Function to install GitHub CLI
install_github_cli() {
    echo "GitHub CLI not found. Installing GitHub CLI..."

    # Detect operating system and install GitHub CLI
    if [ -x "$(command -v apt-get)" ]; then
        # Debian-based systems (e.g., Ubuntu)
        sudo apt-get update
        sudo apt-get install -y gh
    elif [ -x "$(command -v dnf)" ]; then
        # Red Hat-based systems (e.g., Fedora)
        sudo dnf install -y gh
    elif [ -x "$(command -v brew)" ]; then
        # macOS (using conda)
        conda install gh --channel conda-forge
    else
        echo "Unsupported operating system or package manager."
        exit 1
    fi
}

# Function to verify GitHub CLI installation
verify_github_cli() {
    if command -v gh &> /dev/null; then
        echo "GitHub CLI installed."
    else
        echo "GitHub CLI installation failed."
        exit 1
    fi
}

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    install_github_cli
    verify_github_cli
fi

# Go to the repository directory
cd "$REPO_DIR" || { echo "Repository directory not found"; exit 1; }

# Check for uncommitted changes
# if ! git diff-index --quiet HEAD --; then
#   echo "Uncommitted changes detected. Please commit or stash them before running this script."
#   exit 1
# fi

# Create a new branch
git checkout -b "$BRANCH_NAME"

# Add the new files to the repository
git add brainscore_vision/data/data_test_pr/

# Commit the changes
git commit -m "add data"

# Push the new branch to GitHub
git push origin "$BRANCH_NAME"

# Create the pull request using GitHub CLI
gh pr create \
  --title "$PR_TITLE" \
  --body "$PR_BODY" \
  --base "$TARGET_BRANCH" \
  --head "$BRANCH_NAME"

echo "Pull request created successfully."
