#!/bin/bash

# Script to set up branch protection for the main branch
# This requires a GitHub Personal Access Token with repo permissions
#
# Usage:
#   GITHUB_TOKEN=your_token GITHUB_REPO=owner/repo ./setup-branch-protection.sh
#
# Or set environment variables:
#   export GITHUB_TOKEN=your_token
#   export GITHUB_REPO=owner/repo
#   ./setup-branch-protection.sh

set -e

# Check for required environment variables
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN environment variable is not set"
    echo "Create a token at: https://github.com/settings/tokens"
    echo "Required scopes: repo (Full control of private repositories)"
    exit 1
fi

if [ -z "$GITHUB_REPO" ]; then
    echo "Error: GITHUB_REPO environment variable is not set"
    echo "Format: owner/repo (e.g., username/Near-Miss-Detection)"
    exit 1
fi

BRANCH="${BRANCH:-main}"
API_URL="https://api.github.com/repos/${GITHUB_REPO}/branches/${BRANCH}/protection"

echo "Setting up branch protection for '${BRANCH}' branch in ${GITHUB_REPO}..."

# Set up branch protection with required status checks
curl -X PUT "${API_URL}" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d '{
    "required_status_checks": {
      "strict": true,
      "contexts": ["lint-and-type-check"]
    },
    "enforce_admins": true,
    "required_pull_request_reviews": {
      "required_approving_review_count": 0,
      "dismiss_stale_reviews": false,
      "require_code_owner_reviews": false
    },
    "restrictions": null,
    "allow_force_pushes": false,
    "allow_deletions": false,
    "required_linear_history": true,
    "required_conversation_resolution": true
  }'

echo ""
echo "✅ Branch protection configured successfully!"
echo ""
echo "The '${BRANCH}' branch is now protected. Pull requests must pass"
echo "the 'lint-and-type-check' workflow before they can be merged."

