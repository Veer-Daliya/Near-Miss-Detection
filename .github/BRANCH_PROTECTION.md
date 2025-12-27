# Branch Protection Setup

To ensure that code cannot be merged into `main` without passing linting and type checking, you need to set up branch protection rules in GitHub.

## Quick Setup (Automated)

You can use the provided script to automatically set up branch protection:

```bash
export GITHUB_TOKEN=your_github_personal_access_token
export GITHUB_REPO=your-username/Near-Miss-Detection
./.github/setup-branch-protection.sh
```

**Note**: You'll need to create a GitHub Personal Access Token with `repo` permissions at: https://github.com/settings/tokens

## Manual Setup

## Steps to Enable Branch Protection

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Branches**
3. Click **Add rule** or edit the existing rule for `main` branch
4. Configure the following settings:

### Required Settings

- **Branch name pattern**: `main` (or `master` if that's your default branch)
- **Require a pull request before merging**: ✅ Enabled
- **Require status checks to pass before merging**: ✅ Enabled
  - **Require branches to be up to date before merging**: ✅ Enabled
  - Under "Status checks that are required", select:
    - `lint-and-type-check` (the job name from the workflow)

### Recommended Additional Settings

- **Require conversation resolution before merging**: ✅ Enabled (optional but recommended)
- **Require linear history**: ✅ Enabled (optional, keeps a clean git history)
- **Do not allow bypassing the above settings**: ✅ Enabled (prevents admins from bypassing)

## What This Does

Once configured, GitHub will:
- Block any pull request from being merged if the `lint-and-type-check` workflow fails
- Require that the branch is up to date with the base branch
- Prevent force pushes to the protected branch

## Testing

After setting up branch protection:
1. Create a pull request with code that fails linting/type checking
2. Verify that the "Merge" button is disabled until all checks pass
3. Fix the issues and verify the merge becomes available

## Note

The workflow will automatically fail if:
- Ruff finds any linting errors
- Black detects formatting issues (code not formatted with 88-char line length)
- mypy finds type errors

All checks must pass for the workflow to succeed and allow merging.

