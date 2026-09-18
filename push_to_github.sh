#!/bin/bash
# Push Nexus-LLM to GitHub
# Usage: ./push_to_github.sh
#
# Prerequisites:
#   1. Install GitHub CLI: https://cli.github.com/
#   2. Authenticate: gh auth login
#   OR set up SSH key: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

set -e

REPO_URL="https://github.com/brutal-45/Nexus-LLM.git"
BRANCH="master"

echo "========================================="
echo "  Nexus-LLM GitHub Push Script"
echo "========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "nexus_llm" ]; then
    echo "Error: Please run this script from the Nexus-LLM root directory."
    exit 1
fi

# Check for gh CLI
if command -v gh &> /dev/null; then
    echo "GitHub CLI found. Checking authentication..."
    if gh auth status &> /dev/null; then
        echo "Authenticated! Pushing via GitHub CLI..."
        git push -u origin "$BRANCH" --force
        echo "Push complete!"
    else
        echo "Not authenticated. Run: gh auth login"
        echo ""
        echo "Alternatively, you can push manually:"
        echo "  git remote set-url origin git@github.com:brutal-45/Nexus-LLM.git"
        echo "  git push -u origin $BRANCH --force"
        exit 1
    fi
else
    echo "GitHub CLI not found."
    echo ""
    echo "To push to GitHub, choose one of these methods:"
    echo ""
    echo "Method 1 - Install GitHub CLI:"
    echo "  https://cli.github.com/"
    echo "  gh auth login"
    echo "  ./push_to_github.sh"
    echo ""
    echo "Method 2 - Use SSH (recommended):"
    echo "  git remote set-url origin git@github.com:brutal-45/Nexus-LLM.git"
    echo "  git push -u origin $BRANCH --force"
    echo ""
    echo "Method 3 - Use HTTPS with token:"
    echo "  git remote set-url origin https://<YOUR_TOKEN>@github.com/brutal-45/Nexus-LLM.git"
    echo "  git push -u origin $BRANCH --force"
    echo ""
    echo "Generate a token at: https://github.com/settings/tokens"
fi
