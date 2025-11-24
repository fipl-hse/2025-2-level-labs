#!/bin/bash
set -e

REPO_NAME=$1
PR_NUMBER=$2
TARGET_REPO="fipl-hse.github.io"
BRANCH_NAME="auto-update-from-$REPO_NAME-pr-$PR_NUMBER"

COMMENT_BODY=${COMMENT_BODY:-""}

# Clone Target Repo
rm -rf $TARGET_REPO

git clone https://$GH_TOKEN@github.com/fipl-hse/$TARGET_REPO.git
cd $TARGET_REPO
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

# Create Label
if ! gh label list --repo fipl-hse/$TARGET_REPO --json name -q '.[] | select(.name == "automated pr")' | grep -q "automated pr"; then
    gh label create "automated pr" --color "0E8A16" --description "Automated pull request" --repo fipl-hse/$TARGET_REPO
fi

# Check PR and Update Branch
if git show-ref --quiet refs/remotes/origin/$BRANCH_NAME; then
    git checkout $BRANCH_NAME
    git pull origin $BRANCH_NAME
else
    git checkout -b $BRANCH_NAME
fi

if [ -n "$COMMENT_BODY" ] && [ "$COMMENT_BODY" != "" ]; then
    PR_BRANCH=$(gh pr view $PR_NUMBER --repo $GITHUB_REPOSITORY --json headRefName --jq '.headRefName' 2>/dev/null || echo "")
    SOURCE_REF="parent-repo/$PR_BRANCH"
else
    PR_BRANCH="main"
    SOURCE_REF="parent-repo/main"
fi

if [ -z "$PR_BRANCH" ]; then
    exit 0
fi

git remote add parent-repo https://github.com/$GITHUB_REPOSITORY.git
git fetch parent-repo

CHANGED_FILES=$(gh pr view $PR_NUMBER --repo $GITHUB_REPOSITORY --json files --jq '.files[].path' 2>/dev/null || echo "")

if [ -z "$CHANGED_FILES" ]; then
    echo "No changed files found in PR $PR_NUMBER"
    exit 0
fi

JSON_EXISTS=false
if git show $SOURCE_REF:config/external_pr_files/tracked_files.json &>/dev/null; then
    JSON_EXISTS=true
    JSON_CONTENT=$(git show $SOURCE_REF:config/external_pr_files/tracked_files.json 2>/dev/null || echo "")
    
    if ! echo "$JSON_CONTENT" | jq -e . >/dev/null 2>&1; then
        JSON_EXISTS=false
    fi
else
    exit 0
fi

TEST_JSON_CHANGED=false
HAS_CHANGES=false
FILES_TO_SYNC_FOUND=false

if echo "$CHANGED_FILES" | grep -q "config/external_pr_files/tracked_files.json"; then
    if git show $SOURCE_REF:config/external_pr_files/tracked_files.json > config/external_pr_files/tracked_files.json 2>/dev/null; then
        git add config/external_pr_files/tracked_files.json
        TEST_JSON_CHANGED=true
        HAS_CHANGES=true
        
        JSON_CONTENT=$(cat config/external_pr_files/tracked_files.json)
        if ! echo "$JSON_CONTENT" | jq -e . >/dev/null 2>&1; then
            exit 1
        fi
        JSON_EXISTS=true
    fi
fi

for file in $CHANGED_FILES; do
    if [ "$file" = "config/external_pr_files/tracked_files.json" ]; then
        continue
    fi
    
    if [ "$JSON_EXISTS" = true ]; then
        TARGETS=$(echo "$JSON_CONTENT" | jq -r --arg file "$file" '.[] | select(.source == $file) | .target' 2>/dev/null || echo "")
        if [ -n "$TARGETS" ]; then
            FILES_TO_SYNC_FOUND=true
            break
        fi
    fi
done

if [ "$FILES_TO_SYNC_FOUND" = false ] && [ "$TEST_JSON_CHANGED" = false ]; then
    exit 0
fi

for file in $CHANGED_FILES; do
    if [ "$file" = "config/external_pr_files/tracked_files.json" ]; then
        continue
    fi
    
    if [ "$JSON_EXISTS" = true ]; then
        TARGETS=$(echo "$JSON_CONTENT" | jq -r --arg file "$file" '.[] | select(.source == $file) | .target' 2>/dev/null || echo "")
        
        for TARGET_DIR in $TARGETS; do
            if [ -n "$TARGET_DIR" ]; then
                TARGET_DIR_ONLY=$(dirname "$TARGET_DIR")
                mkdir -p "$TARGET_DIR_ONLY"
                if git show $SOURCE_REF:"$file" > "$TARGET_DIR" 2>/dev/null; then
                    git add "$TARGET_DIR"
                    HAS_CHANGES=true
                else
                    echo "Warning: Could not read file $file from $SOURCE_REF"
                fi
            fi
        done
    fi
done

PR_DELETED_FILES=$(gh pr view $PR_NUMBER --repo $GITHUB_REPOSITORY --json files --jq '.files[] | select(.status == "removed") | .path' 2>/dev/null || echo "")

for deleted_file in $PR_DELETED_FILES; do
    if [ "$JSON_EXISTS" = true ]; then
        TARGETS=$(echo "$JSON_CONTENT" | jq -r --arg file "$deleted_file" '.[] | select(.source == $file) | .target' 2>/dev/null || echo "")
        
        for TARGET_PATH in $TARGETS; do
            if [ -n "$TARGET_PATH" ] && [ -f "$TARGET_PATH" ]; then
                git rm "$TARGET_PATH" 2>/dev/null || rm "$TARGET_PATH"
                HAS_CHANGES=true
            fi
        done
    fi
done

if [ "$HAS_CHANGES" = true ]; then
    if [ "$TEST_JSON_CHANGED" = true ] && [ "$FILES_TO_SYNC_FOUND" = false ]; then
        git commit -m "Update sync mapping from $REPO_NAME PR $PR_NUMBER"
    else
        git commit -m "Sync changes from $REPO_NAME PR $PR_NUMBER"
    fi
    git push origin $BRANCH_NAME
else
    echo "No changes to commit"
    exit 0
fi

TARGET_PR_NUMBER=$(gh pr list --repo fipl-hse/$TARGET_REPO --head $BRANCH_NAME --json number -q '.[0].number' 2>/dev/null || true)

if git log --oneline origin/main..$BRANCH_NAME | grep -q .; then
    if [ -z "$TARGET_PR_NUMBER" ]; then
        gh pr create \
            --repo fipl-hse/$TARGET_REPO \
            --head $BRANCH_NAME \
            --base main \
            --title "[Automated] Sync from $REPO_NAME PR $PR_NUMBER" \
            --fill \
            --label "automated pr" \
            --assignee QuietHellsPage \
            --reviewer QuietHellsPage
        echo "Created new PR in target repository"
    else
        gh pr comment $TARGET_PR_NUMBER --repo fipl-hse/$TARGET_REPO --body "Automatically updated"
        echo "Updated existing PR #$TARGET_PR_NUMBER in target repository"
    fi
else
    echo "No commits in branch $BRANCH_NAME - skipping PR creation"
fi
