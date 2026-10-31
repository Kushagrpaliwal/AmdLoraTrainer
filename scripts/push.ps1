$ErrorActionPreference = 'Stop'

$branch = git branch --show-current
if ($LASTEXITCODE -ne 0) {
    throw 'Could not determine the current Git branch.'
}
if ([string]::IsNullOrWhiteSpace($branch)) {
    throw 'No current branch is checked out (detached HEAD).'
}

git push --force-with-lease origin $branch
if ($LASTEXITCODE -ne 0) {
    throw "Push failed for branch '$branch'."
}
