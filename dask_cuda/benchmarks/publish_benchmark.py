import os

from github import Github

g = Github(os.environ["GITHUB_ACCESS_TOKEN"])
repo = g.get_repo("rapidsai/dask-cuda")
label = repo.get_label("benchmark")

repo.create_issue(
    title="[TEST] This issue was created programatically",
    labels=[label],
    body="Nothing to see here 👀",
)
