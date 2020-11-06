import os
from datetime import datetime

from github import Github

g = Github(os.environ["GITHUB_ACCESS_TOKEN"])

today = datetime.now().strftime("%Y%m%d")
d = f"slurm-dask-{today}"
fname_bench = today + "-benchmark.png"
bench_path = os.path.join(d, fname_bench)

profile_name = today + "-dask-cudf-merge-profile.html"
profile_path = os.path.join(d, profile_name)

fname_hist = today + "-benchmark-history.png"
hist_path = os.path.join(d, fname_hist)

with open(os.path.join(d, "raw_data.txt")) as f:
    raw_data = f.read()

repo = g.get_repo("quasiben/dask-cuda")

print("Uploading HTML Profile...")

repo.create_file(
    path=f"assets/{profile_name}",
    message=f"dask profile {today}",
    content=open(f"{profile_path}", "rb").read(),
    branch="benchmark-images",
)

print("Uploading images...")
print(f"\t{fname_bench}")
print(f"\t{bench_path}")

repo.create_file(
    path=f"assets/{fname_bench}",
    message=f"throughput benchmark image {today}",
    content=open(f"{bench_path}", "rb").read(),
    branch="benchmark-images",
)

print(f"\t{fname_hist}")

repo.create_file(
    path=f"assets/{fname_hist}",
    message=f"historical benchmark image {today}",
    content=open(f"{hist_path}", "rb").read(),
    branch="benchmark-images",
)

print("Creating Issue...")
template = f"""
## Historical Throughput
<img width="641" alt="Benchmark Image"
src="https://raw.githubusercontent.com/quasiben/dask-cuda/benchmark-images/assets/{fname_hist}">


## Throughput Runs
<img width="641" alt="Benchmark Image"
src="https://raw.githubusercontent.com/quasiben/dask-cuda/benchmark-images/assets/{fname_bench}">

## Raw Data
{raw_data}

## Dask Profile

[Performance Profile](https://raw.githack.com/quasiben/dask-cuda/benchmark-images/assets/{profile_name})
"""

repo.create_issue(title=f"Nightly Benchmark run {today}", body=template)
