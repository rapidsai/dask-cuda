#!/bin/bash


python -m pip install build --user

# While conda provides these during conda-build, they are also necessary during
# the setup.py build for PyPI
export GIT_DESCRIBE_TAG=$(git describe --abbrev=0 --tags)
export GIT_DESCRIBE_NUMBER=$(git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count)

# Build date for PyPI pre-releases
export BUILD_DATE=$(date +%y%m%d)

# Compute/export RAPIDS_DATE_STRING
source rapids-env-update

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# Update pyproject.toml with pre-release build date
sed_runner "s/^version = \"23.08.00\"/version = \"23.08.00a"${BUILD_DATE}"\"/g" pyproject.toml

python -m build \
  --sdist \
  --wheel \
  --outdir dist/ \
  .

# Revert pyproject.toml pre-release build date
sed_runner "s/^version = \"23.08.00a"${BUILD_DATE}"\"/version = \"23.08.00\"/g" pyproject.toml
