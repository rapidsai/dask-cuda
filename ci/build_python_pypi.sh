#!/bin/bash


python -m pip install build --user

# While conda provides these during conda-build, they are also necessary during
# the setup.py build for PyPI
export GIT_DESCRIBE_TAG=$(git describe --abbrev=0 --tags)
export GIT_DESCRIBE_NUMBER=$(git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count)

# Compute/export VERSION_SUFFIX
source rapids-env-update

python -m build \
  --sdist \
  --wheel \
  --outdir dist/ \
  .
