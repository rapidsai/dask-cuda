#!/bin/bash


python -m pip install build --user

# While conda provides these during conda-build, they are also necessary during
# the setup.py build for PyPI
echo "GIT_DESCRIBE_TAG=$(git describe --abbrev=0 --tags)" >> $GITHUB_ENV
echo "GIT_DESCRIBE_NUMBER=$(git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count)" >> $GITHUB_ENV
# Compute version suffix
source rapids-env-update
echo "VERSION_SUFFIX=${VERSION_SUFFIX}" >> $GITHUB_ENV


python -m build \
  --sdist \
  --wheel \
  --outdir dist/ \
  .
