#!/bin/bash


python -m pip install build --user

# While conda provides these during conda-build, they are also necessary during
# the setup.py build for PyPI
export GIT_DESCRIBE_TAG=$(git describe --abbrev=0 --tags)
export GIT_DESCRIBE_NUMBER=$(git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count)

# Build date for PyPI pre-releases using version from `pyproject.toml` as source.
TOML_VERSION=$(grep "version = .*" pyproject.toml | grep -o '".*"' | sed 's/"//g')
if ! rapids-is-release-build; then
  export BUILD_DATE=$(date +%y%m%d)
  export PACKAGE_VERSION_NUMBER="${TOML_VERSION}a${BUILD_DATE}"
fi

# Compute/export RAPIDS_DATE_STRING
source rapids-env-update

# Update pyproject.toml with pre-release build date
if ! rapids-is-release-build; then
  sed -i "s/^version = \""${TOML_VERSION}".*\"/version = \""${PACKAGE_VERSION_NUMBER}"\"/g" pyproject.toml
fi

python -m build \
  --sdist \
  --wheel \
  --outdir dist/ \
  .

# Revert pyproject.toml pre-release build date
if ! rapids-is-release-build; then
  sed -i "s/^version = \""${PACKAGE_VERSION_NUMBER}"\"/version = \""${TOML_VERSION}"\"/g" pyproject.toml
fi
