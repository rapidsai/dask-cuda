#!/bin/bash


python -m pip install build --user


version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)
# While conda provides these during conda-build, they are also necessary during
# the setup.py build for PyPI
export GIT_DESCRIBE_TAG=$(git describe --abbrev=0 --tags)
export GIT_DESCRIBE_NUMBER=$(git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count)

# Build date for PyPI pre-releases using version from `pyproject.toml` as source.
TOML_VERSION=$(grep "version = .*" pyproject.toml | grep -o '".*"' | sed 's/"//g')
if ! rapids-is-release-build; then
  export PACKAGE_VERSION_NUMBER="${version}"
fi


echo "${version}" | tr -d '"' > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" "${package_name}/_version.py"

# Compute/export RAPIDS_DATE_STRING
source rapids-env-update


python -m build \
  --sdist \
  --wheel \
  --outdir dist/ \
  .
