#!/bin/bash


python -m pip install build --user

# While conda provides these during conda-build, they are also necessary during
# the setup.py build for PyPI
export GIT_DESCRIBE_TAG=$(git describe --abbrev=0 --tags)
export GIT_DESCRIBE_NUMBER=$(git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count)

# Build date for PyPI pre-releases
export RAPIDS_VERSION_NUMBER="23.08"
export RAPIDS_FULL_VERSION_NUMBER="${RAPIDS_VERSION_NUMBER}.00"
if rapids-is-release-build; then
  export PACKAGE_VERSION_NUMBER="${RAPIDS_FULL_VERSION_NUMBER}"
else
  export BUILD_DATE=$(date +%y%m%d)
  export PACKAGE_VERSION_NUMBER="${RAPIDS_FULL_VERSION_NUMBER}a${BUILD_DATE}"
fi

# Compute/export RAPIDS_DATE_STRING
source rapids-env-update

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# Update pyproject.toml with pre-release build date
if ! rapids-is-release-build; then
  sed_runner "s/^version = \""${RAPIDS_FULL_VERSION_NUMBER}".*\"/version = \""${PACKAGE_VERSION_NUMBER}"\"/g" ../pyproject.toml
fi

python -m build \
  --sdist \
  --wheel \
  --outdir dist/ \
  .

# Revert pyproject.toml pre-release build date
if ! rapids-is-release-build; then
  sed_runner "s/^version = \""${PACKAGE_VERSION_NUMBER}"\"/version = \""${RAPIDS_FULL_VERSION_NUMBER}"\"/g" ../pyproject.toml
fi
