#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
################################################################################
# dask-cuda version updater
################################################################################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_UCXPY_VERSION="$(curl -s https://version.gpuci.io/rapids/${NEXT_SHORT_TAG})"

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" | tr -d '"' > VERSION

# Bump cudf and dask-cudf testing dependencies
sed_runner "s/cudf==.*/cudf==${NEXT_SHORT_TAG}.*/g" dependencies.yaml
sed_runner "s/dask-cudf==.*/dask-cudf==${NEXT_SHORT_TAG}.*/g" dependencies.yaml
sed_runner "s/kvikio==.*/kvikio==${NEXT_SHORT_TAG}.*/g" dependencies.yaml
sed_runner "s/ucx-py==.*/ucx-py==${NEXT_UCXPY_VERSION}.*/g" dependencies.yaml
sed_runner "s/ucxx==.*/ucxx==${NEXT_UCXPY_VERSION}.*/g" dependencies.yaml
sed_runner "s/rapids-dask-dependency==.*/rapids-dask-dependency==${NEXT_SHORT_TAG}.*/g" dependencies.yaml

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done
sed_runner "s/RAPIDS_VERSION_NUMBER=\".*/RAPIDS_VERSION_NUMBER=\"${NEXT_SHORT_TAG}\"/g" ci/build_docs.sh
