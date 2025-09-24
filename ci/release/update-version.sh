#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
################################################################################
# dask-cuda version updater
################################################################################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")
NEXT_UCXX_VERSION="$(curl -s https://version.gpuci.io/rapids/"${NEXT_SHORT_TAG}")"

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" | tr -d '"' > VERSION
echo "branch-${NEXT_SHORT_TAG}" > RAPIDS_BRANCH

# Bump testing dependencies
sed_runner "s/ucxx==.*/ucxx==${NEXT_UCXX_VERSION}.*,>=0.0.0a0/g" dependencies.yaml

DEPENDENCIES=(
  cudf
  dask-cuda
  dask-cudf
  kvikio
  rapids-dask-dependency
)
for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" pyproject.toml
done

UCX_DEPENDENCIES=(
  distributed-ucxx
  ucxx
)
for DEP in "${UCX_DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_UCXX_VERSION}.*,>=0.0.0a0/g" "${FILE}"
  done
  sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_UCXX_VERSION}.*,>=0.0.0a0\"/g" pyproject.toml
done

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
  sed_runner "s/:[0-9]*\\.[0-9]*-/:${NEXT_SHORT_TAG}-/g" "${FILE}"
done

# Docs referencing source code
find docs/source/ -type f -name '*.rst' -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s|/branch-[^/]*/|/branch-${NEXT_SHORT_TAG}/|g" "${filename}"
done
