#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
################################################################################
# dask-cuda version updater
################################################################################

## Usage
# NOTE: This script must be run from the repository root, not from the ci/release/ directory
# Primary interface:   bash ci/release/update-version.sh <new_version> [--run-context=main|release]
# Fallback interface:  [RAPIDS_RUN_CONTEXT=main|release] bash ci/release/update-version.sh <new_version>
# CLI arguments take precedence over environment variables
# Defaults to main when no run-context is specified


# Parse command line arguments
CLI_RUN_CONTEXT=""
VERSION_ARG=""

for arg in "$@"; do
    case $arg in
        --run-context=*)
            CLI_RUN_CONTEXT="${arg#*=}"
            shift
            ;;
        *)
            if [[ -z "$VERSION_ARG" ]]; then
                VERSION_ARG="$arg"
            fi
            ;;
    esac
done

# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG="$VERSION_ARG"

# Determine RUN_CONTEXT with CLI precedence over environment variable, defaulting to main
if [[ -n "$CLI_RUN_CONTEXT" ]]; then
    RUN_CONTEXT="$CLI_RUN_CONTEXT"
    echo "Using run-context from CLI: $RUN_CONTEXT"
elif [[ -n "${RAPIDS_RUN_CONTEXT}" ]]; then
    RUN_CONTEXT="$RAPIDS_RUN_CONTEXT"
    echo "Using run-context from environment: $RUN_CONTEXT"
else
    RUN_CONTEXT="main"
    echo "No run-context provided, defaulting to: $RUN_CONTEXT"
fi

# Validate RUN_CONTEXT value
if [[ "${RUN_CONTEXT}" != "main" && "${RUN_CONTEXT}" != "release" ]]; then
    echo "Error: Invalid run-context value '${RUN_CONTEXT}'"
    echo "Valid values: main, release"
    exit 1
fi

# Validate version argument
if [[ -z "$NEXT_FULL_TAG" ]]; then
    echo "Error: Version argument is required"
    echo "Usage: $0 <new_version> [--run-context=<context>]"
    echo "   or: [RAPIDS_RUN_CONTEXT=<context>] $0 <new_version>"
    echo "Note: Defaults to main when run-context is not specified"
    exit 1
fi

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")
NEXT_UCXX_VERSION="$(curl -s https://version.gpuci.io/rapids/"${NEXT_SHORT_TAG}")"

# Set branch references based on RUN_CONTEXT
if [[ "${RUN_CONTEXT}" == "main" ]]; then
    RAPIDS_BRANCH_NAME="main"
    echo "Preparing development branch update $CURRENT_TAG => $NEXT_FULL_TAG (targeting main branch)"
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    RAPIDS_BRANCH_NAME="release/${NEXT_SHORT_TAG}"
    echo "Preparing release branch update $CURRENT_TAG => $NEXT_FULL_TAG (targeting release/${NEXT_SHORT_TAG} branch)"
fi

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" | tr -d '"' > VERSION

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
  sed_runner "/shared-workflows/ s|@.*|@${RAPIDS_BRANCH_NAME}|g" "${FILE}"
  sed_runner "s|:[0-9]*\\.[0-9]*-|:${NEXT_SHORT_TAG}-|g" "${FILE}"
done

# Docs referencing source code - update branch references based on context
find docs/source/ -type f -name '*.rst' -print0 | while IFS= read -r -d '' filename; do
    if [[ "${RUN_CONTEXT}" == "main" ]]; then
        # In main context, convert branch-X.Y to main
        sed_runner "s|/branch-[^/]*/|/main/|g" "${filename}"
        sed_runner "s|/release/[^/]*/|/main/|g" "${filename}"
    elif [[ "${RUN_CONTEXT}" == "release" ]]; then
        # In release context, convert main or branch-X.Y to release/X.Y
        sed_runner "s|/main/|/release/${NEXT_SHORT_TAG}/|g" "${filename}"
        sed_runner "s|/branch-[^/]*/|/release/${NEXT_SHORT_TAG}/|g" "${filename}"
    fi
done
