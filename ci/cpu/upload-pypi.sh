#!/bin/bash
set -e

SOURCE_BRANCH=master

# Restrict uploads to master branch
if [ ${GIT_BRANCH} != ${SOURCE_BRANCH} ]; then
  echo "Skipping upload"
  return 0
fi

if [ -z "$TWINE_PASSWORD" ]; then
    echo "TWINE_PASSWORD not set"
    return 0
fi

echo "Upload pypi"
twine upload --skip-existing -u ${TWINE_USERNAME:-rapidsai} dist/*
