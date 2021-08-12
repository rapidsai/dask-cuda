#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
################################################################################
# dask-cuda Style Tester
################################################################################

# Ignore errors and set path
set +e
PATH=/opt/conda/bin:$PATH

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Print versions
echo -e "\nVersions:"
black --version
echo "isort, `isort --vn`"
echo "flake8, `flake8 --version`"

# Run isort and get results/return code
ISORT=`isort --check-only dask_cuda 2>&1`
ISORT_RETVAL=$?

# Run black and get results/return code
BLACK=`black --check dask_cuda 2>&1`
BLACK_RETVAL=$?

# Run flake8 and get results/return code
FLAKE=`flake8 dask_cuda 2>&1`
FLAKE_RETVAL=$?

# Output results if failure otherwise show pass
if [ "$ISORT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: isort style check; begin output\n\n"
  echo -e "$ISORT"
  echo -e "\n\n>>>> FAILED: isort style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: isort style check\n\n"
fi

if [ "$BLACK_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: black style check; begin output\n\n"
  echo -e "$BLACK"
  echo -e "\n\n>>>> FAILED: black style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: black style check\n\n"
fi

if [ "$FLAKE_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: flake8 style check; begin output\n\n"
  echo -e "$FLAKE"
  echo -e "\n\n>>>> FAILED: flake8 style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: flake8 style check\n\n"
fi

RETVALS=($ISORT_RETVAL $BLACK_RETVAL $FLAKE_RETVAL)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`
exit $RETVAL
