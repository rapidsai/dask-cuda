import os

from setuptools import setup

import versioneer

# dask-cuda uses GIT_DESCRIBE_TAG to override versioneer
# In the RAPIDS pip wheel workflows, we use RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE
# to achieve the same goal (in fact using code copied from this setup.py)
#
# To faciliate building RAPIDS pip wheels in the CI/nightly index, we need
# a copy of dask-cuda available to install before it is released on PyPI.org
#
# We introduced a .github/workflows/wheels.yml workflow to dask-cuda
# to build and upload an early copy of a dask-cuda wheel to the CI/nightly
# index to unblock wheel CI for other RAPIDS projects
version_env_var = (
    "GIT_DESCRIBE_TAG"
    if "GIT_DESCRIBE_TAG" in os.environ
    else "RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE"
)

if version_env_var in os.environ:
    # Disgusting hack. For pypi uploads we cannot use the
    # versioneer-provided version for non-release builds, since they
    # strictly follow PEP440
    # https://peps.python.org/pep-0440/#local-version-identifiers
    # which disallows local version identifiers (as produced by
    # versioneer) in public index servers.
    # We still want to use versioneer infrastructure, so patch
    # in our pypi-compatible version to the output of
    # versioneer.get_versions.

    orig_get_versions = versioneer.get_versions
    version = os.environ[version_env_var] + os.environ.get("VERSION_SUFFIX", "")

    def get_versions():
        data = orig_get_versions()
        data["version"] = version
        return data

    versioneer.get_versions = get_versions


setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
