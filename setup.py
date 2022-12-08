import os

import versioneer
from setuptools import setup

if "GIT_DESCRIBE_TAG" in os.environ:
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
    version = os.environ["GIT_DESCRIBE_TAG"] + os.environ.get("VERSION_SUFFIX", "")

    def get_versions():
        data = orig_get_versions()
        data["version"] = version
        return data

    versioneer.get_versions = get_versions


setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
