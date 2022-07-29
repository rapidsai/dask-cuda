import os
from codecs import open

from setuptools import find_packages, setup

import versioneer

# Get the long description from the README file
with open(os.path.join(os.path.dirname(__file__), "README.md")) as f:
    long_description = f.read()

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
    name="dask-cuda",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Utilities for Dask and CUDA interactions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rapidsai/dask-cuda",
    author="RAPIDS development team",
    author_email="mrocklin@nvidia.com",
    license="Apache-2.0",
    license_files=["LICENSE"],
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(exclude=["docs", "tests", "tests.*", "docs.*"]),
    python_requires=">=3.8",
    install_requires=open("requirements.txt").read().strip().split("\n"),
    entry_points="""
        [console_scripts]
        dask-cuda-worker=dask_cuda.cli.dask_cuda_worker:go
      """,
)
