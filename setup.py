import os
from codecs import open

import versioneer
from setuptools import find_packages, setup

# Get the long description from the README file
with open(os.path.join(os.path.dirname(__file__), "README.md")) as f:
    long_description = f.read()

if "GIT_DESCRIBE_TAG" in os.environ:
    version = os.environ["GIT_DESCRIBE_TAG"] + os.environ.get("VERSION_SUFFIX", "")
    cmdclass = {}
else:
    version = versioneer.get_version()
    cmdclass = versioneer.get_cmdclass()

setup(
    name="dask-cuda",
    version=version,
    cmdclass=cmdclass,
    description="Utilities for Dask and CUDA interactions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rapidsai/dask-cuda",
    author="RAPIDS development team",
    author_email="mrocklin@nvidia.com",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=["docs", "tests", "tests.*", "docs.*"]),
    install_requires=open("requirements.txt").read().strip().split("\n"),
    entry_points="""
        [console_scripts]
        dask-cuda-worker=dask_cuda.dask_cuda_worker:go
      """,
)
