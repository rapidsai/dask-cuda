#!/usr/bin/env bash

pip install git+https://github.com/dask/distributed.git@master
python setup.py install --single-version-externally-managed --record=record.txt
