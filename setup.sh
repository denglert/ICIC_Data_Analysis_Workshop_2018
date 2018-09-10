#!/usr/bin/env bash

CWD=$(pwd)
PYTHON_MODULES_DIR=${CWD}/python_modules/

export PYTHONPATH=${PYTHON_MODULES_DIR}:${PYTHONPATH}
