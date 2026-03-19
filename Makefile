# Copyright (c) 2022-2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of Picanteo
# (see https://github.com/CNES/picanteo).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Initially based on Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# Dependencies : python3 venv
# Some Makefile global variables can be set in make command line
# Recall: .PHONY  defines special targets not associated with files

############### GLOBAL VARIABLES ######################
.DEFAULT_GOAL := help
# Set shell to BASH
SHELL := /bin/bash

# Set Virtualenv directory name
# Example: VENV="other-venv/" make install
ifndef VENV
	VENV = "picanteo_venv"
endif

# Python global variables definition
PYTHON_VERSION_MIN = 3.10

# Set PYTHON if not defined in command line
# Example: PYTHON="python3.10" make venv to use python 3.10 for the venv
# By default the default python3 of the system.
ifndef PYTHON
	PYTHON = "python3"
endif
PYTHON_CMD=$(shell command -v $(PYTHON))

PYTHON_VERSION_CUR=$(shell $(PYTHON_CMD) -c 'import sys; print("%d.%d"% sys.version_info[0:2])')
PYTHON_VERSION_OK=$(shell $(PYTHON_CMD) -c 'import sys; cur_ver = sys.version_info[0:2]; min_ver = tuple(map(int, "$(PYTHON_VERSION_MIN)".split("."))); print(int(cur_ver >= min_ver))')

############### Check python version supported ############

ifeq (, $(PYTHON_CMD))
    $(error "PYTHON_CMD=$(PYTHON_CMD) not found in $(PATH)")
endif

ifeq ($(PYTHON_VERSION_OK), 0)
    $(error "Requires python version >= $(PYTHON_VERSION_MIN). Current version is $(PYTHON_VERSION_CUR)")
endif

################ MAKE targets by sections ######################
.PHONY: help
help: ## help on Picanteo command line usage
	@echo "      PICANTEO MAKE HELP"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: venv
venv: ## create virtualenv in "picanteo_venv" directory if it doesn't exist
	@test -d ${VENV} || $(PYTHON_CMD) -m venv ${VENV}
	@touch ${VENV}/bin/activate
	@${VENV}/bin/python -m pip install --upgrade wheel setuptools pip # no check to upgrade each time


.PHONY: install
install: venv  ## install environment for development target (depends picanteor_venv)
	@test -f ${VENV}/bin/picanteo || echo "Install picanteo package from local directory"
	@test -f ${VENV}/bin/picanteo || ${VENV}/bin/pip install -e .
	@echo "Picanteo installed in dev mode in virtualenv ${VENV}"
	@echo "Picanteo virtual environment usage: source ${VENV}/bin/activate; picanteo -h"

## Notebook section

.PHONY: notebook
notebook: ## install Jupyter notebook kernel with venv and picanteo install
	@echo "Install Jupyter Kernel and run Jupyter notebooks environment"
	@${VENV}/bin/python -m ipykernel install --sys-prefix --name=jupyter-${VENV} --display-name=jupyter-${VENV}
	@echo "Use jupyter kernelspec list to know where is the kernel"
	@echo " --> After Picanteo virtualenv activation, please use following command to run local jupyter notebook to open Notebooks:"
	@echo "jupyter notebook"

.PHONY: notebook-clean-output ## Clean Jupyter notebooks outputs
notebook-clean-output:
	@echo "Clean Jupyter notebooks"
	@${VENV}/bin/jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/tutorials/*.ipynb

## Release section

.PHONY: dist
dist: clean install ## clean, install, builds source and wheel package
	@${VENV}/bin/python -m pip install --upgrade build
	@${VENV}/bin/python -m build --sdist
	ls -l dist

.PHONY: release
release: dist ## package and upload a release
	@${VENV}/bin/twine check dist/*
	@${VENV}/bin/twine upload --verbose --config-file ~/.pypirc -r pypi dist/* ## update your .pypirc accordingly

## Clean section

.PHONY: clean
clean: clean-venv clean-build clean-pyc clean-notebook

.PHONY: clean-venv
clean-venv: ## clean venv
	@echo "+ $@"
	@rm -rf ${VENV}

.PHONY: clean-build
clean-build: ## clean build artifacts
	@echo "+ $@"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf .eggs/
	@find . -name '*.egg-info' -exec rm -rf {} +
	@find . -name '*.egg' -exec rm -rf {} +

.PHONY: clean-pyc
clean-pyc: ## clean Python file artifacts
	@echo "+ $@"
	@find . -type f -name "*.py[co]" -exec rm -rf {} +
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -name '*~' -exec rm -rf {} +

.PHONY: clean-notebook
clean-notebook: ## clean notebooks cache
	@echo "+ $@"
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
