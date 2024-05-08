SHELL := /bin/bash
ENV_NAME := transformers
PYTHON_VERSION := 3.11

default_target: all

.PHONY: all
all: setup-env install-dependencies

.PHONY: setup-env
env:
	@echo "Creating the Pyenv environment..."
	if [ -d "$$HOME/.pyenv/versions/$(ENV_NAME)" ]; then \
		echo "Environment $(ENV_NAME) already exists. Please remove it before running this script."; \
		exit 0; \
	fi
	if [-z "$$(pyenv versions | grep $(PYTHON_VERSION))"]; then \
		pyenv install $(PYTHON_VERSION); \
	fi
	pyenv virtualenv $(PYTHON_VERSION) $(ENV_NAME)
	pyenv activate $(ENV_NAME); \
	pip install poetry
	@echo "Environment $(ENV_NAME) created. Please activate the environment by running 'pyenv activate $(ENV_NAME)'."

.PHONY: install-dependencies
install-dependencies:
	@echo "Configuring Poetry to use the Conda environment's Python interpreter..."
	python3 -m spacy download en > /dev/null
	python3 -m spacy download fr > /dev/null

.PHONY: run
run:
	@echo "Running the application..."
	direnv allow
	python -m src.models.test_model

.PHONY: clean
clean:
	@echo "Removing the Conda environment..."
	conda env remove --name $(ENV_NAME)
	@echo "Environment $(ENV_NAME) has been removed."
