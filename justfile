default:
    just --list

name := "nanoGPT"

alias i := install
alias u := update
alias f := format
alias l := lint
alias t := test
alias r := run
alias b := build
alias p := publish

# install dependencies
install: 
    poetry install

# update dependencies
update: 
    poetry update

# poetry python
python:
    poetry run python

# format the code
format: 
    poetry run black {{name}}
    poetry run black tests

# lint the code
lint:
    poetry run black {{name}} tests --check
    poetry run flake8 {{name}} tests

# run mypy static type analysis
types: 
    poetry run mypy {{name}} tests --show-error-codes

# run all unit tests
test *FLAGS:
    poetry run python -m pytest -v --disable-warnings \
        tests/unit {{FLAGS}}

# run all verifications
verify: format lint types test

# run CLI
run *FLAGS:
    poetry run python -m {{name}} {{FLAGS}}

# run under torch
torchrun *FLAGS:
    poetry run torchrun nanoGPT/__main__.py {{FLAGS}}

# build package
build: 
    poetry build

# publish the package
publish *flags:
    poetry publish {{flags}}
