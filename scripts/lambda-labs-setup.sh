#!/bin/bash
curl -sSL https://install.python-poetry.org | python3 -
git clone git@github.com:wrossmorrow/nanoGPT.git
cd nanoGPT
poetry install
poetry run python data/shakespeare-char/prepare.py
