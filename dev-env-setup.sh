#!/bin/bash

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

# need the full doctr package, not included with pip install

git clone https://github.com/mindee/doctr.git doctr_package
cd doctr_package
pip install -e .[torch]
