#!/bin/bash

rm -rf .env
python3.11 -m venv .env
source .env/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt