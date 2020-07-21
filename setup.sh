#!/bin/bash

sudo apt-get update -y
sudo apt-get install gifsicle
sudo apt-get install -y libopenmpi-dev
sudo apt-get install -y openmpi-bin
sudo apt-get install python-virtualenv
virtualenv -p /usr/bin/python3 venv
source venv/bin/activate
