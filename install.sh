#!/bin/bash
set -e

MAKE_VENV=${1:-true}
SOURCE_VENV=${2:-true}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if $MAKE_VENV; then
    # Virtualenv w/ python3
    export PYTHONPATH=/usr/bin/python3 # point to your python3
    python3 -m pip install virtualenv
    cd $DIR
    virtualenv -p python3 social_vdgnn
fi

if $SOURCE_VENV; then
    cd $DIR
    source social_vdgnn/bin/activate
    export PYTHONPATH=${DIR}/social_vdgnn/bin/python/dist-packages
fi

# Install this pkg and its requirements
python -m pip install -r $DIR/requirements.txt
python -m pip install -e $DIR

python setup.py build
python setup.py install
