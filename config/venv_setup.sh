set -ex

which python

python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv venv

source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
which python

python -m pip install -r requirements.txt
python -m pip install -r requirements_qa.txt

export PYTHONPATH
