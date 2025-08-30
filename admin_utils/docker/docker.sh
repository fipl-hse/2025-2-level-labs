#!/bin/bash
set -e

export PYTHONPATH="/app:$PYTHONPATH"

apt-get update -qqy && \
apt-get install -y --no-install-recommends \
    aspell \
    aspell-en \
    aspell-ru \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*


python -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r /app/requirements_qa.txt -r /app/requirements.txt
