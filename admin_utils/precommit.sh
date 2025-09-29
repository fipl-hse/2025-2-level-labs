set -ex

echo $1
if [[ "$1" == "smoke" ]]; then
  DIRS_TO_CHECK=(
    "config"
    "seminars"
    "lab_1_keywords_tfidf"
    "lab_2_spellcheck"
  )
else
  DIRS_TO_CHECK=(
    "config"
    "seminars"
    "lab_1_keywords_tfidf"
    "lab_2_spellcheck"
  )
fi

export PYTHONPATH=$(pwd)

python config/generate_stubs/generate_labs_stubs.py

python -m black "${DIRS_TO_CHECK[@]}"

isort .

python -m pylint "${DIRS_TO_CHECK[@]}"

mypy "${DIRS_TO_CHECK[@]}"

python config/static_checks/check_docstrings.py

python -m flake8 "${DIRS_TO_CHECK[@]}"

if [[ "$1" != "smoke" ]]; then
  python config/static_checks/check_doc8.py

  sphinx-build -b html -W --keep-going -n . dist -c admin_utils

  python -m pytest -m "mark10 and lab_2_spellcheck"
fi
