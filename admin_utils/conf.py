"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# pylint: disable=invalid-name,redefined-builtin

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

project = "Лабораторный Практикум и Курс Лекций"
copyright = "2025, Демидовский А.В. и другие"
author = "Демидовский А.В. и другие"

extensions = [
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

root_doc = "admin_utils/index"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "requests": ("https://requests.readthedocs.io/en/latest/", None),
    "bs4": ("https://www.crummy.com/software/BeautifulSoup/bs4/doc/", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
}

nitpick_ignore = [
    ("py:class", "spacy.tokens.token.Token"),
    ("py:class", "spacy.tokens.Token"),
    ("py:class", "stanza.models.common.doc.Document"),
    ("py:class", "DiGraph"),
    ("py:class", "optional"),
    ("py:class", "Methods"),
]

exclude_patterns = ["venv/*", "docs/private/*"]

language = "en"

html_theme = "sphinx_rtd_theme"
