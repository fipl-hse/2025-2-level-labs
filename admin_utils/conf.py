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
}

nitpick_ignore = [
    ("py:class", "Methods"),  # lab 2
    ("py:class", "NGramType"),  # lab 4
    ("py:class", "optional"),  # lab 2, 4
    ("py:class", '"TrieNode"'),  # lab 2, 4
]

exclude_patterns = ["venv/*", "docs/private/*"]

language = "en"

html_theme = "sphinx_rtd_theme"
