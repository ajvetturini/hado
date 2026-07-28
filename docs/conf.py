import os
import sys

project = 'hado'
copyright = '2026, A.J. Vetturini'
author = 'A.J. Vetturini'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "myst_nb",
]
autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

nb_execution_mode = "off"

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']