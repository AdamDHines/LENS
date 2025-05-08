# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Adjust if your code is in a subfolder

project = 'LENS'
copyright = '2025, Adam D Hines, Michael Milford, Tobias Fischer'
author = 'Adam D Hines, Michael Milford, Tobias Fischer'
release = '0.1.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # for Google or NumPy style docstrings
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex"
]

bibtex_bibfiles = ['refs.bib']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Set your custom logo
html_logo = '_static/logowhite.svg'

# Optional: Set your custom favicon (for browser tabs)
# html_favicon = '_static/your_logo.svg'