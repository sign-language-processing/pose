# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../src/python'))

project = 'Pose-Format Toolkit'
copyright = '2023, Moryossef, Amit and Müller, Mathias and Fahrni, Rebecka'
author = 'Amit Moryossef, Mathias Müller, Rebecka Fahrni'
release = 'v0.2.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 
              'sphinx.ext.napoleon', 'sphinx.ext.inheritance_diagram',
              'sphinx.ext.viewcode',
              'myst_parser',
              'autodocsumm','sphinxcontrib.bibtex',
              'sphinx_rtd_theme','sphinx_needs','sphinxcontrib.plantuml']
              

autodoc_default_options = {
    'autosummary': True,
    'members': True,
    'show-inheritance':True}
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
master_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autodoc_typehints = "description"
autodoc_mock_imports  = ['mediapipe'] # mocks import of mediapipe for readthedocs
html_theme = 'sphinx_rtd_theme' 
html_static_path = ['_static']
html_theme_options = {
    'logo_only': False,
    'display_version': True}

html_context = {
    'display_github': True,
    'github_user': 'sign-language-processing', 
    'github_repo': 'pose',
    'github_version': 'master/',
    'conf_py_path': '/docs/',
    'source_suffix': '.rst',
}

def setup(app):
    app.add_css_file('nav.css')

bibtex_bibfiles = ['references.bib']

default_role = 'py:obj'
