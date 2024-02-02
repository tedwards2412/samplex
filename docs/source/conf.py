# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -*- coding: utf-8 -*-

# General stuff
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = ".rst"
master_doc = "index"


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "samplex"
copyright = "2024, Thomas Edwards, Nashwan Sabti"
author = "Thomas Edwards, Nashwan Sabti"
release = "0.0.1"


templates_path = ["_templates"]
exclude_patterns = []

exclude_patterns = ["_build"]

# HTML theme
html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "samplex"
html_static_path = ["_static"]
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/tedwards2412/samplex",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "classic",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
nb_execution_mode = "off"
nb_execution_timeout = -1
