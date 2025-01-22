# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'BIDL User Guide'
copyright = '2024, Beijing Lynxi Technologies Co., Ltd & China Nanhu Academy of Electronics and Information Technology. '
author = 'Lynxi & CNAEIT'
release = '1.9.0.0'

# -- General configuration ---------------------------------------------------

# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster' 
# html_theme = "sphinx_rtd_theme"
# html_theme = "piccolo_theme"
html_static_path = ['_static']
html_permalinks_icon = '<span>#</span>'
html_theme = 'sphinx-book-theme'

html_css_files = [
    'table_word_wrap.css',
]
