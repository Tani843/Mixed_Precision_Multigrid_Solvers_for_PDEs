"""
Sphinx configuration file for Mixed-Precision Multigrid API documentation.

This configuration sets up comprehensive API documentation with automatic
module discovery, mathematical notation support, and professional styling.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

# -- Project information -----------------------------------------------------
project = 'Mixed-Precision Multigrid Solvers'
copyright = '2024, Tanisha Gupta'
author = 'Tanisha Gupta'
release = '1.0.0'
version = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',           # Automatic documentation from docstrings
    'sphinx.ext.autosummary',       # Automatic summary tables
    'sphinx.ext.viewcode',          # Source code links
    'sphinx.ext.napoleon',          # Google and NumPy style docstrings
    'sphinx.ext.mathjax',           # Mathematical notation
    'sphinx.ext.intersphinx',       # Links to other projects
    'sphinx.ext.coverage',          # Documentation coverage
    'sphinx.ext.doctest',           # Test code snippets in docs
    'sphinx.ext.githubpages',       # GitHub Pages support
    'sphinx.ext.todo',              # TODO items
    'myst_parser',                  # Markdown support
    'sphinx_rtd_theme',             # Read the Docs theme
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS
html_css_files = [
    'custom.css',
]

# -- Options for autodoc extension ------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Automatically extract typehints
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Mock imports for modules that might not be available during doc build
autodoc_mock_imports = ['cupy', 'numba', 'pycuda']

# -- Options for autosummary extension --------------------------------------
autosummary_generate = True
autosummary_imported_members = True

# -- Options for Napoleon extension -----------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for MathJax extension ------------------------------------------
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'packages': ['base', 'ams', 'noerrors', 'noundefined', 'boldsymbol']
    },
    'options': {
        'ignoreHtmlClass': 'tex2jax_ignore',
        'processHtmlClass': 'tex2jax_process'
    }
}

# -- Options for intersphinx extension --------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pytest': ('https://docs.pytest.org/en/stable/', None),
}

# -- Options for todo extension ---------------------------------------------
todo_include_todos = True

# -- Options for coverage extension -----------------------------------------
coverage_show_missing_items = True

# -- Custom configuration ---------------------------------------------------
def setup(app):
    """Custom Sphinx setup function."""
    app.add_css_file('custom.css')

# Suppress warnings for modules that can't be imported
suppress_warnings = ['autodoc.import_error']

# -- Project-specific configuration -----------------------------------------
# Root module for API documentation
api_root_module = 'multigrid'

# Version information
html_context = {
    'display_github': True,
    'github_user': 'tanishagupta',
    'github_repo': 'Mixed_Precision_Multigrid_Solvers_for_PDEs',
    'github_version': 'main',
    'conf_py_path': '/docs/sphinx/',
}

# LaTeX output configuration
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{bm}
''',
}

latex_documents = [
    (master_doc, 'mixed_precision_multigrid.tex', 
     'Mixed-Precision Multigrid Solvers Documentation',
     'Tanisha Gupta', 'manual'),
]

# -- API Documentation Structure --------------------------------------------
api_sections = {
    'Core Components': [
        'multigrid.core',
        'multigrid.operators', 
        'multigrid.solvers'
    ],
    'Advanced Features': [
        'multigrid.gpu',
        'multigrid.preconditioning',
        'multigrid.optimization'
    ],
    'Applications': [
        'multigrid.applications',
        'multigrid.benchmarks'
    ],
    'Visualization': [
        'multigrid.visualization'
    ],
    'Utilities': [
        'multigrid.utils',
        'multigrid.analysis'
    ]
}