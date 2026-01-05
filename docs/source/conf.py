# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import importlib.util
import os
import sys
from pathlib import Path

import pygit2
from docutils import nodes

# Add necessary directories to Python path
sys.path.insert(0, str(Path(__file__).parent))  # For importing helper module
sys.path.insert(0, str(
    Path(__file__).parent.parent.parent))  # For importing tensorrt_edgellm

# Import helper functions for auto-generating API documentation
from helper import generate_module_rst_files, generate_python_api_rst

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TensorRT Edge-LLM'
copyright = '2025, Nvidia'
author = 'Nvidia'
html_show_sphinx = False

# Get the git commit hash
try:
    repo = pygit2.Repository('.')
    commit_hash = str(
        repo.head.target)[:7]  # Get first 7 characters of commit hash
except Exception:
    commit_hash = 'unknown'

# Get current date
last_updated = datetime.datetime.now(
    datetime.timezone.utc).strftime("%B %d, %Y")

# Get the version from the version.py file
try:
    version_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "../../tensorrt_edgellm/version.py"))
    spec = importlib.util.spec_from_file_location("version_module",
                                                  version_path)
    version_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version_module)
    version = version_module.__version__
except Exception:
    version = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = []

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CPP_XML_INDEX = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "cpp_docs", "xml", "index.xml"))

# C++ documentation is required for this project
if not os.path.exists(CPP_XML_INDEX):
    raise FileNotFoundError(
        f"C++ documentation XML not found at {CPP_XML_INDEX}. "
        "Please generate C++ documentation using Doxygen before building Sphinx docs."
    )

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'myst_parser',  # for markdown support
    'sphinx.ext.todo',
    'sphinx.ext.autosectionlabel',
    'sphinxarg.ext',
    'sphinx_click',
    'sphinx_copybutton',
    'sphinxcontrib.autodoc_pydantic',
    'sphinx_togglebutton',
    'sphinxcontrib.mermaid',  # For mermaid diagram support
    'breathe',  # C++ documentation support (required)
]

autodoc_member_order = 'bysource'
autodoc_pydantic_model_show_json = True
autodoc_pydantic_model_show_config_summary = True
autodoc_pydantic_field_doc_policy = "description"
autodoc_pydantic_model_show_field_list = True  # Display field list with descriptions
autodoc_pydantic_model_member_order = "groupwise"
autodoc_pydantic_model_hide_pydantic_methods = True
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_settings_signature_prefix = ""  # remove any prefix
autodoc_pydantic_settings_hide_reused_validator = True  # hide all the validator should be better

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Mock imports for packages not needed during documentation builds
autodoc_mock_imports = [
    'modelopt',
    'torch',
    'transformers',
    'onnx',
    'onnx_graphsurgeon',
    'onnxruntime',
    'numpy',
    'PIL',
    'huggingface_hub',
    'datasets',
    'tqdm',
    'safetensors',
]

myst_url_schemes = {
    "http":
    None,
    "https":
    None,
    "source":
    "https://github.com/NVIDIA/TensorRT-Edge-LLM/tree/" + commit_hash +
    "/{{path}}",
}

myst_heading_anchors = 4

myst_enable_extensions = [
    "deflist",
    "substitution",
    "dollarmath",
    "amsmath",
    "colon_fence",  # ::: code fences
]

# Enable mermaid code blocks in MyST markdown
myst_fence_as_directive = ["mermaid"]

myst_substitutions = {
    "version": version,
    "version_quote": f"`{version}`",
}

autosummary_generate = True
copybutton_exclude = '.linenos, .gp, .go'
copybutton_prompt_text = ">>> |$ |# "
copybutton_line_continuation_character = "\\"

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
    '.json': 'json',
}

# Add Pygments lexers
pygments_style = 'sphinx'

templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'nvidia_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    "switcher": {
        "json_url": "./_static/switcher.json",
        "version_match": version,
        "check_switcher": True,
    },
    "extra_footer": [
        f'<p>Last updated on {last_updated}.</p>',
        f'<p>This page is generated by TensorRT-Edge-LLM commit <a href="https://github.com/NVIDIA/TensorRT-Edge-LLM/tree/{commit_hash}">{commit_hash}</a>.</p>'
    ]
}

html_css_files = [
    'custom.css',
]

# Breathe configuration
breathe_default_project = "TensorRT Edge-LLM"
breathe_projects = {"TensorRT Edge-LLM": "../cpp_docs/xml"}

# Breathe configuration for C++ API documentation
breathe_default_members = ('members', )
breathe_domain_by_extension = {
    "h": "cpp",
    "cuh": "cpp",
}
breathe_show_define_initializer = True
breathe_show_enumvalue_initializer = True
breathe_order_parameters_first = False


def tag_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """A custom role for displaying tags."""
    options = options or {}
    content = content or []
    tag_name = text.lower()
    node = nodes.literal(text, text, classes=['tag', tag_name])
    return [node], []


def setup(app):
    app.add_role('tag', tag_role)


# -- Auto-generate API documentation at build time --------------------------

# Generate C++ and Python API RST files at configuration time
generated_modules = generate_module_rst_files()
generate_python_api_rst()
