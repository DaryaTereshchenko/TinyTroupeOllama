import os
import logging
import configparser
import rich # for rich console output
import rich.jupyter

# add current path to sys.path
import sys
sys.path.append('.')
from tinytroupe import utils # now we can import our utils

# if not os.getenv("SUPPRESS_TINYTRIOU_STARTUP_OUTPUT"):
#     print(\
#     """
#     !!!!
#     DISCLAIMER: TinyTroupe relies on Artificial Intelligence (AI) models to generate content. 
#     The AI models are not perfect and may produce inappropriate or inacurate results. 
#     For any serious or consequential use, please review the generated content before using it.
#     !!!!
#     """)


config = utils.read_config_file(verbose=False)
# utils.pretty_print_config(config)
utils.start_logger(config)

# fix an issue in the rich library: we don't want margins in Jupyter!
rich.jupyter.JUPYTER_HTML_FORMAT = \
    utils.inject_html_css_style_prefix(rich.jupyter.JUPYTER_HTML_FORMAT, "margin:0px;")

"""TinyTroupe package initialization."""

# Make all core modules available when importing the package
from . import agent
from . import factory
from . import extraction
from . import control

# Explicitly import hardcoded_personas to make it available in the package
try:
    from . import hardcoded_personas
except ImportError as e:
    print(f"Warning: Failed to import hardcoded_personas: {e}")

# Optional: For convenience, import commonly used functions directly
# This allows imports like: from tinytroupe import get_random_persona
from .hardcoded_personas import get_random_persona, get_all_personas

# Package metadata
__version__ = '0.1.0'

