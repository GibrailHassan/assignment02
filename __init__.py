"""
Initialization file for the 'assignment02' package.

This file can serve multiple purposes:
1. Marking the directory as a Python package, allowing modules within it
   to be imported using dot notation (e.g., `from assignment02.agents import ...`).
2. Executing package-level initialization code. In this case, it modifies
   the system path to potentially allow imports from a parent directory if the
   project structure requires it (e.g., if 'assignment02' is a sub-package
   and needs to access modules from its parent).

Note: Modifying `sys.path` directly is generally discouraged in favor of
using virtual environments and proper project setup (e.g., with `setup.py`
or `pyproject.toml`) that handle path management. However, it can be a
quick way to ensure modules are found in certain project layouts, especially
during development or for specific execution contexts.
"""

import os
import sys

# Construct the path to the parent directory of the current file's directory.
# os.path.dirname(__file__) gives the directory of the current file (e.g., '.../assignment02').
# os.pardir is '..' which refers to the parent directory.
# os.path.join combines these to form the path to the parent of 'assignment02'.
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)

# Add the constructed parent directory path to sys.path.
# This allows Python to search for modules in that directory.
# For example, if your project structure is:
# project_root/
#   common_modules/
#   assignment02/
#     __init__.py
#     main.py
# This line would add 'project_root/' to sys.path, making 'common_modules' importable
# from within 'assignment02' (e.g., `import common_modules.some_module`).
sys.path.append(parent_dir_path)

# It's good practice to ensure that the path added is not already in sys.path
# and that it's an absolute path for clarity, though this script doesn't do that.
# Example of a more robust addition:
#
# parent_dir_abs_path = os.path.abspath(parent_dir_path)
# if parent_dir_abs_path not in sys.path:
#     sys.path.insert(0, parent_dir_abs_path) # Insert at the beginning for higher precedence
