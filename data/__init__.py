import sys
from pathlib import Path

# Get the parent directory of the package
parent_dir = str(Path(__file__).resolve().parent.parent)

# Add the parent directory to sys.path if not already included
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
