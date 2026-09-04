import sys
import os
from pathlib import Path

# Add project root directory to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from app import app
