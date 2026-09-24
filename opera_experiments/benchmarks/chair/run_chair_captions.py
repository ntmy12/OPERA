"""
run_chair_captions.py: Compatibility entry point for CHAIR benchmark in OPERA.
Delegates directly to standardized run_chair.py.
"""
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from run_chair import main

if __name__ == "__main__":
    main()
