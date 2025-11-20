#!/usr/bin/env python
"""
Launcher script for doctor_brain.py
This allows running from the project root: python doctor_brain.py
"""
import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# Import and execute
if __name__ == "__main__":
    # Import the module (this will execute the __main__ block in src/doctor_brain.py)
    import runpy
    runpy.run_path(os.path.join(src_path, 'doctor_brain.py'), run_name='__main__')

