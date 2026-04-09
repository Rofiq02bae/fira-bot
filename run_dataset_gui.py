#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launcher script for Dataset Processing GUI
"""

import sys
import os
from pathlib import Path
import tkinter as tk
# Add scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import and run GUI
from scripts.gui_pipeline import DatasetPipelineGUI

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetPipelineGUI(root)
    root.mainloop()
