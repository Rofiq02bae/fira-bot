#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper untuk fix encoding di Windows console.
Import di awal setiap script untuk support emoji dan Unicode.
"""

import sys
import os
from pathlib import Path

def fix_console_encoding():
    """
    Fix Windows console encoding untuk support Unicode/emoji.
    Harus dipanggil sebelum print dengan emoji.
    """
    if sys.platform == 'win32':
        try:
            import io
            # Ganti stdout dan stderr dengan wrapper UTF-8
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, 
                encoding='utf-8', 
                errors='replace',
                line_buffering=True
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, 
                encoding='utf-8', 
                errors='replace',
                line_buffering=True
            )
        except Exception:
            # Jika gagal, tidak apa-apa (fallback ke default)
            pass

def get_data_path(filename: str) -> str:
    """
    Get correct data path regardless of where script is run from.
    
    Args:
        filename: Filename in data/dataset/ folder
        
    Returns:
        str: Correct absolute or relative path to the file
    """
    # Try project root first
    project_root = Path(__file__).parent.parent
    data_file = project_root / "data" / "dataset" / filename
    
    if data_file.exists():
        return str(data_file)
    
    # Fallback to relative path
    return f"data/dataset/{filename}"

# Auto-fix saat import
fix_console_encoding()
