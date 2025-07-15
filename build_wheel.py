#!/usr/bin/env python3
"""
Script to fix imports for packaging YOLOv9
This script temporarily modifies import statements to use absolute imports
for proper packaging.
"""

import os
import re
import glob
import shutil
from pathlib import Path

def backup_files(files):
    """Create backup of files before modification"""
    backups = []
    for file_path in files:
        backup_path = file_path + '.backup'
        shutil.copy2(file_path, backup_path)
        backups.append(backup_path)
    return backups

def restore_files(backup_files):
    """Restore files from backup"""
    for backup_path in backup_files:
        original_path = backup_path.replace('.backup', '')
        shutil.copy2(backup_path, original_path)
        os.remove(backup_path)

def fix_imports_in_file(file_path):
    """Fix imports in a single file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern replacements for common import patterns
    replacements = [
        # from utils.xxx import yyy -> from yolov9_vx.utils.xxx import yyy
        (r'from utils\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*) import', 
         r'from yolov9_vx.utils.\1 import'),
        
        # from models.xxx import yyy -> from yolov9_vx.models.xxx import yyy
        (r'from models\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*) import', 
         r'from yolov9_vx.models.\1 import'),
        
        # from utils import xxx -> from yolov9_vx.utils import xxx
        (r'from utils import', r'from yolov9_vx.utils import'),
        
        # from models import xxx -> from yolov9_vx.models import xxx
        (r'from models import', r'from yolov9_vx.models import'),
        
        # import utils.xxx -> import yolov9_vx.utils.xxx
        (r'import utils\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)', 
         r'import yolov9_vx.utils.\1'),
        
        # import models.xxx -> import yolov9_vx.models.xxx
        (r'import models\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)', 
         r'import yolov9_vx.models.\1'),
    ]
    
    modified = False
    for pattern, replacement in replacements:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            modified = True
            content = new_content
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed imports in: {file_path}")
    
    return modified

def main():
    """Main function to fix all imports"""
    # Find all Python files in models, utils, and tools directories
    python_files = []
    for directory in ['models', 'utils', 'tools']:
        if os.path.exists(directory):
            pattern = os.path.join(directory, '**', '*.py')
            python_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(python_files)} Python files to check")
    
    # Create backups
    backups = backup_files(python_files)
    
    try:
        # Fix imports
        modified_files = []
        for file_path in python_files:
            if fix_imports_in_file(file_path):
                modified_files.append(file_path)
        
        print(f"Modified {len(modified_files)} files")
        
        # Build the wheel
        print("Building wheel...")
        os.system("python3 setup.py bdist_wheel")
        
        print("Wheel built successfully!")
        
    finally:
        # Restore original files
        print("Restoring original files...")
        restore_files(backups)
        print("Files restored")

if __name__ == "__main__":
    main()
