#!/usr/bin/env python3
"""
Script to fix the logging issues in entity_processor.py
"""

import sys
import re
from pathlib import Path

def fix_entity_logging():
    """Fix the mixed logging calls in entity_processor.py"""

    entity_file = Path(__file__).parent / "core" / "entity_processor.py"

    print(f"Fixing logging issues in: {entity_file}")

    # Read the file
    with open(entity_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace all the old-style logging calls with new-style
    replacements = [
        # Replace LogLevel.DEBUG calls
        (r'detailed_logger\.log\(LogLevel\.DEBUG,\s*"([^"]*)",', r'detailed_logger.log_debug("ENTITY", "\1",'),

        # Replace LogLevel.API calls
        (r'detailed_logger\.log\(LogLevel\.API,\s*"([^"]*)"', r'detailed_logger.log_info("API", "\1"'),

        # Replace LogLevel.SUCCESS calls
        (r'detailed_logger\.log\(LogLevel\.SUCCESS,\s*f?"([^"]*)"', r'detailed_logger.log_info("ENTITY", f"\1"'),

        # Replace LogLevel.WARNING calls
        (r'detailed_logger\.log\(LogLevel\.WARNING,\s*"([^"]*)"', r'detailed_logger.log_warning("ENTITY", "\1"'),

        # Replace LogLevel.ERROR calls
        (r'detailed_logger\.log\(LogLevel\.ERROR,\s*"([^"]*)"', r'detailed_logger.log_error("ENTITY", "\1"'),

        # Replace get_phase_logger calls with DetailedLogger
        (r'detailed_logger = get_phase_logger\("ectd"\)', r'detailed_logger = DetailedLogger(phase="ectd")'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    # Write the fixed content back
    with open(entity_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✓ Fixed logging calls in entity_processor.py")
    return True

if __name__ == "__main__":
    success = fix_entity_logging()
    if success:
        print("✓ All logging issues fixed!")
    else:
        print("✗ Failed to fix logging issues!")
        sys.exit(1)