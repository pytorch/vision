#!/usr/bin/env python3
"""
apply-release-changes.py - Cross-platform script to replace main with a specified release version in YML files

This script performs two replacements in YML files in .github/workflows/:
1. Replaces @main with @release/VERSION
2. Replaces 'test-infra-ref: main' with 'test-infra-ref: release/VERSION'

Usage:
  python apply-release-changes.py VERSION

Example:
  python apply-release-changes.py 2.7
"""

import os
import pathlib
import sys
from typing import Optional


def replace_in_file(file_path: pathlib.Path, old_text: str, new_text: str) -> None:
    """Replace all occurrences of old_text with new_text in the specified file."""
    try:
        # Try reading the file without specifying encoding to use the default
        encoding = None
        try:
            content = file_path.read_text()
        except UnicodeDecodeError:
            # If that fails, try with UTF-8
            encoding = "utf-8"
            content = file_path.read_text(encoding=encoding)

        # Perform the replacement
        new_content = content.replace(old_text, new_text)

        # Only write if changes were made
        if new_content != content:
            # Write with the same encoding we used to read
            if encoding:
                file_path.write_text(new_content, encoding=encoding)
            else:
                file_path.write_text(new_content)
            print(f"Updated: {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def find_repo_root() -> Optional[pathlib.Path]:
    """Find the git repository root by searching for .git directory."""
    # Start from the current directory and traverse upwards
    current_path = pathlib.Path.cwd().absolute()

    while current_path != current_path.parent:
        # Check if .git directory exists
        git_dir = current_path / ".git"
        if git_dir.exists() and git_dir.is_dir():
            return current_path

        # Move up one directory
        current_path = current_path.parent

    # If we get here, we didn't find a repository root
    return None


def main() -> None:
    # Check if version is provided as command line argument
    if len(sys.argv) != 2:
        print("Error: Exactly one version parameter is required")
        print(f"Usage: python {os.path.basename(__file__)} VERSION")
        print("Example: python apply-release-changes.py 2.7")
        sys.exit(1)

    # Get version from command line argument
    version = sys.argv[1]
    print(f"Using release version: {version}")

    # Find the repository root by searching for .git directory
    repo_root = find_repo_root()
    if not repo_root:
        print("Error: Not inside a git repository. Please run from within a git repository.")
        sys.exit(1)

    print(f"Repository root found at: {repo_root}")

    # Get path to workflow directory
    workflow_dir = repo_root / ".github" / "workflows"

    # Process all workflow files and perform both replacements on each file
    for yml_file in workflow_dir.glob("*.yml"):
        replace_in_file(yml_file, "@main", f"@release/{version}")
        replace_in_file(yml_file, "test-infra-ref: main", f"test-infra-ref: release/{version}")


if __name__ == "__main__":
    print("Starting YML updates...")
    main()
    print("YML updates completed.")
