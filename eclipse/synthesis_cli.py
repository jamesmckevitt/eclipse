#!/usr/bin/env python3
"""
Command line interface for synthesis script.
"""

def main():
    """Entry point for the synthesis command line script."""
    from .synthesis import main as run_synthesis
    run_synthesis()

if __name__ == "__main__":
    main()
