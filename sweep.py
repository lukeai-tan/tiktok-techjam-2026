#!/usr/bin/env python3
"""Compatibility entry point for the fail-closed manifest benchmark runner."""

from benchmarks.run_matrix import main


if __name__ == "__main__":
    raise SystemExit(main())
