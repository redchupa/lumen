"""Generate reference fixtures for differential testing.

Usage:
    python tools/gen_fixtures.py

Reads test specs from tools/fixtures_spec.toml (TBD) and emits binary
input/output pairs under tests/fixtures/. Each file format:

    magic: b"LUMF"           # 4 bytes
    version: u32 LE          # 1
    rank: u32 LE             # 0..=8
    dims: [u32; rank] LE
    dtype: u32 LE            # 0=f32 1=f16 2=i32
    payload: dims.product() * dtype_size

Two consecutive blocks per file: input, expected_output.
"""

import argparse
import struct
import sys
from pathlib import Path


def main() -> int:
    print("gen_fixtures.py: not implemented yet (Phase 2 work).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
