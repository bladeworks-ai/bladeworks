"""Compatibility entry point for the Bladeworks foreground server.

The supported command is ``fcpxml server run PATH``. Running this module
directly forwards through the same CLI so there is only one server lifecycle.
"""

from __future__ import annotations

import sys

from ..cli import main as cli_main


def main() -> int:
    return cli_main(("server", "run", *sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
