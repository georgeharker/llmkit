"""``python -m llmkit.bridge`` → :func:`.cli.main`."""

import sys

from .cli import main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
