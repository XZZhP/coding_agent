"""支持 python -m coding_agent 启动。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
