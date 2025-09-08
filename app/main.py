from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from .wipe_client import WipeClient


logger = logging.getLogger("client")


async def main() -> None:
    client = WipeClient()
    await client.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass