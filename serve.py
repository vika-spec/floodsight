"""Start the FloodSight web app.

    python serve.py                 # http://127.0.0.1:8000, opens your browser
    python serve.py --port 8080 --no-browser
    python serve.py --host 0.0.0.0  # let other devices on your network open it (judging-room projector)
"""
from __future__ import annotations

import argparse
import threading
import webbrowser


def main(argv=None):
    ap = argparse.ArgumentParser(description="FloodSight web app")
    import os
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    import uvicorn

    from floodsight.core import setup_logging
    from server.api import create_app

    setup_logging()
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}"
    print(f"\n  FloodSight is running at {url}   (Ctrl+C to stop)\n")
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
