"""FastAPI application factory and launcher for the local web app."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import AppConfig, load_config
from ..data.storage import ParquetStore
from ..logging_setup import setup_logging
from .context import AppContext
from .routes import ROUTERS

STATIC = Path(__file__).resolve().parent / "static"


def create_app(cfg: AppConfig | None = None, config_dir: str | None = None) -> FastAPI:
    """Build the app. ``cfg`` wins over ``config_dir`` (default: $QE_CONFIG_DIR or ``config``)."""
    if cfg is None:
        cfg = load_config(config_dir or os.getenv("QE_CONFIG_DIR", "config"))
        setup_logging(cfg.engine.logging.level, cfg.engine.logging.json_format, cfg.engine.logging.file)
    app = FastAPI(title="XAUUSD Quant Engine", docs_url="/api/docs", redoc_url=None)
    app.state.ctx = AppContext(cfg=cfg, store=ParquetStore(cfg.data.storage.root),
                               config_dir=config_dir or os.getenv("QE_CONFIG_DIR", "config"))
    for r in ROUTERS:
        app.include_router(r)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/vendor/plotly.min.js", include_in_schema=False)
    def plotly_js() -> Response:
        """Plotly.js from the installed plotly package, so charts work without a CDN."""
        return Response(_plotly_js(), media_type="application/javascript",
                        headers={"Cache-Control": "max-age=86400"})

    return app


@lru_cache(maxsize=1)
def _plotly_js() -> str:
    from plotly.offline import get_plotlyjs
    return get_plotlyjs()


def serve(config_dir: str | None, port: int, open_browser: bool = True) -> int:
    """Run the web app on 127.0.0.1 only (blocks until Ctrl+C)."""
    import threading
    import webbrowser

    import uvicorn
    app = create_app(config_dir=config_dir)
    url = f"http://localhost:{port}"
    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0
