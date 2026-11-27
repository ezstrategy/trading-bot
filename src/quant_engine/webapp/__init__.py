"""Local web app (FastAPI JSON API + a static single-page front end).

Start it with ``python -m quant_engine dashboard`` (or ``start_dashboard.bat``).
It listens on 127.0.0.1 only. SIGNALS ONLY: no endpoint places, changes or
closes trades; MT5 access goes through the same read-only service layer as the CLI.
"""
