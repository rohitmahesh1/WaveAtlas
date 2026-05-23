from __future__ import annotations

from .api.app_factory import create_app
from .db import init_db

init_db()

app = create_app()
