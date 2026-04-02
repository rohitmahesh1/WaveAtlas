from __future__ import annotations

from .api.app_factory import create_app
from .db import init_db

# For MVP you can set DB_CREATE_ALL=1 to bootstrap tables on startup.
# In production, keep DB_CREATE_ALL=0 and use migrations.
init_db()

app = create_app()
