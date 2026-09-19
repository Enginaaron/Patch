import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_session, init_db
from app.models import Item

init_db()

with get_session() as session:
    item = Item(name="Test Item")
    session.add(item)
    session.commit()
    session.refresh(item)
    print(item.id)
