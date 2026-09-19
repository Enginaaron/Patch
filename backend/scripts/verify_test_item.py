import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_session, init_db
from app.models import Item

init_db()

if len(sys.argv) != 2:
    print("usage: verify_test_item.py <item_id>")
    sys.exit(1)

item_id = sys.argv[1]

with get_session() as session:
    item = session.get(Item, item_id)
    if item is None:
        print("NOT FOUND")
        sys.exit(1)
    print(f"FOUND: {item.id} {item.name} {item.created_at}")
