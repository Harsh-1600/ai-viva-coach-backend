"""Database backup and recovery utility for AI Viva & Interview Preparation Coach."""

import os
import sys
import sqlite3
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so settings can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings


def backup_database(backup_dir: str = "backups") -> str:
    """Creates a consistent point-in-time online snapshot of the SQLite database.
    
    Uses SQLite's online backup API to ensure snapshot consistency even while
    the FastAPI server is serving active candidate requests.
    """
    settings = get_settings()
    if not settings.DATABASE_URL.startswith("sqlite"):
        print(f"[!] Database URL is {settings.DATABASE_URL}. Please use native pg_dump for PostgreSQL.")
        return ""

    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    if not os.path.exists(db_path):
        print(f"[!] Source database file not found at: {db_path}")
        return ""

    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"viva_coach_backup_{timestamp}.db")

    print(f"[*] Starting online backup of '{db_path}' -> '{backup_file}'...")
    src_conn = sqlite3.connect(db_path)
    dst_conn = sqlite3.connect(backup_file)

    with dst_conn:
        src_conn.backup(dst_conn, pages=100, sleep=0.01)

    src_conn.close()
    dst_conn.close()

    size_kb = os.path.getsize(backup_file) / 1024
    print(f"[OK] Backup completed successfully: {backup_file} ({size_kb:.2f} KB)")
    return backup_file


def restore_database(backup_file: str) -> bool:
    """Restores database from a verified backup file."""
    settings = get_settings()
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")

    if not os.path.exists(backup_file):
        print(f"[!] Backup file not found: {backup_file}")
        return False

    print(f"[*] Restoring database from '{backup_file}' -> '{db_path}'...")
    src_conn = sqlite3.connect(backup_file)
    dst_conn = sqlite3.connect(db_path)

    with dst_conn:
        src_conn.backup(dst_conn)

    src_conn.close()
    dst_conn.close()

    print(f"[OK] Database restored successfully.")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--restore":
        restore_database(sys.argv[2])
    else:
        backup_database()
