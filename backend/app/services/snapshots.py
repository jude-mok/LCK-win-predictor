"""Append-only prediction storage. SQLite requires a persistent volume in deployment."""
from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path


class Conflict(Exception):
    pass


class SnapshotStore:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS predictions (series_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, document TEXT NOT NULL)')
            for operation in ('UPDATE', 'DELETE'):
                db.execute(f"CREATE TRIGGER IF NOT EXISTS deny_{operation.lower()} BEFORE {operation} ON predictions BEGIN SELECT RAISE(ABORT, 'Predictions are immutable'); END")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def ready(self):
        with self.connect() as db:
            return db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'

    def get(self, series_id):
        with self.connect() as db:
            row = db.execute('SELECT document FROM predictions WHERE series_id = ?', (series_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def many(self, ids):
        if not ids:
            return {}
        with self.connect() as db:
            rows = db.execute(f"SELECT series_id, document FROM predictions WHERE series_id IN ({','.join('?' for _ in ids)})", ids).fetchall()
        return {key: json.loads(value) for key, value in rows}

    def insert(self, document):
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO predictions VALUES (?, ?, ?)',
                       (document['series_id'], document['request_hash'], json.dumps(document, allow_nan=False)))
            row = db.execute('SELECT request_hash, document FROM predictions WHERE series_id = ?', (document['series_id'],)).fetchone()
            if row[0] != document['request_hash']:
                raise Conflict('A different prediction already exists for this series')
            return json.loads(row[1])
