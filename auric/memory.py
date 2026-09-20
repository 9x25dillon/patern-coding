"""Local SQLite source index and explicit, revisioned project memory."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3

from .workspace import scan


class KnowledgeStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                root UNINDEXED, path, start UNINDEXED, end UNINDEXED,
                content, hash UNINDEXED);
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY, key TEXT NOT NULL, value TEXT NOT NULL,
                evidence TEXT NOT NULL, created TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS memory_key ON memories(key, id);
            CREATE TABLE IF NOT EXISTS roots (root TEXT PRIMARY KEY, indexed TEXT, stats TEXT);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def index(self, root, **limits):
        chunks, stats = scan(root, **limits)
        root = str(Path(root).resolve())
        # Scan completes before mutation. A failed/budget-exceeded scan leaves
        # the previous index untouched; successful refresh removes stale files.
        with self.db:
            self.db.execute("DELETE FROM chunks WHERE root=?", (root,))
            self.db.executemany("INSERT INTO chunks(root,path,start,end,content,hash) VALUES(?,?,?,?,?,?)",
                                [(c["root"], c["path"], c["start"], c["end"], c["content"], c["hash"]) for c in chunks])
            self.db.execute("INSERT OR REPLACE INTO roots VALUES(?,?,?)", (root, self.now(), json.dumps(stats)))
        return {"root": root, "chunks": len(chunks), **stats}

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def search(self, query, limit=5):
        if not 1 <= limit <= 50:
            raise ValueError("Search limit must be between 1 and 50")
        words = list(dict.fromkeys(re.findall(r"[^\W_]+", query, flags=re.UNICODE)))[:32]
        if not words:
            return []
        # Quote individual terms. User input cannot inject FTS query syntax.
        expr = " OR ".join('"'+w+'"' for w in words)
        rows = self.db.execute("SELECT *, bm25(chunks,0,3,0,0,1,0) AS rank FROM chunks "
                               "WHERE chunks MATCH ? ORDER BY rank LIMIT ?", (expr, limit*3))
        results, seen = [], set()
        for row in rows:
            item = dict(row)
            if item["content"] in seen:
                continue
            seen.add(item["content"])
            item["citation"] = f'{Path(item["root"])/item["path"]}:{item["start"]}'
            results.append(item)
            if len(results) == limit:
                break
        return results

    def remember(self, key, value, evidence):
        if not key.strip() or not value.strip() or not evidence.strip():
            raise ValueError("Memory requires a key, value and evidence/source")
        with self.db:
            cursor = self.db.execute("INSERT INTO memories(key,value,evidence,created) VALUES(?,?,?,?)",
                                     (key, value, evidence, self.now()))
        return cursor.lastrowid

    def forget(self, key):
        with self.db:
            self.db.execute("INSERT INTO memories(key,value,evidence,created,deleted) VALUES(?,?,?,?,1)",
                            (key, "", "explicit forget", self.now()))

    def memories(self, *, history=False):
        query = "SELECT * FROM memories ORDER BY id" if history else (
            "SELECT * FROM memories WHERE id IN (SELECT MAX(id) FROM memories GROUP BY key) "
            "AND deleted=0 ORDER BY id DESC")
        return [dict(r) for r in self.db.execute(query)]
