import json
import os
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), "taxonomy.json")
SUGGESTION_TABLE = "category_suggestions"


class TaxonomyManager:
    def __init__(self):
        self.taxonomy = self._load_taxonomy()
        # cache common structures for convenience
        self.aliases = self.taxonomy.get("aliases", {})
        self.tree = self.taxonomy.get("tree", {})

    def _load_taxonomy(self) -> dict:
        if os.path.exists(TAXONOMY_PATH):
            try:
                with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load taxonomy.json: {e}")
        # minimal fallback structure
        return {"aliases": {}, "tree": {}}

    def normalize(self, label: str) -> Optional[List[str]]:
        """Return category path (list) if known; otherwise None."""
        if not label:
            return None
        key = label.strip().lower()
        # check aliases
        if key in self.aliases:
            return self.aliases[key].get("path")

        # try to find node by name in tree
        for node, data in self.tree.items():
            if key == node.lower():
                return data.get("path")
        return None

    def suggest_unknown(self, label: str, db_conn):
        """Insert a review suggestion for unknown label (human-in-loop). Best-effort.
        db_conn should be a psycopg2 connection.
        """
        if not db_conn:
            return
        try:
            with db_conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {SUGGESTION_TABLE} (label, created_at, reviewed) VALUES (%s, current_timestamp, false)",
                    (label,)
                )
            db_conn.commit()
            logger.info(f"Inserted taxonomy suggestion for: {label}")
        except Exception as e:
            logger.error(f"Failed to insert taxonomy suggestion: {e}")
            

    def match_multiple(self, text: str):
        """Return all matching category paths from taxonomy aliases within the text."""
        matches = []
        lower_text = text.lower()
        for alias, info in self.aliases.items():
            if alias in lower_text:
                path = info.get("path")
                if path and path not in matches:
                    matches.append(path)
        return matches

