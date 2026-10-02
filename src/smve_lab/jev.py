"""Thin client for TypeSafe's Jev (System One) API with a disk cache.

Jev answers typed questions about a "state" (text or JSON). A Noul question
returns P(yes); a Choice returns a probability per option. Every response is
cached on disk, keyed by (model, state, questions), so re-running an
experiment costs nothing and gives identical numbers. The first live call's
latency is stored with it, so cached runs still report real latencies.

The API key is read from TYPESAFE_API_KEY (environment or a git-ignored .env).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from typesafe_sdk import TypeSafeClient

from smve_lab.config import ROOT_DIR

MODEL = "jev-1.13.0"  # pinned version (an alias like jev-latest can change under us)


class Jev:
    def __init__(self, cache_path: Path, model: str = MODEL, workers: int = 8):
        load_dotenv(ROOT_DIR / ".env")
        self.client = TypeSafeClient(model=model)
        self.model, self.workers = model, workers
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec
        self._lock = threading.Lock()

    def _key(self, state, questions: dict) -> str:
        blob = json.dumps({"model": self.model, "state": state, "questions": questions}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def ask(self, state, questions: dict) -> dict:
        """One request. `questions` are plain dicts in the HTTP API format. Returns
        {"answers": {name: P(yes) for Nouls / probabilities dict for Choices},
         "latency_ms": ..., "input_tokens": ..., "cached": bool}."""
        key = self._key(state, questions)
        if key in self.cache:
            return {**self.cache[key], "cached": True}
        t = time.perf_counter()
        r = self.client.system_one(state=state, questions=questions)
        latency = (time.perf_counter() - t) * 1000
        answers = {}
        for name, a in r.answers.items():
            answers[name] = a.noul if a.type == "noul" else dict(a.probabilities)
        rec = {"key": key, "answers": answers, "latency_ms": latency,
               "input_tokens": r.usage.input_tokens or 0, "model": r.model}
        with self._lock:
            self.cache[key] = rec
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return {**rec, "cached": False}

    def ask_many(self, items: list[tuple], desc: str = "jev") -> list[dict]:
        """Run many (state, questions) requests in parallel threads, keeping order."""
        with ThreadPoolExecutor(self.workers) as pool:
            return list(tqdm(pool.map(lambda it: self.ask(*it), items), total=len(items), desc=desc))
