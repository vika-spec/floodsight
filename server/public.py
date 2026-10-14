"""Guard rails for a PUBLIC deployment (e.g. a free Hugging Face Space shared with the judges).

Switched on with FLOODSIGHT_PUBLIC=1. Without it (normal local use) nothing here applies.

What it protects: new analyses download Copernicus data under OUR account and use its monthly
processing-unit quota. A public link must not let anyone exhaust it. So, in public mode:
  * finished analyses (the showcase runs) and the offline demo are open to everyone;
  * starting a NEW analysis needs the access code (FLOODSIGHT_ACCESS_CODE), given to the judges with the link;
  * area per analysis is capped (FLOODSIGHT_MAX_KM2, default 1600 km2), resolution and backend are fixed;
  * at most FLOODSIGHT_MAX_RUNS_DAY new analyses (default 20) and FLOODSIGHT_PU_DAY Sentinel Hub
    processing units (default 600) per UTC day; usage is kept in runs/_public_ledger.json;
  * at most FLOODSIGHT_RUNS_PER_IP new analyses (default 5) per visitor address on OUR account, in total;
    addresses are kept only as salted hashes (runs/_public_ip_ledger.json);
  * a visitor who enters their OWN Copernicus (Sentinel Hub) keys runs on their own account: our daily and
    per-address limits do not apply to that run and its processing units are not counted against ours.
When our account's limits are reached, the refusal says so and points to the own-keys option.
The code is compared in constant time and never echoed back or stored with the job.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class PublicGuard:
    def __init__(self, runs: Path):
        self.enabled = os.environ.get("FLOODSIGHT_PUBLIC", "").strip() in ("1", "true", "yes")
        self.code = os.environ.get("FLOODSIGHT_ACCESS_CODE", "").strip()
        self.max_km2 = _int("FLOODSIGHT_MAX_KM2", 1600)
        self.max_runs = _int("FLOODSIGHT_MAX_RUNS_DAY", 20)
        self.max_pu = _int("FLOODSIGHT_PU_DAY", 600)
        self.max_per_ip = _int("FLOODSIGHT_RUNS_PER_IP", 5)
        self.ledger_path = Path(runs) / "_public_ledger.json"
        self.ip_path = Path(runs) / "_public_ip_ledger.json"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ ledger
    @staticmethod
    def _today():
        return datetime.now(timezone.utc).date().isoformat()

    def _load(self) -> dict:
        try:
            d = json.loads(self.ledger_path.read_text())
        except Exception:  # noqa: BLE001 - missing or damaged ledger: start fresh
            d = {}
        return d if d.get("day") == self._today() else {"day": self._today(), "runs": 0, "pu": 0.0}

    def usage(self) -> dict:
        with self._lock:
            d = self._load()
        return {"runs_today": d["runs"], "max_runs_day": self.max_runs, "pu_today": round(d["pu"], 1),
                "max_pu_day": self.max_pu}

    def record(self, pu: float) -> None:
        with self._lock:
            d = self._load()
            d["runs"] += 1
            d["pu"] += float(pu or 0)
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self.ledger_path.write_text(json.dumps(d))

    # ------------------------------------------------------------------ per visitor address
    def ip_key(self, ip: str | None) -> str:
        """Salted hash of the visitor address: counts visitors without storing who they are."""
        salt = (self.code or "floodsight").encode()
        return hashlib.sha256(salt + (ip or "unknown").encode()).hexdigest()[:20]

    def _ips(self) -> dict:
        try:
            return json.loads(self.ip_path.read_text())
        except Exception:  # noqa: BLE001
            return {}

    def ip_used(self, ip: str | None) -> int:
        with self._lock:
            return int(self._ips().get(self.ip_key(ip), 0))

    def reserve_ip(self, ip: str | None) -> None:
        with self._lock:
            d = self._ips()
            k = self.ip_key(ip)
            d[k] = int(d.get(k, 0)) + 1
            self.ip_path.parent.mkdir(parents=True, exist_ok=True)
            self.ip_path.write_text(json.dumps(d))

    # ------------------------------------------------------------------ checks
    def status(self, ip: str | None = None) -> dict:
        if not self.enabled:
            return {"enabled": False}
        return {"enabled": True, "access_required": True, "max_km2": self.max_km2, **self.usage(),
                "max_runs_per_visitor": self.max_per_ip,
                "runs_left_for_you": max(self.max_per_ip - self.ip_used(ip), 0)}

    def check(self, code: str | None, area_km2: float | None, ip: str | None = None,
              own_keys: bool = False) -> str | None:
        """Returns a plain-language refusal, or None if the new analysis may start. Refusals caused by OUR
        account's limits start with OWN_KEYS_HINT so the web app can offer the own-keys option."""
        if not self.enabled:
            return None
        if not self.code:
            return "New analyses are switched off on this public server (no access code configured)."
        if not code or not hmac.compare_digest(code.strip().encode(), self.code.encode()):
            return "This public server needs the judges' access code to start a new analysis. " \
                   "Finished analyses and the demo are open to everyone."
        if area_km2 is not None and area_km2 > self.max_km2:
            return f"On the public server an area can be at most {self.max_km2} km²; draw a smaller box."
        if own_keys:
            return None                              # their own account: our quota limits do not apply
        if self.ip_used(ip) >= self.max_per_ip:
            return (f"{OWN_KEYS_HINT} You have used the {self.max_per_ip} new analyses available per visitor on our "
                    "shared account. Enter your own free Copernicus keys to run more.")
        u = self.usage()
        if u["runs_today"] >= self.max_runs:
            return (f"{OWN_KEYS_HINT} Our shared account's daily limit of {self.max_runs} new analyses is reached. "
                    "Enter your own free Copernicus keys to continue, or try again tomorrow (UTC).")
        if u["pu_today"] >= self.max_pu:
            return (f"{OWN_KEYS_HINT} Our shared account's satellite-processing budget for today is used up. "
                    "Enter your own free Copernicus keys to continue, or try again tomorrow (UTC).")
        return None


OWN_KEYS_HINT = "[own-keys]"
