"""Polite, cached HTTP session shared by every scraper.

Design goals:
  * never hammer a fact-checker's server (they are small non-profits)
  * survive transient 5xx / connection resets without losing a whole run
  * cache raw responses on disk so re-parsing does not mean re-fetching
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


class PoliteSession:
    """Rate-limited requests wrapper with on-disk response caching."""

    def __init__(
        self,
        min_delay: float = 1.5,
        max_delay: float = 3.0,
        timeout: int = 30,
        cache_dir: str | Path | None = None,
        user_agent: str = DEFAULT_UA,
        max_retries: int = 4,
    ) -> None:
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
                "Accept": "text/html,application/json,application/xhtml+xml,*/*;q=0.8",
            }
        )
        retry = Retry(
            total=max_retries,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"]),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_at
        wait = random.uniform(self.min_delay, self.max_delay) - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.time()

    def _cache_path(self, url: str, suffix: str) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(url.encode()).hexdigest()[:24]
        return self.cache_dir / f"{key}{suffix}"

    # ------------------------------------------------------------------
    def get_text(self, url: str, *, use_cache: bool = True, **kw: Any) -> str | None:
        cp = self._cache_path(url, ".txt")
        if use_cache and cp and cp.exists():
            return cp.read_text(encoding="utf-8")

        self._throttle()
        try:
            r = self.session.get(url, timeout=self.timeout, **kw)
        except requests.RequestException as e:
            log.warning("GET failed %s -> %s", url, e)
            return None

        if r.status_code != 200:
            log.warning("GET %s -> HTTP %s", url, r.status_code)
            return None

        if cp:
            cp.write_text(r.text, encoding="utf-8")
        return r.text

    def get_json(self, url: str, *, use_cache: bool = True, **kw: Any):
        """Return (payload, headers). Headers matter for WordPress X-WP-Total."""
        cp = self._cache_path(url, ".json")
        hp = self._cache_path(url, ".hdr.json")
        if use_cache and cp and cp.exists():
            import json

            headers = {}
            if hp and hp.exists():
                headers = json.loads(hp.read_text(encoding="utf-8"))
            return json.loads(cp.read_text(encoding="utf-8")), headers

        self._throttle()
        try:
            r = self.session.get(url, timeout=self.timeout, **kw)
        except requests.RequestException as e:
            log.warning("GET failed %s -> %s", url, e)
            return None, {}

        if r.status_code != 200:
            log.warning("GET %s -> HTTP %s", url, r.status_code)
            return None, dict(r.headers)

        try:
            payload = r.json()
        except ValueError:
            log.warning("Non-JSON body at %s", url)
            return None, dict(r.headers)

        if cp:
            import json

            cp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            if hp:
                hp.write_text(json.dumps(dict(r.headers)), encoding="utf-8")
        return payload, dict(r.headers)

    def download(self, url: str, dest: str | Path) -> bool:
        dest = Path(dest)
        if dest.exists() and dest.stat().st_size > 0:
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)

        self._throttle()
        try:
            r = self.session.get(url, timeout=self.timeout, stream=True)
            if r.status_code != 200:
                log.warning("Image %s -> HTTP %s", url, r.status_code)
                return False
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(65536):
                    fh.write(chunk)
        except requests.RequestException as e:
            log.warning("Image download failed %s -> %s", url, e)
            return False
        return True
