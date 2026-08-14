"""Build scraper instances from configs/sources.yaml."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .article import ArticleScraper
from .base import BaseScraper
from .googlefactcheck import GoogleFactCheckScraper
from .wordpress import WordPressScraper

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sources.yaml"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_scraper(name: str, cfg: dict, defaults: dict) -> BaseScraper | None:
    kind = cfg.get("kind")
    common = {
        "out_dir": cfg.get("out_dir", defaults.get("out_dir", "data")),
        "min_delay": cfg.get("min_delay", defaults.get("min_delay", 1.5)),
        "max_delay": cfg.get("max_delay", defaults.get("max_delay", 3.0)),
    }

    if kind == "wordpress":
        return WordPressScraper(name=name, base_url=cfg["base_url"], **common)
    if kind == "article":
        return ArticleScraper(
            name=name,
            base_url=cfg["base_url"],
            sitemaps=cfg.get("sitemaps", []),
            path_filters=cfg.get("path_filters"),
            article_selector=cfg.get("article_selector"),
            **common,
        )
    if kind == "gfc":
        return GoogleFactCheckScraper(
            restrict_to_india=cfg.get("restrict_to_india", True), **common
        )

    log.error("Unknown source kind %r for %s", kind, name)
    return None


def run_kwargs(cfg: dict) -> dict:
    """Translate config keys into iter_records() kwargs."""
    kind = cfg.get("kind")
    if kind == "wordpress":
        return {
            "search": cfg.get("search"),
            "category_slug": cfg.get("category_slug"),
            "per_page": cfg.get("per_page", 50),
            "max_pages": cfg.get("max_pages", 20),
        }
    if kind == "gfc":
        return {
            "queries": cfg.get("queries"),
            "languages": cfg.get("languages", ["en", "hi"]),
            "publishers": cfg.get("publishers"),
            "max_pages": cfg.get("max_pages", 10),
        }
    return {}
