"""AMFI image candidates with source provenance and pending human review.

AMFI provenance is a candidate source, not a verified class label. The collector
does not invent a social caption or copy approval language into model inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit

from PIL import Image

from ..schema.records import FactCheckRecord, ImageAsset, LABEL_UNKNOWN
from ..utils.http import PoliteSession
from ..utils.logging_setup import setup

SITEMAP_URL = "https://www.mutualfundssahihai.com/sitemap-0.xml"
PAGE_HOSTS = {"www.mutualfundssahihai.com", "mutualfundssahihai.com"}
MEDIA_HOST = "media.mutualfundssahihai.com"
NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9",
      "i": "http://www.google.com/schemas/sitemap-image/1.1"}
_SKIP = re.compile(
    r"(?:^|[-_.])(?:logo|icon|favicon|sprite|arrow|badge|footer|header|menu|calculator)"
    r"(?:[-_.]|$)", re.I,
)


@dataclass(frozen=True)
class Candidate:
    image_url: str
    page_url: str
    language: str
    parent_pages: tuple[str, ...]
    page_languages: tuple[str, ...]


def _trusted_https(url: str, hosts: set[str]) -> bool:
    try:
        parts = urlsplit(url)
        return (parts.scheme == "https" and parts.hostname in hosts
                and parts.port in {None, 443} and not parts.username and not parts.password)
    except ValueError:
        return False


def image_url(raw_url: str) -> str | None:
    """Unwrap a first-party Next image URL and enforce an exact media host."""
    raw_url = raw_url.strip()
    parts = urlsplit(raw_url)
    if _trusted_https(raw_url, PAGE_HOSTS) and parts.path == "/_next/image":
        values = parse_qs(parts.query).get("url", [])
        if len(values) != 1:
            return None
        raw_url = values[0]
    if not _trusted_https(raw_url, {MEDIA_HOST}):
        return None
    parts = urlsplit(raw_url)
    return urlunsplit(("https", MEDIA_HOST, parts.path, parts.query, ""))


def discover_candidates(sitemap_text: str, languages=("en", "hi")) -> list[Candidate]:
    """Keep each image's real parent pages; interleave en/hi, then deduplicate URLs.

    Language is the page-path metadata, not a claim about recognized image text.
    One URL appearing on several pages remains one candidate with all parents.
    """
    root = ET.fromstring(sitemap_text)
    if root.tag != f"{{{NS['s']}}}urlset":
        raise ValueError("expected a URL sitemap with image nodes")
    occurrences, parents, page_languages = [], defaultdict(set), defaultdict(set)
    for node in root.findall("s:url", NS):
        page_url = (node.findtext("s:loc", default="", namespaces=NS) or "").strip()
        if not _trusted_https(page_url, PAGE_HOSTS):
            continue
        segments = urlsplit(page_url).path.strip("/").split("/")
        language = segments[0] if segments else ""
        if language not in languages:
            continue
        for image in node.findall("i:image", NS):
            direct = image_url(image.findtext("i:loc", default="", namespaces=NS) or "")
            if not direct:
                continue
            filename = unquote(Path(urlsplit(direct).path).name)
            if _SKIP.search(filename):
                continue
            occurrences.append((language, page_url, direct))
            parents[direct].add(page_url)
            page_languages[direct].add(language)
    queues = {language: deque(sorted(set((page, url) for lang, page, url in occurrences
                                         if lang == language))) for language in languages}
    ordered, seen = [], set()
    while any(queues.values()):
        for language in languages:
            while queues[language]:
                page_url, direct = queues[language].popleft()
                if direct in seen:
                    continue
                seen.add(direct)
                ordered.append(Candidate(direct, page_url, language,
                                         tuple(sorted(parents[direct])),
                                         tuple(sorted(page_languages[direct]))))
                break
    return ordered


def make_record(candidate: Candidate) -> FactCheckRecord:
    """Represent source metadata without invented text or an automatic real label."""
    filename = unquote(Path(urlsplit(candidate.image_url).path).name)
    source_id = hashlib.sha256(candidate.image_url.encode()).hexdigest()[:24]
    return FactCheckRecord(
        source="amfi", source_id=source_id, url=candidate.page_url,
        title="", claim_text="", body_text="", verdict_raw="", label=LABEL_UNKNOWN,
        language=candidate.language,
        images=[ImageAsset(url=candidate.image_url, role="unknown")],
        raw={"suggested_label": "real", "review_status": "pending",
             "origin": candidate.image_url, "parent_pages": list(candidate.parent_pages),
             "page_languages": list(candidate.page_languages),
             "language_basis": "parent page path; image text not verified",
             "image_filename": filename,
             "filename_title": re.sub(r"[-_]+", " ", Path(filename).stem).strip(),
             "discovery_sitemap": SITEMAP_URL},
    )


def select_manifest_candidates(candidates: list[Candidate], manifest_path: str | Path,
                               tier: str = "first_batch"):
    """Validate every manifest URL/page pair against actual sitemap discovery.

    The file prioritizes retrieval, never labels. Unknown, duplicate or
    untrusted entries block the entire filtered run before any image download.
    A requested parent may differ from discovery's representative page, but it
    must be one of that exact image's sitemap parents.
    """
    tiers = {"first_batch", "boundary_probe", "possible_variant"}
    report = {"path": str(Path(manifest_path).resolve()), "sha256": None,
              "requested_tier": tier, "discovered_candidates": len(candidates),
              "manifest_entries": 0, "selected_candidates": 0,
              "filtered_other_tiers": 0, "errors": []}
    if tier not in tiers | {"all"}:
        report["errors"].append("unknown candidate tier")
        return [], report
    try:
        payload = Path(manifest_path).read_bytes()
        report["sha256"] = hashlib.sha256(payload).hexdigest()
        manifest = json.loads(payload.decode("utf-8-sig"))
        entries = manifest.get("candidates") if isinstance(manifest, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError("candidate file must contain a nonempty candidates list")
    except (OSError, UnicodeError, ValueError) as exc:
        report["errors"].append(f"candidate file could not be read: {exc}")
        return [], report
    report["manifest_entries"] = len(entries)
    known = {candidate.image_url: candidate for candidate in candidates}
    selected, seen = [], set()
    for index, entry in enumerate(entries, 1):
        errors = []
        if not isinstance(entry, dict):
            report["errors"].append(f"entry {index}: expected an object")
            continue
        url = entry.get("image_url")
        page = entry.get("source_page_url", entry.get("page_url"))
        selected_tier = entry.get("priority_tier")
        if not isinstance(url, str) or not _trusted_https(url, {MEDIA_HOST}):
            errors.append("untrusted or missing image_url")
        elif url in seen:
            errors.append("duplicate image_url")
        else:
            seen.add(url)
        if not isinstance(page, str) or not _trusted_https(page, PAGE_HOSTS):
            errors.append("untrusted or missing source_page_url")
        if "page_url" in entry and "source_page_url" in entry and entry["page_url"] != page:
            errors.append("conflicting parent page fields")
        if not isinstance(selected_tier, str) or selected_tier not in tiers:
            errors.append("unknown or missing priority_tier")
        candidate = known.get(url) if isinstance(url, str) else None
        if candidate is None:
            errors.append("image_url absent from actual sitemap candidates")
        elif page not in candidate.parent_pages:
            errors.append("parent page does not match this image in actual sitemap")
        elif entry.get("uid") and entry["uid"] != make_record(candidate).uid:
            errors.append("UID does not match exact image URL identity")
        if errors:
            report["errors"].append(f"entry {index}: {'; '.join(errors)}")
            continue
        if tier != "all" and selected_tier != tier:
            report["filtered_other_tiers"] += 1
            continue
        language = urlsplit(page).path.strip("/").split("/")[0]
        selected.append(Candidate(url, page, language, candidate.parent_pages,
                                  candidate.page_languages))
    report["selected_candidates"] = len(selected)
    if not selected and not report["errors"]:
        report["errors"].append("no candidates match the requested tier")
    return ([] if report["errors"] else selected), report


def download_candidate(http, record: FactCheckRecord, data_dir: Path,
                       min_dimension: int = 120, min_bytes: int = 512) -> bool:
    """Validate decoded dimensions/format before publishing a downloaded file."""
    asset = record.images[0]
    suffix = Path(urlsplit(asset.url).path).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        suffix = ".img"
    destination = data_dir / "images" / "amfi" / f"{record.uid}{suffix}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def inspect(path):
        if path.stat().st_size < min_bytes:
            raise ValueError("image file is too small")
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        if min(width, height) < min_dimension:
            raise ValueError("image dimensions are too small")
        return width, height

    try:
        try:
            dimensions = inspect(destination)
        except (OSError, ValueError, SyntaxError):
            # PoliteSession can reuse a nonempty destination. A separate fresh
            # partial path prevents a prior interrupted transfer being reused.
            temporary.unlink(missing_ok=True)
            if not http.download(asset.url, temporary):
                return False
            dimensions = inspect(temporary)
            temporary.replace(destination)
        asset.width, asset.height = dimensions
        asset.local_path = destination.relative_to(data_dir).as_posix()
        asset.sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
        return True
    except (OSError, ValueError, SyntaxError) as exc:
        logging.getLogger(__name__).warning("Rejected image %s: %s", asset.url, exc)
        return False
    finally:
        temporary.unlink(missing_ok=True)


def collect_amfi(out_dir: str | Path = "data", limit: int = 30,
                 download_images: bool = True, languages=("en", "hi"), http=None,
                 candidate_file: str | Path | None = None, tier: str = "first_batch") -> dict:
    """Append at most limit new pending-review candidates, preserving existing rows."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    data_dir = Path(out_dir)
    raw_path = data_dir / "raw" / "amfi.jsonl"
    report_path = data_dir / "processed" / "amfi_collection_report.json"
    report = {"source": "amfi", "sitemap_url": SITEMAP_URL, "status": "failed",
              "pilot_limit": limit, "candidate_count": 0, "existing_records": 0,
              "new_records": 0, "download_failures": 0, "language_counts": {},
              "legacy_labelled_uids": [], "errors": [],
              "review_notice": "New candidates remain unknown; provenance does not verify a label."}

    def finish():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        return report

    http = http or PoliteSession(cache_dir=data_dir / ".httpcache" / "amfi")
    text = http.get_text(SITEMAP_URL)
    if not text:
        report["errors"].append("sitemap could not be read; inspect HTTP/robots logs")
        return finish()
    try:
        candidates = discover_candidates(text, languages)
    except (ET.ParseError, ValueError) as exc:
        report["errors"].append(f"sitemap parse failure: {exc}")
        return finish()
    report["candidate_count"] = len(candidates)
    if not candidates:
        report["errors"].append("no eligible image candidates discovered")
        return finish()
    if candidate_file is not None:
        candidates, candidate_filter = select_manifest_candidates(candidates, candidate_file, tier)
        report["candidate_filter"] = candidate_filter
        report["candidate_count"] = len(candidates)
        if candidate_filter["errors"]:
            report["errors"].extend(candidate_filter["errors"])
            return finish()
    seen_urls, seen_uids = set(), set()
    if raw_path.exists():
        for line_number, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = FactCheckRecord.from_dict(json.loads(line))
            except (ValueError, TypeError, KeyError) as exc:
                report["errors"].append(f"existing output line {line_number}: {exc}")
                return finish()
            seen_uids.add(rec.uid)
            seen_urls.update(a.url for a in rec.images)
            if rec.label != LABEL_UNKNOWN:
                report["legacy_labelled_uids"].append(rec.uid)
            report["existing_records"] += 1
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    language_counts = Counter()
    with raw_path.open("a", encoding="utf-8") as handle:
        for candidate in candidates:
            if report["new_records"] >= limit:
                break
            rec = make_record(candidate)
            if candidate_file is not None:
                rec.raw["collection_candidate_manifest_sha256"] = report["candidate_filter"]["sha256"]
                rec.raw["collection_candidate_tier"] = tier
            if rec.uid in seen_uids or candidate.image_url in seen_urls:
                continue
            if download_images and not download_candidate(http, rec, data_dir):
                report["download_failures"] += 1
                continue
            handle.write(rec.to_json() + "\n")
            handle.flush()
            seen_uids.add(rec.uid)
            seen_urls.add(candidate.image_url)
            report["new_records"] += 1
            language_counts[rec.language] += 1
    report["language_counts"] = dict(sorted(language_counts.items()))
    if report["download_failures"] and not report["new_records"]:
        report["errors"].append("all attempted image downloads failed validation or access")
    else:
        report["status"] = "partial" if report["download_failures"] else "ok"
    return finish()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--limit", type=int, default=30, help="new candidates in this pilot (default: 30)")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--languages", nargs="+", choices=["en", "hi"], default=["en", "hi"])
    parser.add_argument("--candidate-file", help="validated priority JSON; default selects only first_batch entries")
    parser.add_argument("--tier", choices=["first_batch", "boundary_probe", "possible_variant", "all"],
                        default="first_batch", help="priority tier when --candidate-file is used")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.limit <= 0:
        parser.error("--limit must be positive")
    setup(args.verbose)
    report = collect_amfi(args.data_dir, args.limit, not args.no_images,
                          tuple(dict.fromkeys(args.languages)),
                          candidate_file=args.candidate_file, tier=args.tier)
    print(f"AMFI: {report['status']}; {report['candidate_count']} candidates, "
          f"{report['new_records']} new pending review, {report['download_failures']} download failures")
    print(f"Report: {Path(args.data_dir) / 'processed' / 'amfi_collection_report.json'}")
    return 2 if report["status"] == "failed" else 0
