"""Evidence-assisted SEC review of the confirmed-restatement queue.

Searches public SEC EDGAR full text for restatement/non-reliance disclosures
through each row's preselected horizon, retrieves matching filing documents,
and writes a separate review-ledger copy. High-confidence positives are
assigned only from primary SEC text. Rows with ambiguous matches remain pending.

The search is an auditable screening aid, not a substitute for expert legal or
accounting judgment. Do not treat keyword-search negatives as natural-prevalence
ground truth without a human audit.
"""
from __future__ import annotations

import argparse
from datetime import date
import html
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


EFTS = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
SEARCH_TERMS = (
    'restatement OR restated OR restating OR "non-reliance" OR '
    '"no longer be relied upon" OR "previously issued financial statements" OR '
    '"accounting error" OR "material misstatement" OR "Item 4.02"'
)
FORMS = "8-K,8-K/A,10-K,10-K/A,10-Q,10-Q/A,20-F,20-F/A,40-F,40-F/A,6-K,6-K/A"
EVENT_FORMS = "8-K,8-K/A"
REVIEWER = "OpenAI Codex (AI-assisted SEC review)"
SEC_PAUSE_SECONDS = 0.5


class SecAccess:
    def __init__(self, user_agent: str):
        if not re.search(r"\S+@\S+\.\S+", user_agent) or " " not in user_agent:
            raise ValueError("Provide a real name/organization and contact email for SEC User-Agent.")
        self.user_agent = user_agent.strip()
        self.last_request = 0.0

    def get(self, url: str, limit: int = 30 * 1024 * 1024) -> bytes:
        req = Request(url, headers={
            "User-Agent": self.user_agent,
            "Accept-Encoding": "identity",
            "Accept": "application/json,text/html,text/plain,*/*",
        })
        for attempt in range(4):
            delay = SEC_PAUSE_SECONDS - (time.monotonic() - self.last_request)
            if delay > 0:
                time.sleep(delay)
            self.last_request = time.monotonic()
            try:
                with urlopen(req, timeout=60) as response:
                    chunks, size = [], 0
                    while True:
                        block = response.read(64 * 1024)
                        if not block:
                            break
                        size += len(block)
                        if size > limit:
                            raise ValueError(f"SEC response exceeded {limit:,} bytes")
                        chunks.append(block)
                    return b"".join(chunks)
            except HTTPError as exc:
                if exc.code == 403:
                    raise RuntimeError("SEC returned HTTP 403; stopped all requests as required.") from exc
                if exc.code == 429:
                    raise RuntimeError("SEC returned HTTP 429; stopped to avoid retrying into a rate limit.") from exc
                if exc.code not in (500, 502, 503, 504) or attempt == 3:
                    raise
                retry_after = exc.headers.get("Retry-After", "")
                try:
                    wait = min(120, max(10, int(retry_after)))
                except ValueError:
                    wait = min(120, 10 * 2 ** attempt)
                print(f"SEC HTTP {exc.code}; waiting {wait}s before retry {attempt + 1}/3", flush=True)
                time.sleep(wait)
            except (URLError, TimeoutError) as exc:
                raise RuntimeError(f"SEC request failed; cached progress is retained: {exc}") from exc


def norm_cik(value) -> str:
    return str(value).strip().split(".")[0].lstrip("0") or "0"


def filing_url(cik: str, accession: str, filename: str | None = None) -> str:
    cik_num = str(int(cik))
    accession_dir = accession.replace("-", "")
    suffix = f"/{filename}" if filename else f"/{accession}.txt"
    return f"{ARCHIVES}/{cik_num}/{accession_dir}{suffix}"


def efts_query(client: SecAccess, *, ciks: list[str], start: date, end: date,
               query: str, forms: str, items: str | None = None) -> tuple[list[dict], int]:
    results = []
    offset = 0
    page_size = 100
    total = None
    while total is None or offset < total:
        params = {
            "q": query,
            "forms": forms,
            "dateRange": "custom",
            "startdt": start.isoformat(),
            "enddt": end.isoformat(),
            "ciks": ",".join(ciks),
            "from": str(offset),
            "size": str(page_size),
        }
        if items:
            params["items"] = items
        url = f"{EFTS}?{urlencode(params)}"
        data = json.loads(client.get(url).decode("utf-8"))
        hits = data.get("hits", {})
        total = int(hits.get("total", {}).get("value", 0))
        page = hits.get("hits", [])
        results.extend(page)
        if not page:
            break
        offset += len(page)
    return results, int(total or 0)


def visible_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"(?is)<(script|style|ix:header|ix:hidden)\b.*?</\1\s*>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def excerpt_at(text: str, pattern: str, width: int = 480) -> str:
    found = re.search(pattern, text, flags=re.I)
    if not found:
        return text[:width]
    start = max(0, found.start() - width // 3)
    return text[start:min(len(text), start + width)].strip()


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def assess_event(text: str, form: str, items: list[str]) -> tuple[str, str]:
    """Conservative outcome assessment; ambiguous uses status='uncertain'."""
    lower = text.lower()
    no_restatement = re.search(
        r"\b(?:no|not|without)\s+(?:a\s+)?restatement\b|"
        r"\brestatement\s+(?:is\s+)?not\s+(?:required|necessary|expected)\b|"
        r"\bwill\s+not\s+restate\b", lower)
    has_accounting_error = re.search(
        r"\b(?:material\s+)?(?:accounting\s+)?(?:errors?|misstatements?|irregularit(?:y|ies))\b|"
        r"\bincorrect(?:ly)?\s+(?:recorded|reported|accounted)\b|"
        r"\baccounting\s+adjustment\b", lower)
    has_statement_scope = re.search(
        r"\b(?:financial\s+statements?|previously\s+issued|prior[- ]period|"
        r"annual\s+reports?|quarterly\s+reports?)\b", lower)
    has_nonreliance = re.search(
        r"should\s+no\s+longer\s+be\s+relied\s+upon|non[- ]reliance|"
        r"not\s+be\s+relied\s+upon", lower)
    explicit_restatement = re.search(
        r"\b(?:restatement|restated|restating)\b", lower)
    actual_restatement_action = re.search(
        r"\b(?:will|has|have|is|are|was|were|expects?\s+to|plans?\s+to|"
        r"intends?\s+to|must)\s+(?:re-)?restat(?:e|ed|ing)\b|"
        r"\b(?:financial\s+statements?|prior[- ]period\s+amounts?)\s+"
        r"(?:will\s+be|have\s+been|has\s+been|were|was|are\s+being|is\s+being)\s+restat(?:ed|ing)\b|"
        r"\brestat(?:ement|ed)\b.{0,180}\b(?:financial\s+statements?|prior[- ]period)\b",
        lower)
    possible_language = re.search(
        r"\b(?:potential(?:ly)?|possible|may|might|could|would|risk\s+of)\b.{0,100}\brestat",
        lower)

    if no_restatement:
        return "negative_event", excerpt_at(text, r"no\s+(?:material\s+)?restatement|restatement\s+is\s+not")
    if (actual_restatement_action and has_statement_scope and has_accounting_error
            and not possible_language):
        return "positive", excerpt_at(text, r"\brestat(?:ement|ed|ing)\b")
    if (form in ("8-K", "8-K/A") and "4.02" in set(items)
            and has_nonreliance and has_statement_scope and not re.search(r"\b4\.02\s*\(b\)", lower)):
        # Item 4.02(a) is the issuer's non-reliance disclosure for previously
        # issued financial statements; 4.02(b) alone concerns the auditor/review.
        return "positive", excerpt_at(text, r"should\s+no\s+longer\s+be\s+relied\s+upon|non[- ]reliance")
    if explicit_restatement or has_nonreliance or ("4.02" in set(items)):
        return "uncertain", excerpt_at(text, r"restat|non[- ]reliance|no\s+longer\s+be\s+relied|4\.02")
    return "irrelevant", ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path("data/restatement_review_queue.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/restatement_review_queue_ai_assisted.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/restatement_review_evidence_manifest.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/restatement_review_cache"))
    parser.add_argument("--horizon-years", type=int, choices=(1, 3, 5), default=3)
    parser.add_argument("--user-agent", required=True)
    args = parser.parse_args()

    queue = pd.read_csv(args.queue, dtype=str, keep_default_na=False)
    needed = {"original_adsh", "cik", "original_form", "original_filed_date", "sampling_group"}
    if not needed.issubset(queue.columns):
        raise ValueError(f"Queue missing columns: {sorted(needed - set(queue.columns))}")
    queue["_cik"] = queue.cik.map(norm_cik)
    queue["_original_date"] = pd.to_datetime(queue.original_filed_date, errors="raise")
    queue["_end_date"] = queue["_original_date"].map(lambda x: x + pd.DateOffset(years=args.horizon_years))
    today = pd.Timestamp(date.today())
    if queue["_end_date"].gt(today).any():
        raise ValueError("Some rows have not matured through the selected horizon; they cannot be labelled negative.")

    client = SecAccess(args.user_agent)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    query_cache_path = args.cache_dir / "query_hits.json"
    document_cache_path = args.cache_dir / "document_assessments.json"
    query_cache = json.loads(query_cache_path.read_text(encoding="utf-8")) if query_cache_path.exists() else {}
    document_cache = json.loads(document_cache_path.read_text(encoding="utf-8")) if document_cache_path.exists() else {}
    hits_by_id: dict[str, dict] = {}
    query_log = []
    # Search each original-filing-year cohort with CIK filters, but use exact
    # per-row windows when assigning hits. Chunking keeps SEC query URLs small.
    for year, cohort in queue.groupby(queue["_original_date"].dt.year, sort=True):
        ciks = sorted(cohort["_cik"].unique())
        start = date(int(year), 1, 1)
        end = min(date(int(year) + args.horizon_years, 12, 31), date.today())
        for pos in range(0, len(ciks), 80):
            batch = [cik.zfill(10) for cik in ciks[pos:pos + 80]]
            queries = [
                (SEARCH_TERMS, FORMS, None, "restatement_text"),
                ('"Item 4.02" OR 4.02', EVENT_FORMS, "4.02", "item_4_02"),
            ]
            for query, forms, items, query_kind in queries:
                cache_key = json.dumps([int(year), batch, query_kind, start.isoformat(),
                                        end.isoformat(), query, forms, items], separators=(",", ":"))
                cached_query = query_cache.get(cache_key)
                if cached_query is None:
                    page, total = efts_query(client, ciks=batch, start=start, end=end,
                                             query=query, forms=forms, items=items)
                    cached_query = {"hits": page, "total": total}
                    query_cache[cache_key] = cached_query
                    save_json(query_cache_path, query_cache)
                else:
                    page, total = cached_query["hits"], cached_query["total"]
                query_log.append({"original_year": int(year), "cik_count": len(batch),
                                  "start": start.isoformat(), "end": end.isoformat(),
                                  "query_kind": query_kind, "reported_hits": total,
                                  "returned_hits": len(page)})
                for hit in page:
                    source = hit.get("_source", {})
                    accession = str(source.get("adsh", ""))
                    filename = str(hit.get("_id", "")).split(":", 1)[-1]
                    cik_list = source.get("ciks") or []
                    cik = norm_cik(cik_list[0]) if cik_list else ""
                    form = str(source.get("form", ""))
                    file_type = str(source.get("file_type", ""))
                    if (not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession)
                            or not re.fullmatch(r"[A-Za-z0-9_.-]+", filename)
                            or not cik.isdigit()):
                        continue
                    # Inspect the filing's main document and common press-release
                    # exhibits only. Other exhibits (contracts, auditor consents,
                    # etc.) can match boilerplate query terms without changing the
                    # issuer's restatement outcome.
                    if file_type not in {form, "EX-99.1", "EX-99.2"}:
                        continue
                    if cik not in set(cohort["_cik"]):
                        continue
                    event_date = pd.to_datetime(source.get("file_date"), errors="coerce")
                    if pd.isna(event_date):
                        continue
                    matching_rows = cohort.loc[
                        cohort["_cik"].eq(cik)
                        & cohort["_original_date"].le(event_date)
                        & cohort["_end_date"].ge(event_date)
                    ]
                    if matching_rows.empty:
                        continue
                    url = filing_url(cik, accession, filename)
                    entry = hits_by_id.setdefault(url, {
                        "url": url, "cik": cik, "adsh": accession,
                        "filename": filename, "file_date": event_date.strftime("%Y-%m-%d"),
                        "form": form, "file_type": file_type,
                        "items": ",".join(source.get("items") or []),
                        "matching_original_adsh": set(), "query_kinds": set(),
                    })
                    entry["matching_original_adsh"].update(matching_rows.original_adsh.astype(str))
                    entry["query_kinds"].add(query_kind)
        print(f"Indexed cohort {year}: {len(ciks):,} issuers", flush=True)

    print(f"Relevant SEC document hits: {len(hits_by_id):,}; downloading primary documents...", flush=True)
    evidence_rows = []
    assessments: dict[str, list[dict]] = {str(x): [] for x in queue.original_adsh}
    for i, entry in enumerate(hits_by_id.values(), 1):
        cached_doc = document_cache.get(entry["url"])
        if cached_doc is None:
            try:
                raw = client.get(entry["url"])
                text = visible_text(raw)
                status = "downloaded"
                outcome, excerpt = assess_event(text, entry["form"], entry["items"].split(",") if entry["items"] else [])
            except HTTPError as exc:
                if exc.code == 404:
                    status, outcome, excerpt = "http_404", "uncertain", ""
                else:
                    raise
            except ValueError as exc:
                status, outcome, excerpt = "size_limit", "uncertain", str(exc)
            cached_doc = {"fetch_status": status, "assessment": outcome,
                          "evidence_excerpt": excerpt[:800]}
            document_cache[entry["url"]] = cached_doc
            # Keep progress recoverable if a later request is denied/interrupted.
            if i % 10 == 0 or i == len(hits_by_id):
                save_json(document_cache_path, document_cache)
        status = cached_doc["fetch_status"]
        outcome = cached_doc["assessment"]
        excerpt = cached_doc["evidence_excerpt"]
        detail = {**entry, "matching_original_adsh": sorted(entry["matching_original_adsh"]),
                  "query_kinds": sorted(entry["query_kinds"]), "fetch_status": status,
                  "assessment": outcome, "evidence_excerpt": excerpt[:800]}
        evidence_rows.append(detail)
        if outcome in ("positive", "uncertain"):
            for adsh in entry["matching_original_adsh"]:
                assessments[str(adsh)].append(detail)
        if i % 25 == 0:
            print(f"Downloaded and assessed {i:,}/{len(hits_by_id):,} SEC documents", flush=True)

    out = queue.drop(columns=["_cik", "_original_date", "_end_date"]).copy()
    for col in ("label", "event_type", "event_date", "evidence_url", "evidence_excerpt",
                "reviewer", "review_notes", "outcome_followup_through"):
        if col not in out:
            out[col] = ""
    for i, row in out.iterrows():
        original = pd.Timestamp(queue.at[i, "_original_date"])
        horizon_end = original + pd.DateOffset(years=args.horizon_years)
        relevant = assessments[str(row.original_adsh)]
        positives = [x for x in relevant if x["assessment"] == "positive"]
        uncertain = [x for x in relevant if x["assessment"] == "uncertain"]
        if positives:
            event = min(positives, key=lambda x: x["file_date"])
            out.at[i, "label"] = "1"
            out.at[i, "review_status"] = "reviewed"
            out.at[i, "event_type"] = "restatement"
            out.at[i, "event_date"] = event["file_date"]
            out.at[i, "evidence_url"] = event["url"]
            out.at[i, "evidence_excerpt"] = event["evidence_excerpt"][:800]
            out.at[i, "reviewer"] = REVIEWER
            out.at[i, "review_notes"] = (
                f"AI-assisted primary-source screen; SEC EDGAR full-text search across {FORMS} "
                f"from {original:%Y-%m-%d} through {horizon_end:%Y-%m-%d}. "
                "Positive assigned from explicit restatement/non-reliance language plus accounting-error scope."
            )
            out.at[i, "outcome_followup_through"] = horizon_end.strftime("%Y-%m-%d")
        elif uncertain:
            event = min(uncertain, key=lambda x: x["file_date"])
            out.at[i, "review_status"] = "pending"
            out.at[i, "evidence_url"] = event["url"]
            out.at[i, "evidence_excerpt"] = event["evidence_excerpt"][:800]
            out.at[i, "reviewer"] = REVIEWER
            out.at[i, "review_notes"] = (
                f"SEC full-text search found a potentially relevant filing dated {event['file_date']}; "
                "the retrieved document did not meet the conservative positive rule. Requires human adjudication."
            )
        else:
            out.at[i, "label"] = "0"
            out.at[i, "review_status"] = "reviewed"
            out.at[i, "event_type"] = "no_confirmed_restatement"
            out.at[i, "event_date"] = ""
            if str(row.get("candidate_event_url", "")).startswith("https://www.sec.gov/Archives/"):
                anchor = str(row.get("candidate_event_url"))
            else:
                anchor = filing_url(norm_cik(row.cik), str(row.original_adsh))
            out.at[i, "evidence_url"] = anchor
            out.at[i, "evidence_excerpt"] = (
                f"No confirming restatement/non-reliance filing was returned by the SEC EDGAR full-text search "
                f"for {original:%Y-%m-%d} through {horizon_end:%Y-%m-%d}. This is a documented search result, not a quotation."
            )
            out.at[i, "reviewer"] = REVIEWER
            out.at[i, "review_notes"] = (
                f"AI-assisted SEC EDGAR full-text screen: {SEARCH_TERMS}; forms {FORMS} and Item 4.02 8-K filings; "
                f"issuer CIK {norm_cik(row.cik)}; filing window {original:%Y-%m-%d} to {horizon_end:%Y-%m-%d}. "
                "No matching event document was found. Negative is provisional pending human audit of SEC filing history."
            )
            out.at[i, "outcome_followup_through"] = horizon_end.strftime("%Y-%m-%d")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    ev = pd.DataFrame(evidence_rows)
    if not ev.empty:
        ev["matching_original_adsh"] = ev.matching_original_adsh.map(json.dumps)
        ev["query_kinds"] = ev.query_kinds.map(json.dumps)
    ev.to_csv(args.manifest, index=False)
    logpath = args.manifest.with_suffix(".queries.json")
    logpath.write_text(json.dumps({"horizon_years": args.horizon_years,
                                   "forms": FORMS, "search_terms": SEARCH_TERMS,
                                   "query_log": query_log}, indent=2), encoding="utf-8")
    print("Review counts:", out.review_status.value_counts().to_dict(), flush=True)
    print("Labels:", out.loc[out.review_status.eq("reviewed"), "label"].value_counts().to_dict(), flush=True)
    print("Wrote", args.output, "and", args.manifest, flush=True)


if __name__ == "__main__":
    main()
