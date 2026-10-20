"""Download missing SEC Financial Statement Data Set quarters safely.

Requires an identifying SEC_USER_AGENT, e.g. ``Researcher Name name@example.edu``.
The four requested public archives are downloaded sequentially and unpacked
atomically; existing complete quarters are never overwritten.
"""

import argparse
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile


BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"
DEFAULT_QUARTERS = ("2025Q4", "2026Q1", "2026Q2", "2026Q3")
REQUIRED = {"sub.txt", "num.txt", "pre.txt", "tag.txt"}


def download(url, destination, user_agent):
    headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response, open(destination, "wb") as output:
                if response.status != 200:
                    raise RuntimeError(f"Unexpected HTTP status {response.status} for {url}")
                shutil.copyfileobj(response, output)
            return
        except (HTTPError, URLError, TimeoutError) as exc:
            destination.unlink(missing_ok=True)
            if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 4:
                raise RuntimeError(f"SEC download failed after five attempts: {url}") from exc
            time.sleep(min(60, 3 * (2 ** attempt)))


def install_quarter(quarter, raw_dir, user_agent):
    target = raw_dir / quarter
    if target.exists():
        existing = {p.name.lower() for p in target.iterdir() if p.is_file()}
        if REQUIRED <= existing:
            print(f"Already complete: {quarter}")
            return
        raise RuntimeError(f"Existing quarter is incomplete; inspect it manually: {target}")
    url = f"{BASE}/{quarter.lower()}.zip"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{quarter}_", dir=raw_dir) as temp_name:
        temp = Path(temp_name)
        archive = temp / "source.zip"
        print(f"Downloading {quarter} from {url}", flush=True)
        download(url, archive, user_agent)
        extracted = temp / "extracted"
        extracted.mkdir()
        try:
            with ZipFile(archive) as zip_file:
                members = {Path(name).name.lower(): name for name in zip_file.namelist()
                           if not name.endswith("/") and len(Path(name).parts) == 1}
                if not REQUIRED <= members.keys():
                    raise ValueError(f"Unexpected archive contents for {quarter}: {sorted(members)}")
                for filename in sorted(REQUIRED):
                    with zip_file.open(members[filename]) as source, open(extracted / filename, "wb") as output:
                        shutil.copyfileobj(source, output)
        except BadZipFile as exc:
            raise ValueError(f"Invalid ZIP archive for {quarter}") from exc
        extracted.rename(target)
    print(f"Installed {quarter} at {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quarters", nargs="*", default=DEFAULT_QUARTERS)
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in user_agent or " " not in user_agent:
        raise SystemExit("Set SEC_USER_AGENT to your real name or organization and contact email first.")
    for quarter in args.quarters:
        if len(quarter) != 6 or quarter[4] != "Q" or quarter[:4].isdigit() is False or quarter[5] not in "1234":
            raise SystemExit(f"Invalid quarter: {quarter}; expected YYYYQn")
        install_quarter(quarter, args.raw_dir, user_agent)


if __name__ == "__main__":
    main()
