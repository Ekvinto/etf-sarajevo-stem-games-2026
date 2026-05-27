"""Selenium scraper for public scam-archive sites.

Default target: scamletters.com (a public, long-running archive of romance-scam
letters and pasted chat fragments). The site is sparse and stable, so we can
walk its index pages, follow each letter link, and harvest the body.

USAGE
-----
    python -m src.scrape_selenium                              # scrape default site
    python -m src.scrape_selenium --site scamletters --pages 5
    python -m src.scrape_selenium --site custom --start-url https://example.com/listing --pages 3

ETHICS / TOS
------------
Read each site's robots.txt and terms of service BEFORE scraping. We:
  * delay between requests (--delay)
  * use a clearly identified user agent
  * scrape only publicly accessible pages
  * do NOT bypass logins, paywalls, or CAPTCHAs
  * cache results so we never re-hit the same URL

OUTPUT
------
JSONL at data/scam_letters_raw.jsonl, one record per scraped page:
    {"url": "...", "fetched_at": "ISO-8601", "title": "...", "text": "..."}

After scraping, run:
    python -m src.scrape_selenium --postprocess
to split letters into messages and emit data/scam_letters_corpus.jsonl in the
Conversation schema (single-speaker S, since these are letter dumps).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from src.conversation import Conversation, Message, save_corpus

OUT_RAW = Path("data/scam_letters_raw.jsonl")
OUT_CORPUS = Path("data/scam_letters_corpus.jsonl")
OUT_RAW.parent.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Mozilla/5.0 (stem-games-bot-detector/0.1 academic-research)"


# --- Site configurations -----------------------------------------------------
# Each "site" is a dict of CSS selectors + URL builders. To add a new site,
# fill in the same keys.

SITES: dict[str, dict] = {
    "scamletters": {
        "start_url_template": "https://www.scamletters.com/scam-letters/page/{page}/",
        "listing_link_selector": "h2.entry-title a",
        "article_body_selector": "div.entry-content",
        "article_title_selector": "h1.entry-title",
    },
    # Add more here as you discover sites. KEEP a generic fallback:
    "generic": {
        "start_url_template": None,   # require --start-url
        "listing_link_selector": "a",
        "article_body_selector": "article, main, div#content, div.content",
        "article_title_selector": "h1, h2",
    },
}


# --- Driver --------------------------------------------------------------
def build_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(f"--user-agent={USER_AGENT}")
    opts.add_argument("--window-size=1280,1800")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(30)
    return driver


# --- Cache --------------------------------------------------------------
def load_seen() -> set[str]:
    if not OUT_RAW.exists():
        return set()
    urls = set()
    with open(OUT_RAW, "r", encoding="utf-8") as f:
        for line in f:
            try:
                urls.add(json.loads(line)["url"])
            except Exception:
                pass
    return urls


# --- Core scrape ---------------------------------------------------------
def gather_listing_links(driver: webdriver.Chrome, listing_url: str,
                         link_selector: str, base_url: str) -> list[str]:
    print(f"  Listing: {listing_url}")
    try:
        driver.get(listing_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
    except (TimeoutException, WebDriverException) as e:
        print(f"  [WARN] could not load listing: {e}")
        return []
    elements = driver.find_elements(By.CSS_SELECTOR, link_selector)
    hrefs = []
    for el in elements:
        href = el.get_attribute("href")
        if href and not href.startswith("javascript"):
            hrefs.append(urljoin(base_url, href))
    return list(dict.fromkeys(hrefs))  # de-dup, preserve order


def scrape_article(driver: webdriver.Chrome, url: str,
                   title_sel: str, body_sel: str) -> dict | None:
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
    except (TimeoutException, WebDriverException) as e:
        print(f"  [WARN] failed {url}: {e}")
        return None

    title = ""
    title_els = driver.find_elements(By.CSS_SELECTOR, title_sel)
    if title_els:
        title = title_els[0].text.strip()

    body_els = driver.find_elements(By.CSS_SELECTOR, body_sel)
    if not body_els:
        return None
    body_text = "\n".join(el.text for el in body_els).strip()
    if len(body_text.split()) < 40:
        return None

    return {
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "text": body_text,
    }


def run_scrape(site: str, pages: int, delay: float, headless: bool,
               start_url_override: str | None) -> None:
    if site not in SITES:
        print(f"Unknown site '{site}'. Available: {list(SITES)}")
        sys.exit(1)
    cfg = SITES[site]
    seen = load_seen()
    print(f"Already in cache: {len(seen)} URLs")

    driver = build_driver(headless=headless)
    try:
        all_links: list[str] = []
        if start_url_override:
            base = "{0.scheme}://{0.netloc}".format(urlparse(start_url_override))
            all_links = gather_listing_links(
                driver, start_url_override, cfg["listing_link_selector"], base
            )
            time.sleep(delay)
        else:
            template = cfg["start_url_template"]
            if template is None:
                print("Need --start-url for generic site")
                sys.exit(1)
            base = "{0.scheme}://{0.netloc}".format(urlparse(template))
            for p in range(1, pages + 1):
                url = template.format(page=p)
                links = gather_listing_links(
                    driver, url, cfg["listing_link_selector"], base
                )
                all_links.extend(links)
                time.sleep(delay)

        all_links = [u for u in dict.fromkeys(all_links) if u not in seen]
        print(f"Collected {len(all_links)} new article URLs")

        with open(OUT_RAW, "a", encoding="utf-8") as f:
            for i, url in enumerate(all_links, 1):
                print(f"[{i}/{len(all_links)}] {url}")
                rec = scrape_article(
                    driver, url,
                    cfg["article_title_selector"],
                    cfg["article_body_selector"],
                )
                if rec is not None:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                time.sleep(delay)
    finally:
        driver.quit()


# --- Post-processing ----------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _letter_to_messages(text: str) -> list[Message]:
    """Split a letter into pseudo-messages of 1-3 sentences each."""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    messages: list[Message] = []
    i = 0
    msg_idx = 0
    while i < len(sents):
        # Take 1 to 3 sentences as one "message"
        take = min(3, len(sents) - i)
        text = " ".join(sents[i:i + take])
        if len(text) > 15:
            messages.append(Message(
                speaker="S",
                text=text,
                timestamp=base + timedelta(minutes=15 * msg_idx),
            ))
            msg_idx += 1
        i += take
    return messages


def postprocess() -> None:
    if not OUT_RAW.exists():
        print(f"No raw file at {OUT_RAW}")
        return
    convs: list[Conversation] = []
    with open(OUT_RAW, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            messages = _letter_to_messages(rec["text"])
            if len(messages) >= 6:
                convs.append(Conversation(
                    messages=messages,
                    label=1,
                    source=rec["url"],
                ))
    save_corpus(convs, OUT_CORPUS)
    print(f"Wrote {len(convs)} conversations -> {OUT_CORPUS}")


# --- CLI ----------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="scamletters", choices=list(SITES))
    ap.add_argument("--pages", type=int, default=3, help="Listing pages to walk")
    ap.add_argument("--delay", type=float, default=2.0, help="Seconds between requests")
    ap.add_argument("--no-headless", action="store_true",
                    help="Show the Chrome window (useful for debugging)")
    ap.add_argument("--start-url", default=None,
                    help="Override start URL. Required for site=generic")
    ap.add_argument("--postprocess", action="store_true",
                    help="Run after scraping; converts raw JSONL into Conversation JSONL")
    args = ap.parse_args()

    if args.postprocess:
        postprocess()
        return

    run_scrape(
        site=args.site,
        pages=args.pages,
        delay=args.delay,
        headless=not args.no_headless,
        start_url_override=args.start_url,
    )


if __name__ == "__main__":
    main()
