#!/usr/bin/env python3
"""
fetch_testcases.py
==================
Download the "STEM Games 2026 - Day 1" test-case data from the Kontestis
judge (https://kontestis.ac) and save it locally, for the ETF Sarajevo
STEM Games archive.

PIPELINE (reverse-engineered from the site)
    /temp-login            log in with a temporary account
        v
    /contest/<id>          the contest page -> 4 problem links
        v
    /problem/<id>          a problem page -> a "Verdict" table of submissions
        v
    /submission/<id>       a submission -> a "Sample" table and a "Cluster" table
        v
    (click a row)          -> a cluster page with a "Testcase" table whose
                              Input / Output columns hold the download links

The script drives a real (headless) Chrome browser via Selenium, exactly
as a person clicking through the site would. It is deliberately slow and
single-threaded with a delay between every action, so it never hammers
the server -- a full run taking a few minutes is expected and fine.

CREDENTIALS
    You are prompted for username + password at runtime (the password is
    not echoed). Nothing is stored on disk; nothing is hardcoded.

REQUIREMENTS
    * Google Chrome installed.
    * Python 3.9+  and  `pip install selenium`  (>= 4.6 -- it fetches the
      matching chromedriver automatically).

USAGE
    python fetch_testcases.py                 # headless, default settings
    python fetch_testcases.py --headful       # show the browser window
    python fetch_testcases.py --delay 2.0     # be even gentler
    python fetch_testcases.py --out ../testcases --limit-groups 2

CLOUDFLARE
    kontestis.ac sits behind Cloudflare. Plain headless Chrome normally
    passes fine. If you ever hit a "checking your browser" challenge, run
    with --headful and solve it once by hand. The script does NOT try to
    bypass Cloudflare or any CAPTCHA -- it only detects one and tells you.

NOTE ON THE DOWNLOAD CONTROL
    In the Testcase table the Input column is third-from-last and the
    Output column is second-from-last. The download control in each is a
    clickable <svg> icon (not an <a> or <button>) -- clicking it makes the
    browser download a .in / .out file. `download_from_cell()` clicks that
    icon and catches the downloaded file.
"""

from __future__ import annotations

import argparse
import getpass
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.common.exceptions import TimeoutException, WebDriverException
except ImportError:
    sys.exit("Selenium is not installed. Run:  pip install selenium")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
LOGIN_URL = "https://kontestis.ac/temp-login"
# "Stem Games 2026 - Day 1" -- taken from the contest link on the site.
CONTEST_URL = "https://kontestis.ac/contest/577423564509024256"

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "testcases"

CF_MARKERS = ("just a moment", "checking your browser", "attention required",
              "cf-challenge", "cf-browser-verification")


class NoDownload(Exception):
    """Raised when a testcase cell has no usable download element."""


def slug(text: str) -> str:
    """Filesystem-safe lowercase slug, e.g. 'Cluster 3' -> 'cluster_03'."""
    text = (text or "").strip().lower()
    # zero-pad trailing numbers so folders sort naturally
    m = re.match(r"(.*?)(\d+)\s*$", text)
    if m:
        text = f"{m.group(1)}{int(m.group(2)):02d}"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "item"


class TestcaseFetcher:
    def __init__(self, args):
        self.out: Path = args.out.resolve()
        self.delay: float = args.delay
        self.timeout: int = args.timeout
        self.headful: bool = args.headful
        self.limit_groups: int = args.limit_groups
        self.dl_dir: Path = self.out / ".tmp_downloads"
        self.driver = None
        # counters for the final summary
        self.downloaded = self.skipped = self.empty = 0
        self.errors: list[str] = []

    # -- tiny logging helpers ------------------------------------------------
    @staticmethod
    def log(msg: str) -> None:
        print(msg, flush=True)

    def warn(self, msg: str) -> None:
        print(f"  ! {msg}", flush=True)

    def err(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"  X {msg}", flush=True)

    def sleep(self) -> None:
        """Polite pause between actions -- keeps load on the server low."""
        time.sleep(self.delay)

    # -- browser lifecycle ---------------------------------------------------
    def start(self) -> None:
        self.dl_dir.mkdir(parents=True, exist_ok=True)
        opts = Options()
        if not self.headful:
            opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1400,1000")
        opts.add_argument("--disable-gpu")
        opts.add_experimental_option("prefs", {
            "download.default_directory": str(self.dl_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        })
        try:
            self.driver = webdriver.Chrome(options=opts)
        except WebDriverException as e:
            sys.exit(f"Could not start Chrome: {e}\n"
                     f"Make sure Google Chrome is installed.")
        # make sure downloads are allowed even in headless mode
        try:
            self.driver.execute_cdp_cmd(
                "Page.setDownloadBehavior",
                {"behavior": "allow", "downloadPath": str(self.dl_dir)})
        except WebDriverException:
            pass

    def stop(self) -> None:
        if self.driver:
            try:
                self.driver.quit()
            except WebDriverException:
                pass
        # remove the scratch download folder
        shutil.rmtree(self.dl_dir, ignore_errors=True)

    # -- low-level navigation ------------------------------------------------
    def _wait(self, condition, timeout=None):
        return WebDriverWait(self.driver, timeout or self.timeout).until(condition)

    def _check_cloudflare(self) -> None:
        page = (self.driver.page_source or "").lower()
        title = (self.driver.title or "").lower()
        if any(m in page or m in title for m in CF_MARKERS):
            raise RuntimeError(
                "Cloudflare challenge detected. Re-run with --headful and "
                "solve the check once by hand, then let the script continue.")

    def goto(self, url: str, ready_css: str, optional: bool = False) -> bool:
        """Navigate to a URL and wait for the page to actually render."""
        for attempt in range(3):
            self.driver.get(url)
            self._check_cloudflare()
            try:
                self._wait(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, ready_css)))
                self.sleep()
                return True
            except TimeoutException:
                if optional:
                    return False
                if attempt == 2:
                    raise RuntimeError(f"Timed out loading {url}")
                time.sleep(2)
        return False

    # -- login ---------------------------------------------------------------
    def login(self, username: str, password: str) -> None:
        self.log("Logging in ...")
        self.goto(LOGIN_URL, "input[name='username']")
        user = self.driver.find_element(By.CSS_SELECTOR, "input[name='username']")
        pwd = self.driver.find_element(By.CSS_SELECTOR, "input[name='password']")
        user.clear(); user.send_keys(username)
        pwd.clear(); pwd.send_keys(password)

        # the "Log in" button lives inside the form
        buttons = self.driver.find_elements(By.XPATH, "//form//button")
        clicked = False
        for b in buttons:
            if "log in" in (b.text or "").lower():
                b.click(); clicked = True; break
        if not clicked:
            if buttons:
                buttons[0].click()
            else:
                pwd.send_keys(Keys.RETURN)

        # wait until we leave the login route
        try:
            WebDriverWait(self.driver, self.timeout).until(
                lambda d: "/temp-login" not in d.current_url)
        except TimeoutException:
            raise RuntimeError(
                "Login did not go through -- check the username/password.")
        self.sleep()
        self._check_cloudflare()
        self.log("Login OK.")

    # -- table helpers -------------------------------------------------------
    def _tables(self):
        return self.driver.find_elements(By.TAG_NAME, "table")

    @staticmethod
    def _headers(table):
        return [h.text.strip()
                for h in table.find_elements(By.CSS_SELECTOR, "thead td, thead th")]

    def find_table_by_head(self, prefix: str):
        """Return the first table whose first header cell starts with `prefix`."""
        for t in self._tables():
            heads = self._headers(t)
            if heads and heads[0].lower().startswith(prefix.lower()):
                return t
        return None

    def wait_table_by_head(self, prefix: str):
        self._wait(lambda d: self.find_table_by_head(prefix) is not None)
        return self.find_table_by_head(prefix)

    def wait_for_icons(self, timeout: int = 15) -> bool:
        """Wait for the testcase table's Input/Output download controls.

        Those <svg> icons render a moment AFTER the testcase rows
        themselves, so scraping too early finds empty cells. Returns True
        if any download control appeared, False if the table is genuinely
        empty (e.g. a problem that has no real test data)."""
        def present(_):
            t = self.find_table_by_head("Testcase")
            return bool(t) and bool(t.find_elements(
                By.CSS_SELECTOR, "tbody svg, tbody a, tbody button"))
        try:
            WebDriverWait(self.driver, timeout).until(present)
            return True
        except TimeoutException:
            return False

    # -- contest / problems / submissions ------------------------------------
    def collect_problems(self) -> list[tuple[str, str]]:
        self.goto(CONTEST_URL, "a[href*='/problem/']")
        seen, problems = set(), []
        for a in self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/problem/']"):
            href = a.get_attribute("href")
            name = (a.text or "").strip()
            if href and href not in seen:
                seen.add(href)
                problems.append((name or href.rsplit("/", 1)[-1], href))
        return problems

    def collect_submissions(self) -> list[dict]:
        subs, seen = [], set()
        for a in self.driver.find_elements(By.CSS_SELECTOR,
                                           "a[href*='/submission/']"):
            href = a.get_attribute("href")
            if not href or href in seen:
                continue
            seen.add(href)
            verdict = (a.text or "").strip()
            points = 0
            try:
                row = a.find_element(By.XPATH, "./ancestor::tr")
                m = re.search(r"(\d+)\s*/\s*\d+", row.text)
                if m:
                    points = int(m.group(1))
            except WebDriverException:
                pass
            subs.append({"url": href, "verdict": verdict, "points": points})
        return subs

    @staticmethod
    def pick_submission(subs: list[dict]) -> dict:
        """Prefer a submission that actually executed, then the highest score."""
        return sorted(
            subs,
            key=lambda s: (s["verdict"].lower() != "compilation_error",
                           s["points"]),
        )[-1]

    # -- group (Sample / Cluster) handling -----------------------------------
    def list_groups(self) -> list[tuple[str, int, str]]:
        """On a submission page: every Sample/Cluster row as (kind, index, label)."""
        groups = []
        for kind in ("Sample", "Cluster"):
            table = self.find_table_by_head(kind)
            if not table:
                continue
            rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
            for i, row in enumerate(rows):
                cells = row.find_elements(By.CSS_SELECTOR, "td")
                label = (cells[0].text.strip() if cells else "") or f"{kind} {i + 1}"
                groups.append((kind, i, label))
        return groups

    def open_group(self, sub_url: str, kind: str, index: int) -> None:
        """Reload the submission page and click into one Sample/Cluster row."""
        self.goto(sub_url, "table")
        table = self.find_table_by_head(kind)
        if not table:
            raise RuntimeError(f"'{kind}' table not found")
        rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
        if index >= len(rows):
            raise RuntimeError(f"{kind} row {index} no longer present")
        row = rows[index]
        self.driver.execute_script(
            "arguments[0].scrollIntoView({block:'center'})", row)

        # the rows are React-clickable (no <a href>); try the row, then its
        # inner cells/spans, until a "Testcase" table shows up.
        for target in [row] + row.find_elements(By.CSS_SELECTOR, "td, span"):
            try:
                target.click()
            except WebDriverException:
                try:
                    self.driver.execute_script("arguments[0].click()", target)
                except WebDriverException:
                    continue
            try:
                WebDriverWait(self.driver, 8).until(
                    lambda d: self.find_table_by_head("Testcase") is not None)
                return
            except TimeoutException:
                continue
        raise RuntimeError("could not open the cluster page (no Testcase table)")

    # -- testcase downloading ------------------------------------------------
    def download_testcases(self, problem: str, group_label: str) -> None:
        self.wait_table_by_head("Testcase")
        self.wait_for_icons()          # let the download icons render first
        table = self.find_table_by_head("Testcase")
        heads = [h.lower() for h in self._headers(table)]
        ncol = len(heads)
        # Input is the third-from-last column, Output the second-from-last;
        # fall back to those positions if a header cell isn't plain text.
        in_i = heads.index("input") if "input" in heads else (
            ncol - 3 if ncol >= 3 else None)
        out_i = heads.index("output") if "output" in heads else (
            ncol - 2 if ncol >= 2 else None)
        n_rows = len(table.find_elements(By.CSS_SELECTOR, "tbody tr"))
        self.log(f"    {group_label}: {n_rows} testcase(s)")

        for r in range(n_rows):
            # re-find each time so a download click can't leave us with a
            # stale element reference
            table = self.find_table_by_head("Testcase")
            rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
            if r >= len(rows):
                break
            cells = rows[r].find_elements(By.CSS_SELECTOR, "td")
            tc_label = (cells[0].text or "").strip() or f"Testcase {r + 1}"
            base = self.out / slug(problem) / slug(group_label)

            for col, ext in ((in_i, "in"), (out_i, "out")):
                if col is None or col >= len(cells):
                    continue
                dest = base / f"{slug(tc_label)}.{ext}"
                if dest.exists() and dest.stat().st_size > 0:
                    self.skipped += 1
                    continue
                try:
                    self.download_from_cell(cells[col], dest)
                    self.downloaded += 1
                    self.log(f"      saved {dest.relative_to(self.out)}")
                except NoDownload:
                    self.empty += 1
                    self.warn(f"{tc_label} [{ext}]: no download icon found "
                              f"in cell")
                except Exception as e:  # noqa: BLE001 - keep going
                    self.err(f"{problem}/{group_label}/{tc_label} [{ext}]: {e}")
                self.sleep()

    def _robust_click(self, el) -> None:
        """Click an element three different ways. Needed for the <svg>
        download icons, which a plain Selenium click sometimes won't hit."""
        try:
            el.click()
            return
        except WebDriverException:
            pass
        try:
            self.driver.execute_script("arguments[0].click()", el)
            return
        except WebDriverException:
            pass
        ActionChains(self.driver).move_to_element(el).click().perform()

    def download_from_cell(self, cell, dest: Path) -> None:
        """Download the file behind a single Input/Output table cell.

        The download control is a clickable <svg> icon (an <a> or <button>
        is also accepted, as a fallback). Primary path: click it and let
        Chrome download the file. Secondary path: if the control happens
        to be a link, fetch its URL with the browser's own session cookies.
        """
        # the <svg> download icons can render slightly after the row, so
        # give the cell a few attempts before giving up on it
        elements = []
        for _ in range(10):
            elements = cell.find_elements(By.CSS_SELECTOR, "svg, a, button")
            if not elements:
                elements = cell.find_elements(By.XPATH, ".//*[@role='button']")
            if elements:
                break
            time.sleep(0.4)
        if not elements:
            raise NoDownload()

        el = elements[0]
        href = el.get_attribute("href")
        main_handle = self.driver.current_window_handle
        before = self._snapshot()

        self._robust_click(el)

        new_file = self._wait_for_download(before)

        # close any popup tab the click may have opened
        for handle in list(self.driver.window_handles):
            if handle != main_handle:
                self.driver.switch_to.window(handle)
                if not href:
                    href = self.driver.current_url
                self.driver.close()
        self.driver.switch_to.window(main_handle)

        dest.parent.mkdir(parents=True, exist_ok=True)
        if new_file:
            shutil.move(str(new_file), str(dest))
            return
        if href and href.startswith("http"):
            self._fetch_with_cookies(href, dest)
            return
        raise NoDownload()

    def _snapshot(self) -> set:
        return set(self.dl_dir.iterdir())

    def _wait_for_download(self, before: set, timeout: int = 40):
        """Wait until a new, fully-written file appears in the download dir."""
        end = time.time() + timeout
        while time.time() < end:
            current = set(self.dl_dir.iterdir())
            in_progress = any(p.name.endswith((".crdownload", ".tmp"))
                              for p in current)
            fresh = [p for p in current - before
                     if not p.name.endswith((".crdownload", ".tmp"))]
            if fresh and not in_progress:
                return max(fresh, key=lambda p: p.stat().st_mtime)
            time.sleep(0.4)
        return None

    def _fetch_with_cookies(self, url: str, dest: Path) -> None:
        """Fallback download: GET `url` carrying the browser's session."""
        cookies = "; ".join(f"{c['name']}={c['value']}"
                            for c in self.driver.get_cookies())
        ua = self.driver.execute_script("return navigator.userAgent")
        req = urllib.request.Request(
            url, headers={"Cookie": cookies, "User-Agent": ua})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)

    # -- top-level driver ----------------------------------------------------
    def run(self, username: str, password: str) -> None:
        self.login(username, password)
        problems = self.collect_problems()
        if not problems:
            raise RuntimeError("No problems found on the contest page.")
        self.log(f"Found {len(problems)} problem(s).")

        for name, url in problems:
            self.log(f"\n=== {name} ===")
            try:
                self.process_problem(name, url)
            except Exception as e:  # noqa: BLE001 - one problem must not kill the run
                self.err(f"{name}: {e}")

    def process_problem(self, name: str, url: str) -> None:
        self.goto(url, "table")
        subs = self.collect_submissions()
        if not subs:
            self.warn(f"{name}: no submissions found -- skipping.")
            return
        sub = self.pick_submission(subs)
        self.log(f"  using submission {sub['url'].rsplit('/', 1)[-1]} "
                 f"(verdict='{sub['verdict']}', {sub['points']} pts)")

        self.goto(sub["url"], "table")
        groups = self.list_groups()
        if self.limit_groups:
            groups = groups[:self.limit_groups]
        self.log(f"  {len(groups)} Sample/Cluster group(s) to visit")

        for kind, index, label in groups:
            try:
                self.open_group(sub["url"], kind, index)
                self.download_testcases(name, label)
            except Exception as e:  # noqa: BLE001
                self.err(f"{name}/{label}: {e}")
            self.sleep()

    def summary(self) -> None:
        self.log("\n" + "=" * 56)
        self.log(f"Downloaded : {self.downloaded} file(s)")
        self.log(f"Skipped    : {self.skipped} (already present)")
        self.log(f"Empty cells: {self.empty} (no download link available)")
        self.log(f"Errors     : {len(self.errors)}")
        for e in self.errors:
            self.log(f"  - {e}")
        self.log(f"Output dir : {self.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download STEM Games 2026 Day-1 test cases from Kontestis.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds to pause between actions (default: 1.0)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="page-load timeout in seconds (default: 30)")
    parser.add_argument("--headful", action="store_true",
                        help="show the browser window instead of headless")
    parser.add_argument("--limit-groups", type=int, default=0,
                        help="only visit the first N Sample/Cluster groups per "
                             "problem (0 = all; useful for a quick test run)")
    args = parser.parse_args()

    print("Kontestis test-case downloader")
    print("Credentials are used only for this session and never stored.\n")
    username = input("Kontestis username: ").strip()
    password = getpass.getpass("Kontestis password: ")
    if not username or not password:
        sys.exit("Username and password are both required.")

    fetcher = TestcaseFetcher(args)
    try:
        fetcher.start()
        fetcher.run(username, password)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:  # noqa: BLE001
        print(f"\nFATAL: {e}")
    finally:
        fetcher.summary()
        fetcher.stop()


if __name__ == "__main__":
    main()
