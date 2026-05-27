# Scripts

## `fetch_testcases.py`

Downloads the **STEM Games 2026 - Day 1** test-case data (each problem's
test inputs and expected outputs) from the Kontestis judge at
`https://kontestis.ac`, into [`../testcases/`](../testcases/).

It drives a real (headless) Chrome browser with Selenium, logging in and
clicking through the site the same way a person would, with a delay
between every action so the server is never hammered.

### Setup

- Install Google Chrome.
- Install the Python dependency:

  ```bash
  pip install -r requirements.txt
  ```

  Selenium 4.6+ downloads the matching `chromedriver` automatically.

### Run

```bash
python fetch_testcases.py
```

You will be prompted for your Kontestis username and password. They are
used only for that session and are **never stored**.

Useful options:

| Option | Meaning |
|--------|---------|
| `--headful` | Show the browser window (run this if Cloudflare ever challenges you). |
| `--delay 2.0` | Pause longer between actions. |
| `--out PATH` | Change the output directory (default: `../testcases/`). |
| `--limit-groups N` | Only visit the first N Sample/Cluster groups per problem — handy for a quick test run. |

### Notes

- The script does **not** bypass Cloudflare or any CAPTCHA. If it detects
  a challenge it stops and asks you to run with `--headful` and solve it
  once by hand.
- Re-running is safe: files that are already downloaded are skipped.
- A full run takes a few minutes — that is expected and intentional.
- The downloaded `testcases/` data is git-ignored by default if you add
  it to `.gitignore`; decide per the data's size whether to commit it.
