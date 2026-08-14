# Phase 0 — Step-by-Step Guide

Everything here is Windows/PowerShell, since that's what you're on. Total time
is about **3 days**, most of it waiting on a collection run.

Phase 0 answers one question: **is there enough data for the project as
currently scoped?** Do not start Phase 1 until you have that answer.

Steps 1–7 are mechanical. Step 8 is the decision. Steps 9–12 can happen while
the collection runs.

---

## Step 0 — Deal with OneDrive first

Your project lives in `OneDrive\Desktop\Minor Project`. Collection will write
**thousands of images** into `data/`. OneDrive will try to sync every one of
them, which will slow your machine badly and burn your storage quota.

The code is worth syncing. The data is not — it's re-downloadable and it's
already excluded from Git.

**Fix — keep data outside OneDrive.** Create the folder:

```powershell
mkdir C:\mp-data
```

Then pass `--out-dir C:\mp-data` on every command below. All three scripts
accept it. (This guide already includes the flag in each command.)

> If you'd rather keep everything together, right-click the `data` folder →
> **Free up space** periodically, or exclude it in OneDrive settings. But the
> separate folder is simpler and avoids the problem entirely.

---

## Step 1 — Check your prerequisites *(10 min)*

Open PowerShell and run each of these:

```powershell
python --version
git --version
```

**Expected:** Python 3.10 or newer, and any Git version.

| Problem | Fix |
|---|---|
| `python` not recognised | Install from [python.org](https://python.org). **Tick "Add Python to PATH"** during install — this is the step everyone misses. |
| Python 3.9 or older | Upgrade. The code uses `str \| None` syntax that needs 3.10+. |
| `git` not recognised | Install from [git-scm.com](https://git-scm.com). Accept all defaults. |

---

## Step 2 — Set up the environment *(15 min)*

```powershell
cd "$env:USERPROFILE\OneDrive\Desktop\Minor Project"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Expected:** your prompt now starts with `(.venv)`.

> **If you get a red "running scripts is disabled" error**, run this once, then
> retry the activate line:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

Now install:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Takes 2–3 minutes. If `lxml` fails to build, run
`pip install --only-binary :all: lxml` and then re-run the requirements line.

---

## Step 3 — Confirm the code works *(2 min)*

```powershell
python tests\test_pipeline.py
```

**Expected — this exact line at the end:**

```
  56 passed, 0 failed
```

This runs entirely offline against a local mock server. **If this fails, stop
and fix it before going further** — everything downstream assumes it passes.

You've now verified the whole pipeline without touching the internet or
loading any fact-checker's servers.

---

## Step 4 — Get the Google API key *(20 min)*

This is the key that unlocks the main data source.

1. Go to **[console.cloud.google.com](https://console.cloud.google.com)** and sign in with any Google account.
2. **Create a project** — click the project dropdown in the top bar → **New Project** → name it `minor-project` → **Create**. Wait for it to finish, then make sure it's selected in the dropdown.
3. **Enable the API** — in the search bar at the top, type `Fact Check Tools API` → open it → click **Enable**. Wait for the green confirmation.
4. **Create the key** — left menu → **APIs & Services** → **Credentials** → **+ Create Credentials** → **API key**.
5. **Copy the key** immediately. It looks like `AIzaSy...` and is about 39 characters.

**Optional but sensible:** click **Edit API key** → under *API restrictions*
choose **Restrict key** → tick only **Fact Check Tools API** → Save. If the key
ever leaks it's then useless for anything else.

> No billing card is needed. This API is free.

---

## Step 5 — Set the key as an environment variable *(5 min)*

```powershell
setx FACTCHECK_API_KEY "paste-your-key-here"
```

**Expected:** `SUCCESS: Specified value was saved.`

**Now close PowerShell and open a new window.** `setx` writes the variable
permanently but does *not* apply it to the window you're currently in — this
catches almost everyone.

In the new window:

```powershell
cd "$env:USERPROFILE\OneDrive\Desktop\Minor Project"
.\.venv\Scripts\Activate.ps1
echo $env:FACTCHECK_API_KEY
```

**Expected:** your key prints. If it's blank, `setx` didn't take — re-run it and
open another new window.

---

## Step 6 — Smoke test the real API *(5 min)*

Small run, no images, just checking the key works:

```powershell
python scripts\collect_factchecks.py --source googlefactcheck --limit 10 --no-images --out-dir C:\mp-data
```

**Expected:** log lines like `[gfc] site=boomlive.in page=0 -> 10 claims`, then a
summary table showing ~10 records.

| Problem | Cause and fix |
|---|---|
| `No FACTCHECK_API_KEY set` warning, 0 records | Step 5 didn't stick. Open a fresh window. |
| `HTTP 403` | Key restrictions too tight, or the API isn't enabled. Re-check step 4.3. |
| `HTTP 400` | Key is malformed — re-copy it, watch for a trailing space. |
| 0 records but no error | Key is fine, filters are too tight. Try `--verbose` to see the raw calls. |

Now check what you got:

```powershell
Get-Content C:\mp-data\raw\googlefactcheck.jsonl -TotalCount 1
```

You should see JSON with a `url`, `title`, `claim_text` and `verdict_raw`.

---

## Step 7 — Run the full collection *(4–8 hours, unattended)*

This is the long one. The rate limiting is deliberate — 1.5–4 seconds between
requests, because these are small non-profit newsrooms. **Do not lower it.**

**7a — discovery** (the Google API sweep across all publishers):

```powershell
python scripts\collect_factchecks.py --source googlefactcheck --out-dir C:\mp-data
```

**7b — the direct scrapers** (BOOM, Newschecker, Alt News, Factly, Vishvas):

```powershell
python scripts\collect_factchecks.py --out-dir C:\mp-data
```

**7c — fetch the actual images** for everything the API found:

```powershell
python scripts\enrich_images.py --finance-only --out-dir C:\mp-data
```

This is the slowest step — it visits each article page and downloads the
creatives. `--finance-only` skips articles the filter rejects, which saves
hours of bandwidth.

### If you need to stop

Press `Ctrl+C`. Everything is resumable — re-run the same command and it picks
up where it left off, skipping records already collected. HTTP responses are
cached in `C:\mp-data\.httpcache\`, so re-parsing never re-fetches.

Leave it running overnight. Check on it occasionally; a source failing is
logged and skipped rather than killing the run.

---

## Step 8 — GATE 1: read the yield *(30 min)*

**This is the decision Phase 0 exists to make.**

```powershell
python scripts\build_dataset.py --data-dir C:\mp-data
```

You'll get a table like:

```
source                         raw   finance  labelled  w/image    kept
--------------------------------------------------------------------------
gfc:boomlive.in               1240       312       298      287     284
gfc:newschecker.in             890       141       138      130     128
...
--------------------------------------------------------------------------
TOTAL                         3100                                  520

Label balance: {'fake': 520}
Majority-class baseline: 100.0%
```

### Read three numbers

**1. The `kept` total** — your finance-labelled items.

| Result | What it means | What you do |
|---|---|---|
| **1000+** | Good. Fact-check archives can carry Phase 2. | Proceed with the plan as written. |
| **500–1000** | Workable but thin. | Proceed, but start Telegram collection in parallel rather than in Phase 3. |
| **under 500** | Not enough to train on. | **Re-plan.** Archives become your seed + evaluation set; Telegram moves to Phase 1. Tell Dr. Surati at your next meeting. |

If you're below 500, first try loosening the filter before concluding anything:

```powershell
python scripts\build_dataset.py --data-dir C:\mp-data --threshold 3.0
python scripts\build_dataset.py --data-dir C:\mp-data --threshold 2.5
```

Then **spot-check 20 of the newly-included items by hand.** If they're genuinely
finance-related, keep the lower threshold. If they're political noise, the
original threshold was right and the number is the number.

**2. The `w/image` column** — items with a downloaded image. A multimodal model
needs these. If it's far below `finance`, image extraction is failing on some
site and it's worth investigating.

**3. `Majority-class baseline: 100.0%`** — expected, and it's Gate 2's problem.
Fact-checkers only publish debunks. Note the number and move on.

### Write it down

Put the counts in a file — you'll need them for the report:

```powershell
Copy-Item C:\mp-data\interim\stats.json "docs\gate1_stats.json"
```

---

## Step 9 — Check Factly and Vishvas News *(20 min)*

Both returned HTTP 403 during development — Cloudflare rejecting datacentre
IPs. From your home connection they may work fine.

Look at the step 7b output for those two sources. If they collected records,
nothing to do.

**If they show 0 records or 403 errors**, open `configs\sources.yaml`, find the
`factly` block, and change:

```yaml
  factly:
    kind: article          # was: wordpress
    enabled: true
    base_url: https://factly.in
    sitemaps:
      - https://factly.in/sitemap_index.xml
    path_filters:
      - /
```

Then re-run just that source:

```powershell
python scripts\collect_factchecks.py --source factly --out-dir C:\mp-data
```

Do the same for `vishvasnews` if needed (its sitemap is at
`https://www.vishvasnews.com/sitemap_index.xml`).

---

## Step 10 — Git and GitHub *(30 min)*

Do this while step 7 runs.

```powershell
git init
git add .
git commit -m "Phase 1: fact-check collection pipeline with tests"
```

`.gitignore` already excludes `data/`, `.venv/`, `.session` files and API keys —
check nothing sensitive slipped in:

```powershell
git status
```

**Verify your API key is not in any committed file.** It should only ever live
in the environment variable.

Then create an empty repo on GitHub (no README, no .gitignore — you have them)
and:

```powershell
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

Add your teammate as a collaborator: repo → **Settings** → **Collaborators**.

> Use the **HTTPS** URL, not SSH — it avoids key setup entirely.

---

## Step 11 — Define "lightweight" *(20 min)*

Your title says "Lightweight". Right now that word is unfalsifiable — pick
targets before you build anything, so Phase 4 measures against a commitment
instead of reverse-justifying whatever you end up with.

Reasonable targets for a MuRIL + ViT student model:

| Metric | Target | Why |
|---|---|---|
| Model size | **< 100 MB** after INT8 quantisation | Fits comfortably in a container or a phone |
| CPU inference | **< 100 ms** per post, single thread | Feasible without a GPU |
| Parameters | **< 50 M** | Roughly a third of the teacher |
| Accuracy retained | **> 95%** of teacher F1 | Distillation is worth it only if this holds |

Discuss with Dr. Surati, then write the agreed numbers into `README.md`. Adjust
them if she has a specific deployment scenario in mind.

---

## Step 12 — GATE 2: the negative-class conversation *(1 meeting)*

**The most important conversation in the project.** Bring the Gate 1 numbers.

### The problem in one sentence

Every item collected is labelled `fake`, because fact-checking archives exist to
document falsehoods — nobody publishes an article confirming an ordinary
advertisement is legitimate. A binary classifier cannot be trained on one class.

### Options to put to her

1. **SEBI-registered investment advisors' public posts** — the registry of
   registered IAs and RAs is public, so the label is externally defensible.
   *Strongest option, and it ties back to her interest in the regulatory angle.*
2. **AMFI "Mutual Funds Sahi Hai"** campaign creatives — professionally made,
   unambiguously legitimate, visually similar genre.
3. **Registered broker marketing** — Zerodha, Groww, Upstox.
4. **RBI/SEBI investor-education material.**

### The constraint to state explicitly

> The negatives must be **the same genre of image** — promotional finance
> graphics — differing only in whether they deceive. A model trained on scam
> creatives versus random stock photos will report 97% accuracy and will have
> learned to recognise a screenshot, not a fraud.

Say this out loud in the meeting. It's the kind of reasoning that shows you
understand the failure mode rather than just the method, and it's the single
highest-risk decision in the project.

### What you need to leave with

A written decision on where negatives come from, and roughly how many you're
targeting. Aim for at least 1:3 negative:positive.

---

## Phase 0 done — checklist

- [ ] `56 passed, 0 failed`
- [ ] API key working, smoke test returns records
- [ ] Full collection run complete
- [ ] **Gate 1 number recorded** and phasing decision made
- [ ] Factly and Vishvas either working or switched to sitemap mode
- [ ] Repo pushed, teammate added, no key committed
- [ ] "Lightweight" targets written into the README
- [ ] **Gate 2 decision** in writing from Dr. Surati

Then move to `CHECKLIST.md` Phase 1.

---

## Quick troubleshooting reference

| Symptom | Fix |
|---|---|
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `$env:FACTCHECK_API_KEY` is blank | `setx` needs a **new** terminal window |
| Everything re-downloads on re-run | You deleted `C:\mp-data\.httpcache\` |
| Collection seems frozen | It's rate-limited by design — 1.5–4 s per request |
| Many `unknown` labels | Source emits no ClaimReview rating; check `verdict_raw` in the raw JSONL |
| Machine slows to a crawl during collection | OneDrive is syncing `data/` — see Step 0 |
| A source returns 0 records | Check the log for its error; the run skips failures rather than stopping |

---

*Questions this phase answers: how much data exists, whether the current
phasing survives, and where the negative class comes from. Nothing else in the
project can be planned reliably until all three have answers.*
