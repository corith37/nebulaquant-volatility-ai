# Paper-trading the momentum strategy with Alpaca (3-month run)

This is the easiest **honest** way to let the tool trade for you: an Alpaca
**paper** account (real prices, fake money, identical API to live), driven by a
script that trades the *same* regime-gated momentum picks the dashboard shows.

> **No real money is involved in paper mode.** Read `README` / the dashboard's
> "How it works" tab for the survivorship-bias and small-sample caveats before
> you ever consider `--live`.

---

## What you're running

Each time you run the executor it:

1. Computes today's **top-3 gated momentum picks** (`src.scanner.today_picks` —
   the exact logic behind the dashboard's *Today's Picks* tab).
2. Reads what the paper account already holds.
3. **Sells** any holding past its ~20-trading-day time exit.
4. **Buys** the top free slots, equal-weight, using **notional (fractional)**
   orders so a tiny account works (e.g. "$60 of MU").
5. Records everything to the local ledger so the dashboard marks it to market.

A risk-off regime (SPY below its 200-day average, or elevated VIX) **blocks new
buys** but never force-sells — holds run to their timer, matching the backtest.

---

## One-time setup (~10 minutes)

### 1. Install the broker SDK
```
.venv\Scripts\python.exe -m pip install alpaca-py
```

### 2. Make a free Alpaca paper account
- Sign up at **https://alpaca.markets** → switch to **Paper Trading**.
- **Generate API Keys** (Paper). Copy the **Key ID** and **Secret Key** (the
  secret is shown once).

### 3. Give the script your keys (never commit them)
Create a file named **`.env`** in the project root (it is gitignored):
```
APCA_API_KEY_ID=PKxxxxxxxxxxxxxxxxxx
APCA_API_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
(Or set them as environment variables with the same names.)

### 4. Make sure the data is current
```
.venv\Scripts\python.exe scripts\download_data.py
.venv\Scripts\python.exe scripts\build_dataset.py
```
(Or click **🔄 Refresh to latest prices** on the dashboard's *Today's Picks* tab.)

---

## Daily / weekly use

**See the plan (safe — sends nothing):**
```
.venv\Scripts\python.exe scripts\execute_alpaca.py
```
This prints the account, the regime, the exact SELL/BUY plan, and the hold
timeline, then stops (dry run).

**Actually trade the paper account** (run during US market hours — notional
orders need regular trading hours):
```
.venv\Scripts\python.exe scripts\execute_alpaca.py --submit
```

**Size as if the account were $200** (handy while learning):
```
.venv\Scripts\python.exe scripts\execute_alpaca.py --budget 200 --submit
```

Useful flags: `--positions N` (max concurrent names), `--deploy-fraction 0.9`
(how much of equity to put to work), `--force` (submit even if market closed).

Because the strategy holds ~20 trading days, **once a week is plenty** — daily
just lets it act the moment a position hits its exit or the regime flips.

---

## Automating it (optional)

The strategy only needs attention about weekly, so full automation is optional.
If you want hands-off runs, schedule the `--submit` command on a weekday morning
after the open (~9:35 ET):

- **Windows Task Scheduler** → Create Task → Trigger: weekly, Mon 9:35 AM →
  Action: Program `...\.venv\Scripts\python.exe`, Arguments
  `scripts\execute_alpaca.py --submit`, Start in: the project folder.
- Keep a log: append ` >> logs\exec.log 2>&1` style redirection via a wrapper
  `.bat`, or just re-run the dry run to inspect state.

Start by scheduling the **dry run** (no `--submit`) for a week and read the
output before you let it submit.

---

## Going live later (real money) — not yet

When (if) you've watched paper behave for a few weeks and accept the caveats:
```
.venv\Scripts\python.exe scripts\execute_alpaca.py --live --i-understand --submit
```
`--live` is refused without `--i-understand`. Fund the live account with only
what you're willing to treat as tuition. Everything else is identical.

---

## Reality check (the part that matters)

- **~10–12 trades in 3 months** can't prove an edge — your result will be mostly
  luck. Judge the *process*, not the P&L.
- Headline backtest returns are **survivorship-biased** (curated large-caps).
- The ledger's P&L is an approximation; **Alpaca's dashboard is the truth** for
  fills, fractional quantities, and account value.
- This is research tooling, **not investment advice**.
