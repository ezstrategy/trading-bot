# Quant Lab Phase 2 report: historical data management

Date: 2026-11-26. Approved plan: [ARCHITECTURE.md](ARCHITECTURE.md) section 4.1 (and the module 2
profile skeleton from section 4.2). Branch `claude/project-thread-hdljm2`, stacked on PR #8.

## Completed

| Requirement (module 1) | How | Where |
|---|---|---|
| Import CSV OHLC data | MT5 export, MT5 no-header and any headed CSV; comma / semicolon / tab / pipe; UTF-8 or UTF-16 | `data/csv_import.py` |
| 1m, 5m, 15m | Declared with `--timeframe`; the file's median candle spacing must match, or the import stops | `check_timeframe` |
| Future MT5 imports | `data-snapshot` freezes MT5 bars already downloaded (the existing downloader converts server time to UTC); `--tz mt5_server` for MT5 CSV exports uses the same `ServerClock` rules | `catalog.import_frame`, `cli.py` |
| Normalise timestamps to UTC | UTC, fixed offsets, IANA zones with daylight saving, broker server clock; impossible / doubled local times are reported, not guessed | `csv_import.to_utc` |
| Missing / duplicate candles, OHLC consistency, abnormal prices | Existing checks plus two new ones: open-to-previous-close gap (warning) and price bounds (error) | `data/quality.py` |
| Preserve the original | Byte-for-byte copy in each version folder, SHA-256 in the manifest, files read-only, `data-verify` re-checks | `data/catalog.py` |
| Track source and version | Version id, source kind / name / file name, import settings, server-clock settings, created time, row counts, bars fingerprint | `manifest.json` |
| No manufactured candles or spreads | Missing candles stay missing; no spread column means an empty spread and a `no_spread` warning | tests below |

Also added: `config/instruments.yaml` and `quant_engine/instruments.py` (module 2 skeleton:
contract size, lot limits and step, tick size and value, digits, currencies, commission,
execution assumptions, lot rounding, money per price move, consistency warnings, a builder from
MT5 `symbol_info` for Phase 8). Nothing uses it yet; the backtester will in Phase 3.

New commands: `data-import`, `data-snapshot`, `data-list`, `data-show`, `data-verify`,
`instrument`; launchers `import_csv.bat`, `snapshot_data.bat`. User guide: [DATA_MANAGEMENT.md](DATA_MANAGEMENT.md).

## Design choices worth checking

1. **Blocking vs reported issues.** Duplicates, impossible candles, bad rows and wrong prices
   block a version from research until you decide (`allow_issues=True`). Missing candles do
   not block, because real gold data always has some (holidays, broker outages); they are listed
   in the report. The list is in `config/data.yaml: quality.blocking_issues` if you want it stricter.
2. **Row order.** Rows out of time order are stored sorted (values unchanged) and reported as
   `reordered`. Everything else is stored exactly as it came in.
3. **Unreadable rows** cannot be stored as candles, so they are not in `bars.parquet`. They are
   listed with their line number, they block the version, and they are still in `original.csv`.
4. **Existing storage unchanged.** `data/raw` (the MT5 download store) and every old command
   work as before. The catalog is a separate folder, `data/catalog`.
5. GUESS: `price_bounds: [100, 20000]` for XAUUSD; GUESS: MT5 CSV exports are in server time, so
   `import_csv.bat` defaults to `mt5_server`.

## Tests

- New: `tests/test_data_import.py` (17 tests) and `tests/test_instruments.py` (6 tests), all on
  hand-written fixtures in `tests/fixtures/data/` with known answers (for example server 02:00 in
  January and server 03:00 in July both become 00:00 UTC).
- Full suite: **316 passed** (293 existing + 23 new), 4 min 3 s in the cloud.

## Outstanding

- Real data has not been imported yet: it is only on your PC. Run `snapshot_data.bat` there
  (after `download` for M1 if you have it), then send me the `data-list` output so I can check
  the quality reports. I can run it for you when the PC is on.
- The dashboard has no data-manager page yet (Phase 7). Until then use the commands above.
- The old research tools still read `data/raw` directly; Phase 3's engine reads versions only.

_No performance claims. Signals and research only._
