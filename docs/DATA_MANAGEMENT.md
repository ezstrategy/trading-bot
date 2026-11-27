# Data management (Quant Lab Phase 2)

Research uses **dataset versions**: frozen copies of historical candles with their source, a
checksum and a quality report. The live signal engine keeps using `data/raw` as before.

## Quick start (Windows)

| What | How |
|---|---|
| Freeze the MT5 candles you already downloaded (M1, M5, M15) | double-click `snapshot_data.bat` |
| Import a CSV file | drag the file onto `import_csv.bat` |
| See all versions | `python -m quant_engine data-list` |
| Look at one version and its problems | `python -m quant_engine data-show <version_id>` |
| Check that nothing changed since import | `python -m quant_engine data-verify <version_id>` |
| Show the broker profile | `python -m quant_engine instrument` |

## Where things are

```
data/catalog/XAUUSD/M5/XAUUSD_M5_20261126T200546Z_4a65c52e/
    original.csv      the file exactly as imported (or original.parquet for MT5 snapshots)
    bars.parquet      normalised candles: time = candle OPEN in UTC, open/high/low/close, tick_volume, spread (points), real_volume
    quality.json      every issue found
    manifest.json     source, import settings, SHA-256 of the original, fingerprint of the bars, row counts, quality summary
```

The version id is `<symbol>_<timeframe>_<import time UTC>_<first 8 characters of the bars fingerprint>`.
The files are made read-only. Importing the same file with the same settings again returns the
existing version; anything different (a new export, another timezone) is a new version, and old
versions stay as they were, so every experiment can name the exact data it used.

## CSV formats

Detected automatically:

- **MT5 export** (right-click chart → Bars → Export, or the Symbols window): tab separated,
  `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>`.
- **No header**: `2024.01.08,02:00,open,high,low,close,tickvol[,vol,spread]`.
- **Any CSV with a header**: a time column (`time`, `datetime`, `timestamp`, or `date` + `time`)
  and `open, high, low, close`; optional `volume`/`tick_volume`, `spread`. Comma, semicolon, tab
  or `|`. Epoch seconds or milliseconds also work.

## Timezone of the file (`--tz`)

| Value | Meaning |
|---|---|
| `mt5_server` | broker server clock, converted with the same rules as MT5 downloads (`config/mt5.yaml: server_time`; most brokers: GMT+2 winter, GMT+3 summer) |
| `UTC` | already UTC |
| `+02:00` | a fixed offset all year |
| `Europe/London` etc. | a real timezone with daylight saving |

A local time that does not exist or happens twice (daylight-saving change) is reported as an
unparseable row rather than guessed.

## Quality checks

Blocking (research refuses the version until you decide, `data.yaml: quality.blocking_issues`):
duplicate candle times, unsorted or non-UTC times, times off the candle grid, missing or
non-positive prices, impossible candles (high below open/close, low above), prices outside
`price_bounds`, rows that could not be read.

Reported only: missing candles outside normal market closures, range spikes, large gaps between
a close and the next open, rows that were out of time order in the file (stored sorted, values
unchanged), no spread in the source.

Nothing is repaired: missing candles are not created, duplicates are not removed, and a file
without spreads keeps an empty spread. Backtests must then state the spread they assume
(`config/instruments.yaml: typical_spread`, 0.08).

## Broker profile

`config/instruments.yaml` holds the instrument specification per broker (contract size, lot
limits and step, tick size and value, digits, currencies, commission, execution assumptions).
Only the 0.08 spread and zero commission are confirmed; everything else is marked as a guess
until the broker is chosen. Phase 8 fills a profile from MT5's symbol specification.
