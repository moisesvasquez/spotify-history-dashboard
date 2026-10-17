# My Spotify history dashboard

A recreation of the [r/dataisbeautiful "Twelve years of my Spotify streaming history"](https://www.reddit.com/r/dataisbeautiful/comments/1x3xbr4/oc_twelve_years_of_my_spotify_streaming_history/) graphic, built from my own Spotify Extended Streaming History export.

- `analysis/build.py`: pandas pipeline (load & clean, music/podcast split, skip rate, 7-day earworms, discovery rate, genre shares, double-count check). Writes `dashboard/data.js`.
- `analysis/genres.py`: hand-made artist → genre map (Spotify's export has no genres).
- `dashboard/index.html`: the dashboard (d3 + vanilla JS).

## Run it with your own data

1. Request your *Extended streaming history* from Spotify's privacy page and unzip it into `Spotify Extended Streaming History/` at the repo root.
2. `pip install pandas numpy scipy tzdata`
3. Set `TZ` in `analysis/build.py`, then `python analysis/build.py`
4. `python -m http.server 8765 --directory dashboard` and open http://localhost:8765

The raw export is git-ignored; only aggregated numbers are committed.
