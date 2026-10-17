"""Spotify extended streaming history -> dashboard/data.js

Pipeline follows the original r/dataisbeautiful post:
load & clean, classify music/podcast, aggregate, skip rate, earworms,
discovery rate, hand-made genres, double-count sanity check.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from genres import ARTIST_GENRE, GENRES

ROOT = Path(__file__).resolve().parent.parent
TZ = "America/New_York"  # local time for the heatmap / years
OUT = ROOT / "dashboard" / "data.js"

# 1. Load & clean -----------------------------------------------------------
files = glob.glob(str(ROOT / "Spotify Extended Streaming History" / "Streaming_History_*.json"))
df = pd.concat([pd.read_json(f) for f in files], ignore_index=True)
df = df.drop_duplicates()
df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(TZ)
df = df.sort_values("ts").reset_index(drop=True)

# 2. Classify ---------------------------------------------------------------
df["h"] = df["ms_played"] / 3_600_000
df["year"] = df["ts"].dt.year
df["kind"] = np.where(df["spotify_track_uri"].notna(), "music",
                      np.where(df["spotify_episode_uri"].notna(), "podcast", "other"))
df = df.rename(columns={"master_metadata_album_artist_name": "artist",
                        "master_metadata_track_name": "track"})
music = df[df["kind"] == "music"].copy()
pods = df[df["kind"] == "podcast"]
plays = music[music["ms_played"] >= 30_000]  # a play counts if >= 30 s

years = sorted(df["year"].unique())
last_ts = df["ts"].max()

# 3. Aggregate --------------------------------------------------------------
hours_year = pd.DataFrame({
    "music": music.groupby("year")["h"].sum(),
    "podcasts": pods.groupby("year")["h"].sum(),
}).reindex(years).fillna(0)

artist_hours = music.groupby("artist")["h"].sum().sort_values(ascending=False)
track_plays = plays.groupby(["track", "artist"]).size().sort_values(ascending=False)
top_track = track_plays.index[0]

# 4. Skip rate (share of music plays under 30 s) ----------------------------
skip = (music["ms_played"] < 30_000).groupby(music["year"]).mean()

# 5. Earworms: max plays of one song within any 7-day window ----------------
def max_7d(ts):
    t = np.sort(ts.values)
    end = np.searchsorted(t, t + np.timedelta64(7, "D"))
    i = int(np.argmax(end - np.arange(len(t))))
    return pd.Series({"n": int(end[i] - i), "start": t[i]})

ew = (plays.groupby(["spotify_track_uri"])["ts"].apply(lambda s: max_7d(s.dt.tz_localize(None)))
      .unstack())
ew["n"] = ew["n"].astype(int)
meta = plays.drop_duplicates("spotify_track_uri").set_index("spotify_track_uri")[["track", "artist"]]
ew = ew.join(meta).sort_values("n", ascending=False)
# one row per song title/artist (same song can have several URIs)
ew = ew.drop_duplicates(["track", "artist"])
earworms = [{"n": int(r.n), "track": r.track, "artist": r.artist,
             "week": pd.Timestamp(r.start).strftime("%b %Y")} for r in ew.head(6).itertuples()]
n_earworm_songs = int((ew["n"] >= 20).sum())

# Longest daily streak -------------------------------------------------------
days = pd.Series(sorted(df["ts"].dt.normalize().dt.tz_localize(None).unique()))
grp = (days.diff() != pd.Timedelta(days=1)).cumsum()
streak = days.groupby(grp).agg(["count", "min", "max"]).sort_values("count").iloc[-1]

# 6. Discovery rate & top-10 concentration ----------------------------------
first_year = music.groupby("artist")["year"].min()
music["new"] = music["artist"].map(first_year) == music["year"]
new_share = music.groupby("year").apply(lambda g: g.loc[g["new"], "h"].sum() / g["h"].sum())
top10_share = music.groupby("year").apply(
    lambda g: g.groupby("artist")["h"].sum().nlargest(10).sum() / g["h"].sum())

# 7. Genres -----------------------------------------------------------------
music["genre"] = music["artist"].map(ARTIST_GENRE)
coverage = music.loc[music["genre"].notna(), "h"].sum() / music["h"].sum()
n_mapped = int(music.loc[music["genre"].notna(), "artist"].nunique())
gm = music[music["genre"].notna()]

monthly = gm.groupby([gm["ts"].dt.tz_localize(None).dt.to_period("M"), "genre"])["h"].sum().unstack(fill_value=0)
monthly = monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(), freq="M"), fill_value=0)
# smooth (centered 7-month window) then normalise to shares
smooth = monthly.rolling(7, center=True, min_periods=1, win_type="gaussian").mean(std=2.0)
shares = smooth.div(smooth.sum(axis=1), axis=0).fillna(0)
genre_order = list(GENRES.keys())
shares = shares.reindex(columns=genre_order, fill_value=0)

yearly_genre = gm.groupby(["year", "genre"])["h"].sum().unstack(fill_value=0).reindex(columns=genre_order, fill_value=0)
yearly_share = yearly_genre.div(yearly_genre.sum(axis=1), axis=0)

# eras: consecutive years with the same dominant genre (skip thin partial first year)
full_years = [y for y in years if hours_year.loc[y, "music"] > 100]
eras = []
for y in full_years:
    g = yearly_share.loc[y].idxmax()
    if eras and eras[-1]["genre"] == g:
        eras[-1]["end"] = int(y)
    else:
        eras.append({"genre": g, "start": int(y), "end": int(y)})
for e in eras:
    peak = yearly_share.loc[e["start"]:e["end"], e["genre"]]
    e["peak_year"] = int(peak.idxmax())
    e["peak_share"] = float(peak.max())

# 8. Sanity check: overlapping streams (two devices at once) ----------------
start = df["ts"] - pd.to_timedelta(df["ms_played"], unit="ms")
prev_end = df["ts"].cummax().shift()
overlap_ms = (prev_end - start).dt.total_seconds().clip(lower=0).fillna(0) * 1000
overlap_ms = np.minimum(overlap_ms, df["ms_played"])
double_pct = float(overlap_ms.sum() / df["ms_played"].sum())

# Heatmap: hours by weekday x hour -----------------------------------------
heat = df.groupby([df["ts"].dt.dayofweek, df["ts"].dt.hour])["h"].sum().unstack(fill_value=0)
heat = heat.reindex(index=range(7), columns=range(24), fill_value=0)
peak_day = int(df.groupby(df["ts"].dt.dayofweek)["h"].sum().idxmax())
peak_hour = int(df.groupby(df["ts"].dt.hour)["h"].sum().idxmax())

# Output -------------------------------------------------------------------
first_skip_year = full_years[0]
data = {
    "meta": {
        "start": df["ts"].min().strftime("%B %Y"),
        "end": last_ts.strftime("%B %Y"),
        "end_day": "",
        "plays": int(len(df)),
        "hours": float(df["h"].sum()),
        "n_artists": int(music["artist"].nunique()),
        "tz": TZ,
        "coverage": float(coverage),
        "n_mapped": n_mapped,
        "double_pct": double_pct,
        "n_earworm_songs": n_earworm_songs,
        "peak_day": peak_day,
        "peak_hour": peak_hour,
    },
    "genres": genre_order,
    "stream": {
        "dates": [str(p) for p in shares.index],
        "values": shares.round(4).values.tolist(),
    },
    "eras": eras,
    "stats": {
        "top_song": {"plays": int(track_plays.iloc[0]), "track": top_track[0], "artist": top_track[1]},
        "earworm": earworms[0],
        "streak": {"days": int(streak["count"]), "from": streak["min"].strftime("%b %Y"),
                   "to": streak["max"].strftime("%b %Y")},
        "skip": {"from_year": int(first_skip_year), "from": float(skip[first_skip_year]),
                 "to_year": int(years[-1]), "to": float(skip[years[-1]])},
    },
    "hours_year": {"years": [int(y) for y in years],
                   "music": hours_year["music"].round(1).tolist(),
                   "podcasts": hours_year["podcasts"].round(1).tolist()},
    "explore": {"years": [int(y) for y in full_years],
                "new": [float(new_share[y]) for y in full_years],
                "top10": [float(top10_share[y]) for y in full_years]},
    "artists": [{"name": a, "h": float(h)} for a, h in artist_hours.head(8).items()],
    "earworms": earworms,
    "heat": heat.round(2).values.tolist(),
}

# Windows strftime has no %-d
data["meta"]["end_day"] = f"{last_ts.strftime('%B')} {last_ts.day}"

OUT.parent.mkdir(exist_ok=True)
OUT.write_text("window.DATA = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")
print(json.dumps({k: data[k] for k in ["meta", "eras", "stats", "hours_year", "explore", "artists", "earworms"]},
                 ensure_ascii=False, indent=1, default=str))
