"""
Clean raw AQI datasets and save to data/processed/.
Run from project root: python scripts/clean_raw_data.py
"""
import os
import sys
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

# Paths (run from project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_delhi_ncr_aqi():
    """Clean Delhi NCR hourly AQI. Prefer delhi_ncr_aqi.csv; fallback to delhi_ncr_aqi_dataset.csv."""
    for fname in ["delhi_ncr_aqi.csv", "delhi_ncr_aqi_dataset.csv"]:
        path = RAW_DIR / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        # Parse datetime if string
        if "datetime" in df.columns and df["datetime"].dtype == object:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if "date" in df.columns and df["date"].dtype == object:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Drop rows with null AQI or PM2.5 (critical for analysis)
        df = df.dropna(subset=["aqi", "pm25"])
        # Keep 2020--2024
        if "year" in df.columns:
            df = df[df["year"].between(2020, 2024)]
        # Ensure numeric
        num_cols = ["pm25", "pm10", "no2", "so2", "co", "o3", "latitude", "longitude", "aqi",
                    "temperature", "humidity", "wind_speed", "visibility"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        out = PROCESSED_DIR / "delhi_ncr_aqi_2020_2024_clean.csv"
        df.to_csv(out, index=False)
        print(f"Saved {out.name} ({len(df):,} rows)")
        return
    print("Skip delhi_ncr: no raw file found (delhi_ncr_aqi.csv or delhi_ncr_aqi_dataset.csv)")


def clean_city_day():
    """Clean city_day.csv and derive COVID pre/during summary."""
    path = RAW_DIR / "city_day.csv"
    if not path.exists():
        print("Skip city_day: file not found")
        return
    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["year"] = df["Date"].dt.year
    # Numeric columns
    for c in ["PM2.5", "PM10", "NO", "NO2", "CO", "SO2", "O3", "AQI"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Clean: drop rows where AQI is null for city-day series
    df_clean = df.dropna(subset=["AQI"]).copy()
    df_clean = df_clean[df_clean["City"].notna()]
    out_clean = PROCESSED_DIR / "city_day_clean.csv"
    df_clean.to_csv(out_clean, index=False)
    print(f"Saved {out_clean.name} ({len(df_clean):,} rows)")

    # COVID summary: pre (2015-2019) vs lockdown (Mar-May 2020)
    pre = df[(df["Date"] >= "2015-01-01") & (df["Date"] <= "2019-12-31")]
    covid = df[(df["Date"] >= "2020-03-01") & (df["Date"] <= "2020-05-31")]
    pre_avg = pre.groupby("City")["AQI"].mean().reset_index(name="aqi_pre")
    covid_avg = covid.groupby("City")["AQI"].mean().reset_index(name="aqi_covid")
    city_change = pre_avg.merge(covid_avg, on="City", how="inner")
    city_change["aqi_drop"] = city_change["aqi_pre"] - city_change["aqi_covid"]
    city_change = city_change.sort_values("aqi_drop", ascending=False)
    out_covid = PROCESSED_DIR / "covid_city_aqi_change_2015_2020.csv"
    city_change.to_csv(out_covid, index=False)
    print(f"Saved {out_covid.name} ({len(city_change)} cities)")


def clean_india_varios():
    """Clean india_varios.csv: keep PM2.5 only, rename pollutant_avg to pm25."""
    path = RAW_DIR / "india_varios.csv"
    if not path.exists():
        print("Skip india_varios: file not found")
        return
    df = pd.read_csv(path, low_memory=False)
    # Keep only PM2.5 rows
    pid = df["pollutant_id"].astype(str).str.strip()
    df = df[pid.str.contains("PM2", case=False, na=False)].copy()
    df = df.rename(columns={"pollutant_avg": "pm25"})
    if "pm25" not in df.columns and "pollutant_avg" in df.columns:
        df["pm25"] = pd.to_numeric(df["pollutant_avg"], errors="coerce")
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = df.dropna(subset=["pm25", "latitude", "longitude"])
    out = PROCESSED_DIR / "india_pm25_snapshot_clean.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out.name} ({len(df):,} rows)")


def clean_major_city():
    """Clean major_city.csv: parse dates, drop null AQI."""
    path = RAW_DIR / "major_city.csv"
    if not path.exists():
        print("Skip major_city: file not found")
        return
    df = pd.read_csv(path, low_memory=False)
    date_col = "Datetime" if "Datetime" in df.columns else "Date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, "AQI"])
    for c in ["PM2.5", "PM10", "NO", "NO2", "CO", "SO2", "O3", "AQI"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    out = PROCESSED_DIR / "major_city_clean.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out.name} ({len(df):,} rows)")


def clean_india_dataset():
    """Clean India_dataset.csv: parse date, NA -> NaN, numeric."""
    path = RAW_DIR / "India_dataset.csv"
    if not path.exists():
        print("Skip India_dataset: file not found")
        return
    df = pd.read_csv(path, low_memory=False, na_values=["NA", "N/A", ""])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "sampling_date" in df.columns:
        df["sampling_date"] = df["sampling_date"].astype(str)
    for c in ["so2", "no2", "rspm", "spm", "pm2_5"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all", axis=1)
    out = PROCESSED_DIR / "india_historical_clean.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out.name} ({len(df):,} rows)")


def clean_processed_aqi_data():
    """Move/clean processed_aqi_data.csv from raw to processed (ensure dtypes)."""
    path = RAW_DIR / "processed_aqi_data.csv"
    if not path.exists():
        print("Skip processed_aqi_data: file not found")
        return
    df = pd.read_csv(path, low_memory=False)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["pm25", "pm10", "co", "no2", "o3", "so2", "aqi", "hour", "day", "month", "year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    out = PROCESSED_DIR / "processed_aqi_data_clean.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out.name} ({len(df):,} rows)")


def copy_geojson():
    """Copy india_states.geojson to processed (no structural change)."""
    src = RAW_DIR / "india_states.geojson"
    if not src.exists():
        print("Skip india_states.geojson: file not found")
        return
    dst = PROCESSED_DIR / "india_states.geojson"
    shutil.copy2(src, dst)
    print(f"Copied {dst.name} to processed/")


def main():
    os.chdir(PROJECT_ROOT)
    print("Cleaning raw data -> data/processed/\n")
    clean_delhi_ncr_aqi()
    clean_city_day()
    clean_india_varios()
    clean_major_city()
    clean_india_dataset()
    clean_processed_aqi_data()
    copy_geojson()
    print("\nDone.")


if __name__ == "__main__":
    main()
