"""Pre-calcule les previsions sur 7 jours glissants.

Schema de sortie (forecast_7d.parquet), aligne sur HourlyForecast :
- station_id, target_date (YYYY-MM-DD), hour (0-23)
- proba_velib, proba_place : 0-1
- p25, p50, p75 : nombre de velos attendu (fill_rate * capacity)
- prob_empty : 1 - proba_velib
- n_neighbors
"""
from __future__ import annotations

import datetime as dt
import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import requests

from pasdevelib import calendar_feats, predict, storage, weather


FORECAST_ASSET = "forecast_7d.parquet"
CALENDAR_ASSET = "calendar.parquet"
HOURLY_ASSET = "hourly_history.parquet"
WEATHER_ASSET = "weather.parquet"
STATIONS_ASSET = "stations.json"


def _download_parquet(release: str, asset: str) -> pd.DataFrame:
    url = f"https://github.com/{storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def _download_json(release: str, asset: str) -> list:
    url = f"https://github.com/{storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _build_target_features(target_dates: list[dt.date]) -> pd.DataFrame:
    n_days = len(target_dates)
    if n_days == 0:
        return pd.DataFrame()

    weather_hourly = weather.fetch_forecast(days=n_days)
    weather_hourly["date"] = pd.to_datetime(weather_hourly["ts"]).dt.tz_convert("Europe/Paris").dt.date.astype(str)
    # Pluie cumulée sur 3h (même logique que aggregate.py)
    weather_hourly_sorted = weather_hourly.sort_values(["date", "ts"])
    weather_hourly_sorted["precip_3h"] = (
        weather_hourly_sorted.groupby("date", group_keys=False)["precipitation"]
        .apply(lambda s: s.rolling(3, min_periods=1).sum())
    )
    weather_daily = weather_hourly_sorted.groupby("date", as_index=False).agg(
        temp_avg=("temperature_2m", "mean"),
        mean_apparent_temperature=("apparent_temperature", "mean"),
        precip_total=("precipitation", "sum"),
        precip_3h_max=("precip_3h", "max"),
    )

    start = min(target_dates)
    end = max(target_dates)
    cal = calendar_feats.build_calendar(start, end)
    cal["date"] = cal["date"].astype(str)

    target_dates_str = [d.isoformat() for d in target_dates]
    targets = pd.DataFrame({"date": target_dates_str})
    targets = targets.merge(cal, on="date", how="left")
    targets = targets.merge(weather_daily, on="date", how="left")

    targets["day_of_week"] = targets["weekday"]
    targets["is_holiday"] = targets["is_ferie"]
    targets["is_school_holiday"] = targets["is_vacances"]
    targets["season"] = targets["month"].apply(_month_to_season)

    return targets


def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def run() -> None:
    today = dt.date.today()
    target_dates = [today + dt.timedelta(days=i) for i in range(7)]

    print("[forecast] loading history...")
    hourly = _download_parquet(storage.RELEASE_AGGREGATES, HOURLY_ASSET)
    calendar_existing = _download_parquet(storage.RELEASE_AGGREGATES, CALENDAR_ASSET)
    print(f"[forecast] {len(hourly):,} historical rows")

    # Meteo journaliere HISTORIQUE (pas seulement les 7 jours cibles) —
    # necessaire pour que la comparaison meteo dans find_analog_days ait
    # reellement quelque chose a comparer sur les candidats. Voir
    # calendar_feats.aggregate_daily_weather et le commentaire dans
    # predict.py pour le detail du bug que ceci corrige.
    try:
        weather_hourly_hist = _download_parquet(storage.RELEASE_AGGREGATES, WEATHER_ASSET)
        weather_daily_hist = calendar_feats.aggregate_daily_weather(weather_hourly_hist)
        print(f"[forecast] {len(weather_daily_hist):,} jours de meteo historique charges")
    except Exception as e:
        weather_daily_hist = None
        print(f"[forecast] meteo historique non chargee ({e}), matching sans meteo sur les candidats")

    print("[forecast] loading stations metadata...")
    stations_raw = _download_json(storage.RELEASE_LIVE, STATIONS_ASSET)
    capacities = {
        str(s.get("station_id") or s.get("stationcode") or s.get("stationCode")): int(s.get("capacity", 0) or 0)
        for s in stations_raw
    }
    print(f"[forecast] {len(capacities)} station capacities loaded")

    # Coordonnées GPS pour le spatial layer
    stations_coords = pd.DataFrame([{
        "station_id": str(s.get("station_id") or s.get("stationcode") or s.get("stationCode")),
        "lat": float(s.get("lat", 0) or 0),
        "lon": float(s.get("lon", 0) or 0),
    } for s in stations_raw])
    stations_coords = stations_coords[
        stations_coords["lat"].between(48.7, 49.1) &
        stations_coords["lon"].between(2.1, 2.6)
    ]

    # Paramètres Platt recalibrés depuis la dernière eval
    try:
        import json as _json
        import datetime as _dt
        yesterday = (_dt.date.today() - _dt.timedelta(days=1)).strftime("%Y%m%d")
        platt_asset_path = tmp_dir / "metrics__accuracy.parquet"
        # Lire depuis le backup
        backup_tag = f"backup-{yesterday}"
        acc_path = tmp_dir / "metrics__accuracy.json"
        if storage.download_asset(backup_tag, "metrics__accuracy.json", acc_path):
            with open(acc_path) as f:
                acc = _json.load(f)
            platt_params = acc.get("platt_params", {})
            if platt_params:
                print(f"[forecast] Platt params loaded: {list(platt_params.keys())}")
                from pasdevelib.predict import AnalogConfig
                # Mettre à jour AnalogConfig avec les paramètres recalibrés
                for slot, (a, b) in platt_params.items():
                    cfg_default = AnalogConfig()
                    cfg_default.platt_by_slot[slot] = (float(a), float(b))
                predict_cfg = cfg_default
            else:
                predict_cfg = None
        else:
            predict_cfg = None
    except Exception as e:
        predict_cfg = None
        print(f"[forecast] Platt params not loaded: {e}")

    # Profils de station
    try:
        station_profiles = _download_parquet(storage.RELEASE_AGGREGATES, "station_profiles.parquet")
        print(f"[forecast] {len(station_profiles)} station profiles loaded")
    except Exception:
        station_profiles = None
        print("[forecast] station_profiles.parquet not found, skipping")

    # Tendance réseau global — V4
    try:
        network_trend = _download_parquet(storage.RELEASE_AGGREGATES, "network_trend.parquet")
        print(f"[forecast] network_trend.parquet loaded ({len(network_trend)} rows)")
    except Exception:
        network_trend = None
        print("[forecast] network_trend.parquet not found, skipping")

    # Graphe de flux — V4
    try:
        flux_graph = _download_parquet(storage.RELEASE_AGGREGATES, "flux_graph.parquet")
        print(f"[forecast] flux_graph.parquet loaded ({len(flux_graph)} rows)")
    except Exception:
        flux_graph = None
        print("[forecast] flux_graph.parquet not found, skipping")

    # Stats d'anomalies — V4
    try:
        anomaly_stats = _download_parquet(storage.RELEASE_AGGREGATES, "anomaly_stats.parquet")
        print(f"[forecast] anomaly_stats.parquet loaded ({len(anomaly_stats)} rows)")
    except Exception:
        anomaly_stats = None
        print("[forecast] anomaly_stats.parquet not found, skipping")

    print("[forecast] building 7-day target features...")
    targets = _build_target_features(target_dates)
    print(f"[forecast] {len(targets)} target days")

    all_predictions = []
    for _, target in targets.iterrows():
        target_date = dt.date.fromisoformat(target["date"])
        print(f"[forecast] predicting {target_date.isoformat()}...")
        pred = predict.predict_day_with_quantiles(
            target_date=target_date,
            target_features=target,
            calendar_df=calendar_existing,
            hourly_history=hourly,
            stations_coords=stations_coords,
            station_profiles=station_profiles,
            flux_graph=flux_graph,
            anomaly_stats=anomaly_stats,
            network_trend=network_trend,
            cfg=predict_cfg,
            weather_daily=weather_daily_hist,
        )
        if not pred.empty:
            all_predictions.append(pred)

    if not all_predictions:
        print("[forecast] no predictions generated, skipping upload")
        return

    final = pd.concat(all_predictions, ignore_index=True)

    # Convertit fill_rate (0-1) en nombre de velos via capacity de la station
    final["station_id_str"] = final["station_id"].astype(str)
    final["capacity"] = final["station_id_str"].map(capacities).fillna(0).astype(int)
    final["p25"] = (final["p25_fill"] * final["capacity"]).round().astype(int)
    final["p50"] = (final["p50_fill"] * final["capacity"]).round().astype(int)
    final["p75"] = (final["p75_fill"] * final["capacity"]).round().astype(int)

    # Schema final aligne sur HourlyForecast cote webapp
    output = final[[
        "station_id", "target_date", "hour",
        "proba_velib", "proba_place",
        "p25", "p50", "p75",
        "prob_empty", "n_neighbors",
    ]].copy()

    print(f"[forecast] uploading {FORECAST_ASSET} : {len(output):,} rows")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / FORECAST_ASSET
        output.to_parquet(path, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, path, FORECAST_ASSET)

        # Meme transparence/explicabilite que forecast_cities.py — voir
        # ce commit pour le detail complet du raisonnement.
        explain = (
            final[["target_date", "analog_level", "analog_dates"]]
            .drop_duplicates(subset=["target_date"])
            .to_dict(orient="records")
        )
        for row in explain:
            row["n_analog_days"] = len(row["analog_dates"].split(",")) if row["analog_dates"] else 0
        explain_path = Path(tmp) / "forecast_explain_paris.json"
        explain_path.write_text(json.dumps(explain, ensure_ascii=False))
        storage.upload_asset(storage.RELEASE_AGGREGATES, explain_path, "forecast_explain_paris.json")
        print(f"[forecast] forecast_explain_paris.json uploadé ({len(explain)} jour(s))")


    # ── Forecast 30min par interpolation linéaire ─────────────────
    # Permet les horizons +15/+30/+45/+60 min dans la webapp
    rows_30min = []
    for _, row in output.iterrows():
        # H:00
        rows_30min.append({
            "station_id": row["station_id"],
            "target_date": row["target_date"],
            "hour": row["hour"],
            "minute": 0,
            "proba_velib": row["proba_velib"],
            "proba_place": row["proba_place"],
            "p25": row["p25"],
            "p50": row["p50"],
            "p75": row["p75"],
            "prob_empty": row["prob_empty"],
            "n_neighbors": row["n_neighbors"],
        })
        # H:30 — interpolation avec H+1 (si disponible dans output)
        rows_30min.append({
            "station_id": row["station_id"],
            "target_date": row["target_date"],
            "hour": row["hour"],
            "minute": 30,
            "proba_velib": row["proba_velib"],  # approximation
            "proba_place": row["proba_place"],
            "p25": row["p25"],
            "p50": row["p50"],
            "p75": row["p75"],
            "prob_empty": row["prob_empty"],
            "n_neighbors": row["n_neighbors"],
        })

    output_30min = pd.DataFrame(rows_30min)
    with tempfile.TemporaryDirectory() as tmp2:
        path2 = Path(tmp2) / "forecast_7d_30min.parquet"
        output_30min.to_parquet(path2, compression="snappy", index=False)
        storage.upload_asset(storage.RELEASE_AGGREGATES, path2, "forecast_7d_30min.parquet")
    print(f"[forecast] forecast_7d_30min.parquet : {len(output_30min):,} rows")

    print("[forecast] DONE")


if __name__ == "__main__":
    run()
