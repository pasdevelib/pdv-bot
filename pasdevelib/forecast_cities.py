"""forecast_cities.py — Prévisions 7 jours pour les villes autres que Paris.

Réutilise exactement le même moteur (predict.predict_day_with_quantiles)
que forecast.py, mais alimenté avec les données de chaque ville :
- hourly_history_<ville>.parquet (release cities-history, produit par
  consolidate_cities.py)
- stations_cities.json (release cities-live, produit par scrape_cities.py)
  pour la capacité et les coordonnées GPS (spatial layer)
- météo au centre de la ville (Open-Meteo, cf. cities.city_center_latlon)

Limite connue et acceptée : le calendrier historique utilisé pour choisir
les jours analogues (calendar.parquet) a été construit avec la météo de
Paris (aucun historique météo par ville n'a encore été accumulé pour les
villes secondaires). C'est un compromis raisonnable pour le lancement :
jour de semaine / vacances / jour férié / saison (les facteurs les plus
pondérés dans AnalogConfig) restent corrects pour toute la France ; seule
la correspondance météo fine est approximative. À affiner une fois que
quelques mois d'historique météo par ville auront été accumulés.

Les prévisions n'utilisent pas station_profiles / network_trend /
flux_graph / anomaly_stats (ces aggregats V3/V4 sont pour l'instant
Paris-only) — predict_day_with_quantiles les accepte en optionnel (None)
sans planter, cf. ses signatures. Le spatial layer géographique (voisines
dans un rayon donné), lui, fonctionne bien : les coordonnées par ville
sont disponibles dans stations_cities.json.

Schema de sortie (forecast_7d_<ville>.parquet), identique à Paris :
- station_id, target_date (YYYY-MM-DD), hour (0-23)
- proba_velib, proba_place : 0-1
- p25, p50, p75 : nombre de vélos attendu (fill_rate * capacity)
- prob_empty : 1 - proba_velib
- n_neighbors
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import requests

from pasdevelib import calendar_feats, predict, storage, weather
from pasdevelib.cities import list_cities, city_center_latlon

RELEASE_CITIES_HISTORY = "cities-history"     # produit par consolidate_cities.py
RELEASE_CITIES_LIVE = "cities-live"           # produit par scrape_cities.py
RELEASE_CITIES_AGGREGATES = "cities-aggregates"  # sortie de ce module

CALENDAR_ASSET = "calendar.parquet"           # national, partagé avec Paris


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


def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _build_target_features(target_dates: list[dt.date], lat: float, lon: float) -> pd.DataFrame:
    n_days = len(target_dates)
    if n_days == 0:
        return pd.DataFrame()

    weather_hourly = weather.fetch_forecast(days=n_days, lat=lat, lon=lon)
    weather_hourly["date"] = pd.to_datetime(weather_hourly["ts"]).dt.tz_convert("Europe/Paris").dt.date.astype(str)
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


def run_city(city_id: str, calendar_existing: pd.DataFrame) -> None:
    print(f"[forecast_cities] === {city_id} ===")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        history_name = f"hourly_history_{city_id}.parquet"
        history_path = tmp_dir / history_name
        if not storage.download_asset(RELEASE_CITIES_HISTORY, history_name, history_path):
            print(f"[forecast_cities] {city_id}: {history_name} introuvable, skip (pas encore assez de scrapes ?)")
            return
        hourly = pd.read_parquet(history_path)
        if hourly.empty:
            print(f"[forecast_cities] {city_id}: historique vide, skip")
            return
        print(f"[forecast_cities] {city_id}: {len(hourly):,} lignes d'historique")

        # Stations de la ville (capacité + coordonnées pour le spatial layer)
        stations_all = _download_json(RELEASE_CITIES_LIVE, "stations_cities.json")
        stations_city = [s for s in stations_all if s.get("city_id") == city_id]
        if not stations_city:
            print(f"[forecast_cities] {city_id}: aucune station dans stations_cities.json, skip")
            return
        capacities = {str(s["station_id"]): int(s.get("capacity", 0) or 0) for s in stations_city}
        stations_coords = pd.DataFrame([{
            "station_id": str(s["station_id"]),
            "lat": float(s.get("lat", 0) or 0),
            "lon": float(s.get("lon", 0) or 0),
        } for s in stations_city])

        lat, lon = city_center_latlon(city_id)
        today = dt.date.today()
        target_dates = [today + dt.timedelta(days=i) for i in range(7)]
        targets = _build_target_features(target_dates, lat, lon)
        print(f"[forecast_cities] {city_id}: {len(targets)} jours cibles, météo centrée sur ({lat:.3f}, {lon:.3f})")

        all_predictions = []
        for _, target in targets.iterrows():
            target_date = dt.date.fromisoformat(target["date"])
            pred = predict.predict_day_with_quantiles(
                target_date=target_date,
                target_features=target,
                calendar_df=calendar_existing,
                hourly_history=hourly,
                stations_coords=stations_coords,
                # V3/V4 (profils, tendance réseau, flux, anomalies) : pas
                # encore calculés pour les villes secondaires, cf. docstring
                # du module — predict_day_with_quantiles gère None sans
                # planter, la prédiction se limite au coeur k-NN + spatial.
                station_profiles=None,
                flux_graph=None,
                anomaly_stats=None,
                network_trend=None,
            )
            if not pred.empty:
                all_predictions.append(pred)

        if not all_predictions:
            print(f"[forecast_cities] {city_id}: aucune prédiction générée, skip upload")
            return

        final = pd.concat(all_predictions, ignore_index=True)
        final["station_id_str"] = final["station_id"].astype(str)
        final["capacity"] = final["station_id_str"].map(capacities).fillna(0).astype(int)
        final["p25"] = (final["p25_fill"] * final["capacity"]).round().astype(int)
        final["p50"] = (final["p50_fill"] * final["capacity"]).round().astype(int)
        final["p75"] = (final["p75_fill"] * final["capacity"]).round().astype(int)

        output = final[[
            "station_id", "target_date", "hour",
            "proba_velib", "proba_place",
            "p25", "p50", "p75",
            "prob_empty", "n_neighbors",
        ]].copy()

        out_name = f"forecast_7d_{city_id}.parquet"
        out_path = tmp_dir / out_name
        output.to_parquet(out_path, compression="snappy", index=False)
        storage.upload_asset(RELEASE_CITIES_AGGREGATES, out_path, out_name)
        print(f"[forecast_cities] {city_id}: {out_name} uploadé ({len(output):,} lignes)")

        # Fichier compagnon de transparence/explicabilite (JSON, leger,
        # un enregistrement par jour cible) — expose EXACTEMENT quels
        # jours passes ont ete utilises pour chaque prediction, consomme
        # par une page admin dediee. Separe du parquet principal pour
        # eviter de repeter la meme chaine sur ~150+ lignes par jour et
        # rester trivial a lire cote webapp (JSON simple, pas de lecteur
        # parquet necessaire).
        explain = (
            final[["target_date", "analog_level", "analog_dates"]]
            .drop_duplicates(subset=["target_date"])
            .to_dict(orient="records")
        )
        # n_neighbors variait par (station, hour) — pas une mesure fiable
        # au niveau du jour. Le vrai compte de jours analogues utilises
        # (constant pour un target_date donne) est simplement la longueur
        # de analog_dates.
        for row in explain:
            row["n_analog_days"] = len(row["analog_dates"].split(",")) if row["analog_dates"] else 0
        explain_name = f"forecast_explain_{city_id}.json"
        explain_path = tmp_dir / explain_name
        explain_path.write_text(json.dumps(explain, ensure_ascii=False))
        storage.upload_asset(RELEASE_CITIES_AGGREGATES, explain_path, explain_name)
        print(f"[forecast_cities] {city_id}: {explain_name} uploadé ({len(explain)} jour(s))")


def run(city_ids: list[str] | None = None) -> None:
    if city_ids is None:
        city_ids = [c for c in list_cities() if c != "paris"]

    storage.ensure_release(RELEASE_CITIES_AGGREGATES, "Cities Forecasts — Bordeaux, Lyon, Toulouse, Lille, Rennes, Strasbourg…")

    # Calendrier historique national, partagé avec Paris (mêmes jours
    # analogues candidats — voir limite connue dans la docstring du module).
    print("[forecast_cities] loading national calendar...")
    calendar_existing = _download_parquet(storage.RELEASE_AGGREGATES, CALENDAR_ASSET)

    for city_id in city_ids:
        # Isolation par ville (même principe que consolidate_cities.py) :
        # un échec sur une ville ne doit jamais bloquer les suivantes.
        try:
            run_city(city_id, calendar_existing)
        except Exception as e:
            print(f"[forecast_cities] {city_id}: ECHEC ({e}) — villes suivantes non affectées")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", default=None,
                        help="IDs des villes à prévoir (ex: bordeaux lyon)")
    args = parser.parse_args()
    run(args.cities)
