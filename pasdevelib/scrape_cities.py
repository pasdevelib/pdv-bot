"""scrape_cities.py — Scrape toutes les villes configurées et stocke les données.

Usage : python -m pasdevelib.scrape_cities [--cities paris bordeaux lyon]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import tempfile
from pathlib import Path

import pandas as pd

from pasdevelib import storage
from pasdevelib.cities import list_cities
from pasdevelib.fetch_city import fetch_all_cities

# Release dédiée aux données multi-villes
RELEASE_CITIES = "cities-live"


def run(city_ids: list[str] | None = None) -> None:
    if city_ids is None:
        city_ids = list_cities()  # toutes les villes par défaut

    now = dt.datetime.now(dt.timezone.utc)
    print(f"[scrape_cities] {now.isoformat()} — villes: {city_ids}")

    # Scraper toutes les villes
    df = fetch_all_cities(city_ids)

    if df.empty:
        print("[scrape_cities] Aucune donnée récupérée")
        return

    print(f"[scrape_cities] {len(df)} stations au total")
    by_city = df.groupby("city_id").size()
    for city_id, n in by_city.items():
        print(f"  {city_id}: {n} stations")

    # Créer la release si elle n'existe pas
    storage.ensure_release(RELEASE_CITIES, 'Cities Live Data — Bordeaux, Lyon, Toulouse')

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # 1. Snapshot snapshot_cities.parquet (toutes villes, dernier scrape)
        snap_path = tmp_dir / "snapshot_cities.parquet"
        df.to_parquet(snap_path, compression="snappy", index=False)
        storage.upload_asset(RELEASE_CITIES, snap_path, "snapshot_cities.parquet")
        print("[scrape_cities] snapshot_cities.parquet uploadé")

        # 2. Un fichier par ville pour le current_day multi-villes
        today = now.strftime("%Y-%m-%d")
        for city_id in city_ids:
            city_df = df[df["city_id"] == city_id]
            if city_df.empty:
                continue
            try:
                city_path = tmp_dir / f"current_day_{city_id}.parquet"
                city_df.to_parquet(city_path, compression="snappy", index=False)
                storage.upload_asset(RELEASE_CITIES, city_path, f"current_day_{city_id}.parquet")
                print(f"[scrape_cities] current_day_{city_id}.parquet uploadé ({len(city_df)} stations)")
            except Exception as e:
                # Isolation par ville : un echec d'upload
                # pour une ville ne doit pas empecher les autres.
                print(f"[scrape_cities] {city_id}: ECHEC upload ({e})")

        # 3. stations_cities.json — liste de toutes les stations multi-villes
        #
        # BUG CORRIGE ICI : ce fichier etait entierement RECONSTRUIT a
        # partir de `df`, qui ne contient QUE les villes traitees par
        # CETTE execution (chaque ville a son propre workflow, horaires
        # differents) — a chaque run, toutes les autres villes disparaissaient
        # du fichier, remplacees par la derniere ville dont le scrape avait
        # tourne. Consequence concrete : stations_cities.json ne contenait
        # plus qu'UNE seule ville a la fois, cassant find_spatial_neighbors
        # (KeyError) et la conversion fill_rate -> capacite pour toutes les
        # autres villes secondaires. Fusionne desormais avec l'existant :
        # ne remplace que les stations des villes traitees ici, conserve
        # celles des autres villes telles quelles.
        existing_path = tmp_dir / "stations_cities_existing.json"
        existing_stations: list[dict] = []
        if storage.download_asset(RELEASE_CITIES, "stations_cities.json", existing_path):
            try:
                existing_stations = json.loads(existing_path.read_text())
            except Exception as e:
                print(f"[scrape_cities] stations_cities.json existant illisible ({e}), reconstruction depuis zero")

        cities_in_this_run = set(df["city_id"].unique())
        stations_data = [s for s in existing_stations if s.get("city_id") not in cities_in_this_run]
        for _, row in df.iterrows():
            stations_data.append({
                "station_id": row["station_id"],
                "city_id": row["city_id"],
                "name": row["name"],
                "lat": row["lat"],
                "lon": row["lon"],
                "capacity": int(row.get("capacity", 0)),
            })
        stations_path = tmp_dir / "stations_cities.json"
        stations_path.write_text(json.dumps(stations_data, ensure_ascii=False))
        storage.upload_asset(RELEASE_CITIES, stations_path, "stations_cities.json")
        print(f"[scrape_cities] stations_cities.json uploadé ({len(stations_data)} stations, "
              f"{len(cities_in_this_run)} ville(s) mise(s) à jour dans cette exécution)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", default=None,
                        help="IDs des villes à scraper (ex: paris bordeaux lyon)")
    args = parser.parse_args()
    run(args.cities)
