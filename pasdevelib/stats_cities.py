"""stats_cities.py — Analyses de donnees par ville pour blog.pasdevelib.app/donnees :

1. Classements de stations (vides/pleines/fiables), 3 fenetres glissantes
   (jour/semaine/mois) — stats_<ville>_<periode>.json
2. Classement des quartiers/zones les plus problematiques (si geocodage
   disponible, cf. geocode_stations.py / geocode_cities.py) — inclus dans
   le fichier ci-dessus (cle "neighborhoods")
3. Evolution quotidienne du remplissage moyen du reseau, ~90 derniers
   jours — evolution_<ville>.json (pour un graphe de tendance)
4. Motif hebdomadaire (jour de semaine x heure), tout l'historique
   disponible — patterns_<ville>.json (pour une heatmap)
5. Records (meilleur/pire jour observe) — records_<ville>.json

Tourne pour TOUTES les villes (Paris inclus), avec la meme isolation par
ville que consolidate_cities.py / forecast_cities.py : un echec sur une
ville ne bloque jamais les autres.

Schema de sortie principal (stats_<ville>_<periode>.json, release
"stats-cities") :
{
  "city_id", "period" ("day"|"week"|"month"),
  "generated_at", "window_start", "window_end",
  "city_avg_fill_rate": float,
  "top_empty":      [{station_id, name, pct_empty, n_obs}, ...] (20 max)
  "top_full":       [{station_id, name, pct_full,  n_obs}, ...] (20 max)
  "most_reliable":  [{station_id, name, pct_healthy, n_obs}, ...] (20 max)
  "worst":          [{station_id, name, pct_extreme, n_obs}, ...] (20 max)
  "neighborhoods":  [{zone, pct_extreme, n_stations, n_obs}, ...] (10 max)
                    absent si aucune station de la ville n'a de zone geocodee.
}

"pct_healthy" = part du temps ou la station est ni quasi-vide ni quasi-
pleine (fill_rate entre 0.2 et 0.8) — sert de proxy de fiabilite.
Seuil MIN_OBS : une station avec trop peu d'observations sur la fenetre
est exclue des classements (evite qu'une station en panne 29 jours sur 30
mais "parfaite" le dernier jour ne remonte en tete par accident).
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import requests

from pasdevelib import storage
from pasdevelib import weather
from pasdevelib.cities import list_cities, CITIES

RELEASE_STATS = "stats-cities"
MIN_OBS_RATIO = 0.3  # une station doit avoir des donnees au moins 30% du temps de la fenetre pour etre classee
TOP_N = 20
TOP_N_ZONES = 10
EVOLUTION_DAYS = 90
MIN_ZONE_STATIONS = 3  # une zone avec moins de stations n'est pas assez representative pour etre classee

PERIODS = {
    "day": 1,
    "week": 7,
    "month": 30,
}


# BUG CORRIGE ICI (2026-09-28) : les villes hors Paris tournent désormais
# dans un dépôt séparé (pasdevelib/pdvr-bot). Ce workflow (stats-cities.yml)
# lui, reste dans pdv-bot car il a besoin de lire les DEUX cotés (donnees
# Paris ici + donnees villes dans pdvr-bot) — voir README de pdvr-bot.
# CITIES_REPO cible donc explicitement pdvr-bot pour cities-live/
# cities-history, jamais storage.REPO (qui vaut pdv-bot ici). Lecture
# anonyme sur un repo public, donc aucun token necessaire.
CITIES_REPO = "pasdevelib/pdvr-bot"


def _download_parquet(release: str, asset: str, repo: str | None = None) -> pd.DataFrame | None:
    url = f"https://github.com/{repo or storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def _download_json(release: str, asset: str, repo: str | None = None) -> list | None:
    url = f"https://github.com/{repo or storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def _load_history_and_names(city_id: str) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], dict[str, float]] | None:
    """Retourne (hourly_history, {station_id: name}, {station_id: zone}, {station_id: capacity}).

    Le dict de zones peut etre vide (geocodage pas encore passe, ou champ
    absent) — tout le code appelant doit tolerer ce cas sans planter.
    capacity sert a convertir des variations de fill_rate (ratio) en
    nombre de velos estime (cf. compute_traffic) — defensif sur le nom du
    champ id (station_id vs id selon la version du fichier source).
    """
    if city_id == "paris":
        hourly = _download_parquet(storage.RELEASE_AGGREGATES, "hourly_history.parquet")
        stations_raw = _download_json(storage.RELEASE_LIVE, "stations.json")
        sid = lambda s: str(s.get("station_id") or s.get("id") or "")
        names = {sid(s): s.get("name", sid(s)) for s in (stations_raw or [])}
        zones = {sid(s): s["zone"] for s in (stations_raw or []) if s.get("zone")}
        capacities = {sid(s): float(s.get("capacity") or 0) for s in (stations_raw or [])}
    else:
        hourly = _download_parquet("cities-history", f"hourly_history_{city_id}.parquet", repo=CITIES_REPO)
        stations_raw = _download_json("cities-live", "stations_cities.json", repo=CITIES_REPO)
        city_stations = [s for s in (stations_raw or []) if s.get("city_id") == city_id]
        sid = lambda s: str(s.get("station_id") or s.get("id") or "")
        names = {sid(s): s.get("name", sid(s)) for s in city_stations}
        zones = {sid(s): s["zone"] for s in city_stations if s.get("zone")}
        capacities = {sid(s): float(s.get("capacity") or 0) for s in city_stations}
    if hourly is None or hourly.empty:
        return None
    return hourly, names, zones, capacities


def _rank(df: pd.DataFrame, sort_col: str, value_col: str, names: dict[str, str]) -> list[dict]:
    top = df.sort_values(sort_col, ascending=False).head(TOP_N)
    return [
        {
            "station_id": row.station_id,
            "name": names.get(row.station_id, row.station_id),
            value_col: round(float(getattr(row, sort_col)), 3),
            "n_obs": int(row.n_obs),
        }
        for row in top.itertuples()
    ]


def _rank_zones(window: pd.DataFrame, zones: dict[str, str]) -> list[dict] | None:
    if not zones:
        return None
    df = window.copy()
    df["zone"] = df["station_id"].astype(str).map(zones)
    df = df[df["zone"].notna()]
    if df.empty:
        return None

    grouped = df.groupby("zone").agg(
        pct_empty=("fill_rate", lambda s: float((s <= 0.05).mean())),
        pct_full=("fill_rate", lambda s: float((s >= 0.95).mean())),
        n_obs=("fill_rate", "count"),
        n_stations=("station_id", "nunique"),
    ).reset_index()
    grouped["pct_extreme"] = grouped["pct_empty"] + grouped["pct_full"]
    grouped = grouped[grouped["n_stations"] >= MIN_ZONE_STATIONS]
    if grouped.empty:
        return None

    top = grouped.sort_values("pct_extreme", ascending=False).head(TOP_N_ZONES)
    return [
        {
            "zone": row.zone,
            "pct_extreme": round(float(row.pct_extreme), 3),
            "n_stations": int(row.n_stations),
            "n_obs": int(row.n_obs),
        }
        for row in top.itertuples()
    ]


def compute_period(hourly: pd.DataFrame, names: dict[str, str], zones: dict[str, str], days: int) -> dict:
    hourly = hourly.copy()
    hourly["date"] = pd.to_datetime(hourly["date"])
    window_end = hourly["date"].max()
    window_start = window_end - pd.Timedelta(days=days - 1)
    window = hourly[hourly["date"] >= window_start]

    expected_obs = days * 24  # au mieux, une observation par heure
    min_obs = max(3, int(expected_obs * MIN_OBS_RATIO))

    grouped = window.groupby("station_id").agg(
        mean_fill=("fill_rate", "mean"),
        pct_empty=("fill_rate", lambda s: float((s <= 0.05).mean())),
        pct_full=("fill_rate", lambda s: float((s >= 0.95).mean())),
        pct_healthy=("fill_rate", lambda s: float(((s >= 0.2) & (s <= 0.8)).mean())),
        n_obs=("fill_rate", "count"),
    ).reset_index()
    grouped["station_id"] = grouped["station_id"].astype(str)
    grouped["pct_extreme"] = grouped["pct_empty"] + grouped["pct_full"]

    eligible = grouped[grouped["n_obs"] >= min_obs]

    result = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "city_avg_fill_rate": round(float(window["fill_rate"].mean()), 3) if not window.empty else None,
        # Moyenne de pct_healthy sur TOUTES les stations eligibles (pas
        # seulement le top 20) — sert de score comparable entre villes
        # ("quel reseau est le plus fiable ?"), cf. page Donnees.
        "city_avg_pct_healthy": round(float(eligible["pct_healthy"].mean()), 3) if not eligible.empty else None,
        "top_empty": _rank(eligible, "pct_empty", "pct_empty", names),
        "top_full": _rank(eligible, "pct_full", "pct_full", names),
        "most_reliable": _rank(eligible, "pct_healthy", "pct_healthy", names),
        "worst": _rank(eligible, "pct_extreme", "pct_extreme", names),
    }
    neighborhoods = _rank_zones(window, zones)
    if neighborhoods is not None:
        result["neighborhoods"] = neighborhoods
    return result


def compute_evolution(hourly: pd.DataFrame) -> dict:
    """Serie quotidienne du remplissage moyen du reseau, ~90 derniers jours
    (pour un graphe de tendance sur la page Donnees)."""
    df = hourly.copy()
    df["date"] = pd.to_datetime(df["date"])
    end = df["date"].max()
    start_cutoff = end - pd.Timedelta(days=EVOLUTION_DAYS - 1)
    window = df[df["date"] >= start_cutoff]

    daily = window.groupby("date")["fill_rate"].mean().reset_index()
    daily = daily.sort_values("date")
    # BUG CORRIGE ICI : window_start etait calcule comme "EVOLUTION_DAYS
    # jours avant la derniere donnee", QUE CES JOURS AIENT REELLEMENT DES
    # DONNEES OU NON — trompeur pour une ville dont l'historique vient
    # tout juste de commencer (ex. 1 seul jour reel disponible), qui
    # affichait quand meme une fenetre "89 jours" alors que `series` ne
    # contenait qu'un seul point. window_start reflete desormais la VRAIE
    # date la plus ancienne effectivement presente dans les donnees.
    real_start = daily["date"].min() if not daily.empty else end
    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "window_start": real_start.date().isoformat(),
        "window_end": end.date().isoformat(),
        "series": [
            {"date": row.date.date().isoformat(), "avg_fill_rate": round(float(row.fill_rate), 3)}
            for row in daily.itertuples()
        ],
    }


def compute_patterns(hourly: pd.DataFrame) -> dict:
    """Grille jour-de-semaine x heure (remplissage moyen), tout l'historique
    disponible — pour une heatmap "quel jour/heure est le pire" sur la page
    Donnees. weekday : 0=lundi ... 6=dimanche (convention pandas)."""
    df = hourly.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.dayofweek

    grid = df.groupby(["weekday", "hour"])["fill_rate"].mean().reset_index()
    cells = [
        {"weekday": int(row.weekday), "hour": int(row.hour), "avg_fill_rate": round(float(row.fill_rate), 3)}
        for row in grid.itertuples()
    ]
    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "n_days_covered": int(df["date"].nunique()),
        "cells": cells,
    }


def compute_records(hourly: pd.DataFrame) -> dict:
    """Meilleur et pire jour observes (remplissage moyen du reseau sur
    l'historique disponible) — pour un encart "records" sur la page
    Donnees."""
    df = hourly.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["fill_rate"].agg(["mean", "count"]).reset_index()
    # Un jour avec trop peu d'observations (scrape interrompu) fausserait
    # le record — meme logique de seuil que compute_period.
    daily = daily[daily["count"] >= 24 * 0.3]
    if daily.empty:
        return {"generated_at": dt.datetime.utcnow().isoformat() + "Z", "best_day": None, "worst_day": None}

    best = daily.loc[daily["mean"].idxmax()]
    worst = daily.loc[daily["mean"].idxmin()]
    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "best_day": {"date": best["date"].date().isoformat(), "avg_fill_rate": round(float(best["mean"]), 3)},
        "worst_day": {"date": worst["date"].date().isoformat(), "avg_fill_rate": round(float(worst["mean"]), 3)},
    }


def compute_traffic(hourly: pd.DataFrame, capacities: dict[str, str]) -> dict:
    """Estimation du "trafic" velo (mouvements de velos) a partir des
    variations de fill_rate d'une heure a l'autre, par station.

    IMPORTANT (limite methodologique, a rappeler cote UI) : les flux GBFS
    ne donnent jamais de vrais trajets individuels (pas de "check-out" /
    "check-in" trace), seulement des snapshots d'occupation par station.
    Ce calcul est donc une ESTIMATION par variation nette d'occupation
    (fill_rate(h) - fill_rate(h-1)) x capacite de la station, pas un
    comptage exact. Les operations de rebalancement (camions) et les
    velos deposes hors station ne sont pas isolables de cette methode.

    "Depart" = diminution du nombre de velos disponibles a une station
    (quelqu'un a pris un velo, par hypothese). Choisi plutot que les
    arrivees pour la coherence du proxy "vélos qui roulent" — les deux
    convergent a peu pres sur une pleine journee (departs ~ arrivees).
    """
    df = hourly.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["capacity"] = df["station_id"].astype(str).map(capacities).fillna(0)
    df = df[df["capacity"] > 0]
    df = df.sort_values(["station_id", "date", "hour"])

    # Delta entre heures consecutives POUR LA MEME STATION uniquement
    # (groupby.diff() respecte deja les groupes, mais on verifie aussi
    # que l'heure precedente est bien h-1 ou le jour precedent 23h, pour
    # ne pas compter un trou de plusieurs heures comme un seul "depart").
    df["prev_station"] = df["station_id"].shift(1)
    df["prev_date"] = df["date"].shift(1)
    df["prev_hour"] = df["hour"].shift(1)
    df["is_consecutive"] = (df["station_id"] == df["prev_station"]) & (
        ((df["hour"] - df["prev_hour"] == 1) & (df["date"] == df["prev_date"])) |
        ((df["prev_hour"] == 23) & (df["hour"] == 0) & ((df["date"] - df["prev_date"]).dt.days == 1))
    )
    df["delta_bikes"] = df.groupby("station_id")["fill_rate"].diff() * df["capacity"]
    df.loc[~df["is_consecutive"], "delta_bikes"] = None
    # Departs estimes = diminutions (delta negatif), en valeur absolue.
    # Plafond a la capacite de la station : au-dela, la variation ne
    # peut plus correspondre a de l'usage organique (1 velo = 1 personne),
    # c'est forcement une operation de rebalancement (camion) ou un artefact
    # de donnees — cf. le method_note qui documente deja cette limite.
    # Sans ce plafond, quelques operations de camion la nuit ecrasaient tout
    # signal d'usage reel aux heures de pointe (constate directement : la
    # tranche "reste de la journee" dominait a 66% des departs estimes).
    df["departures"] = (-df["delta_bikes"]).clip(lower=0)
    # BUG CORRIGE ICI : le plafond a la capacite totale de la station
    # etait bien trop permissif pour filtrer les operations de camion de
    # rebalancement (constate directement : ~260 000 "departs" estimes par
    # jour a Paris, contre 93 000-160 000 mouvements officiels Velib' Metropole).
    # Un camion peut deplacer 5-15 velos d'un coup sans jamais depasser la
    # capacite totale d'une station de 30-40 places, donc l'ancien plafond
    # ne filtrait que les cas les plus extremes. Remplace par un plafond
    # statistique auto-calibre (percentile 90 de la distribution observee
    # des baisses horaires par station) : les operations ponctuelles et
    # massives de rebalancement sont ecretees, l'usage organique typique ne
    # l'est pas.
    # REECRIT ICI : le plafond precedent utilisait UN SEUL percentile 90
    # calcule sur TOUTES les stations melangees — une grande station tres
    # frequentee et une petite station peu active n'ont pas la meme echelle
    # naturelle d'activite, donc un seuil global ecretait trop les petites
    # stations et pas assez les grandes. Plafond desormais calcule PAR
    # STATION (chaque station comparee a son propre historique), ce qui
    # respecte l'echelle propre de chacune plutot que d'imposer une norme
    # unique a tout le reseau.
    if len(df) > 0:
        station_p90 = df.groupby("station_id")["departures"].transform(
            lambda x: x.quantile(0.90) if (x > 0).any() else 0
        )
        df["departures"] = np.minimum(df["departures"], station_p90)


    valid = df.dropna(subset=["departures"])

    bikes_per_hour = valid.groupby("hour")["departures"].sum() / max(valid["date"].nunique(), 1)
    bikes_per_hour_list = [
        {"hour": h, "avg_departures": round(float(bikes_per_hour.get(h, 0.0)), 1)}
        for h in range(24)
    ]

    trips_per_day = valid.groupby("date")["departures"].sum().reset_index()
    trips_per_day = trips_per_day.sort_values("date").tail(90)
    trips_per_day_list = [
        {"date": row.date.date().isoformat(), "trips": round(float(row.departures))}
        for row in trips_per_day.itertuples()
    ]

    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "method_note": "Estimation de l'activité par variation d'occupation, station par station — ce n'est "
                       "pas un comptage exact de trajets individuels. Seules les BAISSES de vélos disponibles sont "
                       "comptées (les hausses sont ignorées), et les baisses extrêmes (au-delà du percentile 90 "
                       "propre à chaque station) sont écrêtées pour limiter l'effet des opérations de "
                       "rééquilibrage (camions). Cette estimation reste approximative — à comparer, à titre "
                       "indicatif, aux 93 000-160 000 mouvements quotidiens officiels publiés par Vélib' "
                       "Métropole selon la saison.",
        "bikes_per_hour": bikes_per_hour_list,
        "trips_per_day": trips_per_day_list,
    }


def compute_stuck_stations(hourly: pd.DataFrame, names: dict[str, str]) -> dict:
    """Plus longues sequences CONSECUTIVES a vide (fill_rate <= 0.05) et a
    pleine (fill_rate >= 0.95) par station, sur tout l'historique
    disponible. Repond au besoin "combien de temps une station reste
    bloquee", different d'un simple pourcentage cumule (une station a
    20% de temps vide peut etre soit 20% tous les jours un peu, soit
    bloquee 5 jours d'affilee une fois — les deux racontent une histoire
    tres differente pour un journaliste)."""
    df = hourly.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ts"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    df = df.sort_values(["station_id", "ts"])

    def longest_streak(sub: pd.DataFrame, condition_col: str) -> tuple[float, str, str]:
        """Retourne (duree_heures, debut_iso, fin_iso) de la plus longue
        sequence consecutive (heure a heure, sans trou) ou condition_col
        est vrai."""
        is_true = sub[condition_col].to_numpy()
        ts = sub["ts"].to_numpy()
        best_len, best_start, best_end = 0, None, None
        run_start_idx = None
        for i in range(len(sub)):
            consecutive_hour = i == 0 or (ts[i] - ts[i - 1]) == np.timedelta64(1, "h")
            if is_true[i] and consecutive_hour:
                if run_start_idx is None:
                    run_start_idx = i
            else:
                if run_start_idx is not None:
                    run_len = i - run_start_idx
                    if run_len > best_len:
                        best_len, best_start, best_end = run_len, run_start_idx, i - 1
                    run_start_idx = None
                if is_true[i]:
                    run_start_idx = i
        if run_start_idx is not None:
            run_len = len(sub) - run_start_idx
            if run_len > best_len:
                best_len, best_start, best_end = run_len, run_start_idx, len(sub) - 1
        if best_start is None:
            return 0.0, None, None
        return float(best_len), pd.Timestamp(ts[best_start]).isoformat(), pd.Timestamp(ts[best_end]).isoformat()

    df["is_empty"] = df["fill_rate"] <= 0.05
    df["is_full"] = df["fill_rate"] >= 0.95

    empty_results, full_results = [], []
    for sid, sub in df.groupby("station_id"):
        sub = sub.reset_index(drop=True)
        e_len, e_start, e_end = longest_streak(sub, "is_empty")
        f_len, f_start, f_end = longest_streak(sub, "is_full")
        if e_len >= 6:  # au moins 6h consecutives pour etre significatif
            empty_results.append({"station_id": str(sid), "name": names.get(str(sid), str(sid)), "hours": e_len, "start": e_start, "end": e_end})
        if f_len >= 6:
            full_results.append({"station_id": str(sid), "name": names.get(str(sid), str(sid)), "hours": f_len, "start": f_start, "end": f_end})

    empty_results.sort(key=lambda r: r["hours"], reverse=True)
    full_results.sort(key=lambda r: r["hours"], reverse=True)

    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "method_note": "Seuils « vide » (≤5% de remplissage) et « pleine » (≥95%) : des conventions pratiques "
                       "pour repérer les cas extrêmes, pas des vérités absolues. Minimum 6h consécutives pour "
                       "qu'une séquence soit retenue.",
        "longest_empty": empty_results[:10],
        "longest_full": full_results[:10],
    }


def run_city(city_id: str) -> None:
    print(f"[stats_cities] === {city_id} ===")
    loaded = _load_history_and_names(city_id)
    if loaded is None:
        print(f"[stats_cities] {city_id}: pas d'historique disponible, skip")
        return
    hourly, names, zones, capacities = loaded

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for period, days in PERIODS.items():
            result = compute_period(hourly, names, zones, days)
            result["city_id"] = city_id
            result["period"] = period
            out_name = f"stats_{city_id}_{period}.json"
            out_path = tmp_dir / out_name
            out_path.write_text(json.dumps(result, ensure_ascii=False))
            storage.upload_asset(RELEASE_STATS, out_path, out_name)
            zone_note = f", {len(result['neighborhoods'])} quartiers" if "neighborhoods" in result else ""
            print(f"[stats_cities] {city_id}: {out_name} uploade "
                  f"({len(result['top_empty'])} stations classees{zone_note}, fenetre {result['window_start']}..{result['window_end']})")

        for name, compute_fn in [
            ("evolution", compute_evolution),
            ("patterns", compute_patterns),
            ("records", compute_records),
        ]:
            result = compute_fn(hourly)
            result["city_id"] = city_id
            out_name = f"{name}_{city_id}.json"
            out_path = tmp_dir / out_name
            out_path.write_text(json.dumps(result, ensure_ascii=False))
            storage.upload_asset(RELEASE_STATS, out_path, out_name)
            print(f"[stats_cities] {city_id}: {out_name} uploade")

        # Trafic estime (velos en mouvement) et stations bloquees vides/
        # pleines longtemps — demande explicite, pas dans la premiere
        # version de cette page.
        traffic = compute_traffic(hourly, capacities)
        traffic["city_id"] = city_id
        out_path = tmp_dir / f"traffic_{city_id}.json"
        out_path.write_text(json.dumps(traffic, ensure_ascii=False))
        storage.upload_asset(RELEASE_STATS, out_path, f"traffic_{city_id}.json")
        print(f"[stats_cities] {city_id}: traffic_{city_id}.json uploade")

        stuck = compute_stuck_stations(hourly, names)
        stuck["city_id"] = city_id
        out_path = tmp_dir / f"stuck_{city_id}.json"
        out_path.write_text(json.dumps(stuck, ensure_ascii=False))
        storage.upload_asset(RELEASE_STATS, out_path, f"stuck_{city_id}.json")
        print(f"[stats_cities] {city_id}: stuck_{city_id}.json uploade "
              f"({len(stuck['longest_empty'])} stations vides longtemps, {len(stuck['longest_full'])} pleines longtemps)")

        # Impact meteo (sensibilite a la pluie) — demande explicite suite
        # a un tour d'horizon academique/journalistique sur les VLS.
        # Centre approximatif de la ville = centre de son bbox (deja
        # defini dans cities.py, pas de nouvelle config necessaire).
        city_cfg = CITIES.get(city_id)
        if city_cfg and city_cfg.bbox:
            lat_min, lon_min, lat_max, lon_max = city_cfg.bbox
            center_lat, center_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
            weather_stats = compute_weather_impact(hourly, capacities, center_lat, center_lon)
            weather_stats["city_id"] = city_id
            out_path = tmp_dir / f"weather_{city_id}.json"
            out_path.write_text(json.dumps(weather_stats, ensure_ascii=False))
            storage.upload_asset(RELEASE_STATS, out_path, f"weather_{city_id}.json")
            print(f"[stats_cities] {city_id}: weather_{city_id}.json uploade (ready={weather_stats.get('ready')})")

        # Typologie des stations (pendulaire / loisir-weekend / mixte) —
        # demande explicite, cf. compute_station_typology pour la limite
        # methodologique importante (pas de vraie morphologie urbaine).
        typology = compute_station_typology(hourly, names, zones)
        typology["city_id"] = city_id
        out_path = tmp_dir / f"typology_{city_id}.json"
        out_path.write_text(json.dumps(typology, ensure_ascii=False))
        storage.upload_asset(RELEASE_STATS, out_path, f"typology_{city_id}.json")
        print(f"[stats_cities] {city_id}: typology_{city_id}.json uploade "
              f"({typology['counts']}, {len(typology.get('by_zone') or [])} quartiers)")

        # Signature des trajets du matin (exportateur/importateur par
        # quartier) — cf. compute_morning_signature pour le detail et la
        # limite methodologique.
        morning_sig = compute_morning_signature(hourly, zones)
        morning_sig["city_id"] = city_id
        out_path = tmp_dir / f"morning_signature_{city_id}.json"
        out_path.write_text(json.dumps(morning_sig, ensure_ascii=False))
        storage.upload_asset(RELEASE_STATS, out_path, f"morning_signature_{city_id}.json")
        print(f"[stats_cities] {city_id}: morning_signature_{city_id}.json uploade "
              f"({len(morning_sig.get('zones') or [])} quartiers, ready={morning_sig.get('ready')})")


def compute_weather_impact(hourly: pd.DataFrame, capacities: dict[str, str], lat: float, lon: float) -> dict:
    """Compare l'activite du reseau (trajets estimes/jour, meme methode que
    compute_traffic) les jours de pluie vs les jours secs — demande
    explicite ("sensibilite a la pluie") tiree d'un tour d'horizon de ce
    qui se fait en recherche academique sur les VLS. Meteo historique via
    Open-Meteo (deja utilise pour les predictions, pas de nouvelle
    dependance).
    """
    df = hourly.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["capacity"] = df["station_id"].astype(str).map(capacities).fillna(0)
    df = df[df["capacity"] > 0].sort_values(["station_id", "date", "hour"])

    df["prev_station"] = df["station_id"].shift(1)
    df["prev_date"] = df["date"].shift(1)
    df["prev_hour"] = df["hour"].shift(1)
    df["is_consecutive"] = (df["station_id"] == df["prev_station"]) & (
        ((df["hour"] - df["prev_hour"] == 1) & (df["date"] == df["prev_date"])) |
        ((df["prev_hour"] == 23) & (df["hour"] == 0) & ((df["date"] - df["prev_date"]).dt.days == 1))
    )
    df["delta_bikes"] = df.groupby("station_id")["fill_rate"].diff() * df["capacity"]
    df.loc[~df["is_consecutive"], "delta_bikes"] = None
    # Plafond a la capacite de la station : au-dela, la variation ne
    # peut plus correspondre a de l'usage organique (1 velo = 1 personne),
    # c'est forcement une operation de rebalancement (camion) ou un artefact
    # de donnees — cf. le method_note qui documente deja cette limite.
    # Sans ce plafond, quelques operations de camion la nuit ecrasaient tout
    # signal d'usage reel aux heures de pointe (constate directement : la
    # tranche "reste de la journee" dominait a 66% des departs estimes).
    df["departures"] = (-df["delta_bikes"]).clip(lower=0)
    # BUG CORRIGE ICI : le plafond a la capacite totale de la station
    # etait bien trop permissif pour filtrer les operations de camion de
    # rebalancement (constate directement : ~260 000 "departs" estimes par
    # jour a Paris, contre 93 000-160 000 mouvements officiels Velib' Metropole).
    # Un camion peut deplacer 5-15 velos d'un coup sans jamais depasser la
    # capacite totale d'une station de 30-40 places, donc l'ancien plafond
    # ne filtrait que les cas les plus extremes. Remplace par un plafond
    # statistique auto-calibre (percentile 90 de la distribution observee
    # des baisses horaires par station) : les operations ponctuelles et
    # massives de rebalancement sont ecretees, l'usage organique typique ne
    # l'est pas.
    # REECRIT ICI : le plafond precedent utilisait UN SEUL percentile 90
    # calcule sur TOUTES les stations melangees — une grande station tres
    # frequentee et une petite station peu active n'ont pas la meme echelle
    # naturelle d'activite, donc un seuil global ecretait trop les petites
    # stations et pas assez les grandes. Plafond desormais calcule PAR
    # STATION (chaque station comparee a son propre historique), ce qui
    # respecte l'echelle propre de chacune plutot que d'imposer une norme
    # unique a tout le reseau.
    if len(df) > 0:
        station_p90 = df.groupby("station_id")["departures"].transform(
            lambda x: x.quantile(0.90) if (x > 0).any() else 0
        )
        df["departures"] = np.minimum(df["departures"], station_p90)

    valid = df.dropna(subset=["departures"])

    daily_trips = valid.groupby("date")["departures"].sum().reset_index()
    base = {"generated_at": dt.datetime.utcnow().isoformat() + "Z", "ready": False}
    if daily_trips.empty:
        return base

    start = daily_trips["date"].min().date()
    end = daily_trips["date"].max().date()

    try:
        weather_df = weather.fetch_archive(start, end, lat=lat, lon=lon)
    except Exception as e:
        print(f"[stats_cities] meteo indisponible ({e}), stats meteo ignorees")
        return base
    if weather_df.empty:
        return base

    weather_df["date_local"] = weather_df["ts"].dt.tz_convert("Europe/Paris").dt.date
    daily_precip = weather_df.groupby("date_local")["precipitation"].sum().reset_index()
    daily_precip["date"] = pd.to_datetime(daily_precip["date_local"])

    merged = daily_trips.merge(daily_precip[["date", "precipitation"]], on="date", how="inner")
    if merged.empty:
        return base
    # Seuil pluie : plus de 1mm cumule sur la journee (seuil usuel en
    # climatologie pour distinguer "jour de pluie" d'un simple crachin).
    merged["is_rainy"] = merged["precipitation"] > 1.0

    rainy = merged[merged["is_rainy"]]
    dry = merged[~merged["is_rainy"]]
    avg_rainy = float(rainy["departures"].mean()) if len(rainy) > 0 else None
    avg_dry = float(dry["departures"].mean()) if len(dry) > 0 else None
    ratio = round(avg_rainy / avg_dry, 2) if avg_rainy and avg_dry and avg_dry > 0 else None

    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "ready": True,
        "n_rainy_days": int(len(rainy)),
        "n_dry_days": int(len(dry)),
        "avg_trips_rainy_day": round(avg_rainy) if avg_rainy is not None else None,
        "avg_trips_dry_day": round(avg_dry) if avg_dry is not None else None,
        "rain_sensitivity_ratio": ratio,
        "method_note": "Estimation de l'activité par variation d'occupation (même méthode que la page trafic), "
                       "croisée avec la pluviométrie quotidienne (Open-Meteo). Seuil « jour de pluie » : plus de "
                       "1mm cumulé — une convention pratique parmi d'autres possibles, pas une frontière absolue.",
    }


def compute_station_typology(hourly: pd.DataFrame, names: dict[str, str], zones: dict[str, str]) -> dict:
    """Classe chaque station selon son PROFIL D'USAGE TEMPOREL (pendulaire /
    loisir-weekend / mixte) — demande explicite suite a un tour d'horizon
    academique/journalistique sur les VLS ("typologie des stations").

    LIMITE METHODOLOGIQUE IMPORTANTE (a rappeler cote UI) : ceci n'est PAS
    une classification par morphologie urbaine reelle (densite, POI,
    proximite transports) — ces donnees ne sont pas disponibles dans ce
    projet. C'est une classification purement COMPORTEMENTALE, basee sur
    la volatilite du taux de remplissage a differents moments : une
    station "pendulaire" bouge beaucoup aux heures de pointe en semaine et
    peu le week-end ; une station "loisir" bouge autant ou plus le
    week-end qu'en semaine.
    """
    df = hourly.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.dayofweek  # 0=lundi ... 6=dimanche
    df["is_weekend"] = df["weekday"] >= 5

    commute_mask = (~df["is_weekend"]) & (df["hour"].between(7, 10) | df["hour"].between(17, 20))
    midday_mask = (~df["is_weekend"]) & (df["hour"].between(11, 16))
    weekend_mask = df["is_weekend"] & df["hour"].between(9, 20)

    def _std_by_station(mask: pd.Series) -> pd.Series:
        return df[mask].groupby("station_id")["fill_rate"].std()

    commute_std = _std_by_station(commute_mask)
    midday_std = _std_by_station(midday_mask)
    weekend_std = _std_by_station(weekend_mask)
    n_obs = df.groupby("station_id")["fill_rate"].count()

    EPS = 0.01
    # Premiere passe : calcule les scores de toutes les stations avant de
    # classer, pour pouvoir utiliser des SEUILS RELATIFS (terciles) plutot
    # que des seuils absolus arbitraires. BUG CORRIGE ICI : les seuils
    # absolus choisis initialement (commute_score > 1.3, weekend_score
    # < 0.7 / > 1.1) etaient devines sans avoir vu la distribution reelle
    # des donnees, et faisaient tomber ~82% des stations dans "mixte" (le
    # constat direct sur la page Donnees) — un seuil absolu qui ne
    # correspond pas a l'echelle reelle des scores rend la classification
    # inutilisable. Les terciles s'auto-calibrent quelle que soit l'echelle.
    raw_scores = []
    for sid in n_obs.index:
        if n_obs[sid] < 24 * 7:  # au moins une semaine d'observations pour classer serieusement
            continue
        c = float(commute_std.get(sid, 0.0) or 0.0)
        m = float(midday_std.get(sid, 0.0) or 0.0)
        w = float(weekend_std.get(sid, 0.0) or 0.0)
        raw_scores.append({
            "station_id": sid,
            "commute_score": c / (m + EPS),
            "weekend_score": w / (c + EPS),
        })

    if not raw_scores:
        return {
            "generated_at": dt.datetime.utcnow().isoformat() + "Z",
            "ready": False,
            "method_note": "Pas assez d'historique pour classer les stations (minimum 1 semaine d'observations).",
            "counts": {"pendulaire": 0, "loisir_weekend": 0, "mixte": 0},
            "stations": [],
            "by_zone": None,
        }

    commute_values = [s["commute_score"] for s in raw_scores]
    weekend_values = [s["weekend_score"] for s in raw_scores]
    commute_p66 = float(np.percentile(commute_values, 66))
    weekend_p66 = float(np.percentile(weekend_values, 66))
    weekend_p33 = float(np.percentile(weekend_values, 33))

    stations = []
    for s in raw_scores:
        commute_score, weekend_score = s["commute_score"], s["weekend_score"]
        # Tercile superieur de commute_score ET tercile inferieur de
        # weekend_score = profil le plus marque "actif en semaine, calme
        # le week-end".
        if commute_score >= commute_p66 and weekend_score <= weekend_p33:
            profile = "pendulaire"
        elif weekend_score >= weekend_p66:
            profile = "loisir_weekend"
        else:
            profile = "mixte"

        stations.append({
            "station_id": s["station_id"],
            "name": names.get(s["station_id"], s["station_id"]),
            "profile": profile,
            "commute_score": round(commute_score, 2),
            "weekend_score": round(weekend_score, 2),
            "zone": zones.get(s["station_id"]),
        })

    counts = {"pendulaire": 0, "loisir_weekend": 0, "mixte": 0}
    for s in stations:
        counts[s["profile"]] += 1

    # Par quartier (si le geocodage est disponible pour cette ville) :
    # profil dominant + repartition — sert de proxy a "l'effet quartier",
    # faute de vraies donnees de morphologie urbaine (densite, pente,
    # proximite metro/RER) que ce projet n'a pas.
    by_zone: dict[str, dict] = {}
    if zones:
        for s in stations:
            z = s["zone"]
            if not z:
                continue
            by_zone.setdefault(z, {"pendulaire": 0, "loisir_weekend": 0, "mixte": 0, "n_stations": 0})
            by_zone[z][s["profile"]] += 1
            by_zone[z]["n_stations"] += 1
    zone_summary = []
    for zone, c in by_zone.items():
        dominant = max(("pendulaire", "loisir_weekend", "mixte"), key=lambda k: c[k])
        zone_summary.append({"zone": zone, "dominant_profile": dominant, **c})
    zone_summary.sort(key=lambda z: z["n_stations"], reverse=True)

    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "ready": len(stations) > 0,
        "method_note": "Classification comportementale par terciles relatifs (volatilite du remplissage "
                       "aux heures de pointe vs le week-end, comparee entre stations), PAS une classification "
                       "par morphologie urbaine reelle (densite, POI, proximite transports) — ces donnees ne "
                       "sont pas disponibles ici.",
        "counts": counts,
        "stations": stations,
        "by_zone": zone_summary if by_zone else None,
    }


def compute_morning_signature(hourly: pd.DataFrame, zones: dict[str, str]) -> dict:
    """Par quartier, compare le remplissage moyen a midi en semaine (11h-14h)
    a celui du matin en semaine (7h-9h) — inspire d'un projet tiers
    (kardol.us/velib) repere lors d'une veille sur des dashboards Velib'
    existants.

    signature = remplissage_midi - remplissage_matin
    - Positif ("exportateur") : le remplissage est bas le matin (les velos
      partent) puis remonte a midi -> quartier plutot residentiel.
    - Negatif ("importateur") : le remplissage est deja haut le matin (les
      velos arrivent) puis se stabilise/redescend a midi -> quartier
      plutot zone d'emploi.

    LIMITE METHODOLOGIQUE (a rappeler cote UI, meme esprit que
    compute_station_typology) : lit l'ETAT des bornes (taux de
    remplissage), pas des trajets comptes individuellement — moins
    biaise par le rebalancement par camion qu'une metrique de flux, mais
    une forte activite de camions un matin donne peut quand meme deplacer
    le signal. Signal relatif entre quartiers, pas un comptage de trajets.
    """
    df = hourly.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.dayofweek
    df["zone"] = df["station_id"].astype(str).map(zones).fillna("Non classé")

    weekday_df = df[df["weekday"] < 5]
    morning = weekday_df[weekday_df["hour"].between(7, 9)]
    midday = weekday_df[weekday_df["hour"].between(11, 14)]

    morning_by_zone = morning.groupby("zone")["fill_rate"].mean()
    midday_by_zone = midday.groupby("zone")["fill_rate"].mean()
    n_obs_by_zone = weekday_df.groupby("zone")["fill_rate"].count()

    zones_out = []
    for zone in sorted(set(morning_by_zone.index) & set(midday_by_zone.index)):
        if n_obs_by_zone.get(zone, 0) < 24 * 5:  # au moins une semaine de donnees en semaine
            continue
        m = float(morning_by_zone[zone])
        d = float(midday_by_zone[zone])
        zones_out.append({
            "zone": zone,
            "morning_fill_rate": round(m, 3),
            "midday_fill_rate": round(d, 3),
            "signature": round(d - m, 3),
            "profile": "exportateur" if (d - m) > 0.02 else "importateur" if (d - m) < -0.02 else "équilibré",
        })

    zones_out.sort(key=lambda z: z["signature"], reverse=True)

    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "ready": len(zones_out) > 0,
        "method_note": "Compare le remplissage moyen en semaine a midi (11h-14h) contre le matin (7h-9h), "
                       "par quartier. Lit l'etat des bornes, pas des trajets comptes — signal relatif entre "
                       "quartiers, a lire avec prudence en cas de forte activite de camions de rebalancement.",
        "zones": zones_out,
    }


def run(city_ids: list[str] | None = None) -> None:
    if city_ids is None:
        city_ids = list_cities()  # Paris inclus, contrairement a forecast_cities.py

    storage.ensure_release(RELEASE_STATS, "Classements de stations par ville (vides / pleines / fiables)")

    failures = []
    for city_id in city_ids:
        try:
            run_city(city_id)
        except Exception as e:
            print(f"[stats_cities] {city_id}: ECHEC ({e}) — villes suivantes non affectees")
            failures.append(city_id)

    # Meme correctif de visibilite que consolidate_cities.py : un echec ne
    # doit pas se limiter a un print jamais consulte.
    if failures:
        raise SystemExit(f"[stats_cities] echec pour : {', '.join(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", default=None)
    args = parser.parse_args()
    run(args.cities)
