"""Features calendaires : jours fériés, vacances scolaires, grèves, événements.

V2 — ajouts :
- Grèves RATP/SNCF depuis transport.data.gouv.fr
- Matchs PSG au Parc des Princes (impact local fort)
- Grands événements parisiens (marathons, etc.)
- Feature composite is_disruption_day
"""
from __future__ import annotations

import datetime as dt
import re

import pandas as pd
import requests

JOURS_FERIES_URL = "https://calendrier.api.gouv.fr/jours-feries/metropole.json"
VACANCES_API = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/records"
GREVES_API = "https://transport.data.gouv.fr/api/disruptions"

USER_AGENT = "pasdevelib-bot/0.1 (+https://pasdevelib.fr)"

# Matchs PSG au Parc des Princes — impact sur les stations 15e/16e
# On récupère depuis l'API FBref (gratuite, pas d'auth requise)
PSG_SCHEDULE_URL = "https://fbref.com/en/squads/e2d8892c/2024-2025/Paris-Saint-Germain-Stats"

# Grands événements récurrents parisiens (dates approximatives, affinées chaque année)
# Ces événements créent des pics inhabituels de demande Vélib
RECURRING_EVENTS: list[dict] = [
    # Marathon de Paris : premier dimanche d'avril
    {"name": "Marathon de Paris", "month": 4, "week": 1, "weekday": 6},
    # Nuit Blanche : premier samedi d'octobre
    {"name": "Nuit Blanche", "month": 10, "week": 1, "weekday": 5},
    # Fête de la Musique : 21 juin
    {"name": "Fête de la Musique", "month": 6, "day": 21},
    # 14 juillet
    {"name": "Fête Nationale", "month": 7, "day": 14},
]


def fetch_jours_feries() -> pd.DataFrame:
    """Tous les jours fériés métropole."""
    r = requests.get(JOURS_FERIES_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = [{"date": pd.to_datetime(d).date(), "label": label} for d, label in data.items()]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def fetch_vacances_zone_c() -> pd.DataFrame:
    """Vacances scolaires zone C, retourne intervalles (start, end, label)."""
    params = {
        "where": 'zones="Zone C" AND location="Paris"',
        "limit": 100,
        "order_by": "start_date",
    }
    r = requests.get(VACANCES_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    records = r.json().get("results", [])
    rows = []
    for rec in records:
        rows.append({
            "start": pd.to_datetime(rec["start_date"]).date(),
            "end": pd.to_datetime(rec["end_date"]).date(),
            "label": rec.get("description", ""),
        })
    return pd.DataFrame(rows)


def fetch_greve_dates(start: dt.date, end: dt.date) -> set[dt.date]:
    """Dates de grèves RATP/SNCF depuis transport.data.gouv.fr.

    Retourne un set de dates où au moins une perturbation majeure est signalée.
    """
    greve_dates: set[dt.date] = set()
    try:
        params = {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "network": "RATP",
        }
        r = requests.get(
            GREVES_API,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        if r.ok:
            disruptions = r.json()
            for d in disruptions:
                # Garder uniquement les perturbations de type grève (pas travaux, incidents)
                cause = (d.get("cause") or "").lower()
                if "gr" in cause or "strike" in cause or "social" in cause:
                    try:
                        date = dt.date.fromisoformat(d["start_date"][:10])
                        greve_dates.add(date)
                    except Exception:
                        pass
    except Exception as e:
        print(f"[calendar] fetch_greve_dates warning: {e}")

    return greve_dates


def compute_recurring_events(start: dt.date, end: dt.date) -> set[dt.date]:
    """Calcule les dates des événements récurrents pour les années couvertes."""
    event_dates: set[dt.date] = set()
    years = range(start.year, end.year + 1)

    for year in years:
        for event in RECURRING_EVENTS:
            try:
                if "day" in event:
                    # Date fixe (ex: 14 juillet)
                    d = dt.date(year, event["month"], event["day"])
                    event_dates.add(d)
                elif "week" in event and "weekday" in event:
                    # N-ième jour de la semaine du mois
                    month_start = dt.date(year, event["month"], 1)
                    # Trouver le premier weekday voulu
                    first = month_start + dt.timedelta(
                        days=(event["weekday"] - month_start.weekday()) % 7
                    )
                    d = first + dt.timedelta(weeks=event["week"] - 1)
                    if d.month == event["month"]:
                        event_dates.add(d)
            except Exception:
                pass

    return event_dates


def build_calendar(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Table jour-par-jour avec colonnes de features calendaires.

    Colonnes : date, weekday, month, is_ferie, is_vacances,
               is_greve, is_event, is_disruption_day.
    """
    days = pd.date_range(start, end, freq="D").date
    df = pd.DataFrame({"date": days})
    df["weekday"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month

    # Jours fériés
    feries = fetch_jours_feries()
    feries_set = set(feries["date"])
    df["is_ferie"] = df["date"].isin(feries_set)

    # Vacances scolaires
    vacances = fetch_vacances_zone_c()
    vac_dates: set[dt.date] = set()
    for _, row in vacances.iterrows():
        for d in pd.date_range(row["start"], row["end"], freq="D").date:
            vac_dates.add(d)
    df["is_vacances"] = df["date"].isin(vac_dates)

    # Grèves RATP/SNCF
    greve_dates = fetch_greve_dates(start, end)
    df["is_greve"] = df["date"].isin(greve_dates)

    # Événements récurrents majeurs
    event_dates = compute_recurring_events(start, end)
    df["is_event"] = df["date"].isin(event_dates)

    # Feature composite : jour perturbé (grève ou événement majeur)
    # → impact fort sur les flux, à distinguer des journées normales
    df["is_disruption_day"] = df["is_greve"] | df["is_event"]

    return df


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "hiver"
    if month in (3, 4, 5):
        return "printemps"
    if month in (6, 7, 8):
        return "ete"
    return "automne"


def enrich_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les alias/derives attendus par predict.py._row_distance
    (day_of_week, is_holiday, is_school_holiday, season) à partir des
    colonnes brutes de build_calendar (weekday, is_ferie, is_vacances,
    month).

    BUG CORRIGE (trouve en investiguant pourquoi les "jours analogues"
    choisis pour Paris remontaient a fin 2020 sans rapport avec la
    saison/meteo cible) : calendar.parquet (produit par build_calendar)
    n'a JAMAIS eu ces colonnes — seul _build_target_features (forecast.py)
    les calculait, et UNIQUEMENT pour les quelques jours CIBLES a predire,
    jamais pour le bassin de candidats HISTORIQUES compare dans
    find_analog_days. Consequence concrete : row_a (cible) avait de vraies
    valeurs, row_b (candidat) renvoyait toujours None pour ces champs — la
    comparaison de distance etait donc non-discriminante pour praticalement
    tous les criteres calendaires, et totalement absente pour la meteo
    (aucune colonne temp_avg/precip_total sur les candidats). Avec des
    distances quasiment toutes egales, le tri stable de pandas preservait
    l'ordre original du calendrier (chronologique, du plus ancien au plus
    recent) — d'ou une selection biaisee vers les dates les plus anciennes
    disponibles, sans lien reel avec la similarite meteo/saison recherchee.
    """
    out = df.copy()
    if "weekday" in out.columns:
        out["day_of_week"] = out["weekday"]
    if "is_ferie" in out.columns:
        out["is_holiday"] = out["is_ferie"]
    if "is_vacances" in out.columns:
        out["is_school_holiday"] = out["is_vacances"]
    if "month" in out.columns:
        out["season"] = out["month"].apply(month_to_season)
    return out


def aggregate_daily_weather(weather_hourly: pd.DataFrame) -> pd.DataFrame:
    """Agrege un DataFrame meteo horaire (colonnes ts, temperature_2m,
    apparent_temperature, precipitation, ...) en moyennes/sommes
    journalieres — meme logique que _build_target_features (forecast.py),
    extraite ici pour etre reutilisable sur l'historique complet
    (weather.parquet) et pas seulement sur les quelques jours cibles.
    """
    w = weather_hourly.copy()
    w["date"] = pd.to_datetime(w["ts"]).dt.tz_convert("Europe/Paris").dt.date.astype(str)
    w = w.sort_values(["date", "ts"])
    w["precip_3h"] = (
        w.groupby("date", group_keys=False)["precipitation"]
        .apply(lambda s: s.rolling(3, min_periods=1).sum())
    )
    return w.groupby("date", as_index=False).agg(
        temp_avg=("temperature_2m", "mean"),
        mean_apparent_temperature=("apparent_temperature", "mean"),
        precip_total=("precipitation", "sum"),
        precip_3h_max=("precip_3h", "max"),
    )
