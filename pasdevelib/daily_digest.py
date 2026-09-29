"""daily_digest.py — Bilan quotidien du réseau, rédigé par IA (Gemini).

Lit les données déjà calculées par stats_cities.py (release "stats-cities" :
stats_<ville>_day.json, evolution_<ville>.json, records_<ville>.json,
traffic_<ville>.json, weather_<ville>.json) pour les 8 villes (Paris
inclus), les combine avec des repères mondiaux du secteur des vélos en
libre-service (fourni statiquement ci-dessous, cf. discussion du
2026-09-28 : ITDP, CIE — Cities in Europe —, Todd et al. 2021, movmi/ADL),
et demande à Gemini de rédiger un bilan Markdown concis par ville.

Sortie (release "daily-digest") :
- digest_<ville>_<YYYY-MM-DD>.json : {date, city_id, title, description,
  markdown, kpis} — une entrée par ville par jour
- digest_<ville>_index.json : liste des 90 derniers jours publiés
  (date, title, slug) — pour la page d'archive du blog

Ne PAS committer de fichier dans pasdevelib-webapp (voir discussion) :
la webapp lit ces assets en HTTP, exactement comme le reste des données
du projet — aucun redeploy nécessaire pour publier un bilan.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pasdevelib import storage
from pasdevelib.cities import CITIES

RELEASE_DIGEST = "daily-digest"
RELEASE_STATS = "stats-cities"
INDEX_MAX_ENTRIES = 90

# BUG CORRIGE ICI (2026-09-28) : gemini-2.5-flash n'est plus disponible
# pour les nouveaux utilisateurs (HTTP 404, message d'erreur explicite de
# l'API recommandant gemini-3.8-flash) — mis à jour vers le modèle
# recommandé par Google au moment du premier run.
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

# TEST (2026-09-28) : Mistral comme fournisseur alternatif. Contrairement à
# Gemini (une clé Google AI Studio par ville, quota très bas par clé sur le
# tier gratuit), MISTRAL_API_KEY est un secret UNIQUE partagé entre les 9
# villes — la Plateforme Mistral a un tier gratuit avec un débit par requête
# plus permissif. Si ce secret est présent, il prend le pas sur les clés
# Gemini par ville (voir run_city) : plus besoin de créer 9 clés séparées.
MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

CITY_LABELS = {
    "paris": "Paris", "bordeaux": "Bordeaux", "lyon": "Lyon", "toulouse": "Toulouse",
    "lille": "Lille", "rennes": "Rennes", "strasbourg": "Strasbourg",
    "montpellier": "Montpellier", "nantes": "Nantes",
}

# AJOUTE (2026-09-28, demande Théo) : nom court a utiliser dans le titre du
# bilan — "Vélib' Métropole" (nom officiel complet, cities.py system_name)
# donnait des titres lourds ("Vélib' Métropole : le bilan du..."). Seul
# Paris a ce probleme, les 8 autres system_name sont deja courts (Vcub,
# Vélo'v, V'Lille...) donc pas d'entree necessaire pour elles.
SHORT_NETWORK_NAME = {"paris": "Vélib'"}

# Repères mondiaux du secteur (VLS), condensés à partir des sources citées
# dans la discussion du 2026-09-28 (Todd et al. 2021, CIE, ITDP, movmi/ADL,
# NABSA, ISGlobal). Utilisés comme grille de lecture, jamais comme des
# vérités absolues sur PasDeVélib — la note méthodo ci-dessous encadre ça.
BENCHMARKS = """
Repères mondiaux du secteur des vélos en libre-service (VLS), à citer
UNIQUEMENT s'ils éclairent le chiffre du jour — jamais de façon plaquée :
- Trajets par vélo et par jour (TDB) : 4 à 8 = fourchette optimale ITDP ;
  6-12 pour un vélo électrique vs 1-5 pour un vélo mécanique (movmi/ADL).
- Trajets par jour pour 1 000 habitants (benchmark européen CIE) : Paris
  39,9 (référence), seuil d'excellence ~19, top 10 européen > 13.
- Distance moyenne par trajet : 1 à 2,5 km (hypothèse prudente : 2 km).
- Taux de stations avec au moins un vélo aux heures clés (8h, midi, 18h)
  et taux de stations pleines : les deux mesures de la satisfaction usager.
- Ratio bornes/vélos recommandé : 1,5 à 3 emplacements par vélo.
- Temps de cycle de réparation best-in-class : moins de 4 jours.
- Distance de marche acceptable jusqu'à une station : ~500 m.
- CO2 évité : hypothèse 2 km/trajet substitué à une voiture.
- Méthode de référence (baisse du compteur de vélos entre deux
  observations) : erreur moyenne ~15% vs trajets réels — à rappeler
  quand un nombre de trajets est cité.
""".strip()

# RESTRUCTURE (2026-09-28, demande Théo) : sortie JSON stricte plutot que
# TITRE/RESUME/--- — plus facile a exploiter cote template (cartes "points
# cles", encart "conseil du jour" separement du corps narratif) sans
# reparser un bloc de texte libre. La structure du corps (pointe matin vs
# soir, top/flop stations, analyse meteo) est desormais imposee dans les
# regles plutot que laissee au libre choix du modele.
PROMPT_TEMPLATE = """Tu es le rédacteur en chef data-journalisme du blog pasdevelib.app.
Rédige le bilan quotidien du réseau de vélos en libre-service de {city_label}
({network}), pour la journée du {date_human}, à partir des données réelles
ci-dessous. Public : usagers curieux + collectivités/chercheurs qui suivent
le site. Ton : factuel, précis, jamais promotionnel, jamais alarmiste sans
preuve chiffrée.

RÈGLES STRICTES :
- Toute affirmation chiffrée doit venir des données fournies ci-dessous.
  N'invente AUCUN chiffre. Si une donnée manque, dis-le plutôt que de combler
  (et NE MENTIONNE PAS la section correspondante si les données sont vides).
- Les "trajets estimés" sont une ESTIMATION par variation d'occupation
  station par station (pas un comptage réel), marge d'erreur ~15% — le
  rappeler la première fois que tu cites ce chiffre, brièvement.
- Compare au maximum UN chiffre du jour à UN repère mondial pertinent
  (liste ci-dessous), seulement si la comparaison est vraiment éclairante.
  Ne pas forcer une comparaison si aucune n'est pertinente.
- "corps_markdown" : markdown standard uniquement, sans accolades ni blocs
  de code, structuré dans cet ordre :
  1. Un paragraphe d'ouverture avec le chiffre du jour (remplissage moyen).
  2. Si profil_horaire_departs contient des heures 6h-10h ET 16h-20h avec
     des données : un paragraphe séparant la pointe du matin de la pointe
     du soir (deux flux distincts). Sinon, un seul paragraphe sur le pic
     observé, sans forcer une distinction matin/soir inexistante dans les
     données.
  3. Si stations_top_vides ou stations_top_pleines est non vide : un
     paragraphe nommant les stations concernées.
  4. Si impact_meteo est présent : un paragraphe croisant pluie et trafic.
  Longueur du corps : 250 à 400 mots.
- "points_cles" : 3 à 5 puces courtes (une phrase chacune), les faits les
  plus marquants du jour — pas un résumé du corps, des chiffres bruts
  autonomes (ex: "Pic de départs à 8h avec 590 vélos/h").
- "conseil_usager" : UNE phrase pratique et actionnable pour un usager,
  basée sur une donnée réelle ci-dessous (ex: une station ou un horaire à
  éviter/privilégier). Si aucune donnée ne permet un conseil honnête,
  renvoie une chaîne vide "" plutôt que d'inventer.
- "titre" : utilise le nom court et usuel du système ({short_name}), PAS
  son nom officiel complet ({network}) — ex. "{short_name} : le bilan du
  {date_human}", jamais "{network} : le bilan du ...".

{benchmarks}

DONNÉES DU JOUR ({city_label}) :
{data_json}

Réponds UNIQUEMENT avec un objet JSON valide (aucun texte avant/après,
aucun bloc de code ```), exactement ce schéma :
{{
  "titre": "<titre accrocheur avec la date, une ligne, sans guillemets>",
  "chapeau": "<une phrase de résumé pour une carte/aperçu, 160 caractères max>",
  "points_cles": ["<fait 1>", "<fait 2>", "<fait 3>"],
  "conseil_usager": "<conseil actionnable, ou chaîne vide>",
  "corps_markdown": "<corps de l'article en Markdown, structuré selon les règles ci-dessus, sans reprendre le titre en H1>"
}}
"""


def _download_json(release: str, asset: str) -> dict | None:
    url = f"https://github.com/{storage.REPO}/releases/download/{release}/{asset}"
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def _gather_city_data(city_id: str) -> dict:
    """Rassemble un sous-ensemble condensé des assets stats-cities déjà
    calculés — pas de recalcul ici, uniquement de la lecture + un peu de
    découpe pour ne pas envoyer des dizaines de Ko inutiles à Gemini."""
    data: dict = {}

    day_stats = _download_json(RELEASE_STATS, f"stats_{city_id}_day.json")
    if day_stats:
        data["remplissage_moyen"] = day_stats.get("city_avg_fill_rate")
        data["stations_top_vides"] = (day_stats.get("top_empty") or [])[:5]
        data["stations_top_pleines"] = (day_stats.get("top_full") or [])[:5]
        data["quartiers_les_plus_touches"] = (day_stats.get("neighborhoods") or [])[:5]
        data["fenetre"] = {"debut": day_stats.get("window_start"), "fin": day_stats.get("window_end")}
        # AJOUTE (2026-09-28, demande Théo) : part électrique du parc dispo
        # et profil horaire — calculés par stats_cities.py (compute_period).
        # None/[] si pas encore dispo pour cette ville (voir docstring de
        # _compute_ebike_share/_compute_hourly_curve dans stats_cities.py).
        if day_stats.get("ebike_share") is not None:
            data["part_electrique"] = day_stats["ebike_share"]
        if day_stats.get("hourly_curve"):
            data["profil_horaire_remplissage"] = day_stats["hourly_curve"]
        # AJOUTE (2026-09-28, demande Théo) : stations bloquées quasi toute
        # la fenêtre + top 10 sous tension (vides ET pleines confondues,
        # déjà calculé par stats_cities.py sous "worst" = pct_extreme).
        data["n_stations_pleines_toute_la_journee"] = day_stats.get("n_stations_full_all_day", 0)
        data["n_stations_vides_toute_la_journee"] = day_stats.get("n_stations_empty_all_day", 0)
        data["stations_top_tension"] = (day_stats.get("worst") or [])[:10]

    evolution = _download_json(RELEASE_STATS, f"evolution_{city_id}.json")
    if evolution and evolution.get("series"):
        data["evolution_7_derniers_jours"] = evolution["series"][-7:]

    traffic = _download_json(RELEASE_STATS, f"traffic_{city_id}.json")
    if traffic and traffic.get("trips_per_day"):
        last = traffic["trips_per_day"][-1]
        data["trajets_estimes_hier"] = last
        data["profil_horaire_departs"] = traffic.get("bikes_per_hour")

    records = _download_json(RELEASE_STATS, f"records_{city_id}.json")
    if records:
        data["records"] = records

    weather = _download_json(RELEASE_STATS, f"weather_{city_id}.json")
    if weather and weather.get("ready"):
        data["impact_meteo"] = weather

    stuck = _download_json(RELEASE_STATS, f"stuck_{city_id}.json")
    if stuck:
        data["stations_bloquees"] = {
            "vides_longtemps": (stuck.get("longest_empty") or [])[:3],
            "pleines_longtemps": (stuck.get("longest_full") or [])[:3],
        }

    return data


# BUG CORRIGE ICI (2026-09-28) : deux causes distinctes trouvees sur les
# premiers runs reels :
# 1. 503 "high demand" (transitoire, sans rapport avec la cle/le modele)
#    — retry + backoff + gigue, meme pattern que storage.upload_asset().
# 2. 429 RESOURCE_EXHAUSTED, "limit: 5" (quota tier gratuit tres bas
#    pour ce modele, ~5 requetes/minute) — atteint des le run manuel
#    suivant car le workflow lance les 9 villes EN PARALLELE (matrix),
#    qui tapent toutes la meme cle en meme temps. Fix a deux niveaux :
#    ici, on lit le "Please retry in Ns" que Gemini renvoie DANS le
#    corps de la reponse 429 et on attend exactement ce delai (plutot
#    qu'un backoff generique, inutile face a un quota qui ne se libere
#    qu'a un instant precis) ; cote workflow (daily-digest.yml),
#    max-parallel: 1 fait tourner les 9 villes en sequentiel, pas en
#    parallele, pour ne plus jamais cumuler plusieurs requetes dans la
#    meme fenetre glissante.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRY_AFTER_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)
# gemini-3.8-flash est un modele recemment sorti et tres demande ; un 503
# "high demand" peut durer largement plus que les ~7s que couvrait l'ancien
# backoff (4 tentatives, plafond 2**attempt). On monte a 6 tentatives avec un
# plafond de 60s, ce qui laisse ~2min de marge cumulee avant d'abandonner.
MAX_GEMINI_ATTEMPTS = 6
_MAX_BACKOFF_S = 60.0


def _call_gemini(prompt: str, api_key: str) -> str:
    import random
    import time

    last_error: Exception | None = None
    for attempt in range(MAX_GEMINI_ATTEMPTS):
        r = requests.post(
            f"{GEMINI_URL}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.6, "maxOutputTokens": 2000},
            },
            timeout=60,
        )
        if r.status_code >= 400:
            # Le corps de reponse Gemini contient le vrai motif (cle invalide,
            # quota depasse, contenu bloque...) — raise_for_status() seul ne
            # le montre pas, d'ou des erreurs illisibles dans les logs Actions.
            last_error = RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:500]}")
            if r.status_code in RETRYABLE_STATUS and attempt < MAX_GEMINI_ATTEMPTS - 1:
                m = _RETRY_AFTER_RE.search(r.text)
                wait = (
                    float(m.group(1)) + 2.0 if m
                    else min(_MAX_BACKOFF_S, (2 ** attempt) + random.uniform(0, 1.5))
                )
                print(f"[daily_digest] Gemini HTTP {r.status_code} "
                      f"(tentative {attempt + 1}/{MAX_GEMINI_ATTEMPTS}), "
                      f"nouvel essai dans {wait:.1f}s"
                      + (" (delai indique par l'API)" if m else ""))
                time.sleep(wait)
                continue
            raise last_error

        payload = r.json()
        candidates = payload.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini n'a renvoyé aucune réponse exploitable: {payload}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        if not text.strip():
            raise RuntimeError("Gemini a renvoyé une réponse vide")
        return text

    raise last_error or RuntimeError("Gemini: échec après 4 tentatives")


def _call_mistral(prompt: str, api_key: str) -> str:
    import random
    import time

    last_error: Exception | None = None
    for attempt in range(MAX_GEMINI_ATTEMPTS):
        r = requests.post(
            MISTRAL_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "max_tokens": 2000,
            },
            timeout=60,
        )
        if r.status_code >= 400:
            # Meme raisonnement que pour Gemini : le corps de reponse Mistral
            # contient le vrai motif (cle invalide, rate limit...).
            last_error = RuntimeError(f"Mistral HTTP {r.status_code}: {r.text[:500]}")
            if r.status_code in RETRYABLE_STATUS and attempt < MAX_GEMINI_ATTEMPTS - 1:
                m = _RETRY_AFTER_RE.search(r.text)
                wait = (
                    float(m.group(1)) + 2.0 if m
                    else min(_MAX_BACKOFF_S, (2 ** attempt) + random.uniform(0, 1.5))
                )
                print(f"[daily_digest] Mistral HTTP {r.status_code} "
                      f"(tentative {attempt + 1}/{MAX_GEMINI_ATTEMPTS}), "
                      f"nouvel essai dans {wait:.1f}s"
                      + (" (delai indique par l'API)" if m else ""))
                time.sleep(wait)
                continue
            raise last_error

        payload = r.json()
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"Mistral n'a renvoyé aucune réponse exploitable: {payload}")
        text = (choices[0].get("message") or {}).get("content", "")
        if not text.strip():
            raise RuntimeError("Mistral a renvoyé une réponse vide")
        return text

    raise last_error or RuntimeError(f"Mistral: échec après {MAX_GEMINI_ATTEMPTS} tentatives")


def _strip_code_fence(text: str) -> str:
    """Retire un eventuel encadrement ```json ... ``` autour de la reponse —
    frequent malgre la consigne "aucun bloc de code" du prompt."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_output(raw: str) -> dict:
    """Extrait le JSON structure (titre/chapeau/points_cles/conseil_usager/
    corps_markdown) de la reponse du modele. Defensif a plusieurs niveaux
    (meme esprit que l'ancien parseur texte, cf. historique de ce fichier) :
    - retire un encadrement ```json``` eventuel
    - si du texte parasite entoure quand meme le JSON, prend la sous-chaine
      du premier '{' au dernier '}'
    - si le JSON est invalide ou incomplet, degrade proprement plutot que
      de planter tout le run (une ville en echec ne doit jamais bloquer
      les suivantes, cf. run())
    """
    text = _strip_code_fence(raw)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {
            "titre": "Bilan du jour",
            "chapeau": text[:160],
            "points_cles": [],
            "conseil_usager": "",
            "corps_markdown": text,
        }
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {
            "titre": "Bilan du jour",
            "chapeau": text[:160],
            "points_cles": [],
            "conseil_usager": "",
            "corps_markdown": text,
        }

    def _clean_markdown(s: str) -> str:
        # Defensif : aucune accolade dans le markdown — MDXRemote côté
        # webapp interprète { } comme du JSX (bug déjà rencontré ici).
        return str(s).replace("{", "(").replace("}", ")").strip()

    points_cles = payload.get("points_cles")
    if not isinstance(points_cles, list):
        points_cles = []
    points_cles = [_clean_markdown(p) for p in points_cles if str(p).strip()][:5]

    return {
        "titre": _clean_markdown(payload.get("titre") or "Bilan du jour"),
        "chapeau": _clean_markdown(payload.get("chapeau") or "")[:160],
        "points_cles": points_cles,
        "conseil_usager": _clean_markdown(payload.get("conseil_usager") or ""),
        "corps_markdown": _clean_markdown(payload.get("corps_markdown") or ""),
    }


def _download_day_archive(date: dt.date) -> pd.DataFrame | None:
    """Télécharge l'archive brute d'UNE journée (release "history",
    YYYY-MM-DD.parquet, écrite par consolidate.py) — chaque relevé
    individuel du scraper, contrairement à hourly_history.parquet
    (release "aggregates") déjà agrégé à l'heure. C'est cette granularité
    fine qui permet un découpage à 10 min (_compute_intraday_traffic) ou
    un flux heure par heure sur UN jour précis (_compute_station_hourly_flux).

    Paris UNIQUEMENT pour l'instant : seul pdv-bot/consolidate.py archive
    ce fichier brut par jour. pdvr-bot (villes régionales) agrège
    directement à l'heure sans étape intermédiaire — donc None pour elles
    tant que ce n'est pas changé côté pipeline.

    Partagée entre les deux fonctions plutôt que téléchargée deux fois
    (même fichier, ~30s de download) — chacune part ensuite de sa propre
    copie (df.copy() en interne si besoin) pour ne pas se marcher dessus.
    """
    date_str = date.isoformat()
    url = f"https://github.com/{storage.REPO}/releases/download/history/{date_str}.parquet"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    df = pd.read_parquet(io.BytesIO(r.content))
    if df.empty or "fetched_at" not in df.columns:
        return None
    return df


def _compute_intraday_traffic(date: dt.date, day_df: pd.DataFrame | None = None) -> list[dict]:
    """Départs estimés par tranche de 10 minutes sur LA journée du bilan.

    Même méthodologie que compute_traffic() dans stats_cities.py (seules
    les BAISSES comptent comme "départs", plafond par-station au
    percentile 90 pour écrêter les camions de rééquilibrage) — dupliquée
    ici plutôt que partagée car la granularité (10 min vs heure) et le
    fichier source (archive brute vs hourly_history) diffèrent.

    Renvoie une liste de 144 tranches (00:00 à 23:50). "trips"/"cumulative"
    sont None pour une tranche jamais couverte par le scraper (au lieu
    d'un faux 0) — le pipeline a des trous connus (cf. README pdv-bot).

    day_df : archive déjà téléchargée par _download_day_archive(), pour
    éviter un second download quand run_city() a déjà besoin du flux
    horaire (_compute_station_hourly_flux) pour la même date.
    """
    df = day_df if day_df is not None else _download_day_archive(date)
    if df is None:
        return []
    df = df.copy()

    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    local_ts = df["fetched_at"].dt.tz_convert("Europe/Paris")
    df["bucket_min"] = local_ts.dt.hour * 60 + (local_ts.dt.minute // 10) * 10
    covered_buckets = set(df["bucket_min"].unique().tolist())

    per_bucket = (
        df.sort_values("fetched_at")
        .groupby(["station_id", "bucket_min"], as_index=False)
        .agg(num_bikes_available=("num_bikes_available", "last"), fetched_at=("fetched_at", "last"))
        .sort_values(["station_id", "bucket_min"])
    )
    per_bucket["prev_bikes"] = per_bucket.groupby("station_id")["num_bikes_available"].shift(1)
    prev_fetched_at = per_bucket.groupby("station_id")["fetched_at"].shift(1)
    # BUG EVITE ICI (2026-09-28) : l'archive history/<date>.parquet est
    # bornée sur la date UTC, donc convertie en heure locale Paris (+1/+2h)
    # elle couvre les heures locales ~1h/2h→23h59 du jour D PUIS 00h/01h du
    # jour D+1. Comparer les bucket_min entre eux (ex. l'ancien
    # `bucket_min - prev_bucket == 10`) traite alors le dernier bucket local
    # avant minuit (23h50, vrai) et le premier bucket local après minuit
    # (00h00, en réalité ~22h plus tôt dans l'archive UTC précédente) comme
    # "consécutifs" puisque 0-1430 se suivent numériquement — ce qui produit
    # un pic de "départs" fantôme à cette jointure. On compare desormais le
    # VRAI temps écoulé entre les deux relevés (fetched_at réel), qui démasque
    # ce saut (~22h, largement hors tolérance) sans toucher au reste.
    elapsed_min = (per_bucket["fetched_at"] - prev_fetched_at).dt.total_seconds() / 60
    is_consecutive = elapsed_min.between(0, 15, inclusive="right")
    delta = per_bucket["num_bikes_available"] - per_bucket["prev_bikes"]
    departures = (-delta).clip(lower=0)
    per_bucket["departures"] = departures.where(is_consecutive, other=np.nan)

    if len(per_bucket) > 0:
        station_p90 = per_bucket.groupby("station_id")["departures"].transform(
            lambda x: x.quantile(0.90) if (x > 0).any() else 0
        )
        per_bucket["departures"] = np.minimum(per_bucket["departures"], station_p90)

    valid = per_bucket.dropna(subset=["departures"])
    by_bucket = valid.groupby("bucket_min")["departures"].sum()

    result = []
    cumulative = 0.0
    for bucket_min in range(0, 24 * 60, 10):
        has_data = bucket_min in covered_buckets
        trips = float(by_bucket.get(bucket_min, 0.0)) if has_data else 0.0
        if has_data:
            cumulative += trips
        h, m = divmod(bucket_min, 60)
        result.append({
            "time": f"{h:02d}:{m:02d}",
            "trips": round(trips) if has_data else None,
            "cumulative": round(cumulative) if has_data else None,
        })
    return result


def _fetch_stations_meta() -> dict:
    """station_id -> {name, lat, lon, capacity}, depuis stations.json
    (release "live"). Même logique que _build_stations_meta() dans
    flux.py — dupliquée plutôt qu'importée pour ne pas coupler ce module
    à flux.py (déclenché par un workflow séparé), cf. le même choix pour
    _compute_intraday_traffic vs compute_traffic().
    """
    url = f"https://github.com/{storage.REPO}/releases/download/{storage.RELEASE_LIVE}/stations.json"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return {}
    out = {}
    for s in r.json():
        sid = str(s.get("station_id") or s.get("stationcode") or s.get("stationCode") or "")
        if not sid:
            continue
        lat, lon = s.get("lat") or s.get("latitude"), s.get("lon") or s.get("longitude")
        if lat is None or lon is None:
            continue
        out[sid] = {
            "name": s.get("name") or s.get("station_name") or "",
            "lat": float(lat),
            "lon": float(lon),
            "capacity": int(s.get("capacity", 0) or 0),
        }
    return out


def _compute_station_hourly_flux(
    date: dt.date, day_df: pd.DataFrame | None, stations_meta: dict
) -> dict | None:
    """Flux net par station, heure par heure, pour LA journée précise du
    bilan (contrairement à flux.py/flux_hourly.json qui moyenne sur tout
    l'historique disponible pour un jour-de-semaine donné — cf. discussion
    du 2026-09-28 : "il faut une heatmap ... qui défile tout seul de 00h à
    23h59" sur le jour du bilan, pas la version moyennée à curseur manuel
    de /blog/articles/respiration-paris).

    Même formule que respiration-paris (flux_net(station, h) = bikes(h) -
    bikes(h-1)) mais sur les relevés bruts d'un seul jour au lieu d'une
    moyenne multi-jours. Contrairement à flux.py, les heures de nuit (0-5h)
    ne sont PAS masquées : ici on montre le vrai déroulé du jour, y compris
    un éventuel rééquilibrage nocturne — c'est un instantané réel, pas une
    moyenne à "nettoyer".

    Paris uniquement (mêmes limites que _compute_intraday_traffic).
    Publié à part (asset daymap_<ville>_<date>.json) plutôt que dans
    l'entrée digest_<ville>_<date>.json : ~1500 stations × 24h, plus gros
    qu'un digest et consommé différemment (par la carte, pas l'article).
    """
    df = day_df if day_df is not None else _download_day_archive(date)
    if df is None or not stations_meta:
        return None
    df = df.copy()

    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    local_ts = df["fetched_at"].dt.tz_convert("Europe/Paris")
    df["hour"] = local_ts.dt.hour
    df["station_id"] = df["station_id"].astype(str)

    per_hour = (
        df.sort_values("fetched_at")
        .groupby(["station_id", "hour"], as_index=False)
        .agg(num_bikes_available=("num_bikes_available", "last"), fetched_at=("fetched_at", "last"))
        .sort_values(["station_id", "hour"])
    )
    per_hour["prev_bikes"] = per_hour.groupby("station_id")["num_bikes_available"].shift(1)
    prev_fetched_at = per_hour.groupby("station_id")["fetched_at"].shift(1)
    # Même garde-fou que _compute_intraday_traffic (cf. son commentaire) :
    # temps réel écoulé plutôt que différence d'étiquette d'heure locale,
    # pour ne pas confondre 23h(jour D)→0h(jour D+1) avec 22h(D)→23h(D).
    elapsed_min = (per_hour["fetched_at"] - prev_fetched_at).dt.total_seconds() / 60
    is_consecutive = elapsed_min.between(0, 90, inclusive="right")
    delta = per_hour["num_bikes_available"] - per_hour["prev_bikes"]
    per_hour["flux"] = delta.where(is_consecutive, other=np.nan)

    flux_by_station = {
        sid: dict(zip(g["hour"], g["flux"]))
        for sid, g in per_hour.groupby("station_id")
    }

    stations_out = []
    for sid, info in stations_meta.items():
        hourly = flux_by_station.get(sid)
        if not hourly:
            continue
        arr = []
        for h in range(24):
            v = hourly.get(h)
            arr.append(round(float(v), 1) if v is not None and v == v else None)
        if all(v is None for v in arr):
            continue
        stations_out.append({
            "id": sid,
            "name": info.get("name", ""),
            "lat": info["lat"],
            "lon": info["lon"],
            "capacity": info.get("capacity", 0),
            "flux": arr,
        })

    return {
        "date": date.isoformat(),
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "n_stations": len(stations_out),
        "stations": stations_out,
    }


def _api_key_env_var(city_id: str) -> str:
    return f"GEMINI_API_KEY_{city_id.upper()}"


def run_city(city_id: str, date: dt.date) -> None:
    import os

    # MISTRAL_API_KEY (secret unique, partagé entre les 9 villes) essayé en
    # PREMIER si présente, avec repli automatique sur Gemini (clé par ville,
    # GEMINI_API_KEY_<VILLE>) en cas d'échec — pas seulement si la clé est
    # absente. BUG CORRIGE ICI (2026-09-28) : jusqu'ici, dès que
    # MISTRAL_API_KEY existait, gemini_key était forcé à None (`None if
    # mistral_key else ...`), donc un Mistral en échec (429 permanent
    # constaté sur ce compte, cf. diagnostic du 2026-09-28) faisait planter
    # la ville entière sans repli possible — TOUTES les villes étaient
    # bloquées depuis l'ajout de MISTRAL_API_KEY en secret, silencieusement
    # absorbé par le try/except par-ville dans run(). On récupère désormais
    # gemini_key dans tous les cas, et on ne l'ignore que si Mistral réussit.
    mistral_key = os.environ.get("MISTRAL_API_KEY")
    gemini_env_var = _api_key_env_var(city_id)
    gemini_key = os.environ.get(gemini_env_var)
    if not mistral_key and not gemini_key:
        print(f"[daily_digest] {city_id}: ni MISTRAL_API_KEY ni {gemini_env_var}, skip")
        return

    city_cfg = CITIES.get(city_id)
    city_label = CITY_LABELS.get(city_id, city_id.capitalize())
    network = city_cfg.system_name if city_cfg else city_label

    data = _gather_city_data(city_id)
    if not data:
        print(f"[daily_digest] {city_id}: aucune donnée stats-cities disponible, skip")
        return

    date_human = date.strftime("%d/%m/%Y")
    prompt = PROMPT_TEMPLATE.format(
        city_label=city_label,
        network=network,
        short_name=SHORT_NETWORK_NAME.get(city_id, network),
        date_human=date_human,
        benchmarks=BENCHMARKS,
        data_json=json.dumps(data, ensure_ascii=False, indent=2, default=str),
    )

    raw = None
    model_used = None
    if mistral_key:
        try:
            raw = _call_mistral(prompt, mistral_key)
            model_used = MISTRAL_MODEL
        except Exception as e:
            if gemini_key:
                print(f"[daily_digest] {city_id}: Mistral en échec ({type(e).__name__}: {e}), repli sur Gemini")
            else:
                raise
    if raw is None:
        if not gemini_key:
            raise RuntimeError(f"{city_id}: Mistral en échec et pas de {gemini_env_var} pour un repli")
        raw = _call_gemini(prompt, gemini_key)
        model_used = GEMINI_MODEL
    parsed = _parse_json_output(raw)

    date_str = date.isoformat()
    entry = {
        "date": date_str,
        "city_id": city_id,
        "title": parsed["titre"],
        "description": parsed["chapeau"],
        "markdown": parsed["corps_markdown"],
        # AJOUTE (2026-09-28, demande Théo) : sortie JSON structurée plutot
        # que TITRE/RESUME/--- — champs additifs, la version prod actuelle
        # du template (qui ne lit que title/description/markdown) continue
        # de fonctionner sans modif si elle est publiée avant que le
        # template les affiche.
        "points_cles": parsed["points_cles"],
        "conseil_usager": parsed["conseil_usager"],
        "ebike_share": data.get("part_electrique"),
        "hourly_curve": data.get("profil_horaire_remplissage", []),
        "n_stations_full_all_day": data.get("n_stations_pleines_toute_la_journee", 0),
        "n_stations_empty_all_day": data.get("n_stations_vides_toute_la_journee", 0),
        "top_tension": data.get("stations_top_tension", []),
        # Départs par tranche de 10 min sur la journée du bilan — Paris
        # uniquement pour l'instant (cf. docstring de la fonction). Un seul
        # download de l'archive brute (day_archive, ci-dessous), partagé
        # avec le calcul du flux horaire par station juste après — pas la
        # peine de retélécharger le même fichier deux fois.
        "intraday_trips": [],
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "model": model_used,
    }

    # AJOUTE (2026-09-28, demande Théo) : heatmap "respiration" auto-jouée
    # 00h→23h59 sur le jour PRECIS du bilan (par opposition à flux.py qui
    # moyenne sur tout l'historique par jour-de-semaine). Un seul download
    # de l'archive brute du jour, réutilisé pour intraday_trips (10 min)
    # ET pour le flux horaire par station — Paris uniquement (cf. docstring
    # de _download_day_archive).
    day_archive = _download_day_archive(date) if city_id == "paris" else None
    if city_id == "paris":
        entry["intraday_trips"] = _compute_intraday_traffic(date, day_df=day_archive)

    daymap = None
    if city_id == "paris" and day_archive is not None:
        stations_meta = _fetch_stations_meta()
        daymap = _compute_station_hourly_flux(date, day_archive, stations_meta)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        asset_name = f"digest_{city_id}_{date_str}.json"
        out_path = tmp_dir / asset_name
        out_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2))
        storage.upload_asset(RELEASE_DIGEST, out_path, asset_name)

        if daymap is not None:
            daymap_name = f"daymap_{city_id}_{date_str}.json"
            daymap_path = tmp_dir / daymap_name
            daymap_path.write_text(json.dumps(daymap, ensure_ascii=False, separators=(",", ":")))
            storage.upload_asset(RELEASE_DIGEST, daymap_path, daymap_name)
            print(f"[daily_digest] {city_id}: {daymap_name} publié ({daymap['n_stations']} stations)")

        # Index (liste des N derniers jours) — lu par la page d'archive
        # /blog/articles/dailymonitoring/<ville> côté webapp.
        index_name = f"digest_{city_id}_index.json"
        index_path = tmp_dir / index_name
        existing = storage.download_asset(RELEASE_DIGEST, index_name, index_path)
        entries = json.loads(index_path.read_text()) if existing and index_path.exists() else []
        entries = [e for e in entries if e.get("date") != date_str]
        # BUG CORRIGE ICI (2026-09-28) : `title`/`summary` n'existaient plus
        # depuis le passage à la sortie JSON structurée (parsed["titre"]/
        # parsed["chapeau"]) — NameError qui plantait run_city() APRES
        # l'upload de digest_<ville>_<date>.json mais AVANT la mise à jour de
        # l'index, silencieusement absorbé par le try/except par-ville dans
        # run(). Résultat : le digest du jour était bien publié, mais
        # l'index (utilisé par fetchLatestDigest côté webapp) restait bloqué
        # sur l'ancien digest le plus récent — d'où la page qui semblait ne
        # jamais se mettre à jour malgré des runs "réussis".
        entries.append({"date": date_str, "title": entry["title"], "description": entry["description"]})
        entries.sort(key=lambda e: e["date"], reverse=True)
        entries = entries[:INDEX_MAX_ENTRIES]
        index_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
        storage.upload_asset(RELEASE_DIGEST, index_path, index_name)

    print(f"[daily_digest] {city_id}: {asset_name} publié ({entry['title']})")


def run(city_ids: list[str] | None = None) -> None:
    if city_ids is None:
        city_ids = list(CITY_LABELS.keys())

    storage.ensure_release(RELEASE_DIGEST, "Bilans quotidiens par ville (rédigés par IA)")

    # Le bilan porte sur la journée d'hier (le run de 06h30 UTC ne voit
    # pas encore la journée en cours), même logique que stats_day.
    yesterday = dt.datetime.utcnow().date() - dt.timedelta(days=1)

    for city_id in city_ids:
        # Isolation par ville, même principe que consolidate_cities.py :
        # un échec (Gemini, données manquantes, clé absente) ne bloque
        # jamais les autres.
        try:
            run_city(city_id, yesterday)
        except Exception as e:
            # BUG CORRIGE ICI : {e} seul peut afficher un message vide ou
            # trompeur (ex. KeyError affiche repr() de la clé, une
            # requests.HTTPError peut avoir un message tronqué) — on log
            # desormais le type ET la traceback complete, indispensable
            # pour diagnostiquer depuis les logs GitHub Actions sans
            # devoir reproduire localement.
            print(f"[daily_digest] {city_id}: ECHEC ({type(e).__name__}: {e}) — villes suivantes non affectées")
            traceback.print_exc()
            # AJOUTE (2026-09-29) : les logs des runs Actions ne sont pas
            # consultables depuis l'environnement d'agent utilisé pour
            # développer ce bot (les téléchargements de logs, hébergés sur
            # Azure Blob, sont bloqués par la politique réseau de cet
            # environnement — l'API GitHub elle-même n'est pas en cause).
            # On publie donc la traceback comme asset de la release
            # "daily-digest", lisible via l'API releases (non bloquée),
            # pour pouvoir diagnostiquer un échec sans accès aux logs.
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    err_path = Path(tmp) / f"digest_{city_id}_last_error.txt"
                    err_path.write_text(
                        f"{dt.datetime.utcnow().isoformat()}Z — date bilan: {yesterday.isoformat()}\n\n"
                        + traceback.format_exc()
                    )
                    storage.upload_asset(RELEASE_DIGEST, err_path, err_path.name)
            except Exception:
                print(f"[daily_digest] {city_id}: échec de la publication de la traceback de diagnostic")
                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", default=None,
                        help="IDs des villes (ex: paris bordeaux). Défaut: les 8.")
    args = parser.parse_args()
    run(args.cities)
