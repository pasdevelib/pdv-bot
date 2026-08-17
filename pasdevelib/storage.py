"""Stockage sur GitHub Releases via le CLI `gh`.

Convention :
- release `live`        : current_day.parquet, stations.json
- release `history`     : YYYY-MM-DD.parquet (un par jour)
- release `aggregates`  : medians.parquet, weather.parquet, calendar.parquet
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

REPO = os.environ.get("GITHUB_REPOSITORY", "pasdevelib/pdv-bot")

# Mapping logique
RELEASE_LIVE = "live"
RELEASE_HISTORY = "history"
RELEASE_AGGREGATES = "aggregates"


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Exécute une commande shell."""
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def ensure_release(tag: str, title: str | None = None) -> None:
    """Crée la release si elle n'existe pas."""
    result = _run(["gh", "release", "view", tag, "--repo", REPO], check=False)
    if result.returncode != 0:
        _run([
            "gh", "release", "create", tag,
            "--repo", REPO,
            "--title", title or tag.capitalize(),
            "--notes", f"Auto-generated bucket for `{tag}` data.",
        ])


def upload_asset(tag: str, local_path: Path, asset_name: str | None = None) -> None:
    """Upload un fichier comme asset de la release. Écrase si existe.

    BUG CORRIGE ICI : constate en prod — plusieurs villes tournent sur
    des workflows GitHub Actions independants, dont les horaires peuvent
    se chevaucher, et qui uploadent TOUTES vers la meme release
    (`cities-live`), parfois le meme asset exact (stations_cities.json,
    desormais fusionne plutot qu'ecrase — voir scrape_cities.py). `gh
    release upload --clobber` fait en interne un delete puis un upload ;
    deux executions concurrentes sur le meme nom d'asset peuvent entrer
    en collision cote API GitHub (l'une supprime l'asset pendant que
    l'autre tente de l'uploader), faisant echouer l'une des deux avec un
    simple exit code 1 sans autre contexte. Retente desormais avec un
    backoff + gigue aleatoire avant d'abandonner pour de bon — la
    plupart de ces conflits sont transitoires et se resolvent seuls au
    second essai.
    """
    name = asset_name or local_path.name
    last_result = None
    for attempt in range(4):
        # `--clobber` remplace l'asset s'il existe déjà
        last_result = _run([
            "gh", "release", "upload", tag,
            f"{local_path}#{name}",
            "--repo", REPO,
            "--clobber",
        ], check=False)
        if last_result.returncode == 0:
            return
        if attempt < 3:
            wait = (2 ** attempt) + random.uniform(0, 1.5)
            print(f"[storage] upload_asset({tag}, {name}) échec (tentative {attempt + 1}/4), "
                  f"nouvel essai dans {wait:.1f}s — {(last_result.stderr or '').strip()[:200]}")
            time.sleep(wait)
    raise RuntimeError(
        f"upload_asset({tag}, {name}) a échoué après 4 tentatives : "
        f"{(last_result.stderr or '').strip() if last_result else 'inconnu'}"
    )


def download_asset(tag: str, asset_name: str, dest: Path) -> Path | None:
    """Télécharge un asset. Retourne None s'il n'existe pas."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = _run([
        "gh", "release", "download", tag,
        "--repo", REPO,
        "--pattern", asset_name,
        "--output", str(dest),
        "--clobber",
    ], check=False)
    return dest if result.returncode == 0 else None


def list_assets(tag: str) -> list[str]:
    """Liste les assets d'une release."""
    result = _run(
        ["gh", "release", "view", tag, "--repo", REPO, "--json", "assets"],
        check=False,
    )
    if result.returncode != 0:
        return []
    data = json.loads(result.stdout)
    return [a["name"] for a in data.get("assets", [])]


def append_to_parquet(tag: str, asset_name: str, new_rows: pd.DataFrame) -> None:
    """Télécharge le parquet courant, append, et ré-uploade."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / asset_name
        existing = download_asset(tag, asset_name, tmp_path)
        if existing and existing.exists() and existing.stat().st_size > 0:
            df = pd.read_parquet(tmp_path)
            df = pd.concat([df, new_rows], ignore_index=True)
        else:
            df = new_rows
        df.to_parquet(tmp_path, compression="snappy", index=False)
        upload_asset(tag, tmp_path, asset_name)
