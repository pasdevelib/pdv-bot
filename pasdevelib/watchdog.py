"""watchdog.py — Vérifie que le dernier scrape date de moins de 15 min.

Lit snapshot_prev.parquet depuis la release live-data et vérifie le timestamp.
Si le dernier scrape date de plus de 15 min, log une alerte ET envoie un
email (voir BUG CORRIGE ICI ci-dessous).
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

from pasdevelib import storage


MAX_AGE_MINUTES = 15
ASSET = "snapshot_prev.parquet"

# BUG CORRIGE ICI (2026-09-27) : ce watchdog tourne toutes les 15 min
# depuis des mois et detectait deja tres correctement les trous de
# donnees (16 des 20 derniers runs en echec au moment du diagnostic de
# la regression du 27 juin 2026) — mais se contentait d'un `sys.exit(1)`
# qui fait juste passer le job GitHub Actions en rouge, sans aucune
# notification reelle. Avec ~38 workflows programmes dans ce depot, un
# job rouge de plus est invisible en pratique : la regression est restee
# non detectee pendant 3 mois. Le watchdog envoie desormais un vrai email
# via Resend (meme pattern que la webapp), avec un cooldown pour ne pas
# spammer toutes les 15 minutes pendant une panne prolongee, et un email
# de "retour a la normale" quand le scraper repart.
ALERT_COOLDOWN_MINUTES = 120
ALERT_STATE_ASSET = "watchdog_alert_state.txt"  # contient "ALERTING|<iso timestamp du dernier email>" ou "OK"


def send_alert_email(subject: str, html: str) -> None:
    resend_key = os.environ.get("RESEND_API_KEY")
    alert_to = os.environ.get("WATCHDOG_ALERT_EMAIL", "hello@pasdevelib.app")
    if not resend_key:
        print("[watchdog] RESEND_API_KEY absente — impossible d'envoyer l'alerte par email")
        return
    try:
        res = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
            json={
                "from": "PasDeVélib Watchdog <hello@pasdevelib.app>",
                "to": alert_to,
                "subject": subject,
                "html": html,
            },
            timeout=15,
        )
        if res.status_code >= 300:
            print(f"[watchdog] échec envoi email ({res.status_code}): {res.text[:200]}")
        else:
            print("[watchdog] email d'alerte envoyé")
    except Exception as e:
        print(f"[watchdog] erreur envoi email (non-bloquant): {e}")


def _read_alert_state(tmp_dir: Path) -> tuple[str, datetime | None]:
    path = tmp_dir / ALERT_STATE_ASSET
    downloaded = storage.download_asset(storage.RELEASE_LIVE, ALERT_STATE_ASSET, path)
    if not downloaded or not path.exists():
        return "OK", None
    content = path.read_text().strip()
    if content == "OK" or not content:
        return "OK", None
    try:
        _, ts_str = content.split("|", 1)
        return "ALERTING", datetime.fromisoformat(ts_str)
    except Exception:
        return "OK", None


def _write_alert_state(tmp_dir: Path, state: str, ts: datetime | None = None) -> None:
    path = tmp_dir / ALERT_STATE_ASSET
    content = "OK" if state == "OK" else f"ALERTING|{(ts or datetime.now(timezone.utc)).isoformat()}"
    path.write_text(content)
    storage.upload_asset(storage.RELEASE_LIVE, path, ALERT_STATE_ASSET)


def run() -> None:
    import pandas as pd

    now = datetime.now(timezone.utc)
    print(f"[watchdog] {now.isoformat()}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        path = tmp_dir / ASSET
        if not storage.download_asset(storage.RELEASE_LIVE, ASSET, path):
            print("[watchdog] ⚠️  snapshot_prev.parquet introuvable — scraper possiblement en panne")
            send_alert_email(
                "🚨 PasDeVélib — scraper introuvable",
                "<p>snapshot_prev.parquet est introuvable sur la release live. Le scraper semble ne jamais avoir tourné, ou la release a été supprimée.</p>",
            )
            sys.exit(1)

        df = pd.read_parquet(path)

        ts_col = None
        for col in ["fetched_at", "last_reported", "last_updated", "date"]:
            if col in df.columns:
                ts_col = col
                break

        if ts_col is None:
            print(f"[watchdog] colonnes: {list(df.columns)}")
            print("[watchdog] ⚠️  Aucune colonne timestamp trouvée")
            return

        last_ts = pd.to_datetime(df[ts_col]).max()
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        age_minutes = (now - last_ts).total_seconds() / 60
        print(f"[watchdog] Dernier scrape : {last_ts.isoformat()} ({age_minutes:.1f} min)")

        state, last_alert_ts = _read_alert_state(tmp_dir)

        if age_minutes > MAX_AGE_MINUTES:
            print(f"[watchdog] 🚨 ALERTE — dernier scrape il y a {age_minutes:.0f} min (seuil: {MAX_AGE_MINUTES} min)")
            cooldown_ok = (
                last_alert_ts is None
                or (now - last_alert_ts) >= timedelta(minutes=ALERT_COOLDOWN_MINUTES)
            )
            if state == "OK" or cooldown_ok:
                send_alert_email(
                    f"🚨 PasDeVélib — scraper en panne depuis {age_minutes:.0f} min",
                    f"<p>Le dernier scrape enregistré date de <b>{age_minutes:.0f} minutes</b> "
                    f"(dernier point : {last_ts.isoformat()}).</p>"
                    f"<p>Seuil d'alerte : {MAX_AGE_MINUTES} min. Vérifier les runs du workflow "
                    f"<code>scrape.yml</code> sur GitHub Actions.</p>",
                )
                _write_alert_state(tmp_dir, "ALERTING", now)
            sys.exit(1)
        else:
            print(f"[watchdog] ✅ OK — scraper actif")
            if state == "ALERTING":
                send_alert_email(
                    "✅ PasDeVélib — scraper de nouveau actif",
                    f"<p>Le scraper est de nouveau à jour (dernier point : {last_ts.isoformat()}, "
                    f"{age_minutes:.1f} min).</p>",
                )
                _write_alert_state(tmp_dir, "OK")


if __name__ == "__main__":
    run()
