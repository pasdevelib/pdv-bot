# pdv-bot

Bot de collecte et de prédiction Vélib' pour [pasdevelib.app](https://pasdevelib.app) — **Paris uniquement**. Les 7 autres villes (Bordeaux, Lille, Lyon, Montpellier, Nantes, Rennes, Strasbourg, Toulouse) sont gérées par [`pasdevelib/pdvr-bot`](https://github.com/pasdevelib/pdvr-bot), qui installe ce paquet à distance (`pip install git+https://github.com/pasdevelib/pdv-bot.git`) — aucun code dupliqué entre les deux dépôts.

Aucune base de données : tout est lu/écrit sur des **GitHub Releases** (parquet + json), consommées directement par [`pasdevelib-webapp`](https://github.com/pasdevelib/pasdevelib-webapp) via de simples requêtes HTTP en ISR.

## Pourquoi deux dépôts (contexte, 2026-09-28)

GitHub Actions retarde et laisse tomber la plupart des déclenchements `schedule` quand un dépôt en cumule trop, surtout à haute fréquence (`*/5 * * * *`) — ce qui a provoqué une régression de 3 mois sur le scraping Paris, découverte fin septembre 2026. Isoler les villes en région dans `pdvr-bot` donne à Paris son propre quota de déclenchements programmés, indépendant du reste. Détails dans le README de `pdvr-bot`.

## Pipeline

```
scrape.yml (*/5 min, boucle interne 30s)
   └─ current_day.parquet, stations.json, snapshot_prev.parquet   [release: live]

consolidate.yml (quotidien)
   └─ hourly_history.parquet                                      [release: aggregates]

aggregate.yml (quotidien)
   └─ medians.parquet, weather.parquet, calendar.parquet           [release: aggregates]

forecast.yml (quotidien)
   └─ forecast_7d.parquet, forecast_explain_paris.json             [release: aggregates]

stats-cities.yml (quotidien, TOUTES les villes y compris Paris)
   └─ stats_<ville>_<periode>.json, evolution_<ville>.json, etc.   [release: stats-cities]
   (lit les données villes dans pdvr-bot, en anonyme — pdvr-bot est public)

daily-digest.yml (quotidien, TOUTES les villes)
   └─ bilan du jour rédigé par Gemini, à partir de stats-cities     [release: daily-digest]
   → consommé par blog.pasdevelib.app/blog/articles/dailymonitoring/<ville>

watchdog.yml (*/15 min)
   └─ alerte email (Resend) si le dernier scrape date de +15 min
```

## Convention de stockage (`pasdevelib/storage.py`)

| Release      | Contenu                                                              |
|--------------|-----------------------------------------------------------------------|
| `live`       | `current_day.parquet`, `stations.json`, `snapshot_prev.parquet`      |
| `history`    | `YYYY-MM-DD.parquet` (un par jour)                                    |
| `aggregates` | `medians.parquet`, `weather.parquet`, `calendar.parquet`, `forecast_7d.parquet`, `hourly_history.parquet` |
| `stats-cities` | Classements/analyses par ville, **8 villes** (Paris inclus)         |
| `daily-digest` | Bilans quotidiens rédigés par IA, **8 villes**                      |

## Pages consommant ces données

- `blog.pasdevelib.app` : `/donnees?ville=<id>&periode=week` (dashboard chiffré), `/blog/articles/dailymonitoring/<ville>` (bilan quotidien narratif), `/paris`, `/bordeaux`, `/lyon`, `/lille`, `/rennes`, `/strasbourg`, `/toulouse` (pages vitrine par ville)
- `api.pasdevelib.app` : API développeur publique (clé requise), cf. `pasdevelib-webapp/app/api/v1/`
- `statut.pasdevelib.app` : santé du pipeline (`/api/statut`, lit directement la release `live`)

## Setup

```bash
pip install -e .
```

Secrets requis (GitHub Actions → Settings → Secrets) : `RESEND_API_KEY` (alertes watchdog), `GEMINI_API_KEY` (daily-digest). `GH_TOKEN` est le token par défaut d'Actions (`contents: write` sur ce dépôt) — aucun PAT dédié nécessaire.
