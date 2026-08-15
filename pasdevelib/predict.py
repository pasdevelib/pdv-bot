"""Prediction module — V4.

V2 : Platt Scaling, pondération temporelle, shrinkage, k adaptatif
V3 : spatial layer géographique, profils de station, grèves/événements
V4 : tendance réseau global, flux graph (propagation spatiale par corrélation),
     profil de station dans la distance analogue, détection d'anomalies
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AnalogConfig:
    k: int = 7
    weight_temp: float = 1.0
    weight_rain: float = 2.0
    weight_dow: float = 3.0
    weight_holiday: float = 4.0
    weight_season: float = 1.5
    weight_disruption: float = 5.0
    weight_network: float = 0.0      # désactivé — trop peu de données 2026 (réactiver à 6+ mois)
    weight_station_type: float = 2.0  # profil bureau/résidentiel/touristique
    temporal_halflife_days: float = 365.0
    shrinkage_threshold: int = 3
    shrinkage_alpha: float = 0.3
    # Spatial layer géographique
    spatial_radius_m: float = 400.0
    spatial_k: int = 5
    spatial_weight: float = 0.2
    # Spatial layer par flux (graphe de corrélation)
    flux_weight: float = 0.0         # désactivé — corrélations instables avec < 6 mois de données
    flux_top_k: int = 3              # top K voisines par corrélation
    # Anomalie
    anomaly_confidence_penalty: float = 0.15  # réduction de confiance si anomalie
    platt_by_slot: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "morning":   (1.0, 0.0),
        "midday":    (1.0, 0.0),
        "peak":      (0.80, -0.15),
        "evening":   (0.90, -0.05),
        "night":     (1.0, 0.0),
    })


def _hour_to_slot(hour: int) -> str:
    if 6 <= hour <= 9: return "morning"
    if 10 <= hour <= 15: return "midday"
    if 16 <= hour <= 19: return "peak"
    if 20 <= hour <= 22: return "evening"
    return "night"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _platt_scale(proba: float, hour: int, cfg: AnalogConfig) -> float:
    slot = _hour_to_slot(hour)
    a, b = cfg.platt_by_slot.get(slot, (1.0, 0.0))
    if a == 1.0 and b == 0.0:
        return proba
    eps = 1e-6
    p = np.clip(proba, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    return float(_sigmoid(a * logit_p + b))


def _temporal_weight(date_str: str, today: dt.date, halflife_days: float) -> float:
    try:
        d = dt.date.fromisoformat(date_str)
        age_days = (today - d).days
        return float(np.exp(-age_days * np.log(2) / halflife_days))
    except Exception:
        return 1.0


def _row_distance(
    row_a: pd.Series,
    row_b: pd.Series,
    cfg: AnalogConfig,
    station_type_a: str | None = None,
) -> float:
    d = 0.0
    # Météo
    if pd.notna(row_a.get("temp_avg")) and pd.notna(row_b.get("temp_avg")):
        d += cfg.weight_temp * abs(row_a["temp_avg"] - row_b["temp_avg"]) / 30.0
    if pd.notna(row_a.get("mean_apparent_temperature")) and pd.notna(row_b.get("mean_apparent_temperature")):
        d += cfg.weight_temp * 0.5 * abs(row_a["mean_apparent_temperature"] - row_b["mean_apparent_temperature"]) / 30.0
    if pd.notna(row_a.get("precip_total")) and pd.notna(row_b.get("precip_total")):
        d += cfg.weight_rain * abs(row_a["precip_total"] - row_b["precip_total"]) / 20.0
    if pd.notna(row_a.get("precip_3h_max")) and pd.notna(row_b.get("precip_3h_max")):
        d += cfg.weight_rain * 0.8 * abs(row_a["precip_3h_max"] - row_b["precip_3h_max"]) / 10.0
    # Calendrier
    if row_a.get("day_of_week") != row_b.get("day_of_week"):
        d += cfg.weight_dow
    if row_a.get("is_holiday") != row_b.get("is_holiday"):
        d += cfg.weight_holiday
    if row_a.get("is_school_holiday") != row_b.get("is_school_holiday"):
        d += cfg.weight_holiday * 0.5
    if bool(row_a.get("is_disruption_day")) != bool(row_b.get("is_disruption_day")):
        d += cfg.weight_disruption
    elif bool(row_a.get("is_greve")) != bool(row_b.get("is_greve")):
        d += cfg.weight_disruption * 0.8
    # Tendance réseau global — NOUVEAU V4
    if pd.notna(row_a.get("network_empty_rate")) and pd.notna(row_b.get("network_empty_rate")):
        d += cfg.weight_network * abs(row_a["network_empty_rate"] - row_b["network_empty_rate"])
    # Saison
    if row_a.get("season") != row_b.get("season"):
        d += cfg.weight_season
    return d


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def find_spatial_neighbors(
    station_id: str,
    stations_coords: pd.DataFrame,
    radius_m: float = 400.0,
    max_k: int = 5,
) -> list[str]:
    row = stations_coords[stations_coords["station_id"] == station_id]
    if row.empty:
        return []
    lat, lon = float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])
    others = stations_coords[stations_coords["station_id"] != station_id].copy()
    others["dist_m"] = others.apply(
        lambda r: _haversine_m(lat, lon, r["lat"], r["lon"]), axis=1
    )
    nearby = others[others["dist_m"] <= radius_m].nsmallest(max_k, "dist_m")
    return list(nearby["station_id"])


def find_flux_neighbors(
    station_id: str,
    flux_graph: pd.DataFrame,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Retourne les K voisines les plus corrélées avec leur coefficient."""
    if flux_graph is None or flux_graph.empty:
        return []
    sub = flux_graph[flux_graph["station_a"] == station_id].nlargest(top_k, "correlation")
    return [(str(r["station_b"]), float(r["correlation"])) for _, r in sub.iterrows()]


def compute_station_profiles(hourly_history: pd.DataFrame) -> pd.DataFrame:
    df = hourly_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.dayofweek
    weekdays = df[df["weekday"] < 5]
    morning = weekdays[weekdays["hour"].isin([7, 8, 9])].groupby("station_id")["fill_rate"].mean()
    evening = weekdays[weekdays["hour"].isin([17, 18, 19])].groupby("station_id")["fill_rate"].mean()
    weekend = df[df["weekday"] >= 5].groupby("station_id")["fill_rate"].mean()
    weekday_all = weekdays.groupby("station_id")["fill_rate"].mean()
    profiles = pd.DataFrame({
        "fill_morning": morning,
        "fill_evening": evening,
        "fill_weekend": weekend,
        "fill_weekday": weekday_all,
    }).dropna()

    def classify(row: pd.Series) -> str:
        ratio_me = row["fill_morning"] / (row["fill_evening"] + 0.01)
        we_boost = row["fill_weekend"] / (row["fill_weekday"] + 0.01)
        if we_boost > 1.3: return "touristique"
        if ratio_me < 0.7: return "bureau"
        if ratio_me > 1.4: return "residentiel"
        return "mixte"

    profiles["station_type"] = profiles.apply(classify, axis=1)
    return profiles[["station_type"]].reset_index()


def detect_anomaly(
    station_id: str,
    hour: int,
    weekday: int,
    current_fill: float | None,
    anomaly_stats: pd.DataFrame | None,
) -> bool:
    """True si le fill_rate actuel est anormal (hors intervalle q05-q95)."""
    if anomaly_stats is None or current_fill is None:
        return False
    mask = (
        (anomaly_stats["station_id"].astype(str) == str(station_id)) &
        (anomaly_stats["hour"] == hour) &
        (anomaly_stats["weekday"] == weekday)
    )
    ref = anomaly_stats[mask]
    if ref.empty:
        return False
    q05 = float(ref.iloc[0]["q05_fill"])
    q95 = float(ref.iloc[0]["q95_fill"])
    return current_fill < q05 or current_fill > q95


def _adaptive_k(distances: "pd.Series", cfg: AnalogConfig) -> int:
    sorted_dists = distances.sort_values().values
    k_base = min(cfg.k, len(sorted_dists))
    if k_base <= 2:
        return k_base
    d0 = sorted_dists[0] + 1e-6
    k = k_base
    for i in range(k_base, min(len(sorted_dists), cfg.k * 2)):
        gap = (sorted_dists[i] - sorted_dists[i - 1]) / d0
        if gap > 0.20:
            break
        k = i + 1
    return k


def find_analog_days(
    target_features: pd.Series,
    candidates: pd.DataFrame,
    cfg: AnalogConfig | None = None,
    station_type: str | None = None,
) -> tuple[list[str], str]:
    cfg = cfg or AnalogConfig()
    levels = [
        ("L1 strict", cfg),
        ("L2 sans saison", AnalogConfig(k=cfg.k, weight_temp=cfg.weight_temp,
            weight_rain=cfg.weight_rain, weight_dow=cfg.weight_dow,
            weight_holiday=cfg.weight_holiday, weight_disruption=cfg.weight_disruption,
            weight_network=cfg.weight_network, weight_station_type=cfg.weight_station_type,
            weight_season=0)),
        ("L3 sans pluie", AnalogConfig(k=cfg.k, weight_temp=cfg.weight_temp,
            weight_rain=0, weight_dow=cfg.weight_dow, weight_holiday=cfg.weight_holiday,
            weight_disruption=cfg.weight_disruption, weight_network=cfg.weight_network,
            weight_station_type=0, weight_season=0)),
        ("L4 sans calendrier", AnalogConfig(k=cfg.k, weight_temp=cfg.weight_temp,
            weight_rain=0, weight_dow=0, weight_holiday=0, weight_disruption=0,
            weight_network=0, weight_station_type=0, weight_season=0)),
        ("L5 tout", AnalogConfig(k=max(20, cfg.k), weight_temp=0, weight_rain=0,
            weight_dow=0, weight_holiday=0, weight_disruption=0, weight_network=0,
            weight_station_type=0, weight_season=0)),
    ]
    for label, cur_cfg in levels:
        distances = candidates.apply(
            lambda r: _row_distance(target_features, r, cur_cfg, station_type), axis=1
        )
        if len(distances) == 0:
            continue
        k_adaptive = _adaptive_k(distances, cur_cfg)
        sorted_idx = distances.sort_values().index[:k_adaptive]
        if len(sorted_idx) >= 2:
            return list(candidates.loc[sorted_idx, "date"]), f"{label} (k={k_adaptive})"
    return list(candidates["date"].head(min(20, len(candidates)))), "L5 fallback"


def _count_recent_neighbors(analog_dates: list[str], today: dt.date, recency_days: int = 365) -> int:
    cutoff = today - dt.timedelta(days=recency_days)
    return sum(
        1 for d in analog_dates
        if _safe_date(d) is not None and _safe_date(d) >= cutoff
    )


def _safe_date(s: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        return None


def predict_day_with_quantiles(
    target_date: dt.date,
    target_features: pd.Series,
    calendar_df: pd.DataFrame,
    hourly_history: pd.DataFrame,
    cfg: AnalogConfig | None = None,
    stations_coords: pd.DataFrame | None = None,
    station_profiles: pd.DataFrame | None = None,
    flux_graph: pd.DataFrame | None = None,
    anomaly_stats: pd.DataFrame | None = None,
    network_trend: pd.DataFrame | None = None,
    current_fill_rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Prédit pour chaque (station, hour).

    V4 : tendance réseau, flux graph, profil dans distance, anomalies.
    """
    cfg = cfg or AnalogConfig()
    today = dt.date.today()

    # Enrichir candidates avec tendance réseau
    candidates = calendar_df[calendar_df["date"] != target_date.isoformat()].copy()

    # BUG CORRIGE ICI : candidates venait du calendrier NATIONAL partage
    # (2020 -> aujourd'hui, ~2100 jours), sans jamais etre restreint aux
    # dates REELLEMENT presentes dans hourly_history de CETTE ville.
    # Sans consequence pour Paris (historique large qui couvre la quasi
    # totalite du calendrier), mais catastrophique pour une ville avec
    # seulement quelques semaines d'historique : find_analog_days
    # choisissait les "meilleures" journees analogues sur TOUT le
    # calendrier (ex. un jour de mars 2023 au climat proche), puis
    # `sub = hourly_history[...isin(analog_dates)]` se retrouvait VIDE
    # puisque ces dates n'existaient simplement pas dans les quelques
    # semaines de donnees reelles de la ville — d'ou aucune prediction
    # generee du tout, silencieusement, pour toutes les villes
    # secondaires. Restreint desormais aux dates ou l'on a reellement de
    # la donnee pour cette ville avant de chercher les analogues.
    # BUG CORRIGE ICI (trouve en testant le correctif ci-dessus sur de
    # vraies donnees) : calendar_df["date"] contient des objets
    # datetime.date, hourly_history["date"] contient des chaines — .isin()
    # ne matchait donc JAMAIS, meme pour un jour strictement identique.
    # Normalise les deux cotes en chaines ISO avant comparaison.
    available_dates = set(hourly_history["date"].astype(str).unique())
    candidates["date"] = candidates["date"].astype(str)
    candidates = candidates[candidates["date"].isin(available_dates)]

    if network_trend is not None and "network_empty_rate" in network_trend.columns:
        # Ajouter la tendance réseau moyenne de la journée comme feature
        avg_network = network_trend.groupby("weekday")["network_empty_rate"].mean().to_dict()
        candidates["network_empty_rate"] = candidates["weekday"].map(avg_network)

    if len(candidates) == 0:
        return pd.DataFrame()

    # Enrichir target_features avec tendance réseau cible
    target_weekday = int(target_features.get("day_of_week", target_features.get("weekday", 0)))
    if network_trend is not None:
        avg_net = network_trend[network_trend["weekday"] == target_weekday]["network_empty_rate"].mean()
        target_features = target_features.copy()
        target_features["network_empty_rate"] = avg_net if pd.notna(avg_net) else 0.0

    # Index des profils de station
    profile_map: dict[str, str] = {}
    if station_profiles is not None:
        profile_map = dict(zip(
            station_profiles["station_id"].astype(str),
            station_profiles["station_type"]
        ))

    analog_dates, level = find_analog_days(target_features, candidates, cfg)
    n_recent = _count_recent_neighbors(analog_dates, today)
    print(f"[predict] {target_date.isoformat()} -> {level} : {len(analog_dates)} neighbors "
          f"({n_recent} récents)"
          + (" [GREVE]" if target_features.get("is_greve") else "")
          + (" [EVENT]" if target_features.get("is_event") else ""))

    sub = hourly_history[hourly_history["date"].isin(analog_dates)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["_w"] = sub["date"].apply(
        lambda d: _temporal_weight(d, today, cfg.temporal_halflife_days)
    )

    def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
        w = weights.loc[series.index]
        wsum = w.sum()
        if wsum == 0:
            return float(series.mean())
        return float((series * w).sum() / wsum)

    def weighted_quantile(series: pd.Series, weights: pd.Series, q: float) -> float:
        s = series.sort_values()
        w = weights.loc[s.index].values
        cumw = np.cumsum(w)
        cutoff = cumw[-1] * q
        idx = np.searchsorted(cumw, cutoff)
        return float(s.iloc[min(idx, len(s) - 1)])

    grouped_rows = []
    station_predictions: dict[str, dict[int, float]] = {}

    for (station_id, hour), grp in sub.groupby(["station_id", "hour"]):
        w = grp["_w"]
        p_velib_raw = weighted_mean(grp["has_velib"].astype(float), w)
        p_place_raw = weighted_mean(grp["has_place"].astype(float), w)

        clim_velib = float(grp["has_velib"].mean())
        clim_place = float(grp["has_place"].mean())

        alpha = min(1.0, n_recent / cfg.shrinkage_threshold) if cfg.shrinkage_threshold > 0 else 1.0
        alpha = max(cfg.shrinkage_alpha, alpha)

        p_velib_shrunk = alpha * p_velib_raw + (1 - alpha) * clim_velib
        p_place_shrunk = alpha * p_place_raw + (1 - alpha) * clim_place

        p_velib_cal = _platt_scale(p_velib_shrunk, int(hour), cfg)
        p_place_cal = _platt_scale(p_place_shrunk, int(hour), cfg)

        # Détection d'anomalie — NOUVEAU V4
        sid_str = str(station_id)
        current_fill = (current_fill_rates or {}).get(sid_str)
        weekday_now = today.weekday()
        is_anomaly = detect_anomaly(sid_str, int(hour), weekday_now, current_fill, anomaly_stats)
        if is_anomaly:
            # Réduire la confiance : tirer vers 0.5 (incertitude maximale)
            p_velib_cal = p_velib_cal * (1 - cfg.anomaly_confidence_penalty) + 0.5 * cfg.anomaly_confidence_penalty

        row = {
            "station_id": station_id,
            "hour": hour,
            "proba_velib": round(p_velib_cal, 4),
            "proba_ebike": round(float(grp["has_ebike"].mean()) if "has_ebike" in grp.columns else 0.0, 4),
            "proba_mechanical": round(float(grp["has_mechanical"].mean()) if "has_mechanical" in grp.columns else 0.0, 4),
            "proba_place": round(p_place_cal, 4),
            "p25_fill": weighted_quantile(grp["fill_rate"], w, 0.25),
            "p50_fill": weighted_quantile(grp["fill_rate"], w, 0.50),
            "p75_fill": weighted_quantile(grp["fill_rate"], w, 0.75),
            "p50_ebike": (weighted_quantile(grp["fill_rate_ebike"], w, 0.50) if "fill_rate_ebike" in grp.columns else 0.0),
            "p50_mechanical": (weighted_quantile(grp["fill_rate_mechanical"], w, 0.50) if "fill_rate_mechanical" in grp.columns else 0.0),
            "n_neighbors": len(grp),
            "n_recent_neighbors": n_recent,
            "is_anomaly": is_anomaly,
        }
        grouped_rows.append(row)

        if sid_str not in station_predictions:
            station_predictions[sid_str] = {}
        station_predictions[sid_str][int(hour)] = p_velib_cal

    if not grouped_rows:
        return pd.DataFrame()

    # ── Spatial layer géographique ────────────────────────────────────────────
    if stations_coords is not None and cfg.spatial_weight > 0:
        for row in grouped_rows:
            sid = str(row["station_id"])
            hour = int(row["hour"])
            neighbors = find_spatial_neighbors(sid, stations_coords, cfg.spatial_radius_m, cfg.spatial_k)
            neighbor_probas = [
                station_predictions[n][hour]
                for n in neighbors
                if n in station_predictions and hour in station_predictions[n]
            ]
            if neighbor_probas:
                spatial_mean = float(np.mean(neighbor_probas))
                row["proba_velib"] = round(
                    (1 - cfg.spatial_weight) * row["proba_velib"]
                    + cfg.spatial_weight * spatial_mean, 4
                )

    # ── Flux graph (propagation par corrélation) — NOUVEAU V4 ────────────────
    if flux_graph is not None and cfg.flux_weight > 0:
        for row in grouped_rows:
            sid = str(row["station_id"])
            hour = int(row["hour"])
            flux_neighbors = find_flux_neighbors(sid, flux_graph, cfg.flux_top_k)
            if not flux_neighbors:
                continue
            weighted_sum = 0.0
            weight_total = 0.0
            for neighbor_id, corr in flux_neighbors:
                if neighbor_id in station_predictions and hour in station_predictions[neighbor_id]:
                    w = abs(corr)
                    # Corrélation négative = flux inverse → si voisine se vide, on se remplit
                    contrib = station_predictions[neighbor_id][hour] if corr > 0 else (1 - station_predictions[neighbor_id][hour])
                    weighted_sum += w * contrib
                    weight_total += w
            if weight_total > 0:
                flux_mean = weighted_sum / weight_total
                row["proba_velib"] = round(
                    (1 - cfg.flux_weight) * row["proba_velib"]
                    + cfg.flux_weight * flux_mean, 4
                )

    grouped = pd.DataFrame(grouped_rows)
    grouped["prob_empty"] = 1.0 - grouped["proba_velib"]
    grouped["target_date"] = target_date.isoformat()
    grouped["analog_level"] = level
    # Transparence/explicabilite pour l'admin : les VRAIS jours utilises
    # pour cette prediction (partages entre toutes les stations de ce
    # target_date — voir le commentaire plus haut sur pourquoi
    # find_analog_days n'est appele qu'une fois par jour cible, pas par
    # station). Rejoint en une seule chaine pour rester compatible avec
    # le format parquet en colonnes.
    grouped["analog_dates"] = ",".join(sorted(analog_dates))

    return grouped
