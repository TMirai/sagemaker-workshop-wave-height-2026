#!/usr/bin/env python3
"""擬似データが実データの統計的性質を再現できているかを検証する。

generate_synthetic_data.py が出力した CSV を、実データの目標値と比較する。
実データの CSV を --reference で渡すと、実測値との並列比較も表示する。

使い方:
    python validate_data.py --input ./data/observation.csv
    python validate_data.py --input ./data/observation.csv --reference ./data/real.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 実データを解析して得た目標値と、許容範囲
#   key: (目標値, 許容誤差)
# ---------------------------------------------------------------------------
TARGETS: dict[str, tuple[float, float]] = {
    "corr_wind_wave": (0.72, 0.06),
    "wave_height_mean": (0.212, 0.030),
    "wave_height_std": (0.108, 0.030),
    "wind_speed_mean": (5.26, 0.60),
    "wind_speed_std": (3.16, 0.50),
    "wave_height_skew": (1.93, 0.90),
    "wind_speed_skew": (0.96, 0.40),
    "wave_autocorr_1h": (0.899, 0.06),
    "wave_autocorr_2h": (0.812, 0.09),
    "wave_autocorr_3h": (0.719, 0.11),
    "wave_autocorr_6h": (0.471, 0.15),
    "wind_autocorr_1h": (0.883, 0.06),
    "wind_autocorr_3h": (0.708, 0.11),
    "r2_time_split": (0.502, 0.12),
    "r2_random_split": (0.704, 0.12),
}

DIRECTIONS_16 = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]
DIR_TO_DEGREE = {name: index * 22.5 for index, name in enumerate(DIRECTIONS_16)}


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """学習に使う特徴量を組み立てる（前処理と同じロジック）。"""
    frame = frame.dropna(subset=["wind_speed", "wind_dir", "wave_height"]).copy()
    frame = frame[frame["wind_dir"].isin(DIR_TO_DEGREE)]
    degrees = frame["wind_dir"].map(DIR_TO_DEGREE)
    frame["wind_sin"] = np.sin(np.radians(degrees))
    frame["wind_cos"] = np.cos(np.radians(degrees))
    frame["month"] = frame["timestamp"].dt.month
    return frame


def autocorrelation(series: pd.Series, lag: int) -> float:
    """欠測を含む系列の自己相関を計算する（時刻順に並んでいる前提）。"""
    values = series.to_numpy(dtype=float)
    first, second = values[:-lag], values[lag:]
    mask = np.isfinite(first) & np.isfinite(second)
    if mask.sum() < 100:
        return float("nan")
    return float(np.corrcoef(first[mask], second[mask])[0, 1])


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """検証に使う統計量をまとめて計算する。"""
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    metrics: dict[str, float] = {}

    wave = frame["wave_height"].dropna()
    wind = frame["wind_speed"].dropna()
    metrics["wave_height_mean"] = float(wave.mean())
    metrics["wave_height_std"] = float(wave.std())
    metrics["wave_height_skew"] = float(wave.skew())
    metrics["wind_speed_mean"] = float(wind.mean())
    metrics["wind_speed_std"] = float(wind.std())
    metrics["wind_speed_skew"] = float(wind.skew())

    for lag in (1, 2, 3, 6):
        metrics[f"wave_autocorr_{lag}h"] = autocorrelation(frame["wave_height"], lag)
    for lag in (1, 3):
        metrics[f"wind_autocorr_{lag}h"] = autocorrelation(frame["wind_speed"], lag)

    valid = frame.dropna(subset=["wind_speed", "wave_height"])
    metrics["corr_wind_wave"] = float(
        valid["wind_speed"].corr(valid["wave_height"])
    )

    # --- XGBoost による R²（時系列分割 / ランダム分割）--------------------
    try:
        import xgboost as xgb
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        print(
            "! xgboost / scikit-learn が無いため R² の検証をスキップします",
            file=sys.stderr,
        )
        return metrics

    features = build_features(frame)
    columns = ["wind_speed", "wind_sin", "wind_cos", "month"]
    params = dict(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    years = features["timestamp"].dt.year
    split_year = int(years.min()) + 2  # 先頭 3 年を学習に使う
    train = features[years <= split_year]
    test = features[years > split_year]
    if len(train) > 100 and len(test) > 100:
        model = xgb.XGBRegressor(**params)
        model.fit(train[columns], train["wave_height"])
        metrics["r2_time_split"] = float(
            r2_score(test["wave_height"], model.predict(test[columns]))
        )

    test_ratio = len(test) / len(features) if len(features) else 0.3
    x_train, x_test, y_train, y_test = train_test_split(
        features[columns],
        features["wave_height"],
        test_size=max(min(test_ratio, 0.5), 0.1),
        random_state=42,
    )
    model = xgb.XGBRegressor(**params)
    model.fit(x_train, y_train)
    metrics["r2_random_split"] = float(r2_score(y_test, model.predict(x_test)))

    return metrics


def report(
    metrics: dict[str, float], reference: dict[str, float] | None = None
) -> bool:
    """目標値との比較結果を表示し、全項目が許容範囲内かを返す。"""
    print()
    print("=" * 78)
    print("統計的性質の検証")
    print("=" * 78)
    header = f"{'項目':24s} {'生成データ':>10s} {'目標値':>10s} {'許容':>7s}"
    if reference:
        header += f" {'実データ':>10s}"
    header += "  判定"
    print(header)
    print("-" * 78)

    all_passed = True
    for name, (target, tolerance) in TARGETS.items():
        value = metrics.get(name)
        if value is None or not np.isfinite(value):
            print(f"{name:24s} {'—':>10s} {target:10.3f} {tolerance:7.3f}  スキップ")
            continue
        passed = abs(value - target) <= tolerance
        all_passed = all_passed and passed
        line = f"{name:24s} {value:10.3f} {target:10.3f} {tolerance:7.3f}"
        if reference:
            ref_value = reference.get(name, float("nan"))
            line += f" {ref_value:10.3f}"
        line += f"  {'✅' if passed else '❌'}"
        print(line)

    print("-" * 78)
    print("総合判定:", "✅ すべて許容範囲内" if all_passed else "❌ 範囲外の項目あり")
    return all_passed


def report_drift(frame: pd.DataFrame) -> None:
    """Lab4 のドリフト検知デモに使う期間の分布差を表示する。"""
    try:
        import xgboost as xgb
        from scipy import stats
        from sklearn.metrics import r2_score
    except ImportError:
        return

    features = build_features(frame)
    columns = ["wind_speed", "wind_sin", "wind_cos", "month"]
    years = features["timestamp"].dt.year
    baseline_year = int(years.min()) + 2
    baseline = features[years <= baseline_year]

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42
    )
    model.fit(baseline[columns], baseline["wave_height"])

    print()
    print("=" * 78)
    print(f"Lab4 ドリフト検知デモ用の期間比較（ベースライン: {int(years.min())}〜{baseline_year} 年）")
    print("=" * 78)
    print(
        f"{'期間':22s} {'行数':>7s} {'風速KS':>8s} {'波高KS':>8s} {'R²':>8s}"
    )
    print("-" * 78)

    windows = [
        ("正常系 2024/1-8", "2024-01-01", "2024-08-31"),
        ("中間  2025/4-9", "2025-04-01", "2025-09-30"),
        ("異常系 2025/10-12", "2025-10-01", "2025-12-31"),
    ]
    for label, start, end in windows:
        window = features[
            (features["timestamp"] >= start)
            & (features["timestamp"] <= f"{end} 23:59:59")
        ]
        if len(window) < 50:
            print(f"{label:22s} {len(window):7,}  （データ不足）")
            continue
        ks_wind = stats.ks_2samp(baseline["wind_speed"], window["wind_speed"]).statistic
        ks_wave = stats.ks_2samp(
            baseline["wave_height"], window["wave_height"]
        ).statistic
        r2 = r2_score(window["wave_height"], model.predict(window[columns]))
        print(
            f"{label:22s} {len(window):7,} {ks_wind:8.3f} {ks_wave:8.3f} {r2:8.3f}"
        )

    print("-" * 78)
    print("狙い: 正常系は R² が高くドリフト小、異常系は波高 KS が大きく R² が悪化")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="擬似データの統計的性質を実データの目標値と比較検証する"
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="検証対象の CSV（擬似データ）"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="比較用の実データ CSV（任意）",
    )
    parser.add_argument(
        "--skip-drift", action="store_true", help="ドリフト期間の比較を省略する"
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input, parse_dates=["timestamp"])
    print(f"検証対象: {args.input}（{len(frame):,} 行）")
    metrics = compute_metrics(frame)

    reference_metrics = None
    if args.reference:
        reference_frame = pd.read_csv(args.reference, parse_dates=["timestamp"])
        print(f"比較対象: {args.reference}（{len(reference_frame):,} 行）")
        reference_metrics = compute_metrics(reference_frame)

    passed = report(metrics, reference_metrics)

    if not args.skip_drift:
        report_drift(frame)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
