#!/usr/bin/env python3
"""ワークショップ用の擬似（合成）海象観測データを生成する。

実データ（東京都港湾局「東京港波浪観測所・海象観測データ」）の統計的性質を
再現した合成データを生成する。実データはオープンデータライセンスではなく
配布に許諾が必要なため、ワークショップでは本スクリプトの出力を使用する。

出力スキーマは load_real_data.py と同一。したがって下流の処理
（前処理・学習・デプロイ・モニタリング）は両者で共通に動作する。

再現する統計的性質（実データを解析して得た目標値）:
    corr(wind_speed, wave_height)   0.72
    wave_height  mean/std           0.212 / 0.108 m   skew 1.93
    wind_speed   mean/std           5.26 / 3.16 m/s   skew 0.96
    自己相関 wave_height            1h=0.899 2h=0.812 3h=0.719 6h=0.471
    自己相関 wind_speed             1h=0.883 2h=0.792 3h=0.708 6h=0.482
    XGBoost R²（時系列分割）        0.50 前後
    XGBoost R²（ランダム分割）      0.70 前後
    年ごとの分布変化                2025 年後半に波高が低下（ドリフト検知デモ用）
    欠測パターン                    2024 年 9〜12 月・2025 年 1〜4 月に風向風速欠測

生成の考え方:
    1. 風速   : 自己相関を持つ潜在ガウス過程 → ガンマ分布へ分位点変換
    2. 風向   : 季節性（冬は北風、夏は南風）＋ 時間的な持続性
    3. 波高   : 風速のべき乗則（風波成分）＋ うねり成分（風とは独立で自己相関を
                持つ）＋ 観測ノイズ。うねり成分があることで「同時刻の風だけでは
                変動の約半分しか説明できない」という実データの性質が再現される
    4. その他 : 周期・最高波・流速・潮位（調和成分）を波高・風速から派生

使い方:
    python generate_synthetic_data.py --output ./data/observation.csv
    python generate_synthetic_data.py --start-year 2021 --end-year 2025 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 出力スキーマ（load_real_data.py と一致させること）
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = [
    "timestamp",
    "wave_height_max",
    "wave_period_max",
    "wave_height",
    "wave_period",
    "wave_dir",
    "current_dir",
    "current_speed",
    "wind_dir",
    "wind_speed",
    "tide_level",
]

# 16 方位（時計回り、N から）
DIRECTIONS_16 = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

# 実データの風向出現割合（%）。季節性の重みづけの基準に使う
REAL_WIND_DIR_SHARE = {
    "N": 14.54, "NNE": 9.87, "NE": 5.95, "ENE": 5.38,
    "E": 4.02, "ESE": 5.69, "SE": 5.51, "SSE": 5.16,
    "S": 17.49, "SSW": 5.80, "SW": 2.44, "WSW": 1.83,
    "W": 1.24, "WNW": 1.30, "NW": 2.82, "NNW": 10.95,
}

# --- 生成パラメータ（validate_data.py の検証結果に基づき調整済み）-----------
WIND_AR_PHI = 0.885          # 風速の潜在過程の 1 次自己回帰係数
WIND_GAMMA_SHAPE = 2.78      # 風速のガンマ分布 形状パラメータ
WIND_GAMMA_SCALE = 1.90      # 風速のガンマ分布 尺度パラメータ
WIND_DIR_PERSIST = 0.88      # 風向が前時刻と同じ方向に留まる強さ

WAVE_BASE = 0.040            # 背景波高（常時存在する残存うねり）
WAVE_WIND_COEF = 0.0065      # 風波成分の係数
WAVE_WIND_POWER = 1.50       # 風波成分のべき指数（風速^この値）
SWELL_AR_PHI = 0.955         # うねり成分の自己回帰係数（風より持続する）
SWELL_SCALE = 0.070          # うねり成分の大きさ
SWELL_SIGMA = 0.70           # うねり成分の対数正規の形状（裾の重さ＝時折の大波）
NOISE_SCALE = 0.012          # 観測ノイズの大きさ
WAVE_FLOOR = 0.04            # 波高の下限（実データの最小値）

# 風向ごとの波の立ちやすさ（東京湾内・観測点の地形を模した係数）
# 実データでは風向の東西成分（cos）の寄与が大きかったため、
# 南〜南南西の風で波が立ちやすく、西〜北西の風では立ちにくい形にする
WAVE_DIR_FACTOR = {
    "N": 0.82, "NNE": 0.88, "NE": 0.95, "ENE": 1.02,
    "E": 1.08, "ESE": 1.14, "SE": 1.20, "SSE": 1.26,
    "S": 1.30, "SSW": 1.24, "SW": 1.10, "WSW": 0.92,
    "W": 0.78, "WNW": 0.72, "NW": 0.70, "NNW": 0.75,
}

# 年ごとの波高スケール（ドリフト検知デモ用。実データの年別平均を模す）
YEAR_WAVE_SCALE = {
    2021: 1.05,
    2022: 1.00,
    2023: 1.02,
    2024: 0.97,
    2025: 0.92,
}
# 2025 年 10〜12 月はさらに波高が下がる（モニタリングの「異常系」で使う強いドリフト）
DRIFT_PERIOD = {"year": 2025, "months": (10, 11, 12), "wave_scale": 0.62}

# 風向風速の欠測期間（実データの機器欠測を模す）
WIND_MISSING_PERIODS = [
    ("2024-09-02", "2024-12-31"),
    ("2025-01-01", "2025-04-15"),
]
# 上記以外にも散発的な欠測を入れる割合
SPORADIC_MISSING_RATE = {
    "wind": 0.004,
    "wave": 0.022,
    "current": 0.020,
    "tide": 0.003,
}

# 潮位の調和成分（振幅 cm, 周期 h）。M2・S2・K1・O1 を模した簡易モデル
TIDE_COMPONENTS = [
    (52.0, 12.4206),  # M2 主太陰半日周潮
    (24.0, 12.0000),  # S2 主太陽半日周潮
    (17.0, 23.9345),  # K1 日周潮
    (13.0, 25.8193),  # O1 日周潮
]
TIDE_MEAN = 118.0


def _ar1_series(length: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """定常分散 1 の 1 次自己回帰系列を生成する。"""
    innovation_scale = np.sqrt(1.0 - phi**2)
    noise = rng.standard_normal(length) * innovation_scale
    series = np.empty(length)
    series[0] = rng.standard_normal()
    for index in range(1, length):
        series[index] = phi * series[index - 1] + noise[index]
    return series


def _gaussian_to_gamma(values: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """標準正規の系列をガンマ分布へ分位点変換する（自己相関を保つ）。"""
    from scipy import stats

    uniform = stats.norm.cdf(values)
    # 端点で inf にならないようクリップする
    uniform = np.clip(uniform, 1e-9, 1 - 1e-9)
    return stats.gamma.ppf(uniform, a=shape, scale=scale)


def _generate_wind_direction(
    timestamps: pd.DatetimeIndex, rng: np.random.Generator
) -> np.ndarray:
    """季節性と持続性を持つ風向系列を生成する。

    東京湾は冬季に北〜北北西、夏季に南の風が卓越する。実データの出現割合を
    基準に、季節で重みを変えたうえで前時刻の風向に留まりやすくする。
    """
    base_weight = np.array(
        [REAL_WIND_DIR_SHARE[name] for name in DIRECTIONS_16], dtype=float
    )
    base_weight /= base_weight.sum()

    # 冬（12〜2 月）は北成分、夏（6〜8 月）は南成分を強調する
    north_boost = np.array(
        [2.0, 1.7, 1.2, 0.9, 0.7, 0.6, 0.5, 0.5,
         0.4, 0.5, 0.7, 0.9, 1.2, 1.6, 1.8, 1.9]
    )
    south_boost = np.array(
        [0.4, 0.5, 0.7, 0.9, 1.1, 1.4, 1.7, 1.9,
         2.0, 1.8, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4]
    )

    months = timestamps.month.to_numpy()
    # 1 月を北のピーク、7 月を南のピークとする滑らかな季節係数
    season = np.cos((months - 1) / 12.0 * 2 * np.pi)  # 1月:+1, 7月:-1
    directions = np.empty(len(timestamps), dtype=object)

    previous_index = int(rng.choice(len(DIRECTIONS_16), p=base_weight))
    for step in range(len(timestamps)):
        blend = (1 + season[step]) / 2  # 1月:1.0（北寄り）, 7月:0.0（南寄り）
        weight = base_weight * (north_boost**blend) * (south_boost ** (1 - blend))

        # 前時刻の風向とその隣接方位に留まりやすくする
        persistence = np.zeros(len(DIRECTIONS_16))
        for offset, bonus in ((0, 1.0), (-1, 0.45), (1, 0.45), (-2, 0.15), (2, 0.15)):
            persistence[(previous_index + offset) % len(DIRECTIONS_16)] += bonus
        weight = weight * (1 - WIND_DIR_PERSIST) + persistence * WIND_DIR_PERSIST

        weight /= weight.sum()
        previous_index = int(rng.choice(len(DIRECTIONS_16), p=weight))
        directions[step] = DIRECTIONS_16[previous_index]

    return directions


def _apply_missing(
    frame: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """実データを模した欠測を適用する。"""
    frame = frame.copy()

    # 1) 機器欠測（風向・風速がまとまって落ちる期間）
    for start, end in WIND_MISSING_PERIODS:
        mask = (frame["timestamp"] >= start) & (frame["timestamp"] <= f"{end} 23:59:59")
        frame.loc[mask, ["wind_dir", "wind_speed"]] = np.nan

    # 2) 散発的な欠測
    size = len(frame)
    wind_mask = rng.random(size) < SPORADIC_MISSING_RATE["wind"]
    frame.loc[wind_mask, ["wind_dir", "wind_speed"]] = np.nan

    wave_mask = rng.random(size) < SPORADIC_MISSING_RATE["wave"]
    frame.loc[
        wave_mask,
        ["wave_height", "wave_period", "wave_height_max", "wave_period_max"],
    ] = np.nan

    current_mask = rng.random(size) < SPORADIC_MISSING_RATE["current"]
    frame.loc[current_mask, ["current_dir", "current_speed"]] = np.nan

    tide_mask = rng.random(size) < SPORADIC_MISSING_RATE["tide"]
    frame.loc[tide_mask, "tide_level"] = np.nan

    # 3) 波向は有義波高 25cm 未満では観測値を掲載しない（実データの仕様）
    frame.loc[frame["wave_height"] < 0.25, "wave_dir"] = np.nan
    frame.loc[frame["wave_height"].isna(), "wave_dir"] = np.nan

    return frame


def generate(
    start_year: int = 2021, end_year: int = 2025, seed: int = 42
) -> pd.DataFrame:
    """擬似観測データを生成して DataFrame で返す。"""
    rng = np.random.default_rng(seed)

    # 実データと同じく「時刻 01〜24」を正時として並べる
    timestamps = pd.date_range(
        start=f"{start_year}-01-01 01:00:00",
        end=f"{end_year + 1}-01-01 00:00:00",
        freq="h",
    )
    size = len(timestamps)

    # --- 風速 -------------------------------------------------------------
    wind_latent = _ar1_series(size, WIND_AR_PHI, rng)
    wind_speed = _gaussian_to_gamma(
        wind_latent, WIND_GAMMA_SHAPE, WIND_GAMMA_SCALE
    )
    wind_speed = np.round(np.clip(wind_speed, 0.0, 25.0), 1)

    # --- 風向 -------------------------------------------------------------
    wind_dir = _generate_wind_direction(timestamps, rng)
    dir_factor = np.array([WAVE_DIR_FACTOR[name] for name in wind_dir])

    # --- 波高 -------------------------------------------------------------
    # 風波成分: 風速のべき乗則。風速が上がると波高が非線形に伸びる
    wind_wave = WAVE_WIND_COEF * (wind_speed**WAVE_WIND_POWER) * dir_factor

    # うねり成分: 風とは独立。風より持続時間が長い（自己相関が高い）。
    # これが「同時刻の風だけでは説明できない残り半分」に相当する。
    # 対数正規にすることで、実データと同じく時折大きな波が現れる裾の重い分布になる
    swell_latent = _ar1_series(size, SWELL_AR_PHI, rng)
    swell = np.exp(swell_latent * SWELL_SIGMA) * SWELL_SCALE

    noise = rng.standard_normal(size) * NOISE_SCALE

    wave_height = WAVE_BASE + wind_wave + swell + noise

    # 年・期間ごとのスケール（ドリフト）を適用する
    years = timestamps.year.to_numpy()
    months = timestamps.month.to_numpy()
    scale = np.ones(size)
    for year, year_scale in YEAR_WAVE_SCALE.items():
        scale[years == year] = year_scale
    drift_mask = (years == DRIFT_PERIOD["year"]) & np.isin(
        months, DRIFT_PERIOD["months"]
    )
    scale[drift_mask] = DRIFT_PERIOD["wave_scale"]
    wave_height = wave_height * scale

    wave_height = np.round(np.clip(wave_height, WAVE_FLOOR, 2.0), 2)

    # --- 有義波周期 -------------------------------------------------------
    # 実データでは波高との相関は弱い（0.058）。うねりの周期に引っ張られるため
    period_latent = _ar1_series(size, 0.90, rng)
    wave_period = 2.82 + period_latent * 0.30 + (wave_height - 0.21) * 0.18
    wave_period = np.round(np.clip(wave_period, 1.7, 5.0), 1)

    # --- 最高波（有義波の約 1.9 倍）--------------------------------------
    ratio = 1.87 + rng.standard_normal(size) * 0.16
    wave_height_max = np.round(
        np.clip(wave_height * np.clip(ratio, 1.3, 3.0), WAVE_FLOOR, 3.5), 2
    )
    period_max_latent = rng.standard_normal(size)
    wave_period_max = np.round(
        np.clip(wave_period + period_max_latent * 0.58, 1.0, 10.0), 1
    )

    # --- 流向・流速（波高とはほぼ無相関）--------------------------------
    current_latent = _ar1_series(size, 0.80, rng)
    current_speed = np.round(
        np.clip(7.3 + current_latent * 4.6 + rng.standard_normal(size) * 1.2, 0.0, 50.0),
        1,
    )
    current_dir = _generate_wind_direction(timestamps, rng)

    # --- 潮位（調和成分の重ね合わせ）------------------------------------
    hours_from_start = (
        (timestamps - timestamps[0]).total_seconds() / 3600.0
    ).to_numpy()
    tide = np.full(size, TIDE_MEAN, dtype=float)
    for amplitude, period_hours in TIDE_COMPONENTS:
        phase = rng.uniform(0, 2 * np.pi)
        tide += amplitude * np.sin(2 * np.pi * hours_from_start / period_hours + phase)
    tide += rng.standard_normal(size) * 4.0
    tide_level = np.round(np.clip(tide, -60.0, 280.0), 0)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "wave_height_max": wave_height_max,
            "wave_period_max": wave_period_max,
            "wave_height": wave_height,
            "wave_period": wave_period,
            "wave_dir": _generate_wind_direction(timestamps, rng),
            "current_dir": current_dir,
            "current_speed": current_speed,
            "wind_dir": wind_dir,
            "wind_speed": wind_speed,
            "tide_level": tide_level,
        },
        columns=OUTPUT_COLUMNS,
    )

    return _apply_missing(frame, rng)


def summarize(frame: pd.DataFrame) -> None:
    """生成結果の要約を表示する。"""
    print()
    print("=" * 62)
    print("生成結果")
    print("=" * 62)
    print(f"行数        : {len(frame):,}")
    print(f"期間        : {frame['timestamp'].min()} 〜 {frame['timestamp'].max()}")
    print()
    print("列ごとの欠測状況:")
    for column in frame.columns:
        if column == "timestamp":
            continue
        missing = frame[column].isna().sum()
        print(
            f"  {column:16s} 欠測 {missing:6,} 件 "
            f"({missing / len(frame) * 100:5.1f}%)"
        )

    valid = frame.dropna(subset=["wind_speed", "wind_dir", "wave_height"])
    print()
    print(f"風向・風速・波高がすべて有効な行: {len(valid):,}")
    print(
        "corr(wind_speed, wave_height) = "
        f"{valid['wind_speed'].corr(valid['wave_height']):.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ワークショップ用の擬似海象観測データを生成する"
    )
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/observation.csv"),
        help="出力先 CSV のパス（既定: data/observation.csv）",
    )
    args = parser.parse_args()

    print(f"擬似データを生成します（{args.start_year}〜{args.end_year} 年, seed={args.seed}）")
    frame = generate(args.start_year, args.end_year, args.seed)
    summarize(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print()
    print(f"✅ 出力しました: {args.output}")
    print()
    print("これは合成データです。実データではありません。")
    print("統計的性質は東京都港湾局の公開観測データを参考に調整しています。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
