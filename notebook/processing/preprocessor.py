"""SageMaker Processing ジョブ内で実行される前処理スクリプト。

東京港（擬似）観測データから、風向・風速→有義波波高の回帰モデル用の
学習・検証データセットを作成する。

コンテナの契約:
    入力  : /opt/ml/processing/input/observation.csv
    出力  : /opt/ml/processing/train/{train_features.csv, train_labels.csv}
            /opt/ml/processing/validation/{val_features.csv, val_labels.csv}
            /opt/ml/processing/baseline/baseline.csv  （モニタリング（ドリフト検知）で使う参照データ）

前処理の内容:
    1. 風向・風速・有義波波高が欠測している行を除外する
       （風向風速センサーの欠測期間はモデルの入出力が成立しないため）
    2. 風向（16 方位）を周期特徴 sin/cos に変換する
       （N と NNW は角度としては隣接しているが、文字列や連番エンコードでは
       その隣接関係が表現できないため、三角関数で円環構造を表現する）
    3. 月を特徴量に加える（季節性）
    4. 時系列分割で学習/検証データを分ける
       （2021〜2023 年を学習、2024〜2025 年を検証。ランダム分割はしない。
       理由: 有義波波高の自己相関が高く［1 時間後で 0.90 前後］、ランダム分割では
       学習データと検証データが実質的に重複し、精度が見かけ上高く出てしまうため）

出力される特徴量の列順序（ラベルなしの X）:
    wind_speed, wind_sin, wind_cos, month
出力されるラベル（y）:
    wave_height
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# 16 方位 → 角度（度）。generate_synthetic_data.py / load_real_data.py と揃える
DIRECTIONS_16 = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]
DIR_TO_DEGREE = {name: index * 22.5 for index, name in enumerate(DIRECTIONS_16)}

FEATURE_COLUMNS = ["wind_speed", "wind_sin", "wind_cos", "month"]
LABEL_COLUMN = "wave_height"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2023,
        help="学習データに含める最終年（この年まで学習、翌年以降を検証に使う）",
    )
    args, _ = parser.parse_known_args()
    return args


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """風向・風速・タイムスタンプから学習用の特徴量列を組み立てる。"""
    frame = frame.copy()
    frame = frame[frame["wind_dir"].isin(DIR_TO_DEGREE)]
    degrees = frame["wind_dir"].map(DIR_TO_DEGREE)
    frame["wind_sin"] = np.sin(np.radians(degrees))
    frame["wind_cos"] = np.cos(np.radians(degrees))
    frame["month"] = frame["timestamp"].dt.month
    return frame


def main() -> None:
    args = parse_args()

    input_path = os.path.join("/opt/ml/processing/input", "observation.csv")
    print(f"Reading input data from {input_path}")
    frame = pd.read_csv(input_path, parse_dates=["timestamp"])
    print(f"Loaded {len(frame):,} rows")

    # (1) 風向・風速・波高のいずれかが欠測している行を除外する
    frame = frame.dropna(subset=["wind_dir", "wind_speed", "wave_height"])
    print(f"After dropping missing wind/wave rows: {len(frame):,} rows")

    # (2)(3) 特徴量を組み立てる
    frame = build_features(frame)

    # (4) 時系列分割: train_end_year 以前を学習、それより後を検証に使う
    years = frame["timestamp"].dt.year
    train_frame = frame[years <= args.train_end_year]
    validation_frame = frame[years > args.train_end_year]
    print(
        f"Train: {len(train_frame):,} rows (<= {args.train_end_year}), "
        f"Validation: {len(validation_frame):,} rows (> {args.train_end_year})"
    )

    train_output_dir = "/opt/ml/processing/train"
    validation_output_dir = "/opt/ml/processing/validation"
    baseline_output_dir = "/opt/ml/processing/baseline"
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(validation_output_dir, exist_ok=True)
    os.makedirs(baseline_output_dir, exist_ok=True)

    # 学習コンテナ（train_xgboost.py）はヘッダーなし CSV を期待する
    train_frame[FEATURE_COLUMNS].to_csv(
        os.path.join(train_output_dir, "train_features.csv"), index=False, header=False
    )
    train_frame[[LABEL_COLUMN]].to_csv(
        os.path.join(train_output_dir, "train_labels.csv"), index=False, header=False
    )
    validation_frame[FEATURE_COLUMNS].to_csv(
        os.path.join(validation_output_dir, "val_features.csv"), index=False, header=False
    )
    validation_frame[[LABEL_COLUMN]].to_csv(
        os.path.join(validation_output_dir, "val_labels.csv"), index=False, header=False
    )

    # モニタリング（ドリフト検知）の参照データ（ベースライン）として学習データの
    # 特徴量＋ラベルを列名つきで保存しておく（Evidently の reference_data に使う）
    baseline_frame = train_frame[FEATURE_COLUMNS + [LABEL_COLUMN]]
    baseline_frame.to_csv(os.path.join(baseline_output_dir, "baseline.csv"), index=False)

    print("Preprocessing complete.")
    print(f"  train_features: {train_output_dir}/train_features.csv")
    print(f"  train_labels  : {train_output_dir}/train_labels.csv")
    print(f"  val_features  : {validation_output_dir}/val_features.csv")
    print(f"  val_labels    : {validation_output_dir}/val_labels.csv")
    print(f"  baseline      : {baseline_output_dir}/baseline.csv")


if __name__ == "__main__":
    main()
