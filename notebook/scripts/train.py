"""SageMaker 学習ジョブ内で実行される XGBoost 回帰の学習スクリプト。

風向・風速（＋季節性）から有義波波高を推定する回帰モデルを学習する。
XGBoost の回帰（reg:squarederror）で学習し、
SageMaker Managed MLflow へのロギングを組み込んでいる。

コンテナの契約:
    入力チャネル:
        SM_CHANNEL_TRAIN      /opt/ml/input/data/train
            train_features.csv, train_labels.csv（いずれもヘッダーなし）
        SM_CHANNEL_VALIDATION /opt/ml/input/data/validation
            val_features.csv, val_labels.csv（いずれもヘッダーなし）
    出力:
        SM_MODEL_DIR          /opt/ml/model
            {MLFLOW_MODEL_NAME} という名前で XGBoost の Booster を保存する
            （inference.py の model_fn が同じ名前で読み込む）

MLflow ロギング:
    MLFLOW_TRACKING_URI / MLFLOW_EXP / MLFLOW_MODEL_NAME を環境変数で受け取り、
    「複数実行を比較する」体験のために、ハイパーパラメータ・指標を記録する。
    `mlflow.xgboost.autolog()` を使うことで、明示的な log_params 呼び出しを
    最小限にしつつパラメータ・モデルを自動記録する。
    `mlflow.start_run(log_system_metrics=True)` により、CPU 使用率・メモリ使用量
    などのシステムメトリクス（`system/` プレフィックス）も自動記録する。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

FEATURE_COLUMNS = ["wind_speed", "wind_sin", "wind_cos", "month"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # --- XGBoost ハイパーパラメータ（回帰） ---
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=4)
    parser.add_argument("--min_child_weight", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--num_round", type=int, default=200)
    parser.add_argument("--early_stopping_rounds", type=int, default=20)
    parser.add_argument("--verbosity", type=int, default=0)

    # --- SageMaker が渡すチャネルパス（環境変数から既定値を取得） ---
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
    )
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    args, _ = parser.parse_known_args()
    return args


def load_dataset(channel_dir: str, features_filename: str, labels_filename: str):
    """特徴量 CSV とラベル CSV（いずれもヘッダーなし）を読み込む。"""
    features_path = os.path.join(channel_dir, features_filename)
    labels_path = os.path.join(channel_dir, labels_filename)

    logger.info(f"Loading features from {features_path}")
    features = pd.read_csv(features_path, header=None, names=FEATURE_COLUMNS)
    # month が整数型だと MLflow のモデルシグネチャ推論で欠損値絡みの警告が出るため float に統一する
    features = features.astype("float64")

    logger.info(f"Loading labels from {labels_path}")
    labels = pd.read_csv(labels_path, header=None, names=["wave_height"])

    return features, labels["wave_height"]


def main() -> None:
    args = parse_args()

    # --- MLflow: このジョブの実行を記録する ---
    # autolog は DMatrix を構築する前に有効化する必要がある
    # （後から有効化すると入力サンプル・モデルシグネチャを推論できず警告が出る）。
    mlflow_app_arn = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow_experiment_name = os.environ.get("MLFLOW_EXP")
    mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME", "wave-height-xgboost")

    if mlflow_app_arn:
        mlflow.set_tracking_uri(mlflow_app_arn)
    if mlflow_experiment_name:
        mlflow.set_experiment(mlflow_experiment_name)

    # autolog がハイパーパラメータ・学習曲線・モデルを自動記録する。
    # 明示的な log_metrics は、autolog が拾わない検証指標（MAE, R^2）の分だけ足す。
    mlflow.xgboost.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
        log_datasets=False,
        model_format="json",
        extra_tags={"team": "workshop", "use_case": "tokyo-port-wave-height"},
    )

    X_train, y_train = load_dataset(args.train, "train_features.csv", "train_labels.csv")
    X_val, y_val = load_dataset(
        args.validation, "val_features.csv", "val_labels.csv"
    )
    logger.info(f"Train rows: {len(X_train):,}, Validation rows: {len(X_val):,}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLUMNS)

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "verbosity": args.verbosity,
        # --- 回帰タスク用の設定 ---
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }

    run_name = os.environ.get("MLFLOW_RUN_NAME", f"train_{mlflow_model_name}")
    # log_system_metrics=True で CPU/メモリ使用率などのシステムメトリクスも記録する
    # （psutil が必要。requirements.txt に追加済み）
    # 学習自体が数十秒程度で終わるため、既定のサンプリング間隔（10秒）だと
    # 1 点もサンプリングされないまま終了してしまう。間隔を 2 秒に短縮して
    # 学習中に最低数点は記録されるようにする。
    mlflow.set_system_metrics_sampling_interval(2)
    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=args.num_round,
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=args.early_stopping_rounds,
        )

        y_pred = model.predict(dval)
        metrics = {
            "rmse": mean_squared_error(y_val, y_pred) ** 0.5,
            "mae": mean_absolute_error(y_val, y_pred),
            "r2": r2_score(y_val, y_pred),
        }
        mlflow.log_metrics(metrics)
        logger.info(f"Validation metrics: {metrics}")

        model_dir = args.model_dir or "/opt/ml/model"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, mlflow_model_name)
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
