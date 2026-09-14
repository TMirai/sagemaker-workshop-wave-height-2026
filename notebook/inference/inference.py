"""SageMaker XGBoost 推論コンテナ用の推論スクリプト（回帰）。

風向・風速（sin/cos 変換済み）＋季節性から有義波波高を推定するモデルの
推論ロジック。

コンテナの契約:
    model_fn(model_dir)                    必須。コンテナ起動時に 1 回だけ呼ばれる。
                                            model_dir（モデルファイルが置かれた場所）から
                                            モデルをメモリに読み込んで返す。
    input_fn(input_data, content_type)     任意。リクエストが来るたびに呼ばれる。
                                            既定は SageMaker が提供する XGBoost コンテナの
                                            組み込み処理。生のリクエストボディ（バイト列や
                                            テキスト）を、モデルが処理できる形（DMatrix）に
                                            変換する。
    predict_fn(input_data, model)          任意。リクエストが来るたびに呼ばれる。
                                            既定は SageMaker が提供する XGBoost コンテナの
                                            組み込み処理。input_fn の出力とモデルを受け取り、
                                            実際に予測を実行する。
    output_fn(prediction, accept)          任意。リクエストが来るたびに呼ばれる。
                                            既定は SageMaker が提供する XGBoost コンテナの
                                            組み込み処理。予測結果を、クライアントが要求する
                                            形式（テキスト、JSON 等）に変換してレスポンス
                                            ボディを作る。

このスクリプトでは model_fn のみを定義する。学習・評価と同じ 4 特徴量
（wind_speed, wind_sin, wind_cos, month）を CSV（ヘッダーなし、1 行 1
レコード）で送る前提であれば、SageMaker 組み込み XGBoost 推論コンテナの
既定の input_fn / predict_fn / output_fn がそのまま動作する
（DMatrix への変換、predict()、テキストでの結果返却を代行してくれる）。
"""

import os

import xgboost as xgb


def model_fn(model_dir: str) -> xgb.Booster:
    """モデルディレクトリから XGBoost の Booster をロードする。

    train.py は `{model_dir}/{MLFLOW_MODEL_NAME}` という名前でモデルを
    保存する。ファイル名は学習時のジョブ設定に依存するため、`code/`
    サブディレクトリ（推論スクリプト自身が置かれる場所）を除いた
    最初のファイルをモデルとして読み込む。
    """
    candidates = [
        name
        for name in os.listdir(model_dir)
        if not name.startswith(".") and name != "code"
    ]
    if not candidates:
        raise FileNotFoundError(f"No model file found in {model_dir}")

    model_path = os.path.join(model_dir, candidates[0])
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster
