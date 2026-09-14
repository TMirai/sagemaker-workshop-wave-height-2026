#!/usr/bin/env python3
"""東京都港湾局の海象観測データ（実データ）を共通スキーマ CSV に変換する。

擬似データ（generate_synthetic_data.py）と同一のスキーマで出力するため、
下流の処理（前処理・学習・デプロイ・モニタリング）は変更なしに動作する。

出典（実データを使う場合は必ず表記すること）:
    東京都港湾局「東京港波浪観測所・海象観測データ」
    https://www.kouwan.metro.tokyo.lg.jp/yakuwari/choui/kako1-

⚠️ 実データの利用について
    出典ページには「私的使用等、著作権法上認められた行為を除き、港湾局に
    無断で転載等を行うことはできない」旨の記載がある。オープンデータ
    ライセンス（CC-BY 等）ではないため、ワークショップでの配布には
    港湾局への確認が必要。本ワークショップでは既定で擬似データを使用する。

入力ファイルの形式（実測により確認済み）:
    - エンコーディング: Shift-JIS（cp932 で読む）
    - 1 日ごとに以下のブロックが繰り返される固定レイアウト
        1) 日付行            "2021/01/01"
        2) ヘッダー行        "時刻  ,最高波波高(m),..."
        3) データ行 × 24     "    01,         0.15, ..."
        4) 最大行            "最大  ,         0.30, ..."
        5) 平均行            "平均  ,         0.21, ..."
    - 値は空白パディングあり。欠測は空文字
    - 末尾に余分なカンマがあるため列数は 14（有効列は 13）

使い方:
    python load_real_data.py --input-dir ./raw --output ./data/observation.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 実データの列インデックス → 共通スキーマの列名
#
# 実データのヘッダー:
#   0:時刻 1:最高波波高(m) 2:最高波周期(s) 3:有義波波高(m) 4:有義波周期(s)
#   5:波向 6:流向 7:流速(cm/s) 8:風向 9:風速(m/s) 10:潮位(cm)
#   11:推算潮位(cm) 12:偏差(cm)
#
# 11（推算潮位）と 12（偏差）は全 43,824 行すべて空だったため出力しない。
# ---------------------------------------------------------------------------
COLUMN_MAP: list[tuple[int, str, str]] = [
    # (実データの列インデックス, 出力列名, 型: "float" | "str")
    (1, "wave_height_max", "float"),   # 最高波波高(m)
    (2, "wave_period_max", "float"),   # 最高波周期(s)
    (3, "wave_height", "float"),       # 有義波波高(m)  ★目的変数
    (4, "wave_period", "float"),       # 有義波周期(s)
    (5, "wave_dir", "str"),            # 波向（有義波高 25cm 未満では非表示の仕様）
    (6, "current_dir", "str"),         # 流向
    (7, "current_speed", "float"),     # 流速(cm/s)
    (8, "wind_dir", "str"),            # 風向          ★特徴量
    (9, "wind_speed", "float"),        # 風速(m/s)     ★特徴量
    (10, "tide_level", "float"),       # 潮位(cm)
]

OUTPUT_COLUMNS = ["timestamp"] + [name for _, name, _ in COLUMN_MAP]

DATE_LINE_RE = re.compile(r"^\s*(\d{4})/(\d{2})/(\d{2})\s*$")
DATA_LINE_RE = re.compile(r"^\s*(\d{1,2})\s*,")


def _parse_value(raw: str, kind: str) -> float | str | None:
    """1 セルをパースする。欠測（空文字）は None を返す。"""
    text = raw.strip()
    if not text:
        return None
    if kind == "float":
        try:
            return float(text)
        except ValueError:
            # 想定外の文字列が入っていた場合は欠測扱いにする
            return None
    return text


def parse_file(path: Path, encoding: str = "cp932") -> list[dict]:
    """1 ファイル（1 年分）をパースしてレコードのリストを返す。"""
    records: list[dict] = []
    current_date: pd.Timestamp | None = None
    skipped_lines = 0

    with path.open(encoding=encoding, errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue

            # 1) 日付行
            date_match = DATE_LINE_RE.match(line)
            if date_match:
                year, month, day = (int(g) for g in date_match.groups())
                current_date = pd.Timestamp(year=year, month=month, day=day)
                continue

            fields = line.split(",")

            # 2) データ行（先頭が時刻の数値）。ヘッダー行・最大行・平均行は
            #    先頭が「時刻」「最大」「平均」なのでここで除外される。
            if not DATA_LINE_RE.match(line):
                continue

            if current_date is None:
                skipped_lines += 1
                continue

            hour = int(fields[0].strip())
            if not 1 <= hour <= 24:
                skipped_lines += 1
                continue

            # 時刻は「正時」を表す（観測はその正時前後の平均値）。
            # 時刻 01 → 01:00、時刻 24 → 翌日 00:00 として扱う。
            timestamp = current_date + timedelta(hours=hour)

            record: dict = {"timestamp": timestamp}
            for index, name, kind in COLUMN_MAP:
                record[name] = (
                    _parse_value(fields[index], kind) if index < len(fields) else None
                )
            records.append(record)

    if skipped_lines:
        print(
            f"  ! {path.name}: 解釈できずスキップした行 {skipped_lines} 件",
            file=sys.stderr,
        )
    return records


def load_directory(input_dir: Path, pattern: str = "*-h.txt") -> pd.DataFrame:
    """ディレクトリ内の全ファイルをパースして 1 つの DataFrame にまとめる。"""
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"{input_dir} に {pattern} に一致するファイルがありません。"
        )

    all_records: list[dict] = []
    for path in files:
        records = parse_file(path)
        print(f"  {path.name}: {len(records):,} 行")
        all_records.extend(records)

    frame = pd.DataFrame(all_records, columns=OUTPUT_COLUMNS)
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    duplicated = frame["timestamp"].duplicated().sum()
    if duplicated:
        print(
            f"  ! 重複したタイムスタンプ {duplicated} 件を検出（先頭を残します）",
            file=sys.stderr,
        )
        frame = frame.drop_duplicates(subset="timestamp", keep="first").reset_index(
            drop=True
        )
    return frame


def summarize(frame: pd.DataFrame) -> None:
    """読み込み結果の要約を表示する。"""
    print()
    print("=" * 62)
    print("読み込み結果")
    print("=" * 62)
    print(f"行数        : {len(frame):,}")
    print(
        f"期間        : {frame['timestamp'].min()} 〜 {frame['timestamp'].max()}"
    )
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
    if len(valid) > 1:
        correlation = valid["wind_speed"].corr(valid["wave_height"])
        print(f"corr(wind_speed, wave_height) = {correlation:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="東京都港湾局の海象観測データを共通スキーマ CSV に変換する"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="実データ（YYYY12-h.txt）が置かれたディレクトリ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/observation.csv"),
        help="出力先 CSV のパス（既定: data/observation.csv）",
    )
    parser.add_argument(
        "--pattern",
        default="*-h.txt",
        help="入力ファイルの glob パターン（既定: *-h.txt）",
    )
    parser.add_argument(
        "--encoding",
        default="cp932",
        help="入力ファイルのエンコーディング（既定: cp932 = Shift-JIS）",
    )
    args = parser.parse_args()

    print(f"入力ディレクトリ: {args.input_dir}")
    frame = load_directory(args.input_dir, args.pattern)
    summarize(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print()
    print(f"✅ 出力しました: {args.output}")
    print()
    print("出典表記（手順書・ノートブックに記載してください）:")
    print("  出典: 東京都港湾局「東京港波浪観測所・海象観測データ」")
    print("  https://www.kouwan.metro.tokyo.lg.jp/yakuwari/choui/kako1-")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
