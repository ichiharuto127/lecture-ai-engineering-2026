"""第1回 宿題（発展）: 応答時間とトークン数を、モデル・質問ごとに計測する。

使い方:
    uv run python -m app.measure                       # 既定: flash-lite で各質問3回
    uv run python -m app.measure --models gemini-3.5-flash-lite gemini-3.8-flash --repeat 2

結果は1回の呼び出しごとに CSV（既定 lectures/01/results.csv）へ追記し、最後に集計を表示する。
APIエラー（429・503など）も1行として記録する。レート制限との関係を考察する材料になるため。
"""

import argparse
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx
from google import genai
from google.genai import errors, types

from app.config import get_api_key

# 出力の長さが違う3問。応答時間が出力トークン数にどう比例するかを見るため
QUESTIONS = {
    "short": "LoRAとは何ですか？一文で答えてください。",
    "course": "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
    "long": "LoRAとQLoRAの違いを、仕組み・メモリ使用量・使いどころの観点で800字程度で説明してください。",
}

# 無料枠の毎分リクエスト上限（講座の実測）。呼び出し間隔をこれから決める
RPM = {"gemini-3.5-flash-lite": 15, "gemini-3.8-flash": 5}
DEFAULT_RPM = 5
TIMEOUT_S = 120

FIELDS = ["timestamp", "model", "question", "run", "status",
          "latency_s", "input", "output", "thoughts", "total", "chars"]


def usage_summary(response) -> dict[str, int]:
    """response.usage_metadata をトークン数の辞書にする。

    返ってこない項目は None なので 0 にそろえる（集計で足し算するため）。
    thoughts は思考モデルが回答前に使ったトークンで、出力（candidates）とは別に数えられる。
    """
    usage = response.usage_metadata
    return {
        "input": (usage and usage.prompt_token_count) or 0,
        "output": (usage and usage.candidates_token_count) or 0,
        "thoughts": (usage and usage.thoughts_token_count) or 0,
        "total": (usage and usage.total_token_count) or 0,
    }


def measure_once(client, model: str, question: str) -> dict:
    """1回呼び出して、応答時間・トークン数・状態を返す。エラーでも例外にせず記録用の行を返す。"""
    start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=question)
    except errors.APIError as e:
        return {"status": e.code, "latency_s": round(time.perf_counter() - start, 3)}
    except httpx.TimeoutException:
        return {"status": "timeout", "latency_s": round(time.perf_counter() - start, 3)} 
    latency = time.perf_counter() - start
    text = response.text or ""
    return {"status": "ok", "latency_s": round(latency, 3), "chars": len(text), **usage_summary(response)}


def summarize(rows: list[dict]) -> None:
    """モデル×質問ごとに、成功件数・応答時間の中央値・平均トークン数・出力速度を表示する。"""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["question"]), []).append(r)

    print(f"\n{'model':<24}{'question':<8}{'ok/n':>6}{'中央値s':>9}{'入力':>6}{'出力':>7}{'思考':>7}{'tok/s':>8}")
    for (model, q), rs in groups.items():
        ok = [r for r in rs if r["status"] == "ok"]
        if not ok:
            print(f"{model:<24}{q:<8}{0:>3}/{len(rs):<2}  （全てエラー: {[r['status'] for r in rs]}）")
            continue
        med = statistics.median(r["latency_s"] for r in ok)
        mean = lambda k: statistics.mean(r[k] for r in ok)
        # 出力速度は生成した全トークン（出力＋思考）を応答時間で割る
        tps = statistics.mean((r["output"] + r["thoughts"]) / max(r["latency_s"], 1e-3) for r in ok)
        print(f"{model:<24}{q:<8}{len(ok):>3}/{len(rs):<2}{med:>9.2f}{mean('input'):>6.0f}"
              f"{mean('output'):>7.0f}{mean('thoughts'):>7.0f}{tps:>8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="応答時間とトークン数を計測する")
    parser.add_argument("--models", nargs="+", default=["gemini-3.5-flash-lite"])
    parser.add_argument("--questions", nargs="+", default=list(QUESTIONS), choices=list(QUESTIONS))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("lectures/01/results.csv"))
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key(),
                          http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000))      
    print(f"各呼び出しの上限 {TIMEOUT_S}秒。1回ごとに結果を表示します", flush=True)
    new_file = not args.out.exists()
    rows = []
    with args.out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for model in args.models:
            # 上限ちょうどだと端数で超えるので、1割余裕を持たせる
            interval = 60 / RPM.get(model, DEFAULT_RPM) * 1.1
            for q in args.questions:
                for run in range(1, args.repeat + 1):
                    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                           "model": model, "question": q, "run": run,
                           **measure_once(client, model, QUESTIONS[q])}
                    writer.writerow(row)
                    f.flush()  # 途中で止まっても、それまでの結果は残す
                    rows.append(row)
                    print(f"{model} {q} #{run}: {row['status']} {row['latency_s']}s "
                          f"出力{row.get('output', '-')} 思考{row.get('thoughts', '-')}")
                    time.sleep(interval)
    summarize(rows)
    print(f"\n結果: {args.out}")


if __name__ == "__main__":
    main()