"""計測スクリプトのテスト。偽のクライアントを使い、APIキーも利用枠も使わない。"""

from types import SimpleNamespace

from google.genai import errors

from app.measure import measure_once, summarize, usage_summary


def _response(prompt=10, out=20, thoughts=None, text="回答"):
    usage = SimpleNamespace(prompt_token_count=prompt, candidates_token_count=out,
                            thoughts_token_count=thoughts, total_token_count=prompt + out + (thoughts or 0))
    return SimpleNamespace(usage_metadata=usage, text=text)


class FakeClient:
    def __init__(self, result):
        self.models = SimpleNamespace(generate_content=lambda **_: self._call(result))

    @staticmethod
    def _call(result):
        if isinstance(result, Exception):
            raise result
        return result


def test_usage_summary_fills_missing_with_zero():
    assert usage_summary(_response(thoughts=None))["thoughts"] == 0
    assert usage_summary(SimpleNamespace(usage_metadata=None))["total"] == 0


def test_measure_once_records_tokens_and_latency():
    row = measure_once(FakeClient(_response(thoughts=5)), "m", "q")
    assert row["status"] == "ok"
    assert (row["input"], row["output"], row["thoughts"], row["total"]) == (10, 20, 5, 35)
    assert row["latency_s"] >= 0 and row["chars"] == 2


def test_measure_once_records_api_error_without_raising():
    err = errors.APIError(429, {"error": {"message": "quota"}})
    row = measure_once(FakeClient(err), "m", "q")
    assert row["status"] == 429


def test_summarize_handles_all_error_group(capsys):
    summarize([{"model": "m", "question": "q", "status": 503, "latency_s": 0.1}])
    assert "全てエラー" in capsys.readouterr().out