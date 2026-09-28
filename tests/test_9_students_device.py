"""周3 S3c のテスト（手元）。device が run_student から学習・評価まで届くことだけを見る。数秒。

GPU で本当に動くかは Colab で tests/test_9_students_gpu.py を回して確かめる。
実行: pytest tests/test_9_students_device.py
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")

import students  # noqa: E402
import train  # noqa: E402
from test_9_students import teacher_file  # noqa: E402,F401  （fixture をそのまま使う）


@pytest.fixture()
def device_spies(monkeypatch):
    """train_model / train_distill / evaluate を「受け取った device を記録して、何もせず返す」ものに差し替える。

    device="cuda" を渡すが、差し替えた関数は何も動かさないので GPU が無くても回る。
    落ちて RuntimeError（CUDA が無い）になるときは、run_student の中でモデルを device へ動かしている
    （動かすのは train_model / train_distill / evaluate の仕事。run_student は device を渡すだけ）。
    """
    seen = []

    def fake_train_model(model, x, y, epochs=3, device="cpu", **kw):
        seen.append(("train_model", device))
        return model, [0.0] * epochs

    def fake_train_distill(model, x, t, epochs=3, device="cpu", **kw):
        seen.append(("train_distill", device))
        return model, [0.0] * epochs

    def fake_evaluate(model, x, y, device="cpu"):
        seen.append(("evaluate", device))
        return 0.5

    monkeypatch.setattr(train, "train_model", fake_train_model)
    monkeypatch.setattr(train, "train_distill", fake_train_distill)
    monkeypatch.setattr(train, "evaluate", fake_evaluate)
    return seen


@pytest.mark.parametrize("mode, fn", [("small", "train_model"), ("teacher", "train_distill"), ("full", "train_model")])
def test_device_reaches_training(teacher_file, device_spies, mode, fn):  # noqa: F811
    students.run_student(mode, teacher_path=teacher_file["path"], n_label=50, epochs=1, device="cuda")
    assert (fn, "cuda") in device_spies, f"届いたもの: {device_spies}"


def test_fp32_is_evaluated_on_the_device_and_int8_on_the_cpu(teacher_file, device_spies):  # noqa: F811
    """fp32 の生徒は GPU で測る。int8（torchao）は周2と同じく CPU で測る。

    だから evaluate は2回呼ばれ、1回目が device="cuda"、2回目が device="cpu"。
    """
    students.run_student("teacher", teacher_path=teacher_file["path"], epochs=1, device="cuda")
    evals = [d for f, d in device_spies if f == "evaluate"]
    assert evals == ["cuda", "cpu"], f"evaluate に届いた device（呼ばれた順）: {evals}"


def test_device_defaults_to_cpu(teacher_file, device_spies):  # noqa: F811
    students.run_student("full", teacher_path=teacher_file["path"], epochs=1)
    assert {d for _, d in device_spies} == {"cpu"}, f"届いたもの: {device_spies}"


def test_result_says_where_it_was_trained(teacher_file, device_spies):  # noqa: F811
    """S4 の表で「どこで何秒」を並べるため、返り値に device を入れる。"""
    r = students.run_student("full", teacher_path=teacher_file["path"], epochs=1, device="cuda")
    assert r.get("device") == "cuda", f"返り値: {r}"
