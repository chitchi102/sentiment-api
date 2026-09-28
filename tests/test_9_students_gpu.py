"""周3 S3c のテスト（GPU）。Colab の GPU ランタイムでだけ回る。手元では全部 skip になる。

偽 SST-2 と偽の先生で、学習・評価・保存が GPU で通るかを見る。1分前後。
実行（Colab）: !python -m pytest tests/test_9_students_gpu.py
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")

DEVICE = os.environ.get("TEST_DEVICE", "cuda")  # 確かめる置き場（ふだんは cuda のまま）
if DEVICE == "cuda" and not torch.cuda.is_available():
    pytest.skip("GPU が無い（Colab の「ランタイムのタイプを変更」で T4 GPU を選ぶ）", allow_module_level=True)

import model  # noqa: E402
import students  # noqa: E402
import train  # noqa: E402
from test_9_students import teacher_file  # noqa: E402,F401


def tiny_model():
    torch.manual_seed(0)
    return model.SentimentClassifier(vocab_size=50, d_model=32, nhead=4, num_layers=2, max_len=12)


def tiny_data(n=32):
    g = torch.Generator().manual_seed(0)
    x = torch.randint(1, 50, (n, 12), generator=g)  # 手元（CPU）に置いたまま渡す
    y = torch.tensor([0, 1] * (n // 2))
    return x, y


def param_device(m):
    return next(m.parameters()).device.type


def test_train_model_runs_on_the_device():
    """x と y は CPU に置いたまま渡す。バッチごとに device へ動かすのは train_model の仕事。"""
    m = tiny_model()
    x, y = tiny_data()
    _, history = train.train_model(m, x, y, epochs=2, batch_size=8, device=DEVICE)
    assert param_device(m) == DEVICE, f"モデルの置き場: {param_device(m)}"
    assert len(history) == 2 and all(isinstance(h, float) for h in history)


def test_train_distill_runs_on_the_device():
    m = tiny_model()
    x, y = tiny_data()
    t = torch.nn.functional.one_hot(y, 2).float() * 6 - 3
    _, history = train.train_distill(m, x, t, epochs=2, batch_size=8, device=DEVICE)
    assert param_device(m) == DEVICE, f"モデルの置き場: {param_device(m)}"
    assert len(history) == 2 and all(isinstance(h, float) for h in history)


def test_evaluate_gives_the_same_answer_on_the_device_and_the_cpu():
    """置き場を変えても正解率は同じ。返り値は float（GPU の上の Tensor のままにしない）。"""
    m = tiny_model()
    x, y = tiny_data()
    on_cpu = train.evaluate(m, x, y)
    on_dev = train.evaluate(m, x, y, device=DEVICE)
    assert isinstance(on_dev, float)
    assert on_dev == pytest.approx(on_cpu)


def test_run_student_on_the_device(teacher_file, tmp_path):  # noqa: F811
    """GPU で学んでも、保存する重みは CPU の上（Render には GPU が無い）。int8 の測定も通ること。"""
    r = students.run_student("teacher", teacher_path=teacher_file["path"], epochs=3, out_dir=tmp_path, device=DEVICE)
    assert r["acc"] >= 0.9 and r["int8_acc"] >= 0.9, r
    saved = torch.load(tmp_path / "model.pt")
    places = {t.device.type for t in saved.values()}
    assert places == {"cpu"}, f"保存した重みの置き場: {places}"
