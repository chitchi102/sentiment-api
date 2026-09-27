"""周3 S1b のテスト（GPU）。Colab の GPU ランタイムでだけ回る。手元では全部 skip になる。

実行（Colab）: !python -m pytest tests/test_7_teacher_gpu.py
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

if not torch.cuda.is_available():
    pytest.skip("GPU が無い（Colab の「ランタイムのタイプを変更」で T4 GPU を選ぶ）", allow_module_level=True)

import teacher  # noqa: E402

SENTS = [
    "a wonderful, moving film",
    "a boring, painful mess",
    "the plot is thin but the cast is charming",
    "i wanted to leave after twenty minutes",
] * 10


@pytest.fixture(scope="module")
def on_gpu():
    tok, m = teacher.load_teacher(device="cuda")
    return tok, m, teacher.teacher_logits(tok, m, SENTS, batch_size=16, device="cuda")


def test_teacher_is_on_the_gpu(on_gpu):
    _, m, _ = on_gpu
    assert next(m.parameters()).device.type == "cuda"


def test_logits_come_back_on_the_cpu(on_gpu):
    """GPU の上の Tensor のまま返すと、torch.cat・保存・手元での読み込みでつまずく。.cpu() で戻すこと。"""
    _, _, out = on_gpu
    assert out.device.type == "cpu", f"返ってきた置き場: {out.device}"
    assert out.shape == (len(SENTS), 2)


def test_gpu_gives_the_same_answers_as_cpu(on_gpu):
    """置き場を変えても先生の答えは変わらない（小数の最後の桁が少しずれるだけ）。"""
    _, _, out = on_gpu
    tok, m = teacher.load_teacher()
    cpu_out = teacher.teacher_logits(tok, m, SENTS, batch_size=16)
    assert torch.allclose(out, cpu_out, atol=1e-3), f"最大の差: {(out - cpu_out).abs().max().item()}"
