"""手順5のテスト（quantize.py）＝周2の新要素。torchao で Linear の重みを int8 にする。

学習済みでない（ランダムな重みの）モデルで回る。数秒。
実行: pytest tests/test_5_quantize.py
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")

import model  # noqa: E402
import quantize  # noqa: E402


def is_quantized(weight):
    """torchao が置き換えた重みは、ただの Tensor / Parameter ではない型になる（0.18 では Int8Tensor）。"""
    return type(weight).__name__ not in ("Tensor", "Parameter")


@pytest.fixture()
def m():
    torch.manual_seed(0)
    net = model.SentimentClassifier(vocab_size=8000)
    net.eval()
    return net


@pytest.fixture()
def x():
    return torch.randint(1, 8000, (32, 64), generator=torch.Generator().manual_seed(0))


# ---------- quantize_int8 ----------

def test_linear_weights_become_int8(m):
    q = quantize.quantize_int8(m)
    assert is_quantized(q.head.weight), f"head.weight の型: {type(q.head.weight).__name__}"
    assert is_quantized(q.encoder.layers[0].linear1.weight)


def test_original_model_is_left_alone(m, x):
    """元のモデルは fp32 のまま残すこと（あとで2つを並べて比べるため）。

    落ちるときは、コピーを取らずに渡されたモデルをそのまま量子化している。
    """
    with torch.no_grad():
        before = m(x)
    quantize.quantize_int8(m)
    assert not is_quantized(m.head.weight), "元のモデルまで int8 になった"
    with torch.no_grad():
        assert torch.equal(m(x), before)


def test_quantized_model_answers_almost_the_same(m, x):
    q = quantize.quantize_int8(m)
    with torch.no_grad():
        a, b = m(x), q(x)
    assert (a.argmax(1) == b.argmax(1)).float().mean() >= 0.9
    assert (a - b).abs().max() < 0.1


def test_quantized_model_is_in_eval_mode(m):
    m.train()
    assert quantize.quantize_int8(m).training is False


# ---------- model_size_mb ----------

def test_size_of_fp32_model_is_params_times_4_bytes(m):
    """fp32 は1個4バイト。1MB = 1,000,000 バイトで数える。"""
    size = quantize.model_size_mb(m)
    assert isinstance(size, float)
    expected = sum(t.numel() for t in m.state_dict().values()) * 4 / 1e6
    assert size == pytest.approx(expected, rel=0.02)


def test_int8_model_is_smaller(m):
    assert quantize.model_size_mb(quantize.quantize_int8(m)) < quantize.model_size_mb(m)


# ---------- time_inference ----------

def test_time_inference_returns_seconds(m, x):
    s = quantize.time_inference(m, x)
    assert isinstance(s, float) and 0 < s < 30


def test_time_inference_does_not_record_gradients(x):
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, inp):
            seen["grad"] = torch.is_grad_enabled()
            return inp.float()

    quantize.time_inference(Spy(), x)
    assert seen["grad"] is False, "torch.no_grad() の中で測ること"
