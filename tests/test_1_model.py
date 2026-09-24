"""手順1のテスト（model.py）。偽の番号の列だけで回る。ダウンロードもトークナイザも要らない。

周1の NewsClassifier と中身は同じ。違うのはクラスが 2 つ（negative / positive）になったこと。
実行: pytest tests/test_1_model.py
"""

import pytest

torch = pytest.importorskip("torch")

import model  # noqa: E402

VOCAB = 50
LEN = 12


def fake_batch(n=8, length=LEN):
    g = torch.Generator().manual_seed(0)
    return torch.randint(1, VOCAB, (n, length), generator=g)


@pytest.fixture
def m():
    torch.manual_seed(0)
    net = model.SentimentClassifier(vocab_size=VOCAB, d_model=32, nhead=4, num_layers=2, max_len=LEN)
    net.eval()
    return net


def test_labels_are_negative_then_positive():
    """順番に意味がある: SST-2 のラベルは 0=negative, 1=positive。"""
    assert model.LABELS == ["negative", "positive"]
    assert model.NUM_CLASSES == 2


def test_forward_shape(m):
    assert m(fake_batch(n=5)).shape == (5, 2)


def test_padding_does_not_change_output(m):
    """後ろを 0 で埋めても出力が変わらないこと（周1の R1 と同じ検査）。"""
    x = fake_batch(n=3, length=6)
    padded = torch.zeros(3, LEN, dtype=torch.long)
    padded[:, :6] = x
    with torch.no_grad():
        a, b = m(x), m(padded)
    assert torch.allclose(a, b, atol=1e-5), f"padding で出力が動いた: {(a - b).abs().max().item()}"


def test_count_params(m):
    n = model.count_params(m)
    assert isinstance(n, int) and n > 0


def test_default_size_is_about_1_3_million():
    """既定の大きさ（vocab 8000）で約130万。周1と同じ骨格なら head が 4→2 列に減るだけ。"""
    n = model.count_params(model.SentimentClassifier(vocab_size=8000))
    assert 1_250_000 <= n <= 1_350_000, n
