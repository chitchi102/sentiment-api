"""手順5のテスト（bench.py）。fp32 と int8 を同じ物差しで並べる表を作る。

実行: pytest tests/test_5_bench.py
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")

import bench  # noqa: E402
import model  # noqa: E402


@pytest.fixture()
def rows():
    torch.manual_seed(0)
    m = model.SentimentClassifier(vocab_size=8000)
    m.eval()
    x = torch.randint(1, 8000, (64, 64), generator=torch.Generator().manual_seed(0))
    y = torch.randint(0, 2, (64,), generator=torch.Generator().manual_seed(1))
    return bench.compare(m, x, y), m


def test_one_row_per_model(rows):
    r, _ = rows
    assert [row["name"] for row in r] == ["fp32", "int8"]


def test_each_row_has_the_three_numbers(rows):
    r, _ = rows
    for row in r:
        assert set(row) == {"name", "size_mb", "acc", "seconds"}
        assert 0.0 <= row["acc"] <= 1.0
        assert row["size_mb"] > 0 and row["seconds"] > 0


def test_int8_row_is_smaller(rows):
    r, _ = rows
    assert r[1]["size_mb"] < r[0]["size_mb"]


def test_compare_leaves_the_fp32_model_alone(rows):
    _, m = rows
    assert type(m.head.weight).__name__ in ("Tensor", "Parameter")
