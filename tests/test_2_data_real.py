"""手順2のテスト（本物の SST-2）。初回だけダウンロードする。

偽物のテスト（test_2_data.py）は形しか見ない。こちらは「id と列名と split が本物と合っているか」を見る。
実行: pytest tests/test_2_data_real.py
"""

import pytest

pytest.importorskip("datasets")

import data  # noqa: E402


@pytest.fixture(scope="module")
def loaded():
    return data.load_data(n_train=2000)


def test_lengths(loaded):
    xtr, ytr, xte, yte = loaded
    assert len(xtr) == len(ytr) == 2000
    assert len(xte) == len(yte) > 0


def test_both_labels_and_no_hidden_ones(loaded):
    _, ytr, _, yte = loaded
    assert set(ytr) == {0, 1}
    assert set(yte) == {0, 1}, f"validation のラベル: {sorted(set(yte))}"


def test_sentences_are_non_empty_strings(loaded):
    xtr, _, xte, _ = loaded
    assert all(isinstance(t, str) and t.strip() for t in xtr[:200] + xte[:200])


def test_train_and_validation_do_not_overlap(loaded):
    xtr, _, xte, _ = loaded
    assert not (set(xtr) & set(xte)), "train と validation に同じ文がある"
