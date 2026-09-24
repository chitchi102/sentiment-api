"""手順2のテスト（data.py）。本物はダウンロードせず、conftest.py の偽 SST-2 で回る。

本物で確かめるのは test_2_data_real.py（ネットが要る）。
実行: pytest tests/test_2_data.py
"""

import data


def test_uses_the_sst2_dataset(fake_sst2):
    data.load_data(n_train=100)
    assert fake_sst2["calls"] == ["stanfordnlp/sst2"], f"load_dataset に渡した id: {fake_sst2['calls']}"


def test_returns_four_lists(fake_sst2):
    out = data.load_data(n_train=100)
    assert len(out) == 4
    assert all(isinstance(v, list) for v in out), [type(v).__name__ for v in out]


def test_train_is_cut_to_n_train(fake_sst2):
    xtr, ytr, _, _ = data.load_data(n_train=100)
    assert len(xtr) == 100 and len(ytr) == 100


def test_texts_come_from_the_sentence_column(fake_sst2):
    """SST-2 の文章の列名は "text" ではなく "sentence"（周1の AG News と違う）。"""
    xtr, _, xte, _ = data.load_data(n_train=100)
    assert xtr == fake_sst2["ds"]["train"]["sentence"][:100]
    assert all(isinstance(t, str) for t in xte)


def test_evaluation_uses_the_validation_split_whole(fake_sst2):
    """評価は validation を全部使う。

    SST-2 の test split はラベルが全部 -1（答えが非公開）なので使えない。
    落ちて -1 が見えるときは "test" を読んでいる。
    """
    _, _, xte, yte = data.load_data(n_train=100)
    assert -1 not in yte, "ラベル -1 が入っている＝test split を読んでいる"
    assert len(xte) == len(fake_sst2["ds"]["validation"]) and len(yte) == len(xte)


def test_labels_are_0_and_1(fake_sst2):
    _, ytr, _, yte = data.load_data(n_train=100)
    assert set(ytr) == {0, 1} and set(yte) == {0, 1}
    assert all(isinstance(v, int) for v in ytr)
