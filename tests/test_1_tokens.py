"""手順1のテスト（tokens.py）。その場の短い文で回る。1秒台。

周1の build_tokenizer / encode_batch をファイルごと移す段。
実行: pytest tests/test_1_tokens.py
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")

import tokens  # noqa: E402

TOY_TEXTS = [
    "a wonderful and moving film",
    "the plot is dull and the acting is awful",
    "charming , funny and smart",
    "a tedious mess from start to finish",
]


@pytest.fixture(scope="module")
def tok():
    return tokens.build_tokenizer(TOY_TEXTS, vocab_size=200)


def test_special_token_ids(tok):
    assert tok.token_to_id("[PAD]") == 0
    assert tok.token_to_id("[UNK]") == 1


def test_encode_batch_shape_and_dtype(tok):
    x = tokens.encode_batch(tok, TOY_TEXTS, max_len=16)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (len(TOY_TEXTS), 16)
    assert x.dtype == torch.long


def test_short_is_padded_long_is_cut(tok):
    x = tokens.encode_batch(tok, ["a film", " ".join(["film"] * 100)], max_len=8)
    assert x[0, 0].item() != 0 and x[0, -1].item() == 0
    assert (x[1] != 0).all()


def test_default_max_len_is_64(tok):
    assert tokens.encode_batch(tok, TOY_TEXTS).shape == (len(TOY_TEXTS), 64)


def test_unknown_word_becomes_unk(tok):
    assert tokens.encode_batch(tok, ["zzz qqq"], max_len=8)[0, 0].item() == 1
