"""周3 S1a のテスト（teacher.py）。先生＝HF の DistilBERT（SST-2 で学習済み）。

ここは本物をダウンロードしない。偽の tokenizer と偽の先生を渡して「形」だけ見る。数秒。
本物の先生で確かめるのは test_7_teacher_real.py（初回だけ約 270MB ダウンロード）。
実行: pytest tests/test_7_teacher.py
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import BatchEncoding  # noqa: E402
from transformers.modeling_outputs import SequenceClassifierOutput  # noqa: E402

import teacher  # noqa: E402


class FakeTokenizer:
    """HF の tokenizer と同じ呼ばれ方をする偽物。呼ばれた引数を記録する。

    各文の文字数を 1 トークン目に入れて返す（偽の先生がそれを logits にする）。
    """

    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        texts = list(texts)
        self.calls.append((texts, kwargs))
        n = len(texts)
        ids = torch.zeros(n, 8, dtype=torch.long)
        ids[:, 0] = torch.tensor([len(t) for t in texts])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones(n, 8, dtype=torch.long)})


class FakeTeacher(torch.nn.Module):
    """HF の分類モデルと同じ返し方（.logits を持つ出力）をする偽物。

    logits = [文字数, -文字数]。1回ごとの件数と、勾配を記録していたかを残す。
    """

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.batch_sizes = []
        self.grad_enabled = []

    def forward(self, input_ids, attention_mask=None):
        self.batch_sizes.append(input_ids.shape[0])
        self.grad_enabled.append(torch.is_grad_enabled())
        L = input_ids[:, 0].float()
        return SequenceClassifierOutput(logits=torch.stack([L, -L], dim=1) * self.scale)


def texts_of_length(n):
    """長さ 1, 2, ..., n 文字の文。順番が崩れたら logits で分かる。"""
    return ["a" * (i + 1) for i in range(n)]


@pytest.fixture()
def tok():
    return FakeTokenizer()


@pytest.fixture()
def m():
    return FakeTeacher()


def test_returns_one_row_of_two_logits_per_text(tok, m):
    out = teacher.teacher_logits(tok, m, texts_of_length(10))
    assert isinstance(out, torch.Tensor), f"返り値の型: {type(out).__name__}"
    assert out.shape == (10, 2), f"形: {tuple(out.shape)}"
    assert out.dtype == torch.float32


def test_texts_are_sent_in_batches_of_64_by_default(tok, m):
    """67,349件を一度に先生へ渡すとメモリが足りない。64件ずつに切って渡す。"""
    teacher.teacher_logits(tok, m, texts_of_length(150))
    assert m.batch_sizes == [64, 64, 22], f"先生に渡した件数: {m.batch_sizes}"


def test_batch_size_can_be_changed(tok, m):
    teacher.teacher_logits(tok, m, texts_of_length(50), batch_size=16)
    assert m.batch_sizes == [16, 16, 16, 2], f"先生に渡した件数: {m.batch_sizes}"


def test_order_is_kept_across_batches(tok, m):
    """i 行目が i 番目の文の logits であること（切って渡して、つなぎ直す）。"""
    out = teacher.teacher_logits(tok, m, texts_of_length(150), batch_size=16)
    assert torch.equal(out[:, 0], torch.arange(1, 151).float()), "行の順番が文の順番と合っていない"


def test_tokenizer_cuts_at_64_tokens_and_returns_tensors(tok, m):
    """周2の生徒と同じ長さ（64）で揃える。長い文は切り捨て、短い文は埋める。

    return_tensors="pt" が無いと tokenizer は torch.Tensor でなく list を返す。
    """
    teacher.teacher_logits(tok, m, texts_of_length(3))
    _, kwargs = tok.calls[0]
    assert kwargs.get("truncation") is True, f"truncation: {kwargs.get('truncation')}"
    assert kwargs.get("max_length") == 64, f"max_length: {kwargs.get('max_length')}"
    assert kwargs.get("padding"), f"padding: {kwargs.get('padding')}"
    assert kwargs.get("return_tensors") == "pt", f"return_tensors: {kwargs.get('return_tensors')}"


def test_no_gradients_are_recorded(tok, m):
    """採点するだけで学習しない＝torch.no_grad() の中で先生を呼ぶこと（周2の time_inference と同じ）。"""
    out = teacher.teacher_logits(tok, m, texts_of_length(10))
    assert m.grad_enabled == [False], "torch.no_grad() の中で先生を呼ぶこと"
    assert out.requires_grad is False
