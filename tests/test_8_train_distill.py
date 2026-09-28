"""周3 S3a のテスト（train.py に train_distill を足す）。小さい偽データだけで回す。数秒。

train_distill は train_model の「正解ラベルの代わりに先生の logits で学ぶ」版。
損失は S2 の distill.distill_loss を使う（train.py で `import distill` して呼ぶ）。
実行: pytest tests/test_8_train_distill.py
"""

import math

import pytest

torch = pytest.importorskip("torch")

import distill  # noqa: E402
import model  # noqa: E402
import train  # noqa: E402


def tiny_model(vocab_size=50):
    torch.manual_seed(0)
    return model.SentimentClassifier(vocab_size=vocab_size, d_model=32, nhead=4, num_layers=2, max_len=12)


def two_kinds_of_sentences(n=64):
    """前半の文は単語 1〜24 だけ、後半の文は単語 25〜49 だけでできている。

    先生は前半を「0 寄り」、後半を「1 寄り」と答える。先生の logits の i 行目は x の i 行目の答え。
    """
    g = torch.Generator().manual_seed(0)
    half = n // 2
    x = torch.cat([torch.randint(1, 25, (half, 12), generator=g), torch.randint(25, 50, (n - half, 12), generator=g)])
    t = torch.cat([torch.tensor([[3.0, -3.0]]).repeat(half, 1), torch.tensor([[-3.0, 3.0]]).repeat(n - half, 1)])
    return x, t


# ---------- 返り値 ----------

def test_returns_model_and_history_like_train_model():
    """train_model と同じ形: (学習したモデル, エポックごとの平均損失のリスト)。"""
    m = tiny_model()
    x, t = two_kinds_of_sentences(16)
    out_model, history = train.train_distill(m, x, t, epochs=3, batch_size=8)
    assert out_model is m
    assert len(history) == 3 and all(isinstance(h, float) for h in history), f"history: {history}"


# ---------- 学べること ----------

def test_can_overfit_a_tiny_batch():
    m = tiny_model()
    x, t = two_kinds_of_sentences(8)
    _, history = train.train_distill(m, x, t, epochs=60, batch_size=8, lr=3e-3)
    assert history[-1] < history[0] * 0.5, f"{history[0]:.3f} -> {history[-1]:.3f}"


def test_student_learns_to_agree_with_the_teacher():
    """シャッフルして小さいバッチで回しても、文と先生の答えの組がずれないこと。

    落ちて一致率が 0.5 前後になるときは、x と先生の logits を別々の行で切り出している
    （x はシャッフルした番号で、先生の logits は i:i+batch_size で、など）。
    """
    m = tiny_model()
    x, t = two_kinds_of_sentences(64)
    train.train_distill(m, x, t, epochs=30, batch_size=16, lr=3e-3)
    m.eval()
    with torch.no_grad():
        agree = (m(x).argmax(1) == t.argmax(1)).float().mean().item()
    assert agree >= 0.95, f"先生と答えが一致した割合: {agree:.2f}"


# ---------- distill_loss の使い方 ----------

@pytest.fixture()
def loss_spy(monkeypatch):
    """distill.distill_loss を「本物を呼びつつ、呼ばれ方を記録する」ものに差し替える。"""
    real = distill.distill_loss
    calls = []

    def spy(student_logits, teacher_logits, T=2.0):
        calls.append({"T": T, "n": student_logits.shape[0], "same_rows": teacher_logits.shape[0] == student_logits.shape[0]})
        return real(student_logits, teacher_logits, T=T)

    monkeypatch.setattr(distill, "distill_loss", spy)
    return calls


def test_uses_distill_loss_with_the_given_T(loss_spy):
    """T は distill_loss まで届けること。

    落ちて「呼ばれた回数: 0」のときは、train.py で distill_loss を
    `from distill import distill_loss` の形で読み込んでいる（`import distill` にして distill.distill_loss と呼ぶ）。
    """
    x, t = two_kinds_of_sentences(20)
    train.train_distill(tiny_model(), x, t, epochs=1, batch_size=8, T=5.0)
    assert loss_spy, "呼ばれた回数: 0"
    assert {c["T"] for c in loss_spy} == {5.0}, f"届いた T: {sorted({c['T'] for c in loss_spy})}"


def test_T_defaults_to_2(loss_spy):
    x, t = two_kinds_of_sentences(8)
    train.train_distill(tiny_model(), x, t, epochs=1, batch_size=8)
    assert {c["T"] for c in loss_spy} == {2.0}


def test_every_sentence_is_used_once_per_epoch(loss_spy):
    """20文・バッチ 8 → 1エポック 3 回（8, 8, 4）。2エポックで 6 回。最後の端数の 4 文も使う。"""
    x, t = two_kinds_of_sentences(20)
    train.train_distill(tiny_model(), x, t, epochs=2, batch_size=8)
    assert [c["n"] for c in loss_spy] == [8, 8, 4, 8, 8, 4], f"1回ごとの文の数: {[c['n'] for c in loss_spy]}"
    assert all(c["same_rows"] for c in loss_spy), "生徒と先生で行の数が違う回がある"


def test_history_is_the_average_loss_per_sentence(monkeypatch):
    """history は「そのエポックの1文あたりの平均損失」（train_model と同じ数え方）。

    バッチの損失はそのバッチの文の平均なので、文の数を掛けて足し、最後に全部の文の数で割る。
    ここでは distill_loss を「バッチの文の数」をそのまま返すものに差し替える。
    20文・バッチ 8 → 損失は 8, 8, 4 → (8×8 + 8×8 + 4×4) / 20 = 7.2。

    落ちて 6.67 前後のときは、バッチの損失をそのまま足してバッチの回数（3）で割っている。
    """
    monkeypatch.setattr(distill, "distill_loss", lambda s, t, T=2.0: (s * 0).sum() + s.shape[0])
    x, t = two_kinds_of_sentences(20)
    _, history = train.train_distill(tiny_model(), x, t, epochs=2, batch_size=8)
    assert history == pytest.approx([7.2, 7.2]), f"history: {history}"


def test_teacher_logits_are_left_alone():
    """先生の logits は全エポック・全部の生徒で使い回す。書き換えないこと。"""
    x, t = two_kinds_of_sentences(16)
    t0 = t.clone()
    train.train_distill(tiny_model(), x, t, epochs=2, batch_size=8)
    assert torch.equal(t, t0)


def test_train_model_still_works():
    """train_model（正解ラベルで学ぶ方）は今まで通り。S3 の生徒①③はこちらを使う。"""
    m = tiny_model()
    x, t = two_kinds_of_sentences(8)
    _, history = train.train_model(m, x, t.argmax(1), epochs=5, batch_size=8)
    assert len(history) == 5 and not any(math.isnan(h) for h in history)
