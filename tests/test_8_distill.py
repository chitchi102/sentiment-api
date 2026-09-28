"""周3 S2 のテスト（distill.py の distill_loss）＝周3の新要素「蒸留の損失」。

先生の確率に生徒の確率を近づける損失。小さい tensor だけで回す。1秒以内。
先生も SST-2 も使わない。
実行: pytest tests/test_8_distill.py

手で計算した値（テストの中の 0.1308...）の出し方:
    先生 [ln3, 0] → softmax = [0.75, 0.25] ＝ p
    生徒 [0, 0]   → softmax = [0.5, 0.5]   ＝ q
    KL(p||q) = 0.75*ln(0.75/0.5) + 0.25*ln(0.25/0.5) = 0.130812...
"""

import math

import pytest

torch = pytest.importorskip("torch")

import distill  # noqa: E402

LN3 = math.log(3)
KL_P_Q = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)  # 0.1308...


def row(*xs):
    """1文ぶんの logits（形 (1, クラス数)）。"""
    return torch.tensor([list(xs)])


# ---------- 形 ----------

def test_returns_a_single_number():
    """バッチ全体で1つの数（0次元 tensor）を返す。loss.backward() を呼べる形。"""
    s = torch.randn(64, 2)
    t = torch.randn(64, 2)
    loss = distill.distill_loss(s, t, T=2.0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0, f"返り値の形: {tuple(loss.shape)}"


# ---------- 値 ----------

def test_zero_when_student_equals_teacher():
    """先生と同じ答えなら、もう学ぶことはない＝0。"""
    t = torch.randn(8, 2)
    assert distill.distill_loss(t.clone(), t, T=2.0).item() == pytest.approx(0.0, abs=1e-6)


def test_never_negative():
    torch.manual_seed(0)
    for _ in range(20):
        loss = distill.distill_loss(torch.randn(16, 2) * 3, torch.randn(16, 2) * 3, T=2.0)
        assert loss.item() >= -1e-6, f"負になった: {loss.item()}"


def test_matches_hand_computed_value_at_T1():
    """T=1 は普通の KL(先生 || 生徒)。

    落ちたときの見分け方（期待 0.1308）:
      0.1438 前後 → 先生と生徒が逆（KL(生徒 || 先生) を計算している）
      0.0654 前後（ちょうど半分）→ 文の数だけでなくクラスの数でも割っている
      負の数 → 生徒側に log を取っていない（確率のまま渡している）
    """
    loss = distill.distill_loss(row(0.0, 0.0), row(LN3, 0.0), T=1.0)
    assert loss.item() == pytest.approx(KL_P_Q, rel=1e-4), f"{loss.item():.6f}（期待 {KL_P_Q:.6f}）"


def test_temperature_divides_logits_and_multiplies_T_squared():
    """T=2 では logits を 2 で割ってから softmax し、最後に T²=4 を掛ける。

    先生 [2ln3, 0] を 2 で割ると [ln3, 0] ＝上のテストと同じ p。生徒 [0, 0] は割っても [0, 0]。
    だから答えは 上の KL の 4 倍。

    落ちたときの見分け方（期待 0.5232）:
      0.1308 前後 → T² を掛け忘れている
      1.4723 前後 → logits を T で割り忘れている（T² だけ掛けている）
      0.5754 前後 → 先生と生徒が逆
      0.2616 前後（ちょうど半分）→ 文の数だけでなくクラスの数でも割っている
    """
    loss = distill.distill_loss(row(0.0, 0.0), row(2 * LN3, 0.0), T=2.0)
    assert loss.item() == pytest.approx(4 * KL_P_Q, rel=1e-4), f"{loss.item():.6f}（期待 {4 * KL_P_Q:.6f}）"


def test_averages_over_the_batch():
    """バッチの「平均」（合計ではない）。同じ文を 5 回並べても値は 1 回のときと同じ。

    バッチの大きさで損失の大きさが変わると、学習率の決め方がバッチの大きさに縛られる。
    """
    s, t = row(0.0, 0.0), row(LN3, 0.0)
    one = distill.distill_loss(s, t, T=1.0)
    five = distill.distill_loss(s.repeat(5, 1), t.repeat(5, 1), T=1.0)
    assert five.item() == pytest.approx(one.item(), rel=1e-5), f"1文: {one.item():.6f} / 同じ文を5つ: {five.item():.6f}"


def test_works_for_more_than_two_classes():
    """クラス数を 2 と決め打ちしない（先生の出力が何クラスでも同じ式）。"""
    s = torch.zeros(1, 3)                  # 生徒: 3クラス均等 = [1/3, 1/3, 1/3]
    t = torch.log(torch.tensor([[0.5, 0.25, 0.25]]))  # 先生の確率がそのまま [0.5, 0.25, 0.25]
    expected = 0.5 * math.log(0.5 * 3) + 2 * 0.25 * math.log(0.25 * 3)
    assert distill.distill_loss(s, t, T=1.0).item() == pytest.approx(expected, rel=1e-4)


def test_T_defaults_to_2():
    """S3 の参照解は T=2 で測った。T を渡さなければ 2。"""
    s, t = row(0.0, 0.0), row(2 * LN3, 0.0)
    assert distill.distill_loss(s, t).item() == pytest.approx(4 * KL_P_Q, rel=1e-4)


# ---------- 学習に使えること ----------

def test_one_step_moves_student_toward_teacher():
    """生徒の logits で微分でき、その向きに1歩進むと損失が下がる（S3 で optimizer に渡す）。"""
    torch.manual_seed(0)
    t = torch.randn(32, 2) * 3
    s = torch.randn(32, 2, requires_grad=True)
    loss = distill.distill_loss(s, t, T=2.0)
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    with torch.no_grad():
        s_next = s - 0.1 * s.grad
    assert distill.distill_loss(s_next, t, T=2.0).item() < loss.item()


def test_inputs_are_left_alone():
    """渡した logits を書き換えない（先生の logits は全エポックで使い回す）。"""
    s = torch.randn(4, 2)
    t = torch.randn(4, 2)
    s0, t0 = s.clone(), t.clone()
    distill.distill_loss(s, t, T=2.0)
    assert torch.equal(s, s0) and torch.equal(t, t0)
