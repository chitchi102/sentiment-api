"""手順4のテスト（train.py）。偽 SST-2 で run まで回す。10秒前後。

train.py は data / tokens / model / artifacts を import して、部品を上から順に呼ぶだけのファイル。
実行: pytest tests/test_4_train.py
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")

import conftest  # noqa: E402
import model  # noqa: E402
import train  # noqa: E402


class FixedModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, x):
        self.was_training = self.training
        self.grad_enabled = torch.is_grad_enabled()
        return self.logits


# ---------- train_model ----------

def test_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    m = model.SentimentClassifier(vocab_size=50, d_model=32, nhead=4, num_layers=2, max_len=12)
    x = torch.randint(1, 50, (8, 12), generator=torch.Generator().manual_seed(0))
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    _, history = train.train_model(m, x, y, epochs=60, batch_size=8, lr=3e-3)
    assert len(history) == 60
    assert history[-1] < history[0] * 0.5, f"{history[0]:.3f} -> {history[-1]:.3f}"


# ---------- evaluate ----------

def test_evaluate_counts_correct_answers():
    logits = torch.tensor([[9., 0], [0, 9.], [9., 0], [9., 0]])  # 予測 0,1,0,0
    acc = train.evaluate(FixedModel(logits), torch.zeros(4, 8, dtype=torch.long), torch.tensor([0, 1, 1, 1]))
    assert isinstance(acc, float), f"float で返すこと（いまは {type(acc).__name__}）"
    assert acc == pytest.approx(0.5)


def test_evaluate_switches_to_eval_and_no_grad():
    m = FixedModel(torch.eye(2))
    m.train()
    train.evaluate(m, torch.zeros(2, 8, dtype=torch.long), torch.tensor([0, 1]))
    assert m.was_training is False, "evaluate の中で model.eval() を呼ぶこと"
    assert m.grad_enabled is False, "torch.no_grad() の中でモデルに通すこと"


# ---------- run ----------

@pytest.fixture()
def result(fake_sst2, tmp_path):
    torch.manual_seed(0)
    return train.run(n_train=400, epochs=3, out_dir=tmp_path), tmp_path


def test_run_returns_the_numbers(result):
    r, _ = result
    assert set(r) >= {"acc_before", "acc_after", "params", "train_seconds", "history"}
    assert len(r["history"]) == 3
    assert r["params"] > 0 and r["train_seconds"] > 0


def test_run_learns_an_easy_task(result):
    """偽 SST-2 は「良い単語が2つ入っていたら positive」だけの簡単な問題。3 epoch でほぼ満点になる。

    学習前は 2 択の当てずっぽう付近。落ちるときは、評価に使う文とラベルの組がずれている。
    """
    r, _ = result
    assert r["acc_after"] >= 0.9, r
    assert r["acc_after"] > r["acc_before"]


def test_run_saves_when_out_dir_is_given(result):
    _, out = result
    assert (out / "model.pt").exists() and (out / "tokenizer.json").exists()


def test_tokenizer_never_saw_validation(result):
    """トークナイザは train の文だけで学習する（validation を混ぜるとカンニング）。

    偽 SST-2 の validation にだけ出る単語が語彙に入っていたら、validation を混ぜている。
    """
    from tokenizers import Tokenizer

    _, out = result
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    assert tok.token_to_id(conftest.TEST_ONLY_WORD) is None
