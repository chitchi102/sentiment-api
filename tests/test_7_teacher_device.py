"""周3 S1b のテスト（teacher.py に device と保存を足す）。本物の先生も GPU も使わない。数秒。

GPU が無くても「どこへ動かしたか」だけは確かめられる: torch の "meta" デバイス
（形だけあって中身の無い Tensor の置き場）に動かせと言って、先生に届いた入力の置き場を見る。
GPU で本当に動くかは Colab で test_7_teacher_gpu.py を回して確かめる。
実行: pytest tests/test_7_teacher_device.py
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.modeling_outputs import SequenceClassifierOutput  # noqa: E402

import teacher  # noqa: E402
from test_7_teacher import FakeTeacher, FakeTokenizer, texts_of_length  # noqa: E402


class DeviceSpy(torch.nn.Module):
    """届いた input_ids / attention_mask の置き場を記録する偽の先生。返す logits は CPU の 0。"""

    def __init__(self):
        super().__init__()
        self.devices = []

    def forward(self, input_ids, attention_mask=None):
        self.devices.append((input_ids.device.type, attention_mask.device.type))
        return SequenceClassifierOutput(logits=torch.zeros(input_ids.shape[0], 2))


# ---------- teacher_logits(..., device=) ----------

def test_inputs_are_moved_to_the_device():
    """先生と入力は同じ置き場にないと計算できない。tokenizer の返り値を device へ動かすこと。

    input_ids だけでなく attention_mask も動かす（tokenizer の返り値ごと動かせば両方動く）。
    """
    spy = DeviceSpy()
    teacher.teacher_logits(FakeTokenizer(), spy, texts_of_length(70), device="meta")
    assert spy.devices == [("meta", "meta"), ("meta", "meta")], f"先生に届いた置き場: {spy.devices}"


def test_device_defaults_to_cpu():
    """device を渡さなければ今まで通り CPU（S1a のテストがそのまま通る）。"""
    spy = DeviceSpy()
    teacher.teacher_logits(FakeTokenizer(), spy, texts_of_length(3))
    assert spy.devices == [("cpu", "cpu")]


# ---------- save_teacher_logits ----------

@pytest.fixture()
def fake_teacher(monkeypatch):
    """teacher.load_teacher を偽物に差し替え、渡された device を記録する。

    device="meta" のときは DeviceSpy を返す（中身の無い meta の Tensor では文字数の計算ができないため）。
    """
    made = {"devices": [], "models": []}

    def fake_load_teacher(device="cpu"):
        m = DeviceSpy() if device == "meta" else FakeTeacher()
        made["devices"].append(device)
        made["models"].append(m)
        return FakeTokenizer(), m

    monkeypatch.setattr(teacher, "load_teacher", fake_load_teacher)
    return made


def test_saves_train_and_validation_logits(fake_sst2, fake_teacher, tmp_path):
    """保存するのは辞書1つ。train（生徒に教える分）と validation（S4 の表で先生の正解率を出す分）。"""
    path = tmp_path / "teacher_logits.pt"
    teacher.save_teacher_logits(path, n_train=100)
    saved = torch.load(path)
    assert set(saved) == {"train_logits", "val_logits"}, f"保存したキー: {sorted(saved)}"
    assert saved["train_logits"].shape == (100, 2), f"train の形: {tuple(saved['train_logits'].shape)}"
    val_n = len(fake_sst2["ds"]["validation"])
    assert saved["val_logits"].shape == (val_n, 2), f"validation の形: {tuple(saved['val_logits'].shape)}"


def test_rows_line_up_with_data_load_data(fake_sst2, fake_teacher, tmp_path):
    """i 行目が data.load_data の i 番目の文の logits であること（S3 で生徒の文と突き合わせる）。

    偽の先生は logits の 1 列目に文字数を入れて返す。
    """
    path = tmp_path / "teacher_logits.pt"
    teacher.save_teacher_logits(path, n_train=100)
    saved = torch.load(path)
    train_sents = fake_sst2["ds"]["train"]["sentence"][:100]
    val_sents = list(fake_sst2["ds"]["validation"]["sentence"])
    assert saved["train_logits"][:, 0].tolist() == [float(len(s)) for s in train_sents], "train の行の順番が違う"
    assert saved["val_logits"][:, 0].tolist() == [float(len(s)) for s in val_sents], "validation の行の順番が違う"


def test_device_is_passed_to_load_teacher(fake_sst2, fake_teacher, tmp_path):
    """Colab では device="cuda" で呼ぶ。それが先生の読み込みと、先生に渡す入力の両方に届くこと。"""
    teacher.save_teacher_logits(tmp_path / "t.pt", n_train=10, device="meta")
    assert fake_teacher["devices"] == ["meta"], f"load_teacher に渡った device: {fake_teacher['devices']}"
    spy = fake_teacher["models"][0]
    assert spy.devices and set(spy.devices) == {("meta", "meta")}, f"先生に届いた入力の置き場: {set(spy.devices)}"

