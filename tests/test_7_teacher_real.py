"""周3 S1a のテスト（本物の先生）。初回だけ先生（約 270MB）をダウンロードする。

偽物のテスト（test_7_teacher.py）は形しか見ない。こちらは「本物の先生を正しく読み込めたか」を見る。
CPU で 100件なら数秒。
実行: pytest tests/test_7_teacher_real.py
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")

import data  # noqa: E402
import teacher  # noqa: E402


@pytest.fixture(scope="module")
def loaded():
    return teacher.load_teacher()


def test_it_is_the_sst2_distilbert(loaded):
    """id2label の向きが SST-2 のラベル（0=negative, 1=positive）と同じ先生であること。"""
    _, m = loaded
    assert teacher.TEACHER_NAME == "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    assert m.config.id2label == {0: "NEGATIVE", 1: "POSITIVE"}
    assert sum(p.numel() for p in m.parameters()) == 66_955_010


def test_teacher_is_in_eval_mode(loaded):
    """HF の from_pretrained は最初から eval で返す（周2の自分のモデルは作った直後は train）。

    なので .eval() を書き忘れてもここは通る。書いておくのは「採点役なので dropout を切る」の明示。
    """
    _, m = loaded
    assert m.training is False


def test_obvious_sentences(loaded):
    tok, m = loaded
    out = teacher.teacher_logits(tok, m, ["a wonderful, moving film", "a boring, painful mess"])
    assert out.argmax(1).tolist() == [1, 0], f"logits: {out.tolist()}"


def test_long_text_does_not_crash(loaded):
    """DistilBERT は 512 トークンまでしか読めない。切り捨てていないとここで落ちる。"""
    tok, m = loaded
    out = teacher.teacher_logits(tok, m, ["good " * 600])
    assert out.shape == (1, 2)


def test_teacher_beats_the_student_on_100_validation_sentences(loaded):
    """validation の先頭 100件。先生は全体で 0.911（周2の生徒は 0.688）。"""
    tok, m = loaded
    _, _, X_val, y_val = data.load_data(n_train=10)
    out = teacher.teacher_logits(tok, m, X_val[:100])
    acc = (out.argmax(1) == torch.tensor(y_val[:100])).float().mean().item()
    assert out.shape == (100, 2)
    assert acc >= 0.85, f"先頭100件の正解率: {acc:.3f}"


def test_load_teacher_moves_the_teacher_to_the_device():
    """S1b: load_teacher(device) で先生の重みがその置き場へ動くこと（手元は GPU が無いので meta で見る）。"""
    _, m = teacher.load_teacher(device="meta")
    assert {p.device.type for p in m.parameters()} == {"meta"}
