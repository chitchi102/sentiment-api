"""周3 S3b のテスト（students.py の run_student）。偽 SST-2 と偽の先生 logits で回す。30秒前後。

生徒を3通り作って測る関数。本番（67,349文）は S3c で Colab の GPU で回す。
  "small"   ① 正解ラベルは先頭 n_label 件だけ     → train.train_model
  "teacher" ② 全文。正解は見ず、先生の logits だけ → train.train_distill
  "full"    ③ 全文の正解ラベル（上限の目安）       → train.train_model
実行: pytest tests/test_9_students.py
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")

import artifacts  # noqa: E402
import conftest  # noqa: E402
import students  # noqa: E402
import train  # noqa: E402

N_TRAIN = 400  # conftest の偽 SST-2 の train の文の数
N_VAL = 60


def write_teacher_file(path, labels, n_val=N_VAL):
    """偽の先生: 正解の向きに ±3 の logits を出す（本物の先生と同じ保存形式）。"""
    y = torch.tensor(labels)
    train_logits = torch.stack([(1 - y) * 3.0 - y * 3.0, y * 3.0 - (1 - y) * 3.0], dim=1)
    torch.save({"train_logits": train_logits, "val_logits": torch.zeros(n_val, 2)}, path)
    return train_logits


@pytest.fixture()
def teacher_file(fake_sst2, tmp_path):
    path = tmp_path / "teacher_logits.pt"
    logits = write_teacher_file(path, fake_sst2["ds"]["train"]["label"])
    return {"path": path, "logits": logits, "labels": list(fake_sst2["ds"]["train"]["label"])}


@pytest.fixture()
def spies(monkeypatch):
    """train.train_model / train.train_distill を「呼ばれ方を記録して、何もせず返す」ものに差し替える。

    落ちて「呼ばれた回数: 0」のときは、students.py で `from train import ...` の形で読み込んでいる
    （`import train` にして train.train_model と呼ぶ）。
    """
    calls = []

    def fake_train_model(model, x, y, epochs=3, **kw):
        calls.append({"fn": "train_model", "n": x.shape[0], "x": x.clone(), "y": y.tolist(), "epochs": epochs})
        return model, [0.0] * epochs

    def fake_train_distill(model, x, teacher_logits, epochs=3, **kw):
        calls.append({"fn": "train_distill", "n": x.shape[0], "t": teacher_logits.clone(), "epochs": epochs})
        return model, [0.0] * epochs

    monkeypatch.setattr(train, "train_model", fake_train_model)
    monkeypatch.setattr(train, "train_distill", fake_train_distill)
    return calls


# ---------- 3通りの作り分け ----------

def test_small_uses_only_the_first_n_label_answers(teacher_file, spies):
    """① は先頭の n_label 文とその正解だけ。

    比べるために "full" も回し、full が受け取った文の先頭 50 行と同じかを見る
    （偽 SST-2 のラベルは 0,1,0,1… なので、ラベルだけでは先頭と末尾の区別がつかない）。
    """
    students.run_student("small", teacher_path=teacher_file["path"], n_label=50, epochs=2)
    assert [c["fn"] for c in spies] == ["train_model"], f"呼ばれたもの: {[c['fn'] for c in spies]}"
    assert spies[0]["n"] == 50, f"学習に使った文の数: {spies[0]['n']}"
    assert spies[0]["y"] == teacher_file["labels"][:50], "先頭の 50 件の正解ではない"
    assert spies[0]["epochs"] == 2
    students.run_student("full", teacher_path=teacher_file["path"], epochs=1)
    assert torch.equal(spies[0]["x"], spies[1]["x"][:50]), "先頭の 50 文ではない"


def test_teacher_uses_all_texts_and_never_the_answers(teacher_file, spies):
    """② は正解ラベルを1つも見ない。train_model を呼んだら（正解で学んだら）失格。"""
    students.run_student("teacher", teacher_path=teacher_file["path"], epochs=2)
    assert [c["fn"] for c in spies] == ["train_distill"], f"呼ばれたもの: {[c['fn'] for c in spies]}"
    assert spies[0]["n"] == N_TRAIN
    assert torch.equal(spies[0]["t"], teacher_file["logits"]), "先生の logits がファイルの train_logits と違う"
    assert spies[0]["epochs"] == 2


def test_full_uses_all_answers(teacher_file, spies):
    students.run_student("full", teacher_path=teacher_file["path"], epochs=2)
    assert [c["fn"] for c in spies] == ["train_model"]
    assert spies[0]["n"] == N_TRAIN and spies[0]["y"] == teacher_file["labels"]


def test_unknown_mode_is_an_error(teacher_file, spies):
    with pytest.raises(ValueError):
        students.run_student("big", teacher_path=teacher_file["path"])
    assert spies == [], "間違った mode なのに学習を始めた"


def test_teacher_file_must_match_the_sentences(fake_sst2, tmp_path, spies):
    """先生の logits の行数と train の文の数が違ったら止める（i 行目が i 番目の文の答え、が崩れる）。

    文の数は先生の logits の行数から決める（data.load_data(n_train=行数)）。
    偽 SST-2 は 400 文しか無いので、500 行のファイルを渡すと 400 文しか返ってこない → ValueError。
    """
    path = tmp_path / "t.pt"
    write_teacher_file(path, [0, 1] * 250)
    with pytest.raises(ValueError):
        students.run_student("teacher", teacher_path=path)
    assert spies == [], "数が合わないのに学習を始めた"


# ---------- 本当に回す（学習の差し替えなし） ----------

@pytest.fixture()
def real_run(teacher_file, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    r = students.run_student("teacher", teacher_path=teacher_file["path"], epochs=3, seed=0, out_dir=out)
    return r, out


def test_returns_the_numbers(real_run):
    r, _ = real_run
    assert set(r) >= {"mode", "n_texts", "acc", "int8_acc", "train_seconds", "history"}, f"キー: {sorted(r)}"
    assert r["mode"] == "teacher" and r["n_texts"] == N_TRAIN
    assert isinstance(r["acc"], float) and isinstance(r["int8_acc"], float)
    assert r["train_seconds"] > 0 and len(r["history"]) == 3


def test_student_learns_from_the_teacher_alone(real_run):
    """偽の先生は正解の向きに答える。その答えだけで学んで、validation を当てられること。"""
    r, _ = real_run
    assert r["acc"] >= 0.9, r
    assert abs(r["acc"] - r["int8_acc"]) <= 0.05, "int8 にしたら大きく変わった（周2では変わらなかった）"


def test_saves_a_model_the_api_can_load(real_run):
    """out_dir を渡したら model.pt と tokenizer.json を保存（S5 で API をこの生徒に差し替える）。"""
    _, out = real_run
    m, tok = artifacts.load_artifacts(str(out))
    assert m.tok.num_embeddings == tok.get_vocab_size()


def test_tokenizer_never_saw_validation(real_run):
    """トークナイザは train の文（ラベルなしでよい）だけで作る。validation を混ぜるとカンニング。"""
    from tokenizers import Tokenizer

    _, out = real_run
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    assert tok.token_to_id(conftest.TEST_ONLY_WORD) is None


def test_same_seed_gives_the_same_student(teacher_file):
    """seed を固定すれば同じ生徒ができる（① を seed 0/1/2 で比べるため）。

    モデルを作る直前に torch.manual_seed(seed) を呼ぶ。
    """
    a = students.run_student("small", teacher_path=teacher_file["path"], n_label=100, epochs=1, seed=3)
    b = students.run_student("small", teacher_path=teacher_file["path"], n_label=100, epochs=1, seed=3)
    assert a["history"] == b["history"] and a["acc"] == b["acc"]
