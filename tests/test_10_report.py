"""周3 S4 のテスト（report.py）＝周3の結果を1枚の表にする。新しい概念は無し。

先生も SST-2 も使わない。小さい tensor と手で作った結果だけで回す。1秒以内。
実行: pytest tests/test_10_report.py

表の中身（本番の数字は `python report.py` で出す。ここでは組み立て方だけを見る）:
    先生／① 正解 2,000件（3 seed の平均と幅）／② 先生の確率だけ／③ 全件正解／② int8
    列 = val 正解率・サイズ (MB)・推論 (秒)
① ③ の重みは保存していない。正解率は S3 の結果ファイル（jsonl）から取り、
サイズと速度は生徒が3つとも同じ構造なので ② を bench.compare で1回測った値を使う。
"""

import json
import re

import pytest

torch = pytest.importorskip("torch")

import report  # noqa: E402

ROW_KEYS = {"name", "acc", "acc_min", "acc_max", "n_seeds", "size_mb", "seconds"}


def fake_results():
    """S3 の jsonl と同じ形の結果。① は seed 3本、② ③ は1本ずつ。"""
    return [
        {"mode": "small", "acc": 0.5, "int8_acc": 0.5, "train_seconds": 3.0},
        {"mode": "small", "acc": 0.7, "int8_acc": 0.7, "train_seconds": 3.0},
        {"mode": "teacher", "acc": 0.8, "int8_acc": 0.79, "train_seconds": 24.0},
        {"mode": "small", "acc": 0.6, "int8_acc": 0.6, "train_seconds": 3.0},
        {"mode": "full", "acc": 0.85, "int8_acc": 0.85, "train_seconds": 24.0},
    ]


def fake_bench():
    """bench.compare の戻り値と同じ形（fp32 の行、int8 の行）。"""
    return [
        {"name": "fp32", "size_mb": 5.2, "acc": 0.8, "seconds": 0.30},
        {"name": "int8", "size_mb": 4.8, "acc": 0.79, "seconds": 0.50},
    ]


# ---------- read_jsonl: 1行＝1つの辞書 ----------

def test_read_jsonl_gives_one_dict_per_line(tmp_path):
    path = tmp_path / "r.jsonl"
    rows = fake_results()
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert report.read_jsonl(path) == rows


def test_read_jsonl_skips_blank_lines(tmp_path):
    """ファイルの最後の空行などで落ちない。"""
    path = tmp_path / "r.jsonl"
    path.write_text('{"mode": "small", "acc": 0.5}\n\n{"mode": "full", "acc": 0.8}\n\n', encoding="utf-8")
    assert report.read_jsonl(path) == [{"mode": "small", "acc": 0.5}, {"mode": "full", "acc": 0.8}]


# ---------- acc_from_logits: 先生の正解率を logits から ----------

def test_acc_from_logits_counts_the_bigger_column():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 3.0], [5.0, -1.0]])
    y = torch.tensor([0, 1, 0, 1])  # 当たり: 1文目と2文目だけ
    acc = report.acc_from_logits(logits, y)
    assert isinstance(acc, float), f"型: {type(acc)}（tensor のままではなく .item() で数にする）"
    assert acc == pytest.approx(0.5)


# ---------- summarize: 同じ mode の結果をまとめる ----------

def test_summarize_three_seeds_gives_mean_and_range():
    s = report.summarize(fake_results(), "small")
    assert s == pytest.approx({"acc": 0.6, "acc_min": 0.5, "acc_max": 0.7, "n_seeds": 3})


def test_summarize_one_seed():
    s = report.summarize(fake_results(), "full")
    assert s == pytest.approx({"acc": 0.85, "acc_min": 0.85, "acc_max": 0.85, "n_seeds": 1})


def test_summarize_can_read_int8_acc():
    """key を変えると int8 の正解率をまとめる。"""
    s = report.summarize(fake_results(), "teacher", key="int8_acc")
    assert s["acc"] == pytest.approx(0.79)


def test_summarize_unknown_mode_is_an_error():
    """無い mode を黙って 0 や空で返すと、表の行が1つ静かに壊れる。"""
    with pytest.raises(ValueError):
        report.summarize(fake_results(), "tiny")


# ---------- teacher_row: 先生の行 ----------

def test_teacher_row():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 3.0], [5.0, -1.0]])
    y = torch.tensor([0, 1, 0, 1])
    r = report.teacher_row(logits, y, size_mb=268.0, seconds=15.0)
    assert set(r) == ROW_KEYS
    assert r["acc"] == pytest.approx(0.5)
    assert (r["acc_min"], r["acc_max"], r["n_seeds"]) == (r["acc"], r["acc"], 1)
    assert (r["size_mb"], r["seconds"]) == (268.0, 15.0)


# ---------- student_rows: 生徒の4行（結果 jsonl ＋ bench） ----------

def test_student_rows_order_and_accuracy():
    rows = report.student_rows(fake_results(), fake_bench())
    assert len(rows) == 4, f"行の数: {len(rows)}（①②③と② int8 の4行）"
    for r in rows:
        assert set(r) == ROW_KEYS
    assert [r["n_seeds"] for r in rows] == [3, 1, 1, 1]
    assert [r["acc"] for r in rows] == pytest.approx([0.6, 0.8, 0.85, 0.79])
    assert (rows[0]["acc_min"], rows[0]["acc_max"]) == pytest.approx((0.5, 0.7))


def test_student_rows_names_are_different():
    names = [r["name"] for r in report.student_rows(fake_results(), fake_bench())]
    assert all(isinstance(n, str) and n for n in names)
    assert len(set(names)) == 4, f"名前が重なっている: {names}"


def test_student_rows_size_and_speed_come_from_bench():
    """①②③ は fp32 の行、② int8 は int8 の行のサイズと速度を使う。"""
    rows = report.student_rows(fake_results(), fake_bench())
    assert [r["size_mb"] for r in rows] == [5.2, 5.2, 5.2, 4.8]
    assert [r["seconds"] for r in rows] == [0.30, 0.30, 0.30, 0.50]


def test_student_rows_accuracy_is_not_from_bench():
    """正解率は S3 の結果から取る（bench の acc は ② を読み直して測っただけ）。"""
    b = fake_bench()
    b[0]["acc"], b[1]["acc"] = 0.0, 0.0
    rows = report.student_rows(fake_results(), b)
    assert [r["acc"] for r in rows] == pytest.approx([0.6, 0.8, 0.85, 0.79])


# ---------- to_markdown: 行のリスト → Markdown の表 ----------

def md_rows():
    return [
        {"name": "先生", "acc": 0.91055, "acc_min": 0.91055, "acc_max": 0.91055, "n_seeds": 1,
         "size_mb": 267.83, "seconds": 15.123},
        {"name": "①", "acc": 0.60128, "acc_min": 0.59174, "acc_max": 0.61468, "n_seeds": 3,
         "size_mb": 5.198, "seconds": 0.1234},
    ]


def test_markdown_shape():
    lines = report.to_markdown(md_rows()).strip().split("\n")
    assert len(lines) == 2 + 2, "見出し1行＋区切り1行＋データ2行"
    for line in lines:
        assert line.startswith("|") and line.endswith("|"), f"表の行になっていない: {line!r}"
    assert set(lines[1].replace("|", "").replace(" ", "")) <= {"-", ":"}, f"2行目は区切り線: {lines[1]!r}"
    assert all(line.count("|") == 5 for line in lines), "どの行も4列"


def test_markdown_number_format():
    lines = report.to_markdown(md_rows()).strip().split("\n")
    teacher, small = lines[2], lines[3]
    assert "0.911" in teacher and "267.8" in teacher and "15.12" in teacher
    assert "(" not in teacher, "seed 1本の行に幅は書かない"
    # 幅のつなぎは – / - / ~ / 〜 のどれでもよい
    assert re.search(r"0\.601 \(0\.592\s*[–\-~〜]\s*0\.615\)", small), f"3 seed の行: {small!r}"
    assert "5.2" in small and "0.12" in small
