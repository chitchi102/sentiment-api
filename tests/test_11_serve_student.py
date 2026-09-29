"""周3 S5 のテスト。API が配るモデルを、周2のもの（ルートの model.pt / tokenizer.json）から
生徒②（student_teacher/ の model.pt / tokenizer.json）に替える。

見ていること:
- app が読んだ重みとトークナイザが、両方とも student_teacher/ のもの（モデルとトークナイザは組で替える）
- /predict が ② の答えを返す（周2と ② で答えが割れる文を使う）
- Dockerfile の COPY どおりにファイルを並べたフォルダでも app が起動して ② で答える
  （＝Render のコンテナの中で「student_teacher/ が無い」で落ちない）

実行: pytest tests/test_11_serve_student.py
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("httpx")
import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from conftest import ROOT  # noqa: E402

STUDENT_DIR = ROOT / "student_teacher"

# 周2のモデルは negative 0.656、② は positive 0.645 と答える（2026-09-29 に手元 int8 で実測。
# 同じ日の Render は negative 0.656＝周2のまま）。
SPLIT_TEXT = "the plot is thin but the cast is charming"


@pytest.fixture(scope="module")
def student():
    assert (STUDENT_DIR / "model.pt").exists(), "student_teacher/model.pt が無い（S3 で保存した生徒②）"
    import artifacts
    return artifacts.load_artifacts(str(STUDENT_DIR))


@pytest.fixture(scope="module")
def app_module():
    here = os.getcwd()
    os.chdir(ROOT)
    try:
        import app
        yield app
    finally:
        os.chdir(here)


@pytest.fixture(scope="module")
def client(app_module):
    return TestClient(app_module.app)


def test_served_weights_are_student_teacher(app_module, student):
    """app の重みが ② のもの。Embedding（tok）は量子化の対象外なので、量子化の後でもそのまま比べられる。"""
    s_model, _ = student
    assert torch.equal(app_module.model.tok.weight, s_model.tok.weight), \
        "app が配っている重みが student_teacher/model.pt ではない"


def test_served_weights_are_not_cycle2(app_module):
    """念のため逆側: 周2の重み（ルート）ではない。ルートの model.pt を消した後はこの検査は飛ばす。"""
    if not (ROOT / "model.pt").exists():
        pytest.skip("ルートの model.pt が無い（消した後）")
    import artifacts
    old_model, _ = artifacts.load_artifacts(str(ROOT))
    assert not torch.equal(app_module.model.tok.weight, old_model.tok.weight), \
        "app がまだ周2のモデル（ルートの model.pt）を配っている"


def test_served_tokenizer_is_student_teacher(app_module, student):
    """トークナイザも ② のもの。周2とは単語→番号の対応が違うので、重みだけ替えると別の文として読まれる。"""
    _, s_tok = student
    assert app_module.tokenizer.get_vocab() == s_tok.get_vocab(), \
        "app のトークナイザが student_teacher/tokenizer.json ではない（重みと組になっていない）"


def test_predict_answers_like_student_teacher(client):
    r = client.post("/predict", json={"text": SPLIT_TEXT})
    assert r.status_code == 200, r.text
    assert r.json()["label"] == "positive", \
        f"周2のモデルと同じ答え {r.json()} ＝ ② が配られていない"


# ---- Dockerfile ----------------------------------------------------------------

IGNORE = shutil.ignore_patterns(".venv", ".git", "__pycache__", "*.pyc", ".pytest_cache")


def _copy_lines(dockerfile_text):
    lines = []
    for line in dockerfile_text.splitlines():
        parts = line.split()
        if parts and parts[0].upper() == "COPY":
            lines.append([p for p in parts[1:] if not p.startswith("--")])
    return lines


def _copy_like_docker(parts, src_root, workdir):
    """Docker の COPY と同じ置き方で src_root から workdir（コンテナの /app 役）へ写す。

    - 送り先が「/」で終わる・送り元が2つ以上 → 送り先はフォルダ
    - 送り元がフォルダ → フォルダの「中身」が送り先に入る（フォルダ自体は作られない）
    """
    *srcs, dest = parts
    if dest.startswith("/app"):
        dest = "." + dest[len("/app"):]
    assert not dest.startswith("/"), f"WORKDIR（/app）の外へ COPY している: {dest}"
    dest_path = workdir / dest
    dest_is_dir = dest.endswith("/") or len(srcs) > 1
    for src in srcs:
        matches = sorted(src_root.glob(src.rstrip("/"))) if any(c in src for c in "*?[") \
            else [src_root / src.rstrip("/")]
        for m in matches:
            assert m.exists(), f"COPY の送り元が無い: {src}"
            if m.is_dir():
                shutil.copytree(m, dest_path, dirs_exist_ok=True, ignore=IGNORE)
            elif dest_is_dir:
                dest_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(m, dest_path / m.name)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(m, dest_path)


CONTAINER_SCRIPT = f"""
import json, sys
sys.modules['datasets'] = None
import app
from fastapi.testclient import TestClient
r = TestClient(app.app).post('/predict', json={{'text': {SPLIT_TEXT!r}}})
print(json.dumps(r.json()))
"""


def test_dockerfile_ships_student_teacher(tmp_path):
    """Dockerfile の COPY だけでファイルを並べたフォルダ（＝コンテナの中身）で app を起動し、② で答える。

    落ちたら、そのフォルダに何があるか（下の stderr と listing）を見る。
    """
    copies = _copy_lines((ROOT / "Dockerfile").read_text(encoding="utf-8"))
    assert copies, "Dockerfile に COPY が無い"
    for parts in copies:
        if parts == ["requirements.txt", "."]:
            continue
        _copy_like_docker(parts, ROOT, tmp_path)

    listing = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    r = subprocess.run([sys.executable, "-c", CONTAINER_SCRIPT], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, f"コンテナの中身で app が起動しない。\n中身: {listing}\n" + r.stderr[-800:]
    body = json.loads(r.stdout.strip().splitlines()[-1])
    assert body["label"] == "positive", f"コンテナの中身が周2のモデルで答えた {body}\n中身: {listing}"
