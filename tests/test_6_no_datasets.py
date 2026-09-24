"""手順6のテスト。datasets が入っていない環境（＝コンテナ）でも app が起動すること。

周1では train.py の先頭の `import datasets` でコンテナが落ち、関数の中へ import を移してしのいだ。
周2はファイルを分けたので、datasets を読むのは data.py だけ。
**app.py から import を辿って data.py に届かなければ**、この検査は何もしなくても通る。
実行: pytest tests/test_6_no_datasets.py
"""

import subprocess
import sys

import pytest

from conftest import ROOT

SCRIPT = "import sys; sys.modules['datasets'] = None; import app; print('ok')"


def test_app_starts_without_datasets():
    if not (ROOT / "model.pt").exists():
        pytest.skip("model.pt が無い（先に python train.py）")
    r = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, "datasets が無いと app が起動しない:\n" + r.stderr[-800:]
    assert "ok" in r.stdout
