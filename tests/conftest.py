"""テスト全体で使う道具。pytest がテストの前に自動で読む（import しなくてよい）。

いちばん大事なのは fake_sst2: 本物の SST-2 をダウンロードせずに、
data.py の load_dataset を「小さい偽の SST-2 を返す関数」に差し替える。
だからネットが無くても data / train のテストが数秒で回る。
"""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

POS_WORDS = ["wonderful", "brilliant", "moving", "delightful", "superb", "charming"]
NEG_WORDS = ["awful", "boring", "dull", "tedious", "painful", "clumsy"]
FILLER = ["the", "film", "is", "a", "story", "and", "cast", "with", "its", "plot"]
TEST_ONLY_WORD = "zyxwvq"  # validation にだけ出る単語（トークナイザのカンニング検出用）


def _sentences(n, seed, extra_word=None):
    rng = random.Random(seed)
    sents, labels = [], []
    for i in range(n):
        label = i % 2  # 0 と 1 を交互に並べる（どこで切っても両方入る）
        words = rng.sample(FILLER, 4) + rng.sample(POS_WORDS if label else NEG_WORDS, 2)
        if extra_word:
            words.append(extra_word)
        rng.shuffle(words)
        sents.append(" ".join(words))
        labels.append(label)
    return sents, labels


def make_fake_sst2(n_train=400, n_val=60, n_test=30):
    """本物の SST-2 と同じ形（列名 idx / sentence / label、split 3つ）の小さい偽物。

    本物と同じく test split のラベルは全部 -1（＝答えが公開されていない）。
    """
    from datasets import Dataset, DatasetDict

    tr_s, tr_l = _sentences(n_train, seed=0)
    va_s, va_l = _sentences(n_val, seed=1, extra_word=TEST_ONLY_WORD)
    te_s, _ = _sentences(n_test, seed=2)
    return DatasetDict({
        "train": Dataset.from_dict({"idx": list(range(n_train)), "sentence": tr_s, "label": tr_l}),
        "validation": Dataset.from_dict({"idx": list(range(n_val)), "sentence": va_s, "label": va_l}),
        "test": Dataset.from_dict({"idx": list(range(n_test)), "sentence": te_s, "label": [-1] * n_test}),
    })


@pytest.fixture()
def fake_sst2(monkeypatch):
    """data.load_dataset を偽物に差し替え、呼ばれた id を記録する。

    落ちて AttributeError が出るときは、data.py の先頭に
    `from datasets import load_dataset` が無い（`import datasets` の形だと差し替えられない）。
    """
    pytest.importorskip("datasets")
    import data

    calls = []
    fake = make_fake_sst2()

    def fake_load_dataset(name, *args, **kwargs):
        calls.append(name)
        return fake

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)
    return {"calls": calls, "ds": fake}
