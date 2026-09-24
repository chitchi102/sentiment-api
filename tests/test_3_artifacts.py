"""手順3のテスト（artifacts.py と predict.py）。保存 → 読み戻し → 1文を分類。

実行: pytest tests/test_3_artifacts.py
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")

import artifacts  # noqa: E402
import model  # noqa: E402
import predict  # noqa: E402
import tokens  # noqa: E402

TOY_TEXTS = ["a wonderful and moving film", "dull , boring and awful", "charming and smart"]


@pytest.fixture()
def saved(tmp_path):
    torch.manual_seed(0)
    tok = tokens.build_tokenizer(TOY_TEXTS, vocab_size=200)
    m = model.SentimentClassifier(vocab_size=tok.get_vocab_size())
    m.eval()
    artifacts.save_artifacts(m, tok, out_dir=tmp_path)
    with torch.no_grad():
        expected = m(tokens.encode_batch(tok, TOY_TEXTS))
    return tmp_path, m, expected


def test_save_writes_both_files(saved):
    path, _, _ = saved
    assert (path / "model.pt").exists() and (path / "tokenizer.json").exists()


def test_saves_the_state_dict_not_the_model(saved):
    path, m, _ = saved
    obj = torch.load(path / "model.pt")
    assert isinstance(obj, dict), "model そのものではなく model.state_dict() を保存すること"
    assert set(obj) == set(m.state_dict())


def test_load_gives_back_the_same_outputs(saved):
    path, _, expected = saved
    m, tok = artifacts.load_artifacts(out_dir=path)
    with torch.no_grad():
        got = m(tokens.encode_batch(tok, TOY_TEXTS))
    assert torch.allclose(got, expected, atol=1e-6)
    assert m.training is False, "load_artifacts の最後で model.eval() を呼ぶこと"


def test_predict_returns_label_and_confidence(saved):
    path, _, _ = saved
    m, tok = artifacts.load_artifacts(out_dir=path)
    out = predict.predict(m, tok, "a wonderful film")
    assert set(out) == {"label", "confidence"}
    assert out["label"] in model.LABELS
    assert isinstance(out["confidence"], float)
    assert 0.5 <= out["confidence"] <= 1.0, "2択の大きい方の確率なので 0.5 以上"
