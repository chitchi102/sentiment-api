# sentiment-api

英語の映画レビュー1文を negative / positive に分類する API。
小さい transformer（生徒）を、DistilBERT（先生）の確率から蒸留して作り、int8 に量子化して配っている。

- 稼働先: https://sentiment-api-3ptb.onrender.com （Render 無料枠。15分アクセスが無いと眠るので、最初の1回は起きるまで待つ）
- データ: SST-2（学習 67,349文・評価は validation 872文）

## 使い方

```bash
curl.exe -X POST https://sentiment-api-3ptb.onrender.com/predict -H "Content-Type: application/json" -d '{\"text\":\"the plot is thin but the cast is charming\"}'
```

```json
{"label": "positive", "confidence": 0.645}
```

`GET /healthz` は `{"status": "ok"}` を返す。

## 蒸留の結果（周3）

正解ラベルが 2,000件しか無く、残りの 65,349文はラベルなし、という状況を想定した。
ラベルなしの文を先生に採点させ、その確率（温度 T=2）で生徒を学習させると、正解ラベルだけで学習した生徒を大きく上回る。

| モデル | val 正解率 | サイズ (MB) | 推論 (秒) |
|---|---|---|---|
| 先生 DistilBERT（SST-2 で fine-tune 済み） | 0.911 | 267.9 | 13.50 |
| 生徒① 正解ラベル 2,000件だけ | 0.601 (0.592-0.615) | 5.2 | 0.38 |
| 生徒② 先生の確率だけ（67,349文） | 0.775 | 5.2 | 0.38 |
| 生徒③ 全件の正解ラベル（上限の目安） | 0.784 | 5.2 | 0.38 |
| **生徒② int8（この API が配っているもの）** | **0.775** | **4.8** | **0.20** |

- ②−① = +0.17（① の seed 幅 0.02 を大きく超える）。② と ③ は seed 1本ずつなので「同等」までしか言えない
- 生徒① は seed 0/1/2 の平均（括弧内は最小-最大）。① の train 損失は 0.17 まで下がるのに val は 0.60＝2,000件を丸暗記している
- 生徒3つは同じアーキテクチャ（語彙 8,000・約130万 params）なので、サイズと推論時間は ② で1回測った値を共用している
- 推論時間は CPU で1回測っただけで、ぶれる。先生は「文字→トークン変換込み・64文ずつ」、生徒は「変換済み・872文を1回」で、同じ物差しではない
- int8 にしてもサイズがあまり減らないのは、パラメータの大半が量子化の対象外の Embedding だから
- 学習は Colab の T4 GPU（生徒② は 24秒）。生の結果は `results/cycle3_s3_colab_t4.jsonl`、表は `python report.py` で `results/cycle3_table.md` に作り直せる（`teacher_logits.pt` と `requirements-dev.txt` が要る）

## 手元で動かす

```bash
uv venv
uv pip install -r requirements-dev.txt
pytest tests
uvicorn app:app --port 7860
```

配っているモデルは `student_teacher/`（`model.pt` と `tokenizer.json` の組）。
