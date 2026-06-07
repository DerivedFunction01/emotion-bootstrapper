**Local machine: build and zip**
Run from the repo root:

```bash
.venv/bin/python build_cache.py \
  --zip-path ./emotion_cache.zip 
```

That produces:
- `./emotion_cache/tokenized.parquet`
- `./emotion_cache/texts.parquet`
- `./emotion_cache/cache_meta.json`
- `./emotion_cache.zip`

**Upload location in the cloud**
Upload `emotion_cache.zip` to any writable/readable path on the Colab instance, for example:
- `/home/youruser/emotion_cache.zip`
- `./emotion_cache.zip` in the repo root
- `/mnt/data/emotion_cache.zip` if your cloud setup provides mounted storage

The only requirement is that the Colab can read the file path you pass to `--cache-zip`.

**Use a different source dataset**
The bootstrap pipeline does not depend on the original source name. It only needs a dataset with one text column (for example, `text`) and the same cache format produced by `build_cache.py`.

To use another dataset, point `build_cache.py` at that Hugging Face dataset and choose the right split/config/column:

```bash
.venv/bin/python build_cache.py \
  --dataset-path DerivedFunction01/arxiv_sample_first_sentences \
  --cache-dir ./arxiv_cache \
  --dataset-config default \
  --text-column text \
  --zip-path ./arxiv_cache.zip
```

```bash
.venv/bin/python build_cache.py \
  --dataset-path DerivedFunction01/urgency_synthetic \
  --cache-dir ./urgency_cache \
  --dataset-config default \
  --text-column text \
  --zip-path ./urgency_cache.zip
```

Notes:
- `--dataset-path` is the Hugging Face dataset repo or path.
- `--dataset-config` is the dataset split/config name your source uses (for example `unsplit`, `default`, or another config defined by the dataset).
- `--text-column` must point to the column that contains the text you want to score.
- The rest of the workflow stays the same: upload the generated `emotion_cache.zip`, start the GPU servers, and run the bootstrap job.

If your dataset is already preprocessed and you only want to change the source text, keep the same cache schema and just swap the `--dataset-path` / `--dataset-config` / `--text-column` values.

**Using the Multilingual Model**
You can run the bootstrapper in multilingual mode, which natively uses `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` and automatically maps the text to correctly translated semantic hypotheses.

First, build the cache using the `--multilingual` flag:

```bash
.venv/bin/python build_cache.py \
  --dataset-path DerivedFunction01/dair-ai_emotions_sample \
  --multilingual \
  --zip-path ./emotion_cache_multilingual.zip
```

Then, start the server cluster using the `--multilingual` flag so it serves the matching weights:

```bash
.venv/bin/python server_manager.py start --multilingual
```

**Start the server cluster**
This auto-detects the available CUDA GPUs and starts one server per GPU:

```bash
.venv/bin/python server_manager.py start
```

Useful companion commands:

```bash
.venv/bin/python server_manager.py status
.venv/bin/python server_manager.py stop
```

**Run bootstrap**
`run_bootstrap.py` reads `server_cluster.json` by default, so you do not need to pass server URLs manually:

```bash
.venv/bin/python run_bootstrap.py \
  --cache-zip ./emotion_cache.zip \
  --batch-size 24
```

If you want to keep the output in a different work folder to avoid collisions with an existing run, pass `--work-dir` explicitly:

```bash
.venv/bin/python run_bootstrap.py \
  --cache-zip ./arxiv_cache.zip \
  --work-dir ./bootstrap_work_arxiv \
  --batch-size 4
```

```bash
.venv/bin/python run_bootstrap.py \
  --cache-zip ./urgency_cache.zip \
  --work-dir ./bootstrap_work_urgency \
  --batch-size 8
```

If you want to point at a different registry file, pass `--server-registry`.
