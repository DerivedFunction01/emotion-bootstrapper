**Local machine: build and zip**
Run from the repo root:

```bash
.venv/bin/python build_cache.py \
  --dataset-path dair-ai/emotion \
  --dataset-config unsplit \
  --text-column text \
  --cache-dir ./emotion_cache \
  --zip-path ./emotion_cache.zip \
  --num-proc 8 \
  --tokenize-batch-size 100
```

That produces:
- `./emotion_cache/tokenized.parquet`
- `./emotion_cache/texts.parquet`
- `./emotion_cache/cache_meta.json`
- `./emotion_cache.zip`

**Upload location in the cloud**
Upload `emotion_cache.zip` to any writable/readable path on the A100 instance, for example:
- `/home/youruser/emotion_cache.zip`
- `./emotion_cache.zip` in the repo root
- `/mnt/data/emotion_cache.zip` if your cloud setup provides mounted storage

The only requirement is that the A100 can read the file path you pass to `--cache-zip`.

**On the A100: run inference**
After uploading and unpacking are handled by the script itself, run:

```bash
.venv/bin/python run_bootstrap.py \
  --cache-zip ./emotion_cache.zip \
  --work-dir ./bootstrap_work \
  --output-path ./emotion_bootstrapped.parquet \
  --model facebook/bart-large-mnli \
  --batch-size 32
```