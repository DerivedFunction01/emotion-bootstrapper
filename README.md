**On the local machine: tokenize and zip**
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

Adjust `--num-proc` to match your local CPU cores. If you want, you can also leave it out and let the script choose the default.

What this produces:
- `./emotion_cache/`
- `./emotion_cache.zip`

**On the A100 in the cloud: run inference from the zip**
Put the zip somewhere accessible on the A100 instance, then run:

```bash
.venv/bin/python run_bootstrap.py \
  --cache-zip ./emotion_cache.zip \
  --work-dir ./bootstrap_work \
  --output-path ./emotion_bootstrapped.parquet \
  --model facebook/bart-large-mnli \
  --batch-size 32
```

**Where to upload the zip file**
Upload `emotion_cache.zip` to the A100 machine into any path you can read from, for example:
- the repo root: `./emotion_cache.zip`
- a data folder: `./data/emotion_cache.zip`
- an attached volume or mounted storage path

Then point `--cache-zip` at that exact path.