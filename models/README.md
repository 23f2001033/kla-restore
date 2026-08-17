# models/

`run.py` loads its checkpoint from this directory automatically, preferring
`best.pt` and otherwise taking the first `.pt`/`.pth` file it finds.

**The trained checkpoint is produced by the Kaggle training run** — see
`notebooks/kaggle_train_standalone.ipynb`. After training finishes, download
`best.pt` from the session and place it here:

```
models/best.pt
```

Then verify the submission runs end to end:

```bash
python run.py <input-dir> <output-dir>
```

`run.py` refuses to run against a checkpoint containing non-finite parameters,
so a corrupted or diverged checkpoint fails loudly instead of silently emitting
garbage.
