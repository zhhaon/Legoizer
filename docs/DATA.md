# Data sources and offline use

## Default behavior

No `.data` file is needed for Studio. Clicking Train or acquiring resources from
the UI checks compatible prepared tables, optional local raw-data overrides, and
then the configured download sources. The process tokenizes the required fields
and caches compatible variants. The UI reports preparation progress and failures.

| Dataset | Automatic source / behavior | Optional local override |
| --- | --- | --- |
| MIND, MIND Demo | MIND-small train/dev ZIPs from Recommenders/MIND on Hugging Face, with a legacy Azure fallback; Demo samples up to 650 users | `mind` |
| xMIND | Selected language ZIPs from xMIND, plus MIND-small for alignment/history | `xmind<language>` and optionally `mind` |
| ONCE-MIND | MIND-small with a deterministic 50/50 split of dev impressions when no custom split is configured | `oncemind` |
| Published RecBench variants | Already split Legommenders Parquet exports on Google Drive | Processor name such as `booksrb` |
| Other RecBench variants | Publisher archive when it includes all required files | Processor/domain name |

The explicit published-source map is in
`studio/backend/executor/recbench_sources.py`. It currently includes Automotive,
Books, CDs, EB-NeRD, Goodreads, H&M, LastFM, MicroLens, MIND, Netflix, PENS, POG,
and Yelp. Availability depends on the publisher; an entry is not a guarantee of
network reachability. Other variants can require a manual download. Studio does
not invent missing official splits or silently replace real data with synthetic data.

The MIND source follows the current
[Recommenders dataset loader](https://github.com/recommenders-team/recommenders/blob/main/recommenders/datasets/mind.py),
which points to [Recommenders/MIND](https://huggingface.co/datasets/Recommenders/MIND).
The old gated `yjw1029/MIND` entry is not used by this release.
GloVe uses [Stanford NLP's published archive](https://huggingface.co/stanfordnlp/glove)
on Hugging Face first, with the Stanford download server as a fallback.

## Optional `.data`

Copy `.data.example` to `.data` only if you want local overrides. Blank lines and
lines beginning with `#` are accepted; uncomment entries you need. Paths refer to
the computer running the backend, not the browser's computer. Restart the backend
after editing the file. `.data` is ignored by Git.

MIND:

```text
mind = /datasets/MIND
```

```text
/datasets/MIND/
  train/news.tsv
  train/behaviors.tsv
  dev/news.tsv
  dev/behaviors.tsv
```

xMIND Chinese (`cmn`, mapped to the publisher's `zho` code):

```text
xmindcmn = /datasets/xmind/cmn
```

The directory must contain `xMINDsmall_train/news.tsv` and
`xMINDsmall_dev/news.tsv`. MIND data is also needed and can be acquired automatically.

Published RecBench splits:

```text
booksrb = /datasets/books
```

Supply `items.parquet`, `users.parquet`, `train.parquet`, `valid.parquet`, and
`test.parquet`. The alternative original RecBench layout expects `items.parquet`,
`users.parquet`, `finetune.parquet`, `test.parquet`, and `valid_user_set_0.1.txt`.
Keep the corresponding publisher's split files together.

For a specific ONCE-MIND split:

```text
oncemind = /datasets/MIND$/datasets/once/impressions.json
```

Without this entry Studio uses a fixed-seed split of real MIND dev impressions.
That default does not claim to reproduce a particular paper's split.

## Resource locations

All paths below are relative to the repository:

| Path | Contents |
| --- | --- |
| `studio/backend/workspace/resources/downloads/` | Downloaded raw datasets; Hugging Face dataset ZIPs may be stored in its Hub cache instead |
| `studio/backend/workspace/resources/datasets/` | Prepared UniTok dataset variants |
| `studio/backend/workspace/resources/huggingface/` | Model, tokenizer, and Hub dataset cache |
| `studio/backend/workspace/resources/nltk/` | Automatically acquired NLTK tokenization resources |
| `data/embeddings/` | Converted token embeddings and GloVe assets |
| `cache/` | Framework hidden-state caches |
| `checkpoints/` | Trained models, logs, and metrics |
| `studio/backend/workspace/runs/` | Studio run records, logs, and run-specific inputs |
| `studio/backend/workspace/graphs/` | Graphs saved through Studio |

Caches and run state are ignored by Git. Do not delete resources while training
or acquisition is active. The UI's resource deletion actions restrict removal
to project-managed locations. Keep irreplaceable checkpoints backed up separately.

## Model resources and offline operation

The tracked `.model` contains public model aliases. Use your own authorized model
repository or local checkpoint directory when needed. Models with custom remote
code can require explicit trust/access settings; review the provider's code and
requirements. Dataset-indexed embedding matrices require a matching published
matrix, not an arbitrary token embedding export.

For offline operation, prepare the selected dataset/model once online in this
checkout and reuse its compatible caches, or provide raw data through `.data`
and all required models, tokenizers, embeddings, and Python/npm dependencies
locally. Merely copying raw data does not provide its tokenizer or pretrained
weights. Automatic resource acquisition needs an internet connection when a
required resource is absent. Official Benchmark also needs its external API.

The standalone upstream `process.py` command does not use Studio's automatic
acquisition wrapper. Configure raw data explicitly before using that command.
