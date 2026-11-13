# Latent Feedback Is a Compute and Data Amplifier

When unique data runs out, a Transformer repeats it for more epochs. A feedback Transformer can spend the same FLOPs
on the data it has: each position reads the previous position's top-layer state, trained with two passes on most
updates. This folder trains and evaluates both arms at 60M, 134M, 271M and 778M parameters over a grid of unique tokens per
parameter U/N and FLOPs per unique token F, and releases the paper's checkpoints and data.

Paper: [alphaXiv](https://www.alphaxiv.org/abs/2611.latent-feedback-transformers-scaling-law). Back to the [repository README](../../README.md).

| File | Contents |
| --- | --- |
| `configs/base.yaml`, `configs/feedback.yaml` | The shared recipe: the Transformer, and the latent feedback Transformer (p = 0.15) |
| `configs/runs.csv` | Every run: size, arm, U/N, F, seed, learning rate, weight decay, batch, schedule, decay-branch trunk, validation loss |
| `make_config.py` | The experiment YAML of a run in `runs.csv` |
| `train.py` | Training entry point |
| `data.py` | D3: Dolma 3 deduplicated, stratified and decontaminated against Paloma |
| `measure/task_bpb.py` | Bits per byte of the gold answer on the Olmo 3 Base Easy tasks |
| `measure/requests.py` | Those tasks' requests, built with OLMES |
| `results/` | Paloma and task BPB of every released run |

The latent feedback Transformer is the core feedback Transformer (`lorm/model/feedback.py`), trained with
`lorm.train.algo.SampledPasses`.

## Data

Download D3 and the task requests from [IFM/latent-feedback-data](https://huggingface.co/datasets/IFM/latent-feedback-data), or rebuild D3 from the Dolma 3 mix:

```bash
uv sync --locked --extra gpu --extra prep
hf download IFM/latent-feedback-data --repo-type dataset --local-dir data   # data/d3 and data/task-bpb
# or
uv run python -m lorm.data.prepare_dolma3 download raw/dolma3
uv run python -m lorm.data.prepare_dolma3 tokenize raw/dolma3 data/dolma3-llama2
uv run python -m works.latent_feedback.data data/dolma3-llama2 data/d3
```

D3 has 757 training shards (75.6B tokens, 35.6M distinct documents) and a 100M-token validation shard. The largest
run reads the first 721 shards. Each unique-data budget U is a prefix of the stream, so smaller pools are nested in
larger ones and follow the same mixture.

## Training

```bash
uv run python -m works.latent_feedback.make_config feedback-134m-u100-f4 transformer-134m-u100-f4
torchrun --nproc-per-node 8 -m works.latent_feedback.train --config works/latent_feedback/configs/runs/feedback-134m-u100-f4.yaml
```

- Run names read `<arm>-<size>-u<U/N>-f<F>` with the arm `transformer` or `feedback`, and `-s1` and `-s2` for the
  extra 134M seeds.
- F counts FLOPs per unique token in Transformer epochs, so latent feedback trains F / c epochs, with c its cost
  per token (1.20 to 1.27).
- The configs assume 8 GPUs per run, 16 at 778M; `make_config.py --gpus N` keeps the paper's sequences per update on N
  GPUs.
- The F = 1 to 3 runs are decay branches: each resumes the checkpoint of its trunk (`trunk` in `runs.csv`) at the
  step where its own learning-rate decay starts. Train the trunk first; it keeps those checkpoints.
- The rows with `released = 0` are trunks that are not themselves in the paper.

## Checkpoints

The trained models are at [IFM/latent-feedback-checkpoints](https://huggingface.co/IFM/latent-feedback-checkpoints), one folder per released run of `runs.csv`
(174 runs):

- `checkpoint.pt` holds the fp32 model weights.
- `config.yaml` is the run's config.
- `metrics.jsonl` is its training and validation log.

| Size | U/N | F | Seeds |
| --- | --- | --- | --- |
| 60M | 12.5, 50, 100, 400 | 1 to 4 | 0 |
| 134M | 12.5, 25, 50, 100, 200, 400 | 1 to 4 | 0; also 1 and 2 at U/N 12.5 and 100 |
| 271M | 12.5, 25, 50, 100, 200, 350 | 1 to 4 | 0 |
| 778M | 12.5 | 1 to 4 | 0 |
| 778M | 50 | 1, 2 | 0 |
| 778M | 100 | 1 | 0 |

Each size and U/N has both arms. Each checkpoint is the best learning rate and weight decay of its cell. The paper's
cell value is the minimum of a quadratic fitted to all of the cell's runs (the best run when that minimum lies outside
the grid), so it can differ from the checkpoint's loss (`loss` in `runs.csv`) by up to 0.002.

```bash
hf download IFM/latent-feedback-checkpoints --local-dir runs   # or --include 'feedback-134m-*/*'
```

## Measurements

```bash
V=data/d3/dolma3_val_000000.bin
python -m lorm.eval.shards runs/feedback-134m-u100-f4 --data $V --out val.json --passes 1,4
uv run python -m lorm.eval.paloma tokenize data/paloma && uv run python -m lorm.eval.paloma raw data/paloma
python -m lorm.eval.shards runs/feedback-134m-u100-f4 --data data/paloma/raw/*.bin --out paloma.json --passes 4
python -m works.latent_feedback.measure.task_bpb runs/feedback-134m-u100-f4 --requests data/task-bpb --out task_bpb
```

The training log's `eval/nll@P4` is the validation loss at four passes, which matches token-by-token decoding; the
paper reports it for latent feedback. `measure/requests.py` rebuilds the task requests.
