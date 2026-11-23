# Attention Residuals with Latent Feedback

Block Attention Residuals ([Kimi Team, 2026](https://arxiv.org/abs/2603.15031)) replace the residual sum with a softmax
over depth: every sublayer reads its own mix of the embedding, the finished blocks and its block's running sum. This
folder adds latent feedback to that model, at 134M parameters, and compares how to feed back.

[Interactive map](https://huskydoge.github.io/attnres-feedback-map/) of the model. Back to the [repository README](../../README.md).

| File | Contents |
| --- | --- |
| `model.py` | AttnRes Transformer with optional feedback (`instance.name: attnres`) |
| `configs/*.yaml` | One config per arm |
| `configs/runs.csv` | Every run: arm, U/N, seed, learning rate, weight decay, schedule, validation loss |
| `make_config.py`, `train.py` | A run's config from `runs.csv`; training entry point |
| `measure/patterns.py`, `figures.py` | Each depth mix's softmax weights; their heatmaps |
| `results/` | Task BPB of the seeded runs; the depth patterns |

## Arms

| Config | Model |
| --- | --- |
| `transformer-attnres` | AttnRes, 2-layer blocks, depth logits scaled by dim^-1/2 |
| `feedback-attnres-top` | + the previous position's output mix, through W, added to the embedding |
| `feedback-attnres-blocks` | + the previous position's six blocks, each through W, as extra sources of every mix |
| `feedback-attnres-blocks-identity` | the same blocks fed back unchanged (no RMSNorm, no W) |
| `feedback-attnres-blocks-shift0` | the same blocks from the position's own earlier pass |
| `loop2-attnres` | a 2x loop: the second visit reads the first visit's blocks; both visits train |

Feedback arms train with one pass on 15% of updates and two on the rest, and are scored at 4 passes (token-by-token
decoding). `-unscaled` rows in `runs.csv` drop the logit scale.

## Training

Data and setup are the [latent feedback work](../latent_feedback/README.md)'s (D3, 8 GPUs, 0.5M tokens per update):

```bash
uv run python -m works.attnres_feedback.make_config feedback-attnres-blocks-134m-u100-lr0.177-wd0.0759-s0
torchrun --nproc-per-node 8 -m works.attnres_feedback.train \
    --config works/attnres_feedback/configs/runs/feedback-attnres-blocks-134m-u100-lr0.177-wd0.0759-s0.yaml
```

Every run spends the training FLOPs of one Transformer epoch over its U unique tokens, so arms that cost more per
token see fewer tokens.

## Results

134M, validation loss and task bits per byte (math, code, QA; Olmo 3 Base Easy), mean of 3 seeds at each arm's tuned
learning rate and weight decay. Transformer and latent feedback are the latent feedback work's runs, plus two U/N 400
seeds listed here.

| Model | Loss, U/N 100 ↓ | BPB, U/N 100 ↓ | Loss, U/N 400 ↓ | BPB, U/N 400 ↓ |
| --- | --- | --- | --- | --- |
| Transformer | 2.4690 | 0.9778 | 2.3879 | 0.9236 |
| Transformer + AttnRes | 2.4354 | 0.9511 | 2.3556 | 0.9020 |
| Latent feedback | 2.4679 | 0.9733 | 2.3748 | 0.9121 |
| Latent feedback + AttnRes, top | 2.4443 | 0.9572 | | |
| Latent feedback + AttnRes, blocks | 2.4319 | 0.9473 | 2.3390 | 0.8850 |
| Loop × 2 + AttnRes | 2.4510 | 0.9597 | | |

The `-identity` and `-shift0` runs are training; their rows in `runs.csv` have no loss yet.

## Depth patterns

```bash
python -m works.attnres_feedback.measure.patterns runs/<run> --val data/d3/<validation shard>
python -m works.attnres_feedback.figures
```

With feedback or a loop, the second pass reads layers 5–10 of the first pass, not its top, and only in its own first
8 layers; the output mix reads the last pass.
