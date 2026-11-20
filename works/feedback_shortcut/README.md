# Feedback Shortcut: Speculative Decoding for Latent Feedback Transformers

A latent feedback Transformer feeds each position's top-layer state into the input of the next position. In
speculative decoding, a drafted position needs a state that its own verification call is still computing, so it reads
a *stand-in*: zeros, the state of a multi-token prediction (MTP) head, or that state through state alignment's map.
Pipelined correction then recomputes drafted positions in the calls that verify the next drafts. This folder trains
the paper's models, and measures their quality, their served quality and their decoding speed.

Paper: [alphaXiv](https://www.alphaxiv.org/abs/2611.feedback-shortcut-speculative-decoding). Back to the [repository README](../../README.md).

| File | Contents |
| --- | --- |
| `model.py` | The chain of MTP heads, the latent feedback Transformer with them, and stand-in training: `state_alignment`, `in_place` |
| `algo.py`, `train.py` | Training with sampled passes, the MTP loss and the alignment loss |
| `configs/` | One config per setting and size; `runs.csv` lists the paper's runs, `make_config.py` writes each run's config |
| `measure/engine.py` | Speculative decoding with pipelined correction at depth D, batch size 1, on CUDA graphs: greedy, speculative sampling, or teacher-forced along a given text |
| `measure/batched.py` | The same decoding over many requests with continuous batching (`decode_attention.py`: its Triton attention) |
| `measure/quality.py` | Exact BPB on Paloma, the one-call error of each stand-in, each head's acceptance |
| `measure/served.py` | BPB that speculative decoding serves at each depth, on Paloma and on downstream gold answers |
| `measure/specbench.py` | Spec-Bench wall-clock timing and its speedups, tokens per call and rho |
| `measure/long.py` | 1,024-token generations: KL from exact decoding when sampling, near-ties at greedy departures |
| `measure/stats.py` | Downstream suite scores and their intervals over seeds and items |
| `measure/flops.py`, `prompts.py` | Training FLOPs per token; the batched benchmark's prompt sets |
| `results/` | The paper's measurements of every run, in the formats the measurement scripts write |
| `tables.py`, `figures.py` | The paper's tables (LaTeX rows) and data figures (PDF) from `results/` |

## Settings

| Setting | Config | Stand-ins it is measured with |
| --- | --- | --- |
| Transformer with MTP heads | `transformer-mtp` | none: verification is exact |
| Latent feedback with MTP heads | `feedback` | zeros, MTP state |
| State alignment | `state-alignment` | the map of the MTP state |
| In-place substitution | `in-place`, `in-place-half`, `in-place-decay` | MTP state |

Every setting has a 135M and a 780M config with three seeds. `in-place-decay` branches from the `feedback` run of its seed where the learning-rate decay starts, so train
that run first.

## Data

```bash
uv sync --locked --extra gpu --extra prep
uv run python -m lorm.data.prepare_dolma3 download raw/dolma3 --files 800   # first files of the shuffled order
uv run python -m lorm.data.prepare_dolma3 tokenize raw/dolma3 data/dolma3-llama2
uv run python -m lorm.eval.paloma tokenize data/paloma && uv run python -m lorm.eval.paloma raw data/paloma
hf download IFM/latent-feedback-data --repo-type dataset --include 'task-bpb/*' --local-dir data
```

The Llama 2 tokenizer is gated on the Hugging Face Hub; accept its license and log in first. The 135M runs read the
first 136 training shards (13.4B tokens) and the 780M runs the first 779 (77.8B). Spec-Bench's questions are
`data/spec_bench/question.jsonl` of [Spec-Bench](https://github.com/hemingkx/Spec-Bench) at commit
`fd2c1cd7d2201ef71db4c5f4e455008f017967bf`. The batched benchmark adds the SPEED-Bench qualitative prompts, whose
placeholder rows NVIDIA [Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)'s
`examples/specdec_bench/prepare_data.py` resolves (commit `732f7d6335eb4d5ec87e4bfb3af96359483ded36`); `measure/prompts.py`
then writes the prompt sets.

## Training

```bash
uv run python -m works.feedback_shortcut.make_config --all
torchrun --nproc-per-node 4 -m works.feedback_shortcut.train --config works/feedback_shortcut/configs/runs/state-alignment-135m.yaml
```

Runs are named `<setting>-<size>`, with `-s1` and `-s2` for the extra seeds. The 135M configs assume 4 GPUs and the
780M configs 16; with another count, change `train.batch_size` or `train.gradient_accumulation` so that each update
keeps 256 sequences of 2,048 tokens.

## Checkpoints

The trained models are at [IFM/feedback-shortcut-checkpoints](https://huggingface.co/IFM/feedback-shortcut-checkpoints),
one folder per run of `runs.csv`: `checkpoint.pt` (the fp32 model weights), `config.yaml` (the run's config) and
`metrics.jsonl` (its training and validation log). The measurements read these folders as they are.

```bash
hf download IFM/feedback-shortcut-checkpoints --local-dir runs   # or --include 'state-alignment-135m/*'
```

## Measurements

Every measurement runs in float16 on one GPU unless noted. `R=runs/state-alignment-135m`.

```bash
python -m works.feedback_shortcut.measure.quality $R --paloma data/paloma/raw --out results/quality
python -m works.feedback_shortcut.measure.served $R --stand-in state_alignment --source paloma --data data/paloma/raw \
    --out results/served-paloma/$(basename $R).json
python -m lorm.eval.task_bpb $R --requests data/task-bpb --out results/task_bpb --passes 4 --precision float16 \
    --work works.feedback_shortcut
python -m works.feedback_shortcut.measure.served $R --stand-in state_alignment --source downstream --data data/task-bpb \
    --shard 0 --shards 8 --out results/served-downstream/$(basename $R)-0.json
python -m works.feedback_shortcut.measure.stats --exact results/task_bpb --served results/served-downstream \
    --out results/downstream.json   # needs --extra analysis
python -m works.feedback_shortcut.measure.batched $R --stand-in state_alignment --prompts data/prompts \
    --out results/batched/$(basename $R)
python -m works.feedback_shortcut.measure.specbench run $R --stand-in state_alignment \
    --questions data/spec_bench/question.jsonl --out results/specbench/$(basename $R)
python -m works.feedback_shortcut.measure.long $R --stand-in state_alignment --paloma data/paloma/raw --out long.json
python -m works.feedback_shortcut.measure.flops
```

`python -m works.feedback_shortcut.tables` and `python -m works.feedback_shortcut.figures` rebuild the paper's tables
and figures from `results/` (the figures need `--extra analysis`). `results/downstream.json` is `stats.py`'s output over
the item-level downstream scores, which are not included.

The Spec-Bench timings in `results/specbench/` are the paper's, measured with an earlier runner around Spec-Bench's
`evaluation/eval.py` that ran each depth over all questions in turn and captured CUDA graphs inside the first timed
requests. `specbench.py` instead captures every graph before timing and interleaves exact decoding and every depth
question by question; on the same checkpoints it decodes the same tokens (tokens per call and rho are unchanged) and
measures speedups 3 to 7% higher.

Exact decoding and the timing prefill run 4 parallel passes of the training iteration; the Transformer runs one. Stand-in names are `zeros`, `mtp_state` and
`state_alignment`; a Transformer takes `mtp_state`, which it never reads.
