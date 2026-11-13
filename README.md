<div align="center">

# LoRM

**A research testbed for looped recurrent language models**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](.python-version)
[![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![uv](https://img.shields.io/badge/env-uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-555555)](LICENSE)

</div>

## Works

| Work | Paper |
| --- | --- |
| [**Latent Feedback Is a Compute and Data Amplifier**](works/latent_feedback/README.md) | [alphaXiv](https://www.alphaxiv.org/abs/2611.latent-feedback-transformers-scaling-law) |
| ? | ? |

## Quickstart

```bash
uv sync --locked --extra gpu
uv run --locked --extra gpu python -m works.latent_feedback.make_config feedback-134m-u100-f4
uv run --locked --extra gpu torchrun --nproc-per-node 8 -m works.latent_feedback.train \
    --config works/latent_feedback/configs/runs/feedback-134m-u100-f4.yaml
```

Use `--extra cpu` for CPU-only development; `uv run --locked --extra cpu python -m pytest` runs the tests.

## Core

`lorm/` holds the model, training, data and evaluation code that every work shares. A work registers its models and
training algorithms in `lorm.registry` when imported, so adding one needs no change to the core.

## License

LoRM is released under the [Apache License 2.0](LICENSE).

## Citation

If LoRM helps your research, please cite the paper of the work you build on (linked in the table above) and this
repository:

```bibtex
@misc{huang2026lorm,
  title  = {LoRM: A Research Testbed for Looped Recurrent Language Models},
  author = {Huang, Benhao},
  year   = {2026},
  url    = {https://github.com/ifm-ai/LoRM}
}
```
