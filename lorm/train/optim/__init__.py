"""Build native AdamW/Muon from model.hidden_matrices(); protect model.head.

Muon shares AdamW's base learning rate using Moonlight's update scaling:
https://arxiv.org/html/2502.16982v1#S2.SS2

`matrix_optimizer` ("module:Class") replaces native Muon, e.g. with lorm.train.optim.normuon:NorMuon, and receives the `muon` options unchanged.

`embedding_lr`/`head_lr` and `embedding_weight_decay`/`head_weight_decay` give the
token embedding and the output head their own AdamW groups; unset values fall back to
the shared lr and weight decay. A tied embedding and head is one group, so its
embedding and head settings must agree.
"""

from importlib import import_module

from torch import nn
from torch.optim import AdamW, Muon, Optimizer

# Moonlight (Sec. 2.2) keeps embeddings, the LM head and RMSNorm on AdamW; EmbeddingBag
# and LayerNorm follow the same policy.
ADAMW_ONLY = (nn.Embedding, nn.EmbeddingBag, nn.LayerNorm, nn.RMSNorm)


def build_optimizers(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    adamw: dict | None = None,
    muon: dict | None = None,
    matrix_optimizer: str | None = None,
    embedding_lr: float | None = None,
    head_lr: float | None = None,
    embedding_weight_decay: float | None = None,
    head_weight_decay: float | None = None,
) -> dict[str, Optimizer]:
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("The model has no trainable parameters.")
    if muon is None and matrix_optimizer is not None:
        raise ValueError("matrix_optimizer requires muon options.")
    matrices = set() if muon is None else _select_matrices(model, parameters)
    no_decay = set(getattr(model, "no_weight_decay", lambda: ())())
    if not no_decay.issubset(parameters) or not no_decay.isdisjoint(matrices):
        raise ValueError("no_weight_decay() must select trainable parameters outside the Muon selection.")

    separate = _separate_groups(model, lr, weight_decay, embedding=(embedding_lr, embedding_weight_decay),
                                head=(head_lr, head_weight_decay))
    grouped = {p for group in separate for p in group["params"]}
    rest = [p for p in parameters if p not in matrices and p not in grouped]
    groups = [*separate,
              {"params": [p for p in rest if p.ndim >= 2 and p not in no_decay]},
              {"params": [p for p in rest if p.ndim < 2 or p in no_decay], "weight_decay": 0.0}]
    groups = [group for group in groups if group["params"]]
    optimizers = {}
    if groups:
        optimizers["adamw"] = AdamW(groups, lr=lr, weight_decay=weight_decay, **(adamw or {}))

    if muon is not None:
        if matrix_optimizer is None:
            name, matrix_class, options = "muon", Muon, {"adjust_lr_fn": "match_rms_adamw", **muon}
        else:
            module, _, attribute = matrix_optimizer.partition(":")
            name, matrix_class, options = "matrix", getattr(import_module(module), attribute), muon
        optimizers[name] = matrix_class([p for p in parameters if p in matrices],
                                        lr=lr, weight_decay=weight_decay, **options)
    return optimizers


def _select_matrices(model: nn.Module, parameters: list[nn.Parameter]) -> set:
    """The trainable matrices that model.hidden_matrices() names, for Muon."""
    selected = set(model.hidden_matrices())
    if not selected or not selected.issubset(parameters):
        raise ValueError("The Muon selection must contain trainable model parameters.")
    head = getattr(model, "head", None)
    protected = {p for module in model.modules() if module is head or isinstance(module, ADAMW_ONLY)
                 for p in module.parameters()}
    if not selected.isdisjoint(protected):
        raise ValueError("Embeddings, normalization, and the head cannot use Muon.")
    return selected


def _separate_groups(model: nn.Module, lr: float, weight_decay: float, **settings) -> list[dict]:
    """AdamW groups for model.embedding and model.head from their (lr, weight_decay) overrides.

    A tied embedding and head is one matrix with one group, so their settings must agree.
    """
    groups = {}
    for name, (rate, decay) in settings.items():
        if rate is None and decay is None:
            continue
        module = getattr(model, name, None)
        if module is None or not module.weight.requires_grad:
            raise ValueError(f"Separate {name} settings require a trainable model.{name}.")
        options = {"lr": lr if rate is None else rate, "weight_decay": weight_decay if decay is None else decay}
        group = groups.setdefault(id(module.weight), {"params": [module.weight], **options})
        if any(group[key] != value for key, value in options.items()):
            raise ValueError("Tied embeddings take one lr and weight decay; the embedding and head settings differ.")
    return list(groups.values())
