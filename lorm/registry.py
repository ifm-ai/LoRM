"""Registries through which a work adds model variants and training algorithms without editing the core.

A module registers its classes when it is imported, e.g. `MODELS["feedback"] = FeedbackTransformer` in
lorm/model/feedback.py; a work's entry points import its modules before reading a config that names them.
"""

# instance name -> Transformer subclass with a `config_class` (lorm.model.config.InstanceConfig).
MODELS: dict[str, type] = {}
# algo name -> Algorithm subclass with a `config_class` dataclass of its options.
ALGORITHMS: dict[str, type] = {}
