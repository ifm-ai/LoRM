"""Train a run: python -m works.latent_feedback.train --config CONFIG [--resume auto]; make_config.py writes the
configs. Both arms are core models (lorm.model.feedback for latent feedback), so nothing is registered here."""

from lorm.train.main import main

if __name__ == "__main__":
    main()
