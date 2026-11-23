"""Train a run: python -m works.attnres_feedback.train --config CONFIG [--resume auto]; make_config.py writes the
configs."""

import works.attnres_feedback.model  # noqa: F401  (registers the attnres instance)
from lorm.train.main import main

if __name__ == "__main__":
    main()
