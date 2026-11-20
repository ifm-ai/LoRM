"""Train a Feedback Shortcut model: python -m works.feedback_shortcut.train --config CONFIG [--resume auto]."""

from lorm.train.main import main

from . import algo, model  # noqa: F401  (registration)

if __name__ == "__main__":
    main()
