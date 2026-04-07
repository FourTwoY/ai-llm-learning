from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_domain_learning_assistant.src.dataset.generator import generate_dataset
from lora_domain_learning_assistant.src.utils.logger import log_step, logger


def main() -> None:
    with log_step("prepare_data"):
        stats = generate_dataset()
        logger.info("Dataset generated: %s", stats)


if __name__ == "__main__":
    main()
