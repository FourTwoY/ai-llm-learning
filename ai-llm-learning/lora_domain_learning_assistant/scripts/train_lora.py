from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_domain_learning_assistant.src.training.trainer import train_lora
from lora_domain_learning_assistant.src.utils.logger import log_step, logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA/QLoRA SFT model.")
    parser.add_argument("--config", default=None, help="Path to config yaml.")
    parser.add_argument("--smoke-test", action="store_true", help="Use tiny subset for quick pipeline check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with log_step("train_lora"):
        result = train_lora(config_path=args.config, smoke_test=args.smoke_test)
        logger.info("Train result: %s", result)


if __name__ == "__main__":
    main()
