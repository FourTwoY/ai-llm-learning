from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_domain_learning_assistant.src.dataset.loader import load_sft_records
from lora_domain_learning_assistant.src.inference.predictor import DomainLearningPredictor
from lora_domain_learning_assistant.src.utils.config import load_config
from lora_domain_learning_assistant.src.utils.io import write_jsonl
from lora_domain_learning_assistant.src.utils.logger import log_step, logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single or batch inference.")
    parser.add_argument("--config", default=None, help="Path to config yaml.")
    parser.add_argument("--instruction", default="", help="Single instruction.")
    parser.add_argument("--input", default="", help="Optional single input text.")
    parser.add_argument("--batch-file", default="", help="JSONL file for batch inference.")
    parser.add_argument("--output-file", default="", help="Where to write JSONL results.")
    parser.add_argument("--adapter-dir", default="", help="Optional LoRA adapter directory.")
    parser.add_argument("--base-only", action="store_true", help="Force base model only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    adapter_dir = "" if args.base_only else (args.adapter_dir or cfg["model"]["adapter_dir"])
    output_file = args.output_file or cfg["inference"]["output_file"]
    predictor = DomainLearningPredictor(
        config_path=args.config,
        adapter_dir=adapter_dir,
        lazy_load=True,
        enable_fallback=True,
    )

    with log_step("run_inference"):
        if args.instruction:
            records = [{"instruction": args.instruction, "input": args.input, **predictor.predict(args.instruction, args.input)}]
        else:
            batch_file = args.batch_file or cfg["inference"]["batch_input_file"]
            records = predictor.batch_predict(load_sft_records(batch_file))
        write_jsonl(output_file, records)
        logger.info("Inference results saved to %s", output_file)


if __name__ == "__main__":
    main()
