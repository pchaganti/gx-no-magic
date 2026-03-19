"""
Generate catalog.json from all algorithm scripts in the repository.

Extracts tier, filename, display name, and thesis docstring from each .py file
in the tier directories. Output is written to docs/catalog.json and consumed
by the no-magic-ai.github.io website.

Usage:
    python scripts/generate_catalog.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TIER_DIRS = ["01-foundations", "02-alignment", "03-systems", "04-agents"]
OUTPUT = REPO_ROOT / "docs" / "catalog.json"

DISPLAY_OVERRIDES = {
    "microgpt": "Autoregressive GPT",
    "micrornn": "RNN vs GRU",
    "microlstm": "LSTM",
    "microtokenizer": "BPE Tokenizer",
    "microembedding": "Word Embeddings",
    "microrag": "RAG Pipeline",
    "microbert": "BERT",
    "microconv": "CNN",
    "microdiffusion": "Denoising Diffusion",
    "microgan": "GAN",
    "microoptimizer": "Optimizer Comparison",
    "microvae": "VAE",
    "microvit": "Vision Transformer",
    "microresnet": "ResNet",
    "microlora": "LoRA",
    "microdpo": "DPO",
    "microppo": "PPO (RLHF)",
    "micromoe": "Mixture of Experts",
    "microbatchnorm": "Batch Normalization",
    "microdropout": "Dropout",
    "microgrpo": "GRPO",
    "microqlora": "QLoRA",
    "microreinforce": "REINFORCE",
    "microattention": "Attention Variants",
    "microbeam": "Beam Search",
    "microflash": "Flash Attention",
    "microkv": "KV-Cache",
    "microquant": "Quantization",
    "microrope": "RoPE",
    "microssm": "State Space Models",
    "microcheckpoint": "Activation Checkpointing",
    "micropaged": "PagedAttention",
    "microparallel": "Model Parallelism",
    "microbm25": "BM25",
    "microcomplexssm": "Complex SSM",
    "microdiscretize": "Discretization",
    "microroofline": "Roofline Model",
    "microspeculative": "Speculative Decoding",
    "microvectorsearch": "Vector Search",
    "microbandit": "Multi-Armed Bandit",
    "micromcts": "Monte Carlo Tree Search",
    "micromemory": "Memory-Augmented Network",
    "microminimax": "Minimax + Alpha-Beta",
    "microreact": "ReAct",
    "attention_vs_none": "Attention vs None",
    "adam_vs_sgd": "Adam vs SGD",
    "rnn_vs_gru_vs_lstm": "RNN vs GRU vs LSTM",
}


def name_to_display(name: str) -> str:
    """Convert a script name to a human-readable display name."""
    if name in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[name]
    # Fallback: strip 'micro' prefix and title-case
    clean = name.replace("micro", "", 1) if name.startswith("micro") else name
    return clean.replace("_", " ").title()


def extract_thesis(path: Path) -> str:
    """Extract the first line of the module docstring."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree)
    if not docstring:
        return ""
    # Take the first sentence (up to first period followed by space/newline, or first newline)
    first_line = docstring.strip().split("\n")[0].strip()
    return first_line


def count_lines(path: Path) -> int:
    """Count non-empty lines in a script."""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def build_catalog() -> list[dict]:
    """Scan all tier directories and build the catalog."""
    catalog = []
    for tier in TIER_DIRS:
        tier_path = REPO_ROOT / tier
        if not tier_path.exists():
            continue
        for script in sorted(tier_path.glob("*.py")):
            name = script.stem
            catalog.append({
                "tier": tier,
                "name": name,
                "display": name_to_display(name),
                "thesis": extract_thesis(script),
                "lines": count_lines(script),
            })
    return catalog


def main() -> None:
    catalog = build_catalog()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Generated {OUTPUT} with {len(catalog)} algorithms")


if __name__ == "__main__":
    main()
