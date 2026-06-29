import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.utils import ensure_dir


def _cosine_matrix(profiles: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    profiles = profiles.float()
    norms = torch.linalg.norm(profiles, dim=1, keepdim=True).clamp_min(eps)
    normalized = profiles / norms
    return normalized @ normalized.T


def _save_heatmap(matrix: torch.Tensor, labels: List[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.5), max(3, len(labels) * 0.45)))
    image = ax.imshow(matrix.numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_diagnostics(
    module_names: List[str],
    module_profiles: torch.Tensor,
    atom_profiles: torch.Tensor,
    output_dir: Path,
    rank_patterns: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    output_dir = ensure_dir(Path(output_dir))
    module_cos = _cosine_matrix(module_profiles)
    module_df = pd.DataFrame(module_cos.numpy(), index=module_names, columns=module_names)
    module_df.to_csv(output_dir / "module_cosine.csv")
    _save_heatmap(module_cos, module_names, output_dir / "module_cosine.png", "Module Cosine")

    top1 = atom_profiles[:, 0, :]
    atom_cos = _cosine_matrix(top1)
    atom_df = pd.DataFrame(atom_cos.numpy(), index=module_names, columns=module_names)
    atom_df.to_csv(output_dir / "atom_top1_cosine.csv")
    _save_heatmap(atom_cos, module_names, output_dir / "atom_top1_cosine.png", "Top-1 Atom Cosine")

    pairs = []
    for i, left in enumerate(module_names):
        for j, right in enumerate(module_names):
            if j <= i:
                continue
            if float(module_cos[i, j]) < 0.3 and float(atom_cos[i, j]) > 0.7:
                pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "module_cosine": float(module_cos[i, j]),
                        "atom_top1_cosine": float(atom_cos[i, j]),
                    }
                )
    (output_dir / "redundancy_pairs.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")

    if rank_patterns:
        rows = []
        for module_name in module_names:
            row = {"module_name": module_name}
            for method, pattern in rank_patterns.items():
                row[method] = int(pattern.get(module_name, 0))
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / "rank_pattern_comparison.csv", index=False)

    return {
        "redundancy_pairs": pairs,
        "module_cosine_path": str(output_dir / "module_cosine.csv"),
        "atom_top1_cosine_path": str(output_dir / "atom_top1_cosine.csv"),
    }
