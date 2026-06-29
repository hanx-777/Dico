from typing import Dict, List


def allocate_uniform(module_names: List[str], avg_rank: int = 1) -> Dict[str, int]:
    if int(avg_rank) != avg_rank or avg_rank < 0:
        raise ValueError("uniform avg_rank must be a non-negative integer")
    return {name: int(avg_rank) for name in module_names}
