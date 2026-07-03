import json
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
except ImportError:
    print("Please install required packages first: pip install matplotlib seaborn numpy")
    exit(1)

def plot_heatmap(json_path: str, output_path: str):
    print(f"Loading allocations from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取分配数据
    allocs = data.get('rank_allocation', {})
    if not allocs:
        print("Error: Could not find 'rank_allocation' in the provided JSON.")
        return

    # 解析出层号和模块类型
    layer_module_rank = {}
    max_layer = 0
    mod_types_set = set()
    
    for m, r in allocs.items():
        parts = m.split('.')
        layer = -1
        for p in parts:
            if p.isdigit():
                layer = int(p)
                break
        mod_type = parts[-1]
        
        if layer == -1:
            continue
            
        if layer not in layer_module_rank:
            layer_module_rank[layer] = {}
        layer_module_rank[layer][mod_type] = r
        max_layer = max(max_layer, layer)
        mod_types_set.add(mod_type)

    # 固定横坐标模块类型的顺序 (符合直觉的顺序)
    standard_order = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    mod_types = [m for m in standard_order if m in mod_types_set]
    
    # 补齐未出现在标准顺序中的其他模块
    for m in mod_types_set:
        if m not in mod_types:
            mod_types.append(m)

    layers = list(range(max_layer + 1))
    
    # 填充矩阵
    matrix = np.zeros((len(layers), len(mod_types)))
    for i, l in enumerate(layers):
        for j, m in enumerate(mod_types):
            matrix[i, j] = layer_module_rank.get(l, {}).get(m, 0)

    # 设置绘图尺寸
    plt.figure(figsize=(12, max(8, len(layers) * 0.3)))
    
    # 绘制热力图 (使用自定义颜色映射)
    ax = sns.heatmap(
        matrix, 
        xticklabels=mod_types, 
        yticklabels=layers, 
        annot=True,     # 在格子里显示具体数字
        fmt='g',        # 格式化数字
        cmap='YlGnBu',  # 蓝绿渐变色系
        linewidths=.5,  # 格子边框
        cbar_kws={'label': 'Allocated Rank'}
    )
    
    plt.title('DiCo Rank Allocation Distribution (Per Layer & Module)', fontsize=14, pad=20)
    plt.xlabel('Module Type', fontsize=12)
    plt.ylabel('Layer Index', fontsize=12)
    
    # 调整 x 轴标签的角度，防止重叠
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap successfully saved to: {output_path}")
    
    # 打印一些统计信息
    budget_error = data.get('budget_error_ratio')
    if budget_error is not None:
        print(f"Budget Error Ratio: {budget_error:.4f}")
        
    relaxation = data.get('rank_beyond_selected_evidence_total', 0)
    if relaxation > 0:
        total_rank = sum(sum(row.values()) for row in layer_module_rank.values())
        print(f"Evidence Relaxation Ratio: {relaxation / max(1, total_rank):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rank allocation heatmap from DiCo JSON log.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="outputs/preallocations/dico_pre_rank8_seed42.json",
        help="Path to the preallocation JSON file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="outputs/preallocations/rank_heatmap.png",
        help="Path to save the output PNG heatmap."
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
    else:
        plot_heatmap(str(input_path), args.output)
