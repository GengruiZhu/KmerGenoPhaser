#!/usr/bin/env python3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_tsv', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--species_name', default="Subgenome_Analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_pdf = os.path.join(args.output_dir, f"{args.species_name}_Advanced_Heatmap.pdf")

    # 直接读取训练脚本输出的距离矩阵
    dist_matrix = pd.read_csv(args.input_tsv, sep='\t', index_col=0)
    
    # 确保矩阵对称并修正微小数值偏差
    dist_val = (dist_matrix.values + dist_matrix.values.T) / 2.0
    np.fill_diagonal(dist_val, 0)
    
    # 层次聚类
    dist_vec = squareform(dist_val)
    linkage_matrix = linkage(dist_vec, method='ward')

    # 绘图配置
    sns.set_theme(style="white")
    num_items = len(dist_matrix.index)
    fig_size = max(10, num_items * 0.5)

    # cmap='magma': 0(黑/深)表示近，亮色表示远(特异)
    g = sns.clustermap(
        dist_matrix,
        cmap='magma', 
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        linewidths=.5,
        figsize=(fig_size, fig_size),
        square=True,
        cbar_kws={'label': 'Dissimilarity (Higher = More Specific)'},
        dendrogram_ratio=0.15,
        annot=False
    )

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=10)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10)
    g.fig.suptitle(f"Specificty Mapping: {args.species_name}", fontsize=16, y=1.02)

    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    print(f"Heatmap saved to: {out_pdf}")

if __name__ == "__main__":
    main()
