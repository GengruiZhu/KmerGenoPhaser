#!/usr/bin/env python3
"""
血缘热图可视化 - 修正版（含 Nodata 推断标签）
每个block单独显示，标签包含起止位置
若提供 --nodata_inferred_tsv，则 Nodata block 标签格式变为：
    Chr1N:ND->Of:113.0M-114.0M
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
import re
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def load_block_info_from_annotations(block_dir):
    """
    从block注释文件加载完整信息
    返回: {block_id: {'chrom', 'bloodline', 'start', 'end'}}
    """
    block_info_map = {}

    for filename in os.listdir(block_dir):
        if not filename.endswith('.txt'):
            continue

        chrom = filename.replace('.txt', '')
        filepath = os.path.join(block_dir, filename)

        try:
            df = pd.read_csv(filepath, sep='\t')

            bloodline_counter = {}

            for _, row in df.iterrows():
                start = int(row['Start'])
                end = int(row['End'])
                raw_bloodline = str(row['Bloodline'])

                # 清理血缘名称（去掉后缀数字和括号内容）
                bloodline = re.sub(r'\(.*\)', '', raw_bloodline)  # 去掉括号及内容
                bloodline = re.sub(r'\d+$', '', bloodline).capitalize()

                if bloodline not in bloodline_counter:
                    bloodline_counter[bloodline] = 0
                bloodline_counter[bloodline] += 1

                block_id = f"{chrom}_{bloodline}_{bloodline_counter[bloodline]}"

                block_info_map[block_id] = {
                    'chrom': chrom,
                    'bloodline': bloodline,
                    'start': start,
                    'end': end
                }

        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            continue

    return block_info_map


def load_nodata_inferred(tsv_path):
    """
    读取 Nodata 推断结果 TSV
    返回: {block_id -> inferred_bloodline}
    """
    if not tsv_path or not os.path.exists(tsv_path):
        return {}
    df = pd.read_csv(tsv_path, sep='\t')
    return dict(zip(df['block_id'], df['inferred_bloodline']))


def format_position(bp):
    """格式化碱基位置，智能选择单位"""
    if bp >= 1_000_000:
        return f"{bp/1_000_000:.1f}M"
    elif bp >= 1_000:
        return f"{bp/1_000:.0f}K"
    else:
        return str(bp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_tsv', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--block_dir', required=True, help="Block annotation directory")
    parser.add_argument('--species_name', default="Bloodline_Analysis")
    parser.add_argument('--show_chromosome', action='store_true')
    parser.add_argument('--nodata_inferred_tsv', default=None,
                        help="Nodata 推断结果 TSV（来自 assign_nodata_bloodline.py）")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取距离矩阵
    print("Loading distance matrix...")
    dist_matrix = pd.read_csv(args.input_tsv, sep='\t', index_col=0)

    print(f"Distance matrix shape: {dist_matrix.shape}")
    print(f"First 5 block IDs from matrix:")
    for i, bid in enumerate(dist_matrix.index[:5]):
        print(f"  {i+1}. {bid}")

    dist_val = (dist_matrix.values + dist_matrix.values.T) / 2.0
    np.fill_diagonal(dist_val, 0)

    # 加载 Nodata 推断结果
    inferred_map = load_nodata_inferred(args.nodata_inferred_tsv)
    if inferred_map:
        print(f"Loaded inferred bloodlines for {len(inferred_map)} Nodata blocks")
    else:
        print("No Nodata inferred file provided (or file not found), using original labels")

    # 加载block信息
    print("Loading block annotations...")
    block_info_map = load_block_info_from_annotations(args.block_dir)

    print(f"Loaded {len(block_info_map)} blocks from annotations")
    if len(block_info_map) == 0:
        print("ERROR: No block information loaded!")
        print(f"Check that block files exist in: {args.block_dir}")
        return

    sample_ids = list(block_info_map.keys())[:3]
    print(f"Sample block IDs: {sample_ids}")
    for bid in sample_ids:
        print(f"  {bid}: {block_info_map[bid]}")

    # 解析所有block
    block_info_list = []
    missing_blocks = []

    for block_id in dist_matrix.index:
        if block_id in block_info_map:
            info = block_info_map[block_id]
            block_info_list.append({
                'block_id': block_id,
                'chrom': info['chrom'],
                'bloodline': info['bloodline'],
                'start': info['start'],
                'end': info['end']
            })
        else:
            missing_blocks.append(block_id)
            parts = block_id.split('_')
            chrom = parts[0] if len(parts) > 0 else 'Unknown'

            try:
                bloodline = '_'.join(parts[1:-1]) if len(parts) > 2 else 'Unknown'
                block_num = int(parts[-1]) if len(parts) > 1 else 0
            except:
                bloodline = 'Unknown'
                block_num = 0

            block_info_list.append({
                'block_id': block_id,
                'chrom': chrom,
                'bloodline': bloodline,
                'start': 0,
                'end': 0
            })

    if missing_blocks:
        print(f"\nWarning: {len(missing_blocks)} blocks not found in annotations:")
        for bid in missing_blocks[:5]:
            print(f"  {bid}")
        if len(missing_blocks) > 5:
            print(f"  ... and {len(missing_blocks) - 5} more")

    block_df = pd.DataFrame(block_info_list)

    print(f"\nTotal blocks: {len(block_df)}")
    print(f"Bloodlines: {sorted(block_df['bloodline'].unique())}")
    print(f"Chromosomes: {sorted(block_df['chrom'].unique())}")

    unique_bloodlines = sorted(block_df['bloodline'].unique())
    unique_chroms = sorted(block_df['chrom'].unique())

    # 层次聚类
    dist_vec = squareform(dist_val)
    linkage_matrix = linkage(dist_vec, method='ward')

    # 配色
    if args.show_chromosome:
        color_map = {}
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_chroms)))
        for i, chrom in enumerate(unique_chroms):
            color_map[chrom] = colors[i]
        row_colors = [color_map[row['chrom']] for _, row in block_df.iterrows()]
        legend_title = 'Chromosome'
        legend_labels = unique_chroms
        legend_colors = [color_map[c] for c in unique_chroms]
    else:
        color_map = {}
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_bloodlines)))
        for i, bloodline in enumerate(unique_bloodlines):
            color_map[bloodline] = colors[i]
        row_colors = [color_map[row['bloodline']] for _, row in block_df.iterrows()]
        legend_title = 'Bloodline'
        legend_labels = unique_bloodlines
        legend_colors = [color_map[b] for b in unique_bloodlines]

    # ── 创建标签（核心修改：Nodata block 加推断标签）──
    print("Creating labels...")
    full_labels = []
    for _, row in block_df.iterrows():
        chrom = row['chrom']
        bloodline = row['bloodline']
        start = row['start']
        end = row['end']
        bid = row['block_id']

        # 血缘缩写
        if bloodline == 'Spontaneum':
            bl_short = 'Sp'
        elif bloodline == 'Robustum':
            bl_short = 'Ro'
        elif bloodline == 'Officinarum':
            bl_short = 'Of'
        elif bloodline == 'Nodata':
            # 如果有推断结果，格式：ND->Of
            if bid in inferred_map:
                inferred_bl = inferred_map[bid]
                inferred_short = {
                    'Spontaneum': 'Sp',
                    'Robustum': 'Ro',
                    'Officinarum': 'Of',
                }.get(inferred_bl, inferred_bl[:2])
                bl_short = f"ND->{inferred_short}"
            else:
                bl_short = 'ND'
        else:
            bl_short = bloodline[:2]

        start_str = format_position(start)
        end_str = format_position(end)

        label = f"{chrom}:{bl_short}:{start_str}-{end_str}"
        full_labels.append(label)

    # ===== 绘制热图 =====
    print("\nGenerating heatmap...")

    num_items = len(dist_matrix.index)
    base_size = max(18, num_items * 0.3)

    dist_matrix_labeled = dist_matrix.copy()
    dist_matrix_labeled.index = full_labels
    dist_matrix_labeled.columns = full_labels

    g = sns.clustermap(
        dist_matrix_labeled,
        cmap='RdYlBu_r',
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        row_colors=row_colors,
        col_colors=row_colors,
        linewidths=0,
        figsize=(base_size, base_size),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        dendrogram_ratio=0.08,
        colors_ratio=0.01
    )

    label_fontsize = max(4, min(8, 60 / num_items))
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=label_fontsize, ha='right')
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=label_fontsize)

    legend_patches = [mpatches.Patch(color=color, label=label)
                     for label, color in zip(legend_labels, legend_colors)]
    g.ax_heatmap.legend(
        handles=legend_patches,
        title=legend_title,
        bbox_to_anchor=(1.25, 0.5),
        loc='center left',
        frameon=True,
        fontsize=9,
        title_fontsize=10
    )

    plt.suptitle(f"{args.species_name}: Block Similarity", fontsize=16, y=0.98)

    out_pdf = os.path.join(args.output_dir, f"{args.species_name}_Bloodline_Heatmap.pdf")
    plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
    print(f"Heatmap saved: {out_pdf}")
    plt.close()

    # ===== 绘制树状图 =====
    print("Generating dendrograms...")

    linkage_transformed = linkage_matrix.copy()
    linkage_transformed[:, 2] = np.sqrt(linkage_matrix[:, 2])

    fig, ax = plt.subplots(figsize=(20, 12))

    dend = dendrogram(
        linkage_transformed,
        labels=full_labels,
        leaf_rotation=90,
        leaf_font_size=max(4, min(8, 60 / num_items)),
        color_threshold=None,
        above_threshold_color='#000000',
        ax=ax
    )

    ax.set_title(f"{args.species_name}: Dendrogram (sqrt-scaled distances)",
                 fontsize=14, pad=20)
    ax.set_xlabel('Block ID', fontsize=12)
    ax.set_ylabel('Distance (sqrt-scaled)', fontsize=12)

    plt.tight_layout()

    dendrogram_pdf = os.path.join(args.output_dir, f"{args.species_name}_Dendrogram.pdf")
    plt.savefig(dendrogram_pdf, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Dendrogram saved: {dendrogram_pdf}")

    # 着色版树状图
    fig, ax = plt.subplots(figsize=(20, 12))

    dend = dendrogram(
        linkage_transformed,
        labels=full_labels,
        leaf_rotation=90,
        leaf_font_size=max(4, min(8, 60 / num_items)),
        no_plot=True
    )

    dend = dendrogram(
        linkage_transformed,
        labels=full_labels,
        leaf_rotation=90,
        leaf_font_size=max(4, min(8, 60 / num_items)),
        above_threshold_color='#000000',
        ax=ax
    )

    xlbls = ax.get_xmajorticklabels()
    leaf_order = dend['leaves']

    if not args.show_chromosome:
        for i, lbl in enumerate(xlbls):
            block_idx = leaf_order[i]
            bloodline = block_df.iloc[block_idx]['bloodline']
            lbl.set_color(color_map.get(bloodline, 'black'))
    else:
        for i, lbl in enumerate(xlbls):
            block_idx = leaf_order[i]
            chrom = block_df.iloc[block_idx]['chrom']
            lbl.set_color(color_map.get(chrom, 'black'))

    legend_patches = [mpatches.Patch(color=color, label=label)
                     for label, color in zip(legend_labels, legend_colors)]
    ax.legend(
        handles=legend_patches,
        title=legend_title,
        bbox_to_anchor=(1.01, 0.5),
        loc='center left',
        frameon=True,
        fontsize=9
    )

    ax.set_title(f"{args.species_name}: Dendrogram (colored by {legend_title})",
                 fontsize=14, pad=20)
    ax.set_xlabel('Block ID', fontsize=12)
    ax.set_ylabel('Distance (sqrt-scaled)', fontsize=12)

    plt.tight_layout()

    colored_pdf = os.path.join(args.output_dir, f"{args.species_name}_Dendrogram_Colored.pdf")
    plt.savefig(colored_pdf, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Colored dendrogram saved: {colored_pdf}")

    # 统计报告
    print("Generating statistics...")
    report_file = os.path.join(args.output_dir, f"{args.species_name}_bloodline_stats.txt")
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BLOODLINE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total blocks: {len(block_df)}\n\n")

        f.write("Bloodline distribution:\n")
        for bloodline in sorted(unique_bloodlines):
            count = (block_df['bloodline'] == bloodline).sum()
            pct = count / len(block_df) * 100
            f.write(f"  {bloodline}: {count} blocks ({pct:.1f}%)\n")

        if inferred_map:
            f.write("\nNodata inferred bloodline distribution:\n")
            inferred_counts = {}
            for bl in inferred_map.values():
                inferred_counts[bl] = inferred_counts.get(bl, 0) + 1
            for bl, cnt in sorted(inferred_counts.items()):
                f.write(f"  Nodata→{bl}: {cnt} blocks\n")

        f.write("\nChromosome distribution:\n")
        for chrom in sorted(unique_chroms):
            count = (block_df['chrom'] == chrom).sum()
            f.write(f"  {chrom}: {count} blocks\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("BLOODLINE SIMILARITY MATRIX\n")
        f.write("=" * 80 + "\n\n")

        bloodline_dists = {}
        for bl1 in unique_bloodlines:
            bloodline_dists[bl1] = {}
            for bl2 in unique_bloodlines:
                idx1 = block_df[block_df['bloodline'] == bl1].index
                idx2 = block_df[block_df['bloodline'] == bl2].index

                dists = []
                for i in idx1:
                    for j in idx2:
                        if i != j:
                            i_pos = list(dist_matrix.index).index(block_df.iloc[i]['block_id'])
                            j_pos = list(dist_matrix.index).index(block_df.iloc[j]['block_id'])
                            dists.append(dist_val[i_pos, j_pos])

                bloodline_dists[bl1][bl2] = np.mean(dists) if dists else 0.0

        f.write(f"{'Bloodline':<15}")
        for bl in unique_bloodlines:
            f.write(f"{bl:<15}")
        f.write("\n" + "-" * 80 + "\n")

        for bl1 in unique_bloodlines:
            f.write(f"{bl1:<15}")
            for bl2 in unique_bloodlines:
                f.write(f"{bloodline_dists[bl1][bl2]:<15.4f}")
            f.write("\n")

    print(f"Report saved: {report_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
