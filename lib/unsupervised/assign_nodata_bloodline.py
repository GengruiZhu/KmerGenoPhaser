#!/usr/bin/env python3
"""
assign_nodata_bloodline.py

功能：
1. 读取距离矩阵，对每个 Nodata block 找最近的非Nodata邻居，推断其血缘
2. 输出更新后的 block 注释文件（格式同输入 Chr1X.txt，Nodata行Bloodline列追加推断标签）
3. 读取亚基因组分配 JSON，检查每个 block 是否与预期亚基因组一致（基于最近邻）
   - 不一致的 block 打印到 log 并输出到 TSV 文件

用法：
    python assign_nodata_bloodline.py \
        --input_tsv FJDY_Chr1_block_distances.tsv \
        --block_dir input/block \
        --subgenome_json FJDY_Chr1_subgenomes.json \
        --output_annotation_dir output/bloodline/FJDY_Chr1/updated_blocks \
        --output_inconsistent_tsv output/bloodline/FJDY_Chr1/subgenome_inconsistent.tsv \
        --species_name FJDY_Chr1
"""

import argparse
import os
import re
import json
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def format_position(bp):
    """碱基位置格式化"""
    if bp >= 1_000_000:
        return f"{bp / 1_000_000:.1f}M"
    elif bp >= 1_000:
        return f"{bp / 1_000:.0f}K"
    else:
        return str(bp)


def clean_bloodline(raw):
    """去掉尾部数字，首字母大写"""
    return re.sub(r'\d+$', '', str(raw)).capitalize()


def block_label(chrom, bloodline_short, start, end):
    """生成人类可读的 block 标签，格式：Chr1H:ND:43.0M-44.0M"""
    return f"{chrom}:{bloodline_short}:{format_position(start)}-{format_position(end)}"


BLOODLINE_SHORT = {
    'Spontaneum': 'Sp',
    'Robustum': 'Ro',
    'Officinarum': 'Of',
    'Nodata': 'ND',
}


def short(bl):
    return BLOODLINE_SHORT.get(bl, bl[:2])


# ─────────────────────────────────────────────
# 加载 block 注释
# ─────────────────────────────────────────────

def load_block_annotations(block_dir):
    """
    返回：
        block_info_map  : {block_id -> dict(chrom, bloodline, start, end, raw_bloodline)}
        chrom_dfs       : {chrom -> DataFrame}  原始 DataFrame（用于后续输出）
        chrom_block_ids : {chrom -> [block_id, ...]}  按顺序排列
    """
    block_info_map = {}
    chrom_dfs = {}
    chrom_block_ids = {}

    for filename in sorted(os.listdir(block_dir)):
        if not filename.endswith('.txt'):
            continue
        chrom = filename.replace('.txt', '')
        filepath = os.path.join(block_dir, filename)

        try:
            df = pd.read_csv(filepath, sep='\t')
        except Exception as e:
            print(f"[WARN] 无法读取 {filename}: {e}")
            continue

        bloodline_counter = {}
        ids_in_order = []

        for idx, row in df.iterrows():
            start = int(row['Start'])
            end = int(row['End'])
            raw_bl = str(row['Bloodline'])
            bl = clean_bloodline(raw_bl)

            if bl not in bloodline_counter:
                bloodline_counter[bl] = 0
            bloodline_counter[bl] += 1

            block_id = f"{chrom}_{bl}_{bloodline_counter[bl]}"
            ids_in_order.append(block_id)

            block_info_map[block_id] = {
                'chrom': chrom,
                'bloodline': bl,
                'raw_bloodline': raw_bl,
                'start': start,
                'end': end,
                'df_idx': idx,
            }

        chrom_dfs[chrom] = df
        chrom_block_ids[chrom] = ids_in_order

    return block_info_map, chrom_dfs, chrom_block_ids


# ─────────────────────────────────────────────
# 推断 Nodata 血缘
# ─────────────────────────────────────────────

def infer_nodata_bloodlines(dist_matrix, block_info_map):
    """
    对每个 Nodata block，在距离矩阵中找距离最近的非Nodata block，取其血缘。
    返回：{block_id -> inferred_bloodline}
    """
    all_ids = list(dist_matrix.index)
    dist_val = (dist_matrix.values + dist_matrix.values.T) / 2.0
    np.fill_diagonal(dist_val, np.inf)  # 对角线设为inf，不选自身

    inferred = {}

    for i, bid in enumerate(all_ids):
        info = block_info_map.get(bid, {})
        if info.get('bloodline', '') != 'Nodata':
            continue

        # 按距离排序，找最近的非Nodata block
        sorted_j = np.argsort(dist_val[i])
        assigned = None
        for j in sorted_j:
            neighbor_id = all_ids[j]
            neighbor_info = block_info_map.get(neighbor_id, {})
            neighbor_bl = neighbor_info.get('bloodline', '')
            if neighbor_bl != 'Nodata' and neighbor_bl != '':
                assigned = neighbor_bl
                break

        if assigned is None:
            assigned = 'Unknown'

        inferred[bid] = assigned
        chrom = info.get('chrom', '?')
        start = info.get('start', 0)
        end = info.get('end', 0)
        print(f"  [Nodata→] {block_label(chrom, 'ND', start, end)}  →  {assigned}")

    return inferred


# ─────────────────────────────────────────────
# 输出更新后的 block 注释文件
# ─────────────────────────────────────────────

def write_updated_annotations(chrom_dfs, chrom_block_ids, block_info_map,
                               inferred_map, output_dir):
    """
    对每条染色体输出一个更新后的 txt 文件。
    Nodata 行的 Bloodline 列在原值后面追加推断结果，例如：
        NoData8  →  NoData8(Spontaneum)
    """
    os.makedirs(output_dir, exist_ok=True)

    for chrom, df in chrom_dfs.items():
        block_ids = chrom_block_ids.get(chrom, [])
        if len(block_ids) != len(df):
            print(f"[WARN] {chrom}: block_ids 数量({len(block_ids)}) ≠ df 行数({len(df)})，跳过")
            continue

        new_df = df.copy()

        for i, bid in enumerate(block_ids):
            info = block_info_map.get(bid, {})
            if info.get('bloodline', '') == 'Nodata' and bid in inferred_map:
                original_val = new_df.at[i, 'Bloodline']
                inferred_bl = inferred_map[bid]
                new_df.at[i, 'Bloodline'] = f"{original_val}({inferred_bl})"

        out_path = os.path.join(output_dir, f"{chrom}.txt")
        new_df.to_csv(out_path, sep='\t', index=False)
        print(f"  [输出] {out_path}")


# ─────────────────────────────────────────────
# 亚基因组一致性检查
# ─────────────────────────────────────────────

def extract_chrom_name(raw_key):
    """
    从 JSON 里的 key 提取纯染色体名。
    兼容两种格式：
      - "Chr1A"               → "Chr1A"   （旧格式，纯染色体名）
      - "Chr1A_Robustum"      → "Chr1A"   （新格式，训练脚本用 rpartition('_')[0] 生成）
      - "Chr1A_Nodata"        → "Chr1A"
    规则：取第一个 '_' 前面的部分；如果没有 '_' 则原样返回。
    """
    raw = raw_key.strip()
    # 染色体名一般是 ChrXY 格式（字母+数字+字母），不含下划线
    # 只要第一段（按 '_' 分割）就是染色体名
    return raw.split('_')[0]


def load_subgenome_json(json_path):
    """
    读取亚基因组 JSON。兼容两种 value 格式：
      格式A（纯染色体名，旧）：
        {"subgenome_0": ["Chr1G", "Chr1H", ...], ...}
      格式B（血缘级别 key，训练脚本实际输出）：
        {"subgenome_0": ["Chr1A_Nodata", "Chr1A_Robustum", "Chr1A_Spontaneum", ...], ...}

    返回：
        chrom_to_sg : {染色体名 -> subgenome_id}，同一染色体只记录一次
        sg_to_chroms: {subgenome_id -> [染色体名, ...]}，去重后的染色体列表
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    chrom_to_sg = {}
    sg_to_chroms = {}

    for sg_key, entries in data.items():
        sg_id = sg_key  # e.g. "subgenome_0"
        unique_chroms_in_sg = []

        for entry in entries:
            chrom = extract_chrom_name(entry)
            if chrom not in chrom_to_sg:
                chrom_to_sg[chrom] = sg_id
                unique_chroms_in_sg.append(chrom)
            # 如果同一染色体在同一亚基因组里出现多次（多血缘），只记一次，忽略重复

        sg_to_chroms[sg_id] = unique_chroms_in_sg

    return chrom_to_sg, sg_to_chroms


def check_subgenome_consistency(dist_matrix, block_info_map, chrom_to_sg):
    """
    对每个 block，找距离最近的非同染色体 block，
    判断其所属亚基因组是否与自身染色体预期亚基因组一致。

    返回：inconsistent_list，每项为 dict：
        {block_id, chrom, bloodline, start, end,
         expected_sg, nearest_id, nearest_chrom, nearest_sg, distance}
    """
    all_ids = list(dist_matrix.index)
    dist_val = (dist_matrix.values + dist_matrix.values.T) / 2.0
    np.fill_diagonal(dist_val, np.inf)

    inconsistent = []

    for i, bid in enumerate(all_ids):
        info = block_info_map.get(bid, {})
        chrom = info.get('chrom', '')
        expected_sg = chrom_to_sg.get(chrom, None)
        if expected_sg is None:
            continue  # 该染色体不在亚基因组分配里，跳过

        # 找最近的邻居（不限血缘）
        sorted_j = np.argsort(dist_val[i])
        nearest_j = sorted_j[0]
        nearest_id = all_ids[nearest_j]
        nearest_dist = dist_val[i, nearest_j]

        nearest_info = block_info_map.get(nearest_id, {})
        nearest_chrom = nearest_info.get('chrom', '')
        nearest_sg = chrom_to_sg.get(nearest_chrom, None)

        if nearest_sg is None:
            continue

        if nearest_sg != expected_sg:
            inconsistent.append({
                'block_id': bid,
                'chrom': chrom,
                'bloodline': info.get('bloodline', ''),
                'start': info.get('start', 0),
                'end': info.get('end', 0),
                'expected_sg': expected_sg,
                'nearest_id': nearest_id,
                'nearest_chrom': nearest_chrom,
                'nearest_sg': nearest_sg,
                'distance': nearest_dist,
            })

    return inconsistent


def print_and_save_inconsistent(inconsistent_list, output_tsv, inferred_map):
    """
    打印不一致 block，并保存为 TSV。
    标签格式：Chr1H:ND->Sp:43.0M-44.0M（Nodata block 带推断标签）
    """
    print("\n" + "=" * 72)
    print("SUBGENOME CONSISTENCY CHECK")
    print("=" * 72)

    if not inconsistent_list:
        print("  ✓ 所有 block 均与预期亚基因组一致，无异常！")
    else:
        print(f"  发现 {len(inconsistent_list)} 个 block 与预期亚基因组不一致：\n")
        for item in inconsistent_list:
            bl = item['bloodline']
            bid = item['block_id']

            # 构造显示标签
            if bl == 'Nodata' and bid in inferred_map:
                bl_display = f"ND->{inferred_map[bid][:2]}"
            else:
                bl_display = short(bl)

            label = block_label(item['chrom'], bl_display, item['start'], item['end'])
            nearest_info_str = f"{item['nearest_chrom']}({item['nearest_sg']})"

            print(f"  [不一致] {label}")
            print(f"           预期亚基因组: {item['expected_sg']}  |  最近邻: {item['nearest_id']} → {nearest_info_str}  |  距离: {item['distance']:.4f}")

    print("=" * 72 + "\n")

    # 写 TSV
    if inconsistent_list:
        rows = []
        for item in inconsistent_list:
            bl = item['bloodline']
            bid = item['block_id']
            if bl == 'Nodata' and bid in inferred_map:
                bl_display = f"ND->{inferred_map[bid]}"
            else:
                bl_display = bl

            label = block_label(item['chrom'], bl_display, item['start'], item['end'])
            rows.append({
                'block_label': label,
                'block_id': bid,
                'chrom': item['chrom'],
                'bloodline': item['bloodline'],
                'inferred_bloodline': inferred_map.get(bid, ''),
                'start': item['start'],
                'end': item['end'],
                'expected_subgenome': item['expected_sg'],
                'nearest_block_id': item['nearest_id'],
                'nearest_chrom': item['nearest_chrom'],
                'nearest_subgenome': item['nearest_sg'],
                'distance': round(item['distance'], 6),
            })
        df_out = pd.DataFrame(rows)
        df_out.to_csv(output_tsv, sep='\t', index=False)
        print(f"  不一致 block 已保存至: {output_tsv}")
    else:
        # 输出空 TSV（保证文件存在）
        pd.DataFrame(columns=['block_label', 'block_id', 'chrom', 'bloodline',
                               'inferred_bloodline', 'start', 'end',
                               'expected_subgenome', 'nearest_block_id',
                               'nearest_chrom', 'nearest_subgenome', 'distance']
                     ).to_csv(output_tsv, sep='\t', index=False)
        print(f"  （空）一致性文件已保存至: {output_tsv}")


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nodata血缘推断 + 亚基因组一致性检查")
    parser.add_argument('--input_tsv', required=True,
                        help="距离矩阵 TSV（来自 train_adaptive_unsupervised.py 输出）")
    parser.add_argument('--block_dir', required=True,
                        help="Block 注释目录（含 Chr1A.txt 等文件）")
    parser.add_argument('--subgenome_json', required=True,
                        help="亚基因组分配 JSON 文件（由训练脚本输出）")
    parser.add_argument('--output_annotation_dir', required=True,
                        help="更新后 block 注释文件的输出目录")
    parser.add_argument('--output_inconsistent_tsv', required=True,
                        help="亚基因组不一致 block 的输出 TSV")
    parser.add_argument('--species_name', default="Species",
                        help="物种/分析名称（用于日志标题）")
    args = parser.parse_args()

    print("=" * 72)
    print(f"NODATA BLOODLINE ASSIGNMENT + SUBGENOME CONSISTENCY CHECK")
    print(f"Species: {args.species_name}")
    print("=" * 72)

    # ── 1. 读取距离矩阵 ──
    print("\n[1/4] 读取距离矩阵...")
    dist_matrix = pd.read_csv(args.input_tsv, sep='\t', index_col=0)
    print(f"  矩阵大小: {dist_matrix.shape}")

    # ── 2. 读取 block 注释 ──
    print("\n[2/4] 读取 block 注释...")
    block_info_map, chrom_dfs, chrom_block_ids = load_block_annotations(args.block_dir)
    print(f"  共加载 {len(block_info_map)} 个 blocks")

    nodata_count = sum(1 for v in block_info_map.values() if v['bloodline'] == 'Nodata')
    print(f"  其中 Nodata blocks: {nodata_count}")

    # ── 3. 推断 Nodata 血缘 ──
    print("\n[3/4] 推断 Nodata 血缘（最近非Nodata邻居）...")
    inferred_map = infer_nodata_bloodlines(dist_matrix, block_info_map)
    print(f"  共推断 {len(inferred_map)} 个 Nodata blocks")

    # ── 3b. 输出更新后的 block 注释文件 ──
    print("\n  输出更新后的 block 注释文件...")
    write_updated_annotations(chrom_dfs, chrom_block_ids, block_info_map,
                               inferred_map, args.output_annotation_dir)

    # ── 4. 亚基因组一致性检查 ──
    print("\n[4/4] 亚基因组一致性检查...")
    chrom_to_sg, sg_to_chroms = load_subgenome_json(args.subgenome_json)

    print(f"  亚基因组分配：")
    for sg, chroms in sg_to_chroms.items():
        print(f"    {sg}: {', '.join(chroms)}")

    inconsistent = check_subgenome_consistency(dist_matrix, block_info_map, chrom_to_sg)
    print_and_save_inconsistent(inconsistent, args.output_inconsistent_tsv, inferred_map)

    # ── 输出 inferred_map 供绘图脚本使用 ──
    inferred_tsv = os.path.join(
        os.path.dirname(args.output_inconsistent_tsv),
        f"{args.species_name}_nodata_inferred.tsv"
    )
    rows = []
    for bid, bl in inferred_map.items():
        info = block_info_map.get(bid, {})
        rows.append({
            'block_id': bid,
            'chrom': info.get('chrom', ''),
            'start': info.get('start', 0),
            'end': info.get('end', 0),
            'inferred_bloodline': bl,
        })
    pd.DataFrame(rows).to_csv(inferred_tsv, sep='\t', index=False)
    print(f"\n  Nodata 推断结果已保存至: {inferred_tsv}")
    print("\n✓ 全部完成！")


if __name__ == "__main__":
    main()
