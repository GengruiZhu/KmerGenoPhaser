#!/usr/bin/env Rscript
# =============================================================================
# block_identification.R
# Ancestry block calling via Maximum-Likelihood + Diagnostic-Dosage fusion.
#
# Fully generic: supports N ancestor groups (N >= 2).
# Groups are defined by --group_names and matched by --group_patterns.
#
# Usage (2 groups):
#   Rscript block_identification.R \
#       --input_dir     /path/to/ad_matrices \
#       --output_dir    /path/to/output \
#       --chrom_sizes   genome.sizes \
#       --target_sample MyHybrid.1 \
#       --sample_names  "ParentA.1,ParentA.2,ParentB.1,ParentB.2,MyHybrid.1" \
#       --group_names   "GroupA,GroupB" \
#       --group_patterns "ParentA,ParentB"
#
# Usage (3 groups — classic sugarcane):
#   Rscript block_identification.R \
#       --input_dir     /path/to/ad_matrices \
#       --output_dir    /path/to/output \
#       --chrom_sizes   FJDY.genome.size \
#       --target_sample Fjdy.1 \
#       --sample_names  "Ssp.1,Ssp.2,Sof.1,Sof.2,Sro.1,Sro.2" \
#       --group_names   "Spontaneum,Officinarum,Robustum" \
#       --group_patterns "Ssp,Sof,Sro" \
#       --diag_dir      /path/to/bedgraph_files
# =============================================================================

suppressPackageStartupMessages({
  library(stringr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
  library(data.table)
})

# =============================================================================
# 0) Parse command-line arguments
# =============================================================================
args_raw <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  result <- list(
    input_dir            = NULL,
    output_dir           = NULL,
    chrom_sizes          = NULL,
    target_sample        = NULL,
    sample_names         = NULL,     # comma-separated
    group_names          = NULL,     # comma-separated: "Spontaneum,Officinarum,Robustum"
    group_patterns       = NULL,     # comma-separated grep patterns, same order as group_names
    diag_dir             = NULL,     # directory with bedGraph files (optional)
    win_size             = 1000000,
    min_call_rate        = 0.50,
    min_group_depth      = 20,
    min_delta            = 0.30,
    min_sites_per_window = 20,
    min_inf_per_window   = 20,
    complex_margin_thr   = 0.05,
    min_reads_thr        = 30,
    min_ratio_thr        = 0.40
  )
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    val <- if (i + 1 <= length(args)) args[i + 1] else NA
    if (key %in% names(result)) {
      result[[key]] <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  for (nm in c("win_size","min_call_rate","min_group_depth","min_delta",
               "min_sites_per_window","min_inf_per_window","complex_margin_thr",
               "min_reads_thr","min_ratio_thr")) {
    result[[nm]] <- as.numeric(result[[nm]])
  }
  result
}

p <- parse_args(args_raw)

for (req in c("input_dir","output_dir","chrom_sizes","target_sample",
              "sample_names","group_names","group_patterns")) {
  if (is.null(p[[req]]) || is.na(p[[req]]))
    stop("Missing required argument: --", req)
}

dir.create(p$output_dir, showWarnings = FALSE, recursive = TRUE)

# Parse sample and group arrays
sample_names   <- trimws(strsplit(p$sample_names,   ",")[[1]])
group_names    <- trimws(strsplit(p$group_names,    ",")[[1]])
group_patterns <- trimws(strsplit(p$group_patterns, ",")[[1]])

if (length(group_names) != length(group_patterns))
  stop("--group_names and --group_patterns must have the same number of entries.")
if (length(group_names) < 2)
  stop("At least 2 ancestor groups are required.")

n_groups <- length(group_names)
message("Ancestor groups (", n_groups, "):")
for (gi in seq_len(n_groups))
  message("  [", gi, "] ", group_names[gi], "  (pattern: '", group_patterns[gi], "')")
message("Samples (", length(sample_names), "): ", paste(sample_names, collapse=", "))
message("Target sample: ", p$target_sample)

# =============================================================================
# 1) Load chromosome sizes
# =============================================================================
if (!file.exists(p$chrom_sizes))
  stop("Chromosome sizes file not found: ", p$chrom_sizes)
chrom_sizes_df     <- read.table(p$chrom_sizes, header=FALSE, col.names=c("Chrom","Len"))
chrom_sizes_lookup <- setNames(chrom_sizes_df$Len, chrom_sizes_df$Chrom)
message("Loaded ", nrow(chrom_sizes_df), " chromosome sizes.")

# Winner level order: NoData/LowInfo/Complex/Unknown first, then group names
winner_levels <- c("NoData","LowInfo","Complex","Unknown", group_names)

# =============================================================================
# 2) Utility functions
# =============================================================================
parse_ad <- function(df, cols) {
  for (col in cols) {
    sd  <- str_split_fixed(df[[col]], ",", 2)
    ref <- suppressWarnings(as.numeric(sd[,1])); ref[is.na(ref)] <- 0
    alt <- suppressWarnings(as.numeric(sd[,2])); alt[is.na(alt)] <- 0
    df[[paste0(col,"_R")]]     <- ref
    df[[paste0(col,"_A")]]     <- alt
    df[[paste0(col,"_Total")]] <- ref + alt
  }
  df
}

calc_group_freq <- function(df, group_cols) {
  r <- rowSums(df[, paste0(group_cols,"_R"),     drop=FALSE])
  a <- rowSums(df[, paste0(group_cols,"_A"),     drop=FALSE])
  pmin(pmax((a + 1) / (r + a + 2), 1e-6), 1 - 1e-6)
}

calc_smooth <- function(curr, prev, nxt) {
  0.25 * ifelse(is.na(prev), 0, prev) +
  0.50 * curr +
  0.25 * ifelse(is.na(nxt),  0, nxt)
}

process_dosage <- function(path, min_reads) {
  if (is.null(path) || !file.exists(path)) return(NULL)
  d <- tryCatch(read.table(path, header=FALSE, stringsAsFactors=FALSE),
                error=function(e) NULL)
  if (is.null(d)) return(NULL)
  if (ncol(d) >= 6) {
    colnames(d)[1:6] <- c("CHROM","Start","End","Ratio","Num","Den")
    d <- d[d$Den >= min_reads, ]
    return(d[, c("Start","Ratio")])
  } else if (ncol(d) == 4) {
    colnames(d)[1:4] <- c("CHROM","Start","End","Ratio")
    return(d[, c("Start","Ratio")])
  }
  NULL
}

# =============================================================================
# 3) Build a fixed NPG-style color palette for up to 10 groups + status levels
# =============================================================================
npg_palette <- c(
  "#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F",
  "#8491B4","#91D1C2","#DC0000","#7E6148","#B09C85"
)
status_colors <- c(
  NoData  = "gray90",
  LowInfo = "gray40",
  Complex = "#7E57C2",
  Unknown = "#000000"
)
group_colors <- setNames(npg_palette[seq_len(n_groups)], group_names)
all_colors   <- c(status_colors, group_colors)

# =============================================================================
# 4) Main loop over AD-matrix files
# =============================================================================
files_list <- list.files(p$input_dir, pattern="_AD_matrix\\.txt$", full.names=TRUE)
if (length(files_list) == 0)
  stop("No *_AD_matrix.txt files found in: ", p$input_dir)
message("\nFound ", length(files_list), " AD-matrix file(s). Starting processing...\n")

for (input_file in files_list) {
  tryCatch({
    current_chrom <- gsub("_AD_matrix\\.txt$", "", basename(input_file))
    file_prefix   <- file.path(p$output_dir, current_chrom)
    message("========================================================")
    message("Processing: ", current_chrom)

    REAL_CHROM_LEN <- if (current_chrom %in% names(chrom_sizes_lookup))
      chrom_sizes_lookup[[current_chrom]] else {
        warning(current_chrom, " not in chrom_sizes, using data max.")
        0
      }

    # --- read AD matrix ---
    raw_df <- read.table(input_file, header=FALSE, stringsAsFactors=FALSE,
                         na.strings=".", check.names=FALSE)
    expected_cols <- 4 + length(sample_names)
    if (ncol(raw_df) != expected_cols)
      stop("Column mismatch in ", basename(input_file),
           ": expected ", expected_cols, ", got ", ncol(raw_df))
    colnames(raw_df) <- c("CHROM","POS","REF","ALT", sample_names)
    colnames(raw_df) <- gsub(".*\\]", "", colnames(raw_df))
    colnames(raw_df) <- gsub(":AD",   "", colnames(raw_df))
    sample_cols      <- colnames(raw_df)[5:ncol(raw_df)]

    # --- parse AD & call-rate filter ---
    df_parsed  <- parse_ad(raw_df, sample_cols)
    depth_cols <- paste0(sample_cols, "_Total")
    valid_cnt  <- rowSums(df_parsed[, depth_cols] > 0)
    df_clean   <- df_parsed[valid_cnt >= length(sample_cols) * p$min_call_rate, ]
    message("  Sites after call-rate filter: ", nrow(df_clean), " / ", nrow(df_parsed))

    # --- identify group columns ---
    group_cols_list <- lapply(group_patterns, function(pat)
      sample_cols[grepl(pat, sample_cols, fixed=FALSE)])
    names(group_cols_list) <- group_names

    for (gi in seq_len(n_groups)) {
      if (length(group_cols_list[[gi]]) == 0)
        stop("No columns matched pattern '", group_patterns[gi],
             "' for group '", group_names[gi], "'. Check --group_patterns.")
    }

    # --- group allele frequencies ---
    for (gi in seq_len(n_groups)) {
      freq_col <- paste0("Freq_", group_names[gi])
      df_clean[[freq_col]] <- calc_group_freq(df_clean, group_cols_list[[gi]])
    }

    # --- log-likelihoods per group ---
    tgt_total_col <- paste0(p$target_sample, "_Total")
    df_calc <- df_clean[df_clean[[tgt_total_col]] > 0, ]
    tgt_r   <- df_calc[[paste0(p$target_sample, "_R")]]
    tgt_a   <- df_calc[[paste0(p$target_sample, "_A")]]
    tgt_n   <- tgt_r + tgt_a

    for (gi in seq_len(n_groups)) {
      freq_col <- paste0("Freq_",  group_names[gi])
      ll_col   <- paste0("LL_",    group_names[gi])
      df_calc[[ll_col]] <- dbinom(tgt_a, tgt_n, df_calc[[freq_col]], log=TRUE)
    }

    # --- informative sites ---
    depth_sum_cols <- sapply(seq_len(n_groups), function(gi) {
      gc <- group_cols_list[[gi]]
      paste0("Depth_", group_names[gi])
    })

    for (gi in seq_len(n_groups)) {
      dc <- paste0("Depth_", group_names[gi])
      df_calc[[dc]] <- rowSums(df_calc[, paste0(group_cols_list[[gi]], "_Total"), drop=FALSE])
    }

    freq_mat <- as.matrix(df_calc[, paste0("Freq_", group_names), drop=FALSE])
    df_calc$Delta <- apply(freq_mat, 1, max) - apply(freq_mat, 1, min)

    all_deep_enough <- Reduce(`&`, lapply(seq_len(n_groups), function(gi)
      df_calc[[paste0("Depth_", group_names[gi])]] >= p$min_group_depth))
    df_calc$IsInformative <- all_deep_enough & (df_calc$Delta >= p$min_delta)

    # --- window summary ---
    ll_cols <- paste0("LL_", group_names)
    df_window <- df_calc %>%
      mutate(Window_Start = floor(POS / p$win_size) * p$win_size) %>%
      group_by(Window_Start) %>%
      summarise(
        n_sites = n(),
        n_inf   = sum(IsInformative),
        across(all_of(ll_cols), ~ sum(.x[IsInformative], na.rm=TRUE),
               .names="Sum_{.col}"),
        .groups = "drop"
      ) %>%
      filter(n_sites >= p$min_sites_per_window)

    sum_ll_cols <- paste0("Sum_LL_", group_names)

    # --- smoothing ---
    df_smooth <- df_window %>% arrange(Window_Start) %>% mutate(
      across(all_of(sum_ll_cols), ~ calc_smooth(.x, lag(.x), lead(.x)),
             .names="S_{.col}"),
      S_n_inf = calc_smooth(n_inf, lag(n_inf), lead(n_inf))
    )

    s_cols <- paste0("S_Sum_LL_", group_names)

    df_smooth <- df_smooth %>% rowwise() %>% mutate(
      s_vec      = list(c_across(all_of(s_cols))),
      best_i     = which.max(s_vec[[1]]),
      second_i   = which.max(replace(s_vec[[1]], best_i, -Inf)),
      Margin     = s_vec[[1]][best_i] - s_vec[[1]][second_i],
      Margin_per_inf = ifelse(S_n_inf > 0, Margin / S_n_inf, NA_real_),
      BaseWinner = group_names[best_i],
      Winner = case_when(
        S_n_inf < p$min_inf_per_window    ~ "LowInfo",
        is.na(Margin_per_inf)             ~ "LowInfo",
        Margin_per_inf < p$complex_margin_thr ~ "Complex",
        TRUE                              ~ BaseWinner
      )
    ) %>% ungroup() %>% select(-s_vec)

    # --- full skeleton 0..REAL_CHROM_LEN ---
    data_max  <- if (nrow(df_smooth) > 0) max(df_smooth$Window_Start) + p$win_size else 0
    final_len <- max(REAL_CHROM_LEN, data_max)
    full_skeleton <- tibble(
      Window_Start = seq(0, max(0, final_len - 1), by=p$win_size)
    )

    df_fixed <- full_skeleton %>%
      left_join(df_smooth, by="Window_Start") %>%
      mutate(
        Window_End = Window_Start + p$win_size,
        Window_Mb  = Window_Start / 1e6,
        Winner     = replace_na(as.character(Winner), "NoData"),
        Winner     = factor(Winner, levels=winner_levels)
      )

    # --- integrate diagnostic dosage bedGraphs (optional) ---
    # Expects files named: <target>_<chrom>_<GroupName>diag_1Mb.bedGraph
    diag_data <- list()
    if (!is.null(p$diag_dir) && !is.na(p$diag_dir)) {
      for (gname in group_names) {
        bg_path <- file.path(p$diag_dir,
          paste0(p$target_sample, "_", current_chrom, "_", gname, "diag_1Mb.bedGraph"))
        diag_data[[gname]] <- process_dosage(bg_path, p$min_reads_thr)
      }
    }

    safe_join <- function(base, d, col_name) {
      if (is.null(d)) { base[[col_name]] <- 0; return(base) }
      colnames(d)[2] <- col_name
      left_join(base, d, by=c("Window_Start"="Start")) %>%
        mutate(across(all_of(col_name), ~ replace_na(.x, 0)))
    }

    df_merged <- df_fixed
    diag_cols <- character(0)
    for (gname in group_names) {
      dcol <- paste0("Diag_Ratio_", gname)
      diag_cols <- c(diag_cols, dcol)
      df_merged <- safe_join(df_merged, diag_data[[gname]], dcol)
    }

    df_merged <- df_merged %>% rowwise() %>% mutate(
      Diag_Max = max(c_across(all_of(diag_cols))),
      Diag_Winner = if (Diag_Max < p$min_ratio_thr) "NoSignal" else {
        idx <- which.max(c_across(all_of(diag_cols)))
        group_names[idx]
      },
      Final_Winner     = ifelse(Diag_Winner != "NoSignal",
                                Diag_Winner, as.character(Winner)),
      Final_Confidence = ifelse(Diag_Winner != "NoSignal",
                                10, Margin_per_inf),
      Final_Winner     = factor(Final_Winner, levels=winner_levels)
    ) %>% ungroup()

    # --- plots ---
    real_end <- df_merged %>% filter(Final_Winner != "NoData") %>%
      summarise(v=max(Window_End)) %>% pull(v)
    if (is.na(real_end) || length(real_end) == 0) real_end <- max(df_merged$Window_End)
    x_lim    <- real_end / 1e6
    shared_x <- scale_x_continuous(limits=c(0, x_lim), expand=c(0,0))

    p1 <- ggplot(df_merged) +
      geom_rect(aes(xmin=Window_Start/1e6, xmax=Window_End/1e6,
                    ymin=0, ymax=1, fill=Final_Winner), color=NA) +
      scale_fill_manual(values=all_colors, drop=FALSE) +
      labs(title=paste0("Ancestry | ", current_chrom, " | target: ", p$target_sample),
           y=NULL, fill=NULL) +
      theme_minimal() +
      theme(axis.text=element_blank(), panel.grid=element_blank()) +
      shared_x

    p2 <- ggplot(df_merged, aes(x=Window_Mb, y=Final_Confidence)) +
      geom_area(aes(fill=Final_Winner), alpha=0.3) +
      geom_line(aes(color=Final_Winner), linewidth=0.8) +
      scale_fill_manual(values=all_colors) +
      scale_color_manual(values=all_colors) +
      labs(y="Confidence") + theme_classic() +
      theme(legend.position="none") + shared_x

    # Diagnostic ratio panel — one line per group
    diag_long <- df_merged %>%
      select(Window_Mb, all_of(diag_cols)) %>%
      pivot_longer(all_of(diag_cols), names_to="Group", values_to="Ratio") %>%
      mutate(Group = str_remove(Group, "^Diag_Ratio_"),
             Group = factor(Group, levels=group_names))

    p3 <- ggplot(diag_long, aes(x=Window_Mb, y=Ratio, color=Group)) +
      geom_line(alpha=0.8, linewidth=0.6) +
      scale_color_manual(values=group_colors) +
      labs(y="Diag Ratio", x="Position (Mb)", color=NULL) +
      theme_minimal() + shared_x

    final_plot <- p1 / p2 / p3 + plot_layout(heights=c(1,2,1))
    out_pdf    <- paste0(file_prefix, "_Final_Merged.pdf")
    ggsave(out_pdf, final_plot, width=12, height=9)
    message("  Plot saved: ", basename(out_pdf))

    # --- export block CSV (RLE-merged) ---
    df_export   <- df_merged %>% arrange(Window_Start)
    rle_res     <- rle(as.character(df_export$Final_Winner))
    end_i       <- cumsum(rle_res$lengths)
    start_i     <- c(1, head(end_i, -1) + 1)

    simple_blocks <- data.frame(
      Chrom     = current_chrom,
      Start_Mb  = df_export$Window_Start[start_i] / 1e6,
      End_Mb    = df_export$Window_End[end_i]      / 1e6,
      Bloodline = rle_res$values
    )
    out_csv <- paste0(file_prefix, "_Final_Blocks.csv")
    write.csv(simple_blocks, out_csv, row.names=FALSE, quote=TRUE)
    message("  CSV saved:  ", basename(out_csv))

    simple_blocks_bp <- data.frame(
      Start_bp  = df_export$Window_Start[start_i],
      End_bp    = df_export$Window_End[end_i],
      Bloodline = rle_res$values
    )
    out_txt <- paste0(file_prefix, ".txt")
    write.table(simple_blocks_bp, out_txt, sep="\t", row.names=FALSE,
                col.names=FALSE, quote=FALSE)
    message("  Block txt saved: ", basename(out_txt), " (for autoencoder)")

  }, error=function(e) {
    message("ERROR processing ", basename(input_file), ": ", e$message)
  })
}

message("\n========================================================")
message("All chromosomes processed. Output in: ", p$output_dir)
message("========================================================")
