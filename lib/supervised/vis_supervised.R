#!/usr/bin/env Rscript
# =============================================================================
# vis_supervised.R  —  KmerGenoPhaser Supervised Visualization Module
# =============================================================================
# Produces two plot types per homologous chromosome group:
#   1. Proportional stacked bar  (continuous k-mer proportion)
#   2. Dominant ancestry map     (window majority call >= dominance_threshold)
#
# Fully generic: species names, colors, column mapping, titles are all CLI args.
#
# Usage:
#   Rscript vis_supervised.R \
#       --data_dir      /path/to/mapping_tsv \
#       --output_dir    /path/to/output \
#       --species_names "SSPON,SOFFI" \
#       --species_cols  "SES208,B48" \
#       --species_colors "#2ca02c,#1f77b4" \
#       --genome_title  "R570" \
#       [--block_file   /path/to/block_boundaries.txt] \
#       [--dominance_threshold 0.55] \
#       [--chrom_pattern "Chr[0-9]+"] \
#       [--ncols 6] \
#       [--simsun_font  simsun.ttc] \
#       [--times_font   times.ttf] \
#       [--lang en]       # en | cn  (axis/legend language)
# =============================================================================
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(fs)
  library(stringr)
  library(ggrepel)
})

# =============================================================================
# 0) Argument parsing
# =============================================================================
parse_args <- function(args) {
  defaults <- list(
    data_dir             = NULL,
    output_dir           = NULL,
    species_names        = NULL,     # "SSPON,SOFFI"  or  "AA,BB,DD"
    species_cols         = NULL,     # column names in TSV matching species order
    species_colors       = NULL,     # hex colors, comma-separated
    genome_title         = "Target",
    block_file           = NULL,     # optional boundary annotation
    dominance_threshold  = 0.55,
    chrom_pattern        = "Chr[0-9]+",
    ncols                = 6,
    simsun_font          = NULL,
    times_font           = NULL,
    lang                 = "en"      # en | cn
  )
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    val <- if (i + 1 <= length(args)) args[i + 1] else NA
    if (key %in% names(defaults)) {
      defaults[[key]] <- val; i <- i + 2
    } else {
      i <- i + 1
    }
  }
  defaults$dominance_threshold <- as.numeric(defaults$dominance_threshold)
  defaults$ncols               <- as.integer(defaults$ncols)
  defaults
}

p <- parse_args(commandArgs(trailingOnly = TRUE))

for (req in c("data_dir", "output_dir", "species_names", "species_cols", "species_colors")) {
  if (is.null(p[[req]]) || is.na(p[[req]]))
    stop("Missing required argument: --", req)
}

dir_create(p$output_dir)

# Parse comma-separated lists
sp_names  <- trimws(strsplit(p$species_names,  ",")[[1]])
sp_cols   <- trimws(strsplit(p$species_cols,   ",")[[1]])
sp_colors <- trimws(strsplit(p$species_colors, ",")[[1]])

if (length(sp_names) != length(sp_cols) || length(sp_names) != length(sp_colors))
  stop("--species_names, --species_cols, --species_colors must all have the same length.")

n_sp <- length(sp_names)

# Build named vectors
species_map    <- setNames(sp_colors, sp_names)
mixed_color    <- "gray70"
dominant_map   <- c(species_map, "Mixed" = mixed_color)
dominant_levels <- c(sp_names, "Mixed")

# =============================================================================
# 1) Font setup (optional — skip if fonts not found)
# =============================================================================
use_showtext <- FALSE
tryCatch({
  library(showtext)
  if (!is.null(p$simsun_font) && !is.na(p$simsun_font) && file.exists(p$simsun_font))
    font_add("SimSun", p$simsun_font)
  else
    font_add("SimSun", "sans")
  if (!is.null(p$times_font)  && !is.na(p$times_font)  && file.exists(p$times_font))
    font_add("Times", p$times_font)
  else
    font_add("Times", "serif")
  showtext_auto()
  options(device = "cairo")
  use_showtext <- TRUE
  message("[INFO] showtext enabled.")
}, error = function(e) {
  message("[INFO] showtext not available, using default fonts.")
})

fam_title <- if (use_showtext) "SimSun" else "sans"
fam_axis  <- if (use_showtext) "Times"  else "serif"

# =============================================================================
# 2) Labels  (en / cn)
# =============================================================================
if (p$lang == "cn") {
  label_mixed    <- "混合/推断 (Mixed)"
  lbl_xaxis      <- "物理位置 (Mb)"
  lbl_yaxis_prop <- "K-mer 占比 (%)"
  lbl_yaxis_dom  <- "血缘属性"
  lbl_legend_prop <- "血缘来源"
  lbl_legend_dom  <- "窗口主导血缘"
  title_suffix_prop <- "血缘连续比例分布"
  title_suffix_dom  <- "窗口主要血缘分布"
  make_title <- function(chrom_id)
    paste0(p$genome_title, " ", chrom_id, " 同源染色体组 ")
  fmt_sp  <- function(nm, pct) sprintf("%s: %.1f%%", nm, pct)
  fmt_mix <- function(pct)     sprintf("Mixed: %.1f%%", pct)
} else {
  label_mixed    <- "Mixed"
  lbl_xaxis      <- "Physical position (Mb)"
  lbl_yaxis_prop <- "K-mer proportion (%)"
  lbl_yaxis_dom  <- "Ancestry"
  lbl_legend_prop <- "Ancestry"
  lbl_legend_dom  <- "Dominant ancestry"
  title_suffix_prop <- "Proportional Ancestry"
  title_suffix_dom  <- "Dominant Ancestry"
  make_title <- function(chrom_id)
    paste0(p$genome_title, " ", chrom_id, " — ")
  fmt_sp  <- function(nm, pct) sprintf("%s: %.1f%%", nm, pct)
  fmt_mix <- function(pct)     sprintf("Mixed: %.1f%%", pct)
}

legend_labels_prop <- setNames(sp_names, sp_names)
legend_labels_dom  <- c(setNames(sp_names, sp_names), Mixed = label_mixed)

# =============================================================================
# 3) Load data files
# FIX: pattern changed from "_mapping_counts\\.tsv$" to "_mapping\\.tsv$"
#      to match actual output of map_kmers_to_genome.py (*_mapping.tsv)
# =============================================================================
all_files <- dir_ls(p$data_dir, regexp = "_mapping\\.tsv$")
if (length(all_files) == 0)
  stop("No *_mapping.tsv files found in: ", p$data_dir)

file_data <- tibble(
  file_path   = all_files,
  filename    = path_file(file_path),
  major_chrom = str_extract(filename, p$chrom_pattern)
)
major_chroms <- sort(unique(file_data$major_chrom[!is.na(file_data$major_chrom)]))
message("Found ", length(major_chroms), " major chromosome groups: ",
        paste(major_chroms, collapse = ", "))

# =============================================================================
# 4) Block boundaries (optional)
# =============================================================================
block_boundaries <- NULL
if (!is.null(p$block_file) && !is.na(p$block_file) && file.exists(p$block_file)) {
  block_boundaries <- tryCatch({
    read_tsv(p$block_file, skip = 1,
             col_names = c("Chr", "Progenitor", "blockStart", "blockEnd"),
             col_types = "ccii", show_col_types = FALSE) %>%
      select(Chr, blockStart, blockEnd) %>%
      pivot_longer(starts_with("block"), names_to = "Type", values_to = "Position") %>%
      mutate(Position_Mb = round(Position / 1e6, 2)) %>%
      distinct() %>% filter(Position > 1)
  }, error = function(e) { message("[WARN] Could not load block file: ", e$message); NULL })
}

# =============================================================================
# 5) Core loop
# =============================================================================
walk(major_chroms, function(major_chrom_id) {

  message("\n--- Processing: ", major_chrom_id, " ---")

  files_to_load <- file_data %>%
    filter(major_chrom == major_chrom_id) %>%
    pull(file_path)

  # Read TSV files; expected columns: Start, End, <sp_col_1>, <sp_col_2>, ...
  data_raw <- tryCatch(
    map_dfr(files_to_load, function(f) {
      d <- read_tsv(f, show_col_types = FALSE)
      colnames(d) <- trimws(colnames(d))
      col_lower <- tolower(colnames(d))
      start_c <- colnames(d)[col_lower == "start"][1]
      end_c   <- colnames(d)[col_lower == "end"][1]
      if (is.na(start_c) || is.na(end_c))
        stop("Cannot find Start/End columns in ", basename(f))
      missing_cols <- sp_cols[!sp_cols %in% colnames(d)]
      if (length(missing_cols) > 0)
        stop("Missing species columns: ", paste(missing_cols, collapse = ", "),
             " in ", basename(f))
      d %>%
        select(Start = all_of(start_c), End = all_of(end_c),
               all_of(setNames(sp_cols, sp_names))) %>%
        mutate(Chr = str_extract(basename(f), paste0(p$chrom_pattern, "[A-Z]?")))
    }),
    error = function(e) { message("[ERROR] ", e$message); return(NULL) }
  )

  if (is.null(data_raw) || nrow(data_raw) == 0) {
    message("[SKIP] No data for ", major_chrom_id); return(invisible(NULL))
  }

  # ── Proportional data ────────────────────────────────────────────────────
  data_prop <- data_raw %>%
    mutate(
      Total = rowSums(across(all_of(sp_names))),
      Total = if_else(Total == 0, 1, Total),
      across(all_of(sp_names), ~ .x / Total * 100, .names = "Prop_{.col}"),
      Start_Mb = Start / 1e6, End_Mb = End / 1e6
    ) %>%
    pivot_longer(starts_with("Prop_"), names_to = "Species", values_to = "Pct") %>%
    mutate(Species = factor(str_remove(Species, "^Prop_"), levels = sp_names)) %>%
    group_by(Start_Mb, Chr) %>% arrange(Species) %>%
    mutate(ymax = cumsum(Pct), ymin = ymax - Pct) %>% ungroup()

  # ── Dominant ancestry data ────────────────────────────────────────────────
  data_dom <- data_raw %>%
    mutate(
      Total = rowSums(across(all_of(sp_names))),
      Total = if_else(Total == 0, 1, Total),
      across(all_of(sp_names), ~ .x / Total, .names = "P_{.col}"),
      Start_Mb = Start / 1e6, End_Mb = End / 1e6
    )

  p_cols <- paste0("P_", sp_names)
  data_dom <- data_dom %>%
    rowwise() %>%
    mutate(
      max_p    = max(c_across(all_of(p_cols))),
      which_sp = if_else(max_p >= p$dominance_threshold,
                         sp_names[which.max(c_across(all_of(p_cols)))],
                         "Mixed")
    ) %>% ungroup() %>%
    mutate(Dominant_Status = factor(which_sp, levels = dominant_levels))

  # ── Summary stats per chromosome ─────────────────────────────────────────
  overall_props <- data_raw %>%
    group_by(Chr) %>%
    summarise(across(all_of(sp_names), sum), .groups = "drop") %>%
    rowwise() %>%
    mutate(
      total_k = sum(c_across(all_of(sp_names))),
      stats   = list(setNames(
        sprintf("%s: %.1f%%", sp_names,
                c_across(all_of(sp_names)) / total_k * 100),
        sp_names))
    ) %>% ungroup()

  overall_dom <- data_dom %>%
    mutate(window_len = End - Start) %>%
    group_by(Chr) %>%
    summarise(
      total_len = sum(window_len),
      across(all_of(sp_names),
             ~ sum(window_len[Dominant_Status == .y], na.rm = TRUE),
             .names = "len_{.col}"),
      len_mixed = sum(window_len[Dominant_Status == "Mixed"], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    rowwise() %>%
    mutate(
      dom_stats = list(c(
        setNames(sprintf("%s: %.1f%%", sp_names,
                         c_across(paste0("len_", sp_names)) / total_len * 100),
                 sp_names),
        Mixed = sprintf("Mixed: %.1f%%", len_mixed / total_len * 100)
      ))
    ) %>% ungroup()

  all_fragments <- sort(unique(data_prop$Chr))
  num_cols <- min(p$ncols, length(all_fragments))
  num_rows <- ceiling(length(all_fragments) / num_cols)

  # ── Plotting ──────────────────────────────────────────────────────────────
  generate_plot <- function(plot_type) {
    plot_list <- list()

    for (i in seq_along(all_fragments)) {
      f_name    <- all_fragments[i]
      is_bottom <- i > (length(all_fragments) - num_cols)
      is_left   <- (i %% num_cols == 1)

      if (plot_type == "proportional") {
        df_f  <- data_prop %>% filter(Chr == f_name)
        stats <- overall_props %>% filter(Chr == f_name) %>% pull(stats) %>% .[[1]]

        p_plot <- ggplot(df_f, aes(fill = Species)) +
          geom_rect(aes(xmin = Start_Mb, xmax = End_Mb,
                        ymin = ymin, ymax = ymax), color = NA) +
          geom_hline(yintercept = 50, linetype = "dashed",
                     color = "grey50", linewidth = 0.4) +
          scale_fill_manual(values = species_map, name = lbl_legend_prop,
                            labels = legend_labels_prop, breaks = sp_names) +
          scale_y_continuous(limits = c(-18, 158), breaks = c(0, 50, 100))

        x_mid <- mean(range(df_f$Start_Mb))
        y_pos <- seq(147 - 17 * (n_sp - 1), 147, by = 17)
        for (s_idx in seq_along(sp_names)) {
          p_plot <- p_plot +
            annotate("text", x = x_mid, y = y_pos[s_idx],
                     label = stats[sp_names[s_idx]],
                     hjust = 0.5, vjust = 1, size = 3.45,
                     color = species_map[sp_names[s_idx]], family = fam_title)
        }
        p_plot <- p_plot +
          annotate("text", x = x_mid, y = max(y_pos) + 14, label = f_name,
                   hjust = 0.5, vjust = 1, size = 4.6,
                   color = "black", family = fam_axis, fontface = "bold")

      } else {  # dominant
        df_d   <- data_dom %>% filter(Chr == f_name)
        d_stats <- overall_dom %>% filter(Chr == f_name) %>% pull(dom_stats) %>% .[[1]]

        p_plot <- ggplot(df_d, aes(fill = Dominant_Status)) +
          geom_rect(aes(xmin = Start_Mb, xmax = End_Mb, ymin = 0, ymax = 100), color = NA) +
          scale_fill_manual(values = dominant_map, name = lbl_legend_dom,
                            labels = legend_labels_dom, breaks = dominant_levels) +
          scale_y_continuous(limits = c(-18, 180), breaks = c(0, 100), labels = c("", ""))

        x_mid <- mean(range(df_d$Start_Mb))
        all_sp_mixed <- c(sp_names, "Mixed")
        y_dom <- seq(170 - 17 * length(all_sp_mixed), 170, by = 17)
        for (s_idx in seq_along(all_sp_mixed)) {
          nm  <- all_sp_mixed[s_idx]
          col <- if (nm == "Mixed") mixed_color else species_map[nm]
          p_plot <- p_plot +
            annotate("text", x = x_mid, y = y_dom[s_idx],
                     label = d_stats[nm], hjust = 0.5, vjust = 1,
                     size = 3.45, color = col, family = fam_title)
        }
        p_plot <- p_plot +
          annotate("text", x = x_mid, y = max(y_dom) + 14, label = f_name,
                   hjust = 0.5, vjust = 1, size = 4.6,
                   color = "black", family = fam_axis, fontface = "bold")
      }

      # ── Common theme ──────────────────────────────────────────────────────
      if (!is.null(block_boundaries)) {
        bb_f <- block_boundaries %>% filter(Chr == f_name)
        if (nrow(bb_f) > 0) {
          p_plot <- p_plot +
            geom_text_repel(data = bb_f,
                            aes(x = Position_Mb, y = 0, label = Position_Mb),
                            inherit.aes = FALSE, direction = "y", nudge_y = -8,
                            size = 2.5, family = fam_axis)
        }
      }

      p_plot <- p_plot +
        theme_classic() +
        theme(
          text         = element_text(family = fam_axis),
          axis.title.x = element_text(family = fam_title, size = 11),
          axis.title.y = element_text(family = fam_title, size = 11),
          axis.text    = element_text(family = fam_axis, color = "black", size = 9),
          legend.title = element_text(family = fam_title, size = 11, face = "bold"),
          legend.text  = element_text(family = fam_title, size = 10),
          panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.7)
        ) +
        labs(
          x = if (is_bottom) lbl_xaxis else NULL,
          y = if (is_left) (if (plot_type == "proportional") lbl_yaxis_prop
                            else lbl_yaxis_dom) else NULL
        )

      if (!is_bottom) p_plot <- p_plot +
        theme(axis.title.x = element_blank(), axis.text.x = element_blank())
      if (!is_left) p_plot <- p_plot +
        theme(axis.title.y = element_blank(), axis.text.y = element_blank())

      plot_list[[f_name]] <- p_plot
    }

    suffix <- if (plot_type == "proportional") title_suffix_prop else title_suffix_dom
    ttl    <- paste0(make_title(major_chrom_id), suffix)

    wrap_plots(plot_list, ncol = num_cols) +
      plot_layout(guides = "collect") +
      plot_annotation(title = ttl) &
      theme(
        legend.position = "bottom",
        plot.title = element_text(family = fam_title, size = 18,
                                  face = "bold", hjust = 0.5)
      )
  }

  # ── Save ──────────────────────────────────────────────────────────────────
  h_prop <- 4.5 * num_rows + 1.8
  h_dom  <- 4.5 * num_rows + 2.4

  out_prop <- path(p$output_dir, paste0(major_chrom_id, "_Proportional.pdf"))
  out_dom  <- path(p$output_dir, paste0(major_chrom_id, "_Dominant.pdf"))

  ggsave(out_prop, generate_plot("proportional"),
         width = 16, height = h_prop, device = cairo_pdf, dpi = 300)
  message("  Saved: ", basename(out_prop))

  ggsave(out_dom,  generate_plot("dominant"),
         width = 16, height = h_dom,  device = cairo_pdf, dpi = 300)
  message("  Saved: ", basename(out_dom))
})

if (use_showtext) showtext_auto(FALSE)
message("\n=== vis_supervised.R complete. Output in: ", p$output_dir, " ===")
