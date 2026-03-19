#!/usr/bin/env Rscript
# =============================================================================
# vis_karyotype.R  —  KmerGenoPhaser Karyotype Visualization Module
# =============================================================================
# Reads per-chromosome block .txt files (output of unsupervised module or
# assign_nodata_bloodline.py) and draws idiogram-style chromosome karyotypes,
# one PDF per homologous group.
#
# Usage:
#   Rscript vis_karyotype.R \
#       --input_dir      /path/to/block_txt_dir \
#       --output_dir     /path/to/output \
#       --genome_title   "MySpecies" \
#       [--centromere_file  /path/to/centromeres.csv] \
#       [--bloodline_colors "GroupA=#3C5488,GroupB=#E64B35,NoData=#fbb4ae"] \
#       [--chrom_pattern    "Chr[0-9]+"] \
#       [--group_pattern    "Chr[0-9]+"] \
#       [--inferred_alpha   0.55]
#
# Input block .txt format (tab-separated, with header):
#   Start   End   Bloodline   [Chrom]
#   0       1000000   GroupA
#   1000000 2000000   NoData8(GroupB)
#   ...
#   (Chrom column optional — inferred from filename if absent)
#
# Centromere CSV format (optional):
#   Chrom,Centromere_Start_Mb,Centromere_End_Mb
#   Chr1A,45.2,48.1
#   ...
# =============================================================================
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(stringr)
})

# =============================================================================
# 0) Argument parsing
# =============================================================================
parse_args <- function(args) {
  defaults <- list(
    input_dir       = NULL,
    output_dir      = NULL,
    genome_title    = "Target",
    centromere_file = NULL,       # optional
    bloodline_colors = NULL,      # "Name1=#hex1,Name2=#hex2,..."
    chrom_pattern   = "Chr[0-9]+[A-Za-z]?",  # matches individual chroms
    group_pattern   = "Chr[0-9]+",            # matches the homologous group
    inferred_alpha  = "0.55"      # lightness multiplier for Inferred_ colors
  )
  i <- 1L
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    val <- if (i + 1L <= length(args)) args[i + 1L] else NA_character_
    if (key %in% names(defaults)) { defaults[[key]] <- val; i <- i + 2L }
    else i <- i + 1L
  }
  defaults$inferred_alpha <- as.numeric(defaults$inferred_alpha)
  defaults
}

p <- parse_args(commandArgs(trailingOnly = TRUE))

# Usage / validation
if (is.null(p$input_dir) || is.null(p$output_dir)) {
  cat("
vis_karyotype.R — Idiogram-style karyotype visualization

Usage:
  Rscript vis_karyotype.R \\
      --input_dir      <path>   Block .txt directory (required)
      --output_dir     <path>   Output directory (required)
      --genome_title   <str>    Title prefix in plots  [default: Target]
      --centromere_file <path>  Optional centromere CSV
      --bloodline_colors <str>  'Name=#hex,...' comma-separated
      --chrom_pattern  <regex>  Regex for individual chrom names  [Chr[0-9]+[A-Za-z]?]
      --group_pattern  <regex>  Regex for homologous group extraction  [Chr[0-9]+]
      --inferred_alpha <float>  Lightness for Inferred_ colors  [0.55]

Input .txt format (tab-separated, header required):
  Start   End   Bloodline   [optional: Chrom]

Centromere CSV format:
  Chrom,Centromere_Start_Mb,Centromere_End_Mb
")
  quit(status = 0)
}

dir.create(p$output_dir, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# 1) Build color palette
# =============================================================================
# Default fallback palette (NPG-style)
default_npg <- c(
  "#3C5488","#E64B35","#00A087","#4DBBD5","#F39B7F",
  "#8491B4","#91D1C2","#DC0000","#7E6148","#B09C85"
)
nodata_color   <- "#fbb4ae"
complex_color  <- "#7E57C2"
lowinfo_color  <- "#aaaaaa"

# Parse user-supplied colors: "GroupA=#3C5488,GroupB=#E64B35"
user_colors <- list()
if (!is.null(p$bloodline_colors) && !is.na(p$bloodline_colors)) {
  pairs <- trimws(strsplit(p$bloodline_colors, ",")[[1]])
  for (pair in pairs) {
    kv <- strsplit(pair, "=")[[1]]
    if (length(kv) == 2) user_colors[[trimws(kv[1])]] <- trimws(kv[2])
  }
}

# Helper: make a lighter version of a hex color for Inferred_X
lighten_hex <- function(hex, alpha = 0.55) {
  rgb_vals <- col2rgb(hex) / 255
  light    <- rgb_vals + (1 - rgb_vals) * (1 - alpha)
  rgb(light[1], light[2], light[3])
}

# We build the palette lazily as we encounter bloodline names in the data
build_palette <- function(bloodline_names) {
  # Known status levels get fixed colors
  status_map <- c(
    NoData  = nodata_color,
    Complex = complex_color,
    LowInfo = lowinfo_color,
    Unknown = "#000000"
  )

  # Identify real ancestor names (not status, not Inferred_*)
  ancestor_names <- bloodline_names[
    !bloodline_names %in% names(status_map) &
    !grepl("^Inferred_", bloodline_names)
  ]
  ancestor_names <- unique(ancestor_names)

  # Assign colors to ancestors: user-supplied first, then NPG
  anc_colors <- character(0)
  npg_idx    <- 1L
  for (nm in ancestor_names) {
    if (!is.null(user_colors[[nm]])) {
      anc_colors[nm] <- user_colors[[nm]]
    } else {
      anc_colors[nm] <- default_npg[((npg_idx - 1L) %% length(default_npg)) + 1L]
      npg_idx <- npg_idx + 1L
    }
  }

  # Build Inferred_X entries as lightened versions
  inferred_names <- bloodline_names[grepl("^Inferred_", bloodline_names)]
  inf_colors <- character(0)
  for (nm in unique(inferred_names)) {
    base_nm <- sub("^Inferred_", "", nm)
    base_col <- if (!is.null(user_colors[[nm]])) {
      user_colors[[nm]]
    } else if (base_nm %in% names(anc_colors)) {
      lighten_hex(anc_colors[[base_nm]], p$inferred_alpha)
    } else {
      lighten_hex(nodata_color, p$inferred_alpha)
    }
    inf_colors[nm] <- base_col
  }

  c(status_map, anc_colors, inf_colors)
}

# =============================================================================
# 2) Read and pre-process block files
# =============================================================================
txt_files <- list.files(p$input_dir, pattern = "\\.txt$", full.names = TRUE)
if (length(txt_files) == 0)
  stop("No .txt files found in: ", p$input_dir)

message("Reading ", length(txt_files), " block file(s)...")

raw_data <- bind_rows(lapply(txt_files, function(f) {
  d <- tryCatch(
    read.table(f, header = TRUE, sep = "\t", fill = TRUE,
               stringsAsFactors = FALSE, comment.char = "#"),
    error = function(e) {
      message("[WARN] Could not read ", basename(f), ": ", e$message)
      return(NULL)
    }
  )
  if (is.null(d) || nrow(d) == 0) return(NULL)

  # Normalise column names (case-insensitive)
  cn_low <- tolower(colnames(d))
  if ("start" %in% cn_low) colnames(d)[cn_low == "start"] <- "Start"
  if ("end"   %in% cn_low) colnames(d)[cn_low == "end"]   <- "End"
  if ("bloodline" %in% cn_low) colnames(d)[cn_low == "bloodline"] <- "Bloodline"

  # Convert to Mb (assume bp if values > 1e4)
  if ("Start" %in% colnames(d)) {
    vals <- suppressWarnings(as.numeric(d$Start))
    if (any(!is.na(vals) & vals > 1e4))
      d$Start_Mb <- vals / 1e6
    else
      d$Start_Mb <- vals
  }
  if ("End" %in% colnames(d)) {
    vals <- suppressWarnings(as.numeric(d$End))
    if (any(!is.na(vals) & vals > 1e4))
      d$End_Mb <- vals / 1e6
    else
      d$End_Mb <- vals
  }

  # Infer Chrom from filename if not in columns
  if (!"Chrom" %in% colnames(d)) {
    chrom_name <- tools::file_path_sans_ext(basename(f))
    chrom_name <- gsub("_blocks.*|_Final.*", "", chrom_name)
    d$Chrom <- chrom_name
  }
  d
}))

if (is.null(raw_data) || nrow(raw_data) == 0)
  stop("No valid data loaded from: ", p$input_dir)

# =============================================================================
# 3) Standardise bloodline labels
# =============================================================================
# Parse labels like "NoData3(GroupB)" → "Inferred_GroupB"
#                   "Spontaneum1"     → "Spontaneum"
standardise_bloodline <- function(label) {
  label <- trimws(as.character(label))

  # "NoData*(SomeName)" → "Inferred_SomeName"
  m <- regmatches(label, regexpr("(?<=\\().*(?=\\))", label, perl = TRUE))
  if (length(m) == 1 && nchar(m) > 0 &&
      grepl("NoData", label, ignore.case = TRUE))
    return(paste0("Inferred_", m))

  # "NoData" variants with no parenthetical
  if (grepl("NoData", label, ignore.case = TRUE)) return("NoData")

  # Strip trailing digits: "Spontaneum1" → "Spontaneum"
  label <- sub("[0-9]+$", "", label)
  label
}

raw_data <- raw_data %>%
  mutate(
    Bloodline = sapply(Bloodline, standardise_bloodline),
    Group     = str_extract(Chrom, p$group_pattern)
  )

# Build color palette from all observed bloodline names
all_bloodlines <- unique(raw_data$Bloodline)
color_pal      <- build_palette(all_bloodlines)

all_groups <- str_sort(unique(raw_data$Group[!is.na(raw_data$Group)]),
                       numeric = TRUE)
if (length(all_groups) == 0)
  stop("No chromosome groups found. Check --group_pattern: '", p$group_pattern, "'")
message("Chromosome groups: ", paste(all_groups, collapse = ", "))

# =============================================================================
# 4) Centromere data (optional)
# =============================================================================
centromere_df <- NULL
if (!is.null(p$centromere_file) && !is.na(p$centromere_file) &&
    file.exists(p$centromere_file)) {
  centromere_df <- tryCatch(
    read.csv(p$centromere_file, stringsAsFactors = FALSE),
    error = function(e) {
      message("[WARN] Could not read centromere file: ", e$message)
      NULL
    }
  )
  if (!is.null(centromere_df))
    message("Centromere data loaded: ", nrow(centromere_df), " entries")
}

# =============================================================================
# 5) Chromosome outline helper
# =============================================================================
chromosome_outline_df <- function(x_center, y_max, width,
                                  centromere_pos, n_points = 200) {
  y       <- seq(0, y_max, length.out = n_points)
  cap_frac <- 0.03    # rounded cap fraction of total length
  neck_frac <- 0.08   # constriction half-width around centromere

  x_hw <- sapply(y, function(pos) {
    if (pos < y_max * cap_frac) {
      w <- (width / 2) * sqrt(pos / (y_max * cap_frac))
    } else if (pos > y_max * (1 - cap_frac)) {
      w <- (width / 2) * sqrt((y_max - pos) / (y_max * cap_frac))
    } else {
      dist <- abs(pos - centromere_pos)
      if (dist < y_max * neck_frac) {
        constrict <- exp(-dist^2 / (y_max * 0.02)^2)
        w <- (width / 2) * (1 - 0.35 * constrict)
      } else {
        w <- width / 2
      }
    }
    w
  })
  data.frame(
    x = c(x_center + x_hw, rev(x_center - x_hw)),
    y = c(y, rev(y))
  )
}

# =============================================================================
# 6) Block polygon helper (follows chromosome outline shape)
# =============================================================================
block_polygon_df <- function(x_center, y_start, y_end, y_max,
                             centromere_pos, width, n_points = 30) {
  y_start <- max(0, y_start)
  y_end   <- min(y_max, y_end)
  if (y_end <= y_start) return(NULL)

  cap_frac  <- 0.03
  neck_frac <- 0.08
  n <- max(n_points, ceiling((y_end - y_start) * 10))
  ys <- seq(y_start, y_end, length.out = n)

  x_hw <- sapply(ys, function(pos) {
    if (pos < y_max * cap_frac) {
      w <- (width / 2) * sqrt(pos / (y_max * cap_frac))
    } else if (pos > y_max * (1 - cap_frac)) {
      w <- (width / 2) * sqrt((y_max - pos) / (y_max * cap_frac))
    } else {
      dist <- abs(pos - centromere_pos)
      if (dist < y_max * neck_frac) {
        constrict <- exp(-dist^2 / (y_max * 0.02)^2)
        w <- (width / 2) * (1 - 0.35 * constrict)
      } else {
        w <- width / 2
      }
    }
    w
  })

  data.frame(
    x = c(x_center + x_hw, rev(x_center - x_hw)),
    y = c(ys, rev(ys))
  )
}

# =============================================================================
# 7) Main loop — one PDF per homologous group
# =============================================================================
chrom_width <- 0.5

for (curr_group in all_groups) {
  message("\n--- Processing: ", curr_group, " ---")

  group_data   <- raw_data %>% filter(Group == curr_group)
  uniq_chroms  <- str_sort(unique(group_data$Chrom), numeric = TRUE)
  group_data$Chrom <- factor(group_data$Chrom, levels = uniq_chroms)
  group_data$x_pos <- as.integer(group_data$Chrom)

  chrom_info <- group_data %>%
    group_by(Chrom, x_pos) %>%
    summarise(Max_Mb = max(End_Mb, na.rm = TRUE), .groups = "drop")

  # Centromere positions
  if (!is.null(centromere_df)) {
    chrom_info <- chrom_info %>%
      left_join(centromere_df, by = "Chrom") %>%
      mutate(
        Centromere_Start = coalesce(Centromere_Start_Mb, Max_Mb * 0.48),
        Centromere_End   = coalesce(Centromere_End_Mb,   Max_Mb * 0.52),
        Centromere_Pos   = (Centromere_Start + Centromere_End) / 2
      )
  } else {
    chrom_info <- chrom_info %>%
      mutate(
        Centromere_Pos   = Max_Mb / 2,
        Centromere_Start = Max_Mb * 0.48,
        Centromere_End   = Max_Mb * 0.52
      )
  }

  max_length <- max(chrom_info$Max_Mb)

  # Build outline polygons
  all_outlines <- bind_rows(lapply(seq_len(nrow(chrom_info)), function(i) {
    ci <- chrom_info[i, ]
    df <- chromosome_outline_df(ci$x_pos, ci$Max_Mb,
                                chrom_width, ci$Centromere_Pos)
    df$Chrom <- ci$Chrom
    df
  }))

  # Build block polygons
  all_blocks <- bind_rows(lapply(seq_len(nrow(group_data)), function(i) {
    row <- group_data[i, ]
    ci  <- chrom_info[chrom_info$Chrom == row$Chrom, ]
    df  <- block_polygon_df(row$x_pos, row$Start_Mb, row$End_Mb,
                            ci$Max_Mb, ci$Centromere_Pos, chrom_width)
    if (is.null(df)) return(NULL)
    df$Bloodline <- as.character(row$Bloodline)
    df$block_id  <- paste(row$Chrom, i, sep = "_")
    df
  }))

  if (is.null(all_blocks) || nrow(all_blocks) == 0) {
    message("[WARN] No block polygons for ", curr_group, " — skipping.")
    next
  }

  # Only keep colors for bloodlines actually present in this group
  present_bl <- unique(all_blocks$Bloodline)
  pal_here   <- color_pal[names(color_pal) %in% present_bl]
  # Any bloodline without a color gets gray
  missing    <- setdiff(present_bl, names(pal_here))
  if (length(missing) > 0) {
    extra <- setNames(rep("#cccccc", length(missing)), missing)
    pal_here <- c(pal_here, extra)
  }

  # Plot
  plot_w <- length(uniq_chroms) * 0.6 + 2.5

  plt <- ggplot() +
    geom_polygon(data = all_blocks,
                 aes(x = x, y = y, fill = Bloodline, group = block_id),
                 color = NA) +
    geom_polygon(data = all_outlines,
                 aes(x = x, y = y, group = Chrom),
                 fill = NA, color = "black", linewidth = 0.3) +
    scale_fill_manual(
      values = pal_here,
      name   = "Ancestry (solid = raw, lighter = inferred)",
      breaks = names(pal_here)
    ) +
    scale_x_continuous(
      breaks   = seq_along(uniq_chroms),
      labels   = uniq_chroms,
      expand   = expansion(add = 0.8)
    ) +
    scale_y_reverse(
      breaks = seq(0, ceiling(max_length / 20) * 20, by = 20),
      expand = expansion(mult = c(0.02, 0.02))
    ) +
    labs(
      title    = paste0(p$genome_title, " — Chromosome Ancestry Karyotype"),
      subtitle = paste0(curr_group, " Homologous Group"),
      x = NULL,
      y = "Physical Position (Mb)"
    ) +
    theme_minimal(base_family = "serif") +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
      axis.text.x  = element_text(size = 12, face = "bold", angle = 45,
                                  vjust = 1, hjust = 1, color = "black"),
      axis.text.y  = element_text(size = 10, color = "black"),
      axis.title.y = element_text(size = 12, face = "bold",
                                  margin = margin(r = 10)),
      legend.position = "bottom",
      legend.text  = element_text(size = 10),
      legend.title = element_text(size = 10, face = "bold"),
      plot.title   = element_text(hjust = 0.5, size = 16, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 12, color = "grey40"),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA)
    )

  out_pdf <- file.path(p$output_dir,
                       paste0(curr_group, "_Karyotype.pdf"))
  ggsave(out_pdf, plt,
         width = plot_w, height = 10, units = "in",
         dpi = 600, device = cairo_pdf)
  message("  Saved: ", basename(out_pdf))
}

message("\n=== vis_karyotype.R complete. Output in: ", p$output_dir, " ===")
