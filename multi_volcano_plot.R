# Multi-group Differential Volcano Plot
# 左：非共生（M）
# 右：共生（MS）
# 时间：10, 20, 30, 40, 50 vs 5

# =========================
# Load required packages
# =========================
required_pkgs <- c("ggplot2", "dplyr", "patchwork")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(ggplot2)
library(dplyr)
library(patchwork)

# =========================
# Helper: get script directory
# =========================
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  return(getwd())
}

base_dir <- get_script_dir()

# =========================
# Read and combine DEG files
# =========================
read_deg_files <- function(file_list, group_labels) {
  all_data <- data.frame()

  for (i in seq_along(file_list)) {
    file_path <- file_list[i]

    if (!file.exists(file_path)) {
      stop(paste("File not found:", file_path))
    }

    df <- read.csv(file_path, header = TRUE, check.names = FALSE)

    if (ncol(df) < 9) {
      stop(paste("File has fewer than 9 columns:", file_path))
    }

    # Standardize first 9 columns
    colnames(df)[1:9] <- c(
      "GeneID", "GeneName", "GeneDesc", "FC", "Log2FC",
      "Pvalue", "Padjust", "Significant", "Regulate"
    )

    df$Log2FC <- suppressWarnings(as.numeric(df$Log2FC))
    df$Pvalue <- suppressWarnings(as.numeric(df$Pvalue))
    df$Padjust <- suppressWarnings(as.numeric(df$Padjust))

    df <- df %>%
      filter(!is.na(Log2FC), !is.na(Pvalue), Pvalue > 0) %>%
      mutate(
        Group = group_labels[i],
        neg_log10_pvalue = -log10(Pvalue),
        Direction = case_when(
          tolower(Regulate) == "up" ~ "Up",
          tolower(Regulate) == "down" ~ "Down",
          TRUE ~ "Other"
        )
      ) %>%
      select(
        GeneID, GeneName, Log2FC, Pvalue, Padjust,
        neg_log10_pvalue, Significant, Regulate, Direction, Group
      )

    all_data <- bind_rows(all_data, df)
  }

  all_data$Group <- factor(all_data$Group, levels = group_labels)
  return(all_data)
}

# =========================
# Create one multi-group volcano plot
# =========================
create_multi_volcano <- function(data, title, output_file, group_colors = NULL) {
  groups <- levels(data$Group)
  n_groups <- length(groups)

  if (is.null(group_colors)) {
    group_colors <- c("#7B68EE", "#F0C040", "#2E8B8B", "#66CDAA", "#FF7F50")[1:n_groups]
  }
  names(group_colors) <- groups

  set.seed(42)
  data <- data %>%
    mutate(
      group_num = as.numeric(Group),
      x_jitter = group_num + runif(n(), -0.32, 0.32),
      neg_log10_pvalue = pmin(neg_log10_pvalue, 50)
    )

  label_data <- data.frame(
    Group = groups,
    group_num = 1:n_groups,
    color = group_colors[groups],
    stringsAsFactors = FALSE
  )

  p <- ggplot(data, aes(x = x_jitter, y = Log2FC)) +
    annotate(
      "rect",
      xmin = seq(0.5, n_groups - 0.5, by = 2),
      xmax = seq(1.5, n_groups + 0.5, by = 2),
      ymin = -Inf, ymax = Inf,
      fill = "grey95", alpha = 0.8
    ) +
    geom_rect(
      data = label_data,
      aes(
        xmin = group_num - 0.38,
        xmax = group_num + 0.38,
        ymin = -1.2,
        ymax = 1.2
      ),
      fill = label_data$color,
      color = NA,
      inherit.aes = FALSE
    ) +
    geom_text(
      data = label_data,
      aes(x = group_num, y = 0, label = Group),
      color = "white",
      fontface = "bold",
      size = 3.2,
      inherit.aes = FALSE
    ) +
    geom_point(
      aes(color = Direction, size = neg_log10_pvalue),
      alpha = 0.75,
      shape = 16
    ) +
    scale_color_manual(
      values = c(
        "Up" = "#F5A623",
        "Down" = "#4FC3F7",
        "Other" = "grey70"
      )
    ) +
    scale_x_continuous(
      breaks = 1:n_groups,
      labels = groups,
      expand = c(0.04, 0.04)
    ) +
    scale_size_continuous(
      name = "-log10(Pvalue)",
      range = c(0.8, 3.5),
      breaks = c(1, 5, 10, 20, 50)
    ) +
    labs(
      x = NULL,
      y = "log2FoldChange",
      title = title,
      color = "Direction"
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(size = 9, face = "bold"),
      axis.text.y = element_text(size = 10),
      axis.title.y = element_text(size = 12),
      legend.position = "right",
      panel.border = element_rect(color = "grey80"),
      strip.background = element_blank()
    )

  ggsave(output_file, plot = p, width = 10, height = 7, dpi = 300)
  cat("Saved:", output_file, "\n")
  return(p)
}

# =========================
# Input files
# =========================
m_files <- file.path(base_dir, c(
  "M10VSM5.csv", "M20VSM5.csv", "M30VSM5.csv", "M40VSM5.csv", "M50VSM5.csv"
))
m_labels <- c("M10_vs_M5", "M20_vs_M5", "M30_vs_M5", "M40_vs_M5", "M50_vs_M5")

ms_files <- file.path(base_dir, c(
  "MS10VSM5.csv", "MS20VSM5.csv", "MS30VSM5.csv", "MS40VSM5.csv", "MS50VSM5.csv"
))
ms_labels <- c("MS10_vs_M5", "MS20_vs_M5", "MS30_vs_M5", "MS40_vs_M5", "MS50_vs_M5")

# =========================
# Read data
# =========================
cat("Reading non-symbiotic data...\n")
m_data <- read_deg_files(m_files, m_labels)
cat("Rows:", nrow(m_data), "\n")

cat("Reading symbiotic data...\n")
ms_data <- read_deg_files(ms_files, ms_labels)
cat("Rows:", nrow(ms_data), "\n")

# =========================
# Create individual plots
# =========================
m_colors <- c("#7B68EE", "#F0C040", "#2E8B8B", "#66CDAA", "#FF7F50")
ms_colors <- c("#7B68EE", "#F0C040", "#2E8B8B", "#66CDAA", "#FF7F50")

p1 <- create_multi_volcano(
  data = m_data,
  title = "非共生差异基因火山图 (Non-symbiotic DEGs)",
  output_file = file.path(base_dir, "volcano_non_symbiotic.png"),
  group_colors = m_colors
)

p2 <- create_multi_volcano(
  data = ms_data,
  title = "共生差异基因火山图 (Symbiotic DEGs)",
  output_file = file.path(base_dir, "volcano_symbiotic.png"),
  group_colors = ms_colors
)

# =========================
# Combine left-right
# =========================
combined_plot <- p1 + p2 +
  plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "right")

combined_plot <- combined_plot +
  plot_annotation(
    title = "多组差异火山图（10–50）",
    subtitle = "左：非共生    右：共生",
    theme = theme(
      plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 12)
    )
  )

ggsave(
  filename = file.path(base_dir, "volcano_combined_non_sym_vs_sym_10_50.png"),
  plot = combined_plot,
  width = 18,
  height = 7,
  dpi = 300
)

cat("\nDone! Generated files:\n")
cat("1. volcano_non_symbiotic.png\n")
cat("2. volcano_symbiotic.png\n")
cat("3. volcano_combined_non_sym_vs_sym_10_50.png\n")
