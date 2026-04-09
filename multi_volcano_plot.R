# Multi-group Differential Volcano Plot
# 共生 (Symbiotic) and 非共生 (Non-symbiotic) groups
# Comparing 10-50 days vs 5 days

# Load required libraries
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2", repos = "https://cloud.r-project.org")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr", repos = "https://cloud.r-project.org")
if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr", repos = "https://cloud.r-project.org")

library(ggplot2)
library(dplyr)
library(tidyr)

# ============================================================
# Function: Read and combine differential expression data
# ============================================================
read_deg_files <- function(file_list, group_labels) {
  all_data <- data.frame()
  for (i in seq_along(file_list)) {
    df <- read.csv(file_list[i], header = TRUE, check.names = FALSE)
    # Standardize column names
    colnames(df)[1:9] <- c("GeneID", "GeneName", "GeneDesc", "FC", "Log2FC",
                           "Pvalue", "Padjust", "Significant", "Regulate")
    df <- df %>%
      filter(Significant == "yes") %>%
      mutate(
        Group = group_labels[i],
        Log2FC = as.numeric(Log2FC),
        Pvalue = as.numeric(Pvalue),
        Padjust = as.numeric(Padjust),
        neg_log10_pvalue = -log10(Pvalue),
        Direction = ifelse(Regulate == "up", "Up", "Down")
      ) %>%
      select(GeneID, GeneName, Log2FC, Pvalue, Padjust, neg_log10_pvalue,
             Significant, Regulate, Direction, Group)
    all_data <- rbind(all_data, df)
  }
  # Set group as ordered factor
  all_data$Group <- factor(all_data$Group, levels = group_labels)
  return(all_data)
}

# ============================================================
# Function: Create multi-group volcano plot (matching reference style)
# ============================================================
create_multi_volcano <- function(data, title, output_file,
                                 group_colors = NULL) {
  groups <- levels(data$Group)
  n_groups <- length(groups)

  if (is.null(group_colors)) {
    group_colors <- c("#7B68EE", "#F0C040", "#2E8B8B", "#66CDAA", "#FF7F50")[1:n_groups]
  }
  names(group_colors) <- groups

  # Assign numeric x positions per group and jitter within
  set.seed(42)
  data <- data %>%
    mutate(
      group_num = as.numeric(Group),
      x_jitter = group_num + runif(n(), -0.35, 0.35)
    )

  # Cap extreme values for display
  data <- data %>%
    mutate(
      neg_log10_pvalue = pmin(neg_log10_pvalue, 50)
    )

  # Size mapping based on -log10(pvalue)
  data <- data %>%
    mutate(
      size_cat = cut(neg_log10_pvalue,
                     breaks = c(-Inf, 1, 5, 10, Inf),
                     labels = c("1", "5", "10", "top"),
                     right = FALSE)
    )

  # Create label data for middle rectangles
  label_data <- data.frame(
    Group = groups,
    group_num = 1:n_groups,
    color = group_colors[groups],
    stringsAsFactors = FALSE
  )

  # Build the plot
  p <- ggplot(data, aes(x = x_jitter, y = Log2FC)) +

    # Background panels (light gray alternating)
    annotate("rect",
             xmin = seq(0.5, n_groups - 0.5, by = 2),
             xmax = seq(1.5, n_groups + 0.5, by = 2),
             ymin = -Inf, ymax = Inf,
             fill = "grey90", alpha = 0.5) +

    # Middle label rectangles
    geom_rect(data = label_data,
              aes(xmin = group_num - 0.4, xmax = group_num + 0.4,
                  ymin = -1.2, ymax = 1.2),
              fill = label_data$color, color = NA, inherit.aes = FALSE) +

    # Group name labels
    geom_text(data = label_data,
              aes(x = group_num, y = 0, label = Group),
              color = "white", fontface = "bold", size = 3.5,
              inherit.aes = FALSE) +

    # All genes with color mapped to Direction
    geom_point(aes(color = Direction, size = neg_log10_pvalue),
               alpha = 0.8, shape = 15) +

    scale_color_manual(
      name = "Direction",
      values = c("Up" = "#F5A623", "Down" = "#4FC3F7")
    ) +

    # Scale and theme
    scale_x_continuous(breaks = 1:n_groups, labels = groups, expand = c(0.05, 0.05)) +
    scale_size_continuous(
      name = "-log10(Pvalue)",
      range = c(1, 4),
      breaks = c(1, 5, 10),
      labels = c("1", "5", "10")
    ) +

    labs(
      x = NULL,
      y = "log2FoldChange",
      title = title
    ) +

    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.minor.y = element_blank(),
      panel.border = element_rect(color = "grey80"),
      axis.text.x = element_text(size = 10, face = "bold"),
      axis.text.y = element_text(size = 10),
      axis.title.y = element_text(size = 12),
      legend.position = "right",
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white")
    )

  ggsave(output_file, plot = p, width = 10, height = 7, dpi = 300)
  cat("Saved:", output_file, "\n")
  return(p)
}

# ============================================================
# Main: Process non-symbiotic (非共生, M) files
# ============================================================
# Determine the script's directory for portable file paths
base_dir <- tryCatch({
  normalizePath(dirname(sys.frame(1)$ofile))
}, error = function(e) {
  getwd()
})

m_files <- file.path(base_dir, c("M10VSM5.csv", "M20VSM5.csv", "M30VSM5.csv",
                                  "M40VSM5.csv", "M50VSM5.csv"))
m_labels <- c("M10_vs_M5", "M20_vs_M5", "M30_vs_M5", "M40_vs_M5", "M50_vs_M5")

ms_files <- file.path(base_dir, c("MS10VSM5.csv", "MS20VSM5.csv", "MS30VSM5.csv",
                                   "MS40VSM5.csv", "MS50VSM5.csv"))
ms_labels <- c("MS10_vs_M5", "MS20_vs_M5", "MS30_vs_M5", "MS40_vs_M5", "MS50_vs_M5")

# Read data
cat("Reading non-symbiotic (非共生) data...\n")
m_data <- read_deg_files(m_files, m_labels)
cat("  Total significant genes:", nrow(m_data), "\n")
cat("  Up:", sum(m_data$Direction == "Up"), " Down:", sum(m_data$Direction == "Down"), "\n")

cat("Reading symbiotic (共生) data...\n")
ms_data <- read_deg_files(ms_files, ms_labels)
cat("  Total significant genes:", nrow(ms_data), "\n")
cat("  Up:", sum(ms_data$Direction == "Up"), " Down:", sum(ms_data$Direction == "Down"), "\n")

# Create plots
cat("\nGenerating plots...\n")

# Non-symbiotic plot (非共生)
m_colors <- c("#7B68EE", "#F0C040", "#2E8B8B", "#66CDAA", "#FF7F50")
p1 <- create_multi_volcano(
  m_data,
  title = "非共生差异基因火山图 (Non-symbiotic DEGs)",
  output_file = file.path(base_dir, "volcano_non_symbiotic.png"),
  group_colors = m_colors
)

# Symbiotic plot (共生)
ms_colors <- c("#7B68EE", "#F0C040", "#2E8B8B", "#66CDAA", "#FF7F50")
p2 <- create_multi_volcano(
  ms_data,
  title = "共生差异基因火山图 (Symbiotic DEGs)",
  output_file = file.path(base_dir, "volcano_symbiotic.png"),
  group_colors = ms_colors
)

cat("\nDone! Two plots generated:\n")
cat("  1. volcano_non_symbiotic.png (非共生)\n")
cat("  2. volcano_symbiotic.png (共生)\n")
