# Updated multi_volcano_plot.R

# Install required packages
if(!requireNamespace("patchwork", quietly=TRUE)) {
    install.packages("patchwork")
}

# Define the function create_multi_volcano
create_multi_volcano <- function(m_data, ms_data, return_plot_only=FALSE, output_file="volcano_combined_non_sym_vs_sym_10_50.png") {
    # ... (existing code for individual plots p1 and p2)
    
    # Create combined plot
    combined_plot <- p1 | p2 + plot_layout(ncol = 2) + plot_annotation(title = "Volcano Plots")
    
    # Save individual plots as before
    ggsave("non_sym_volcano_plot.png", plot = p1, width = 10, height = 7, dpi = 300)
    ggsave("sym_volcano_plot.png", plot = p2, width = 10, height = 7, dpi = 300)
    
    # Save the combined plot if not returning only the plot
    if (!return_plot_only) {
        ggsave(output_file, plot = combined_plot, width = 20, height = 7, dpi = 300)
    }
    
    # Return the combined plot if return_plot_only is TRUE
    if (return_plot_only) {
        return(combined_plot)
    }
}