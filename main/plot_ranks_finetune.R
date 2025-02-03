rm(list = ls())
set.seed(7)

library(dplyr)
library(ggplot2)
library(tidyr)

get_task_rank <- function(path2csv){
  # process the csv table
  tab <- read.csv(path2csv)
  rownames(tab) <- tab[, 1]
  tab <- tab[, -1]
  tab <- tab[, c(1, 3, 5, 7)] # choose the columns of mean values
  
  # Calculate ranks for each column
  ranks <- as.data.frame(apply(tab, 2, function(x) rank(-x)))
  colnames(ranks) <- c("TF_range", 
                       "Dosage_sensitivity",
                       "Bivalent_vs_Lys4",
                       "Bivalent_vs_NonMethyl")
  return(ranks)
}
rank_tab <- get_task_rank("./res/2024_1225/csv/MultiMod_test_811_Finetuning.csv")

rank_tab$overall_rank <- rowSums(rank_tab)/ncol(rank_tab)
rank_tab$Method <- rownames(rank_tab)

# Reorder Method based on OverallRank
rank_tab <- rank_tab %>%
  arrange(overall_rank) %>%
  mutate(Method = factor(Method, levels = Method))

rank_tab$Method <- factor(rank_tab$Method, levels = rev(levels(rank_tab$Method)))

rank_tab <- rank_tab[, !(names(rank_tab) == "overall_rank")]

data_long <- rank_tab %>%
  pivot_longer(cols = -Method, names_to = "Category", values_to = "Rank")

# Add bar length based on rank
data_long <- data_long %>%
  mutate(BarWidth = Rank / max(Rank)) # Normalize rank for bar width (scaled 0-1)

data_long$Category <- factor(data_long$Category, 
                             levels = c("TF_range",
                                        "Dosage_sensitivity",
                                        "Bivalent_vs_Lys4",
                                        "Bivalent_vs_NonMethyl"))


# Plot with uniform tiles and horizontal histograms
pdf("./res/2025_0115/finetune_ranks.pdf", height = 5, width = 8)
ggplot(data_long, aes(x = Category, y = Method)) +
  # Uniform light gray tiles
  geom_tile(fill = "white", color = "black") +
  # Add horizontal histograms inside each tile with rank-based gradient
  geom_rect(aes(
    xmin = as.numeric(factor(Category)) - 0.5,
    xmax = as.numeric(factor(Category)) - 0.55 + BarWidth,
    ymin = as.numeric(factor(Method)) - 0.4,
    ymax = as.numeric(factor(Method)) + 0.4,
    fill = Rank
  )) +
  # Color gradient for the histogram bars
  scale_fill_gradient(low = "darkred", high = "lightcoral", name = "Rank") +
  # Formatting
  labs(x = NULL, y = NULL) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 20, hjust = 1, vjust = 1.1),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    legend.position = "right"
  )
dev.off()





