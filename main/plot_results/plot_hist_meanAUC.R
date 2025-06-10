rm(list = ls())
set.seed(7)

library(dplyr)
library(tidyr)
library(ggplot2)

embed_levels <- c(# "OpenAI", 
                  "GIST-small-Embedding-v0", "NoInstruct-small-Embedding-v0", 
                  "stella-base-en-v2", "e5-small-v2", "GIST-all-MiniLM-L6-v2", 
                  "gte-small", "bge-small-en-v1.5", "MedEmbed-small-v0.1", 
                  "gte-tiny", "e5-small")

# # Manually define the table
# df_LR_NoCV <- data.frame(
#   Embedding = embed_levels,
#   MeanAUC = c(0.845, 0.846, 0.870, 0.862, 0.852, 0.837, 0.831, 0.836, 0.835, 0.863, 0.864),
#   Rank = c(7, 6, 1, 4, 5, 8, 11, 9, 10, 3, 2)
# )
# 
# df_RF_NoCV <- data.frame(
#   Embedding = embed_levels,
#   MeanAUC = c(
#     0.840, 0.839, 0.848,
#     0.880, 0.857, 0.826,
#     0.819, 0.852, 0.848,
#     0.860, 0.861
#   ),
#   Rank = c(
#     8, 9, 6.5,
#     1, 4, 10,
#     11, 5, 6.5,
#     3, 2
#   )
# )
# 
# df_LR_CV <- data.frame(
#   Embedding = embed_levels,
#   MeanAUC = c(
#     0.843, 0.867, 0.872,
#     0.866, 0.860, 0.844,
#     0.855, 0.849, 0.855,
#     0.863, 0.876
#   ),
#   Rank = c(
#     11, 3, 2,
#     4, 6, 10,
#     7.5, 9, 7.5,
#     5, 1
#   )
# )
# 
# df_RF_CV <- data.frame(
#   Embedding = embed_levels,
#   MeanAUC = c(
#     0.839, 0.855, 0.829,
#     0.869, 0.862, 0.811,
#     0.844, 0.866, 0.849,
#     0.845, 0.859
#   ),
#   Rank = c(
#     9, 5, 10,
#     1, 3, 11,
#     8, 2, 6,
#     7, 4
#   )
# )

df_finetune <- data.frame(
  Embedding = c(
    "GIST-small-Embedding-v0",
    "NoInstruct-small-Embedding-v0",
    "stella-base-en-v2",
    "e5-small-v2",
    "GIST-all-MiniLM-L6-v2",
    "gte-small",
    "bge-small-en-v1.5",
    "MedEmbed-small-v0.1",
    "gte-tiny",
    "e5-small"
  ),
  MeanAUC = c(
    0.825,
    0.827,
    0.831,
    0.823,
    0.818,
    0.820,
    0.817,
    0.813,
    0.810,
    0.806
  ),
  Rank = c(
    3,
    2,
    1,
    4,
    6,
    5,
    7,
    8,
    9,
    10
  )
)

df <- df_finetune
df$Embedding <- factor(df$Embedding, levels = embed_levels)
df$Embedding <- factor(df$Embedding, levels = rev(levels(df$Embedding)))

pastel_colors <- c(
  "#AEC6CF", 
  "#FFB347", "#77DD77", "#CBAACB", "#FF6961", 
  "#FDFD96", "#84B6F4", "#FFD1DC", "#B39EB5", "#B0E0E6"# , "#F0E68C"
)

# Plot
p <- ggplot(df, aes(x = Embedding, y = MeanAUC, fill = Embedding)) +
  geom_col() +
  geom_text(aes(label = paste0(Rank)), hjust = -1.1, size = 5) +
  coord_flip(ylim = c(min(df$MeanAUC) - 0.005, max(df$MeanAUC) + 0.005)) +
  scale_fill_manual(values = pastel_colors) +
  theme_minimal(base_size = 12) +
  theme(
    axis.title.y = element_blank(),
    axis.text.y = element_text(size = 12),
    legend.position = "none"
  ) +
  labs(y = "Mean AUC", title = "")
pdf("./res/2025_0610_hist/Fig3_finetune.pdf", height = 5, width = 7)
print(p)
dev.off()