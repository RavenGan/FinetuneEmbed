rm(list = ls())
set.seed(7)

library(dplyr)
library(tidyr)
library(ggplot2)

tasks <- c("long_vs_shortTF", "DosageSensitivity", "MethylationState/bivalent_vs_lys4",
           "MethylationState/bivalent_vs_no_methyl", "Multi_class")
renamed_tasks <- c("Task 1", "Task 2", "Task 3",
                   "Task 4", "Task 5")

tab <- read.csv("./res/2025_0603_All_Num_Res/NoPCA_CV_text_embedding_NoTruncation_final.csv")
tab$AUC_se <- tab$AUC_sd / sqrt(10)
tab$Precision_se <- tab$Precision_sd / sqrt(10)
tab$Recall_se <- tab$Recall_sd / sqrt(10)
tab$F1_se <- tab$F1_sd / sqrt(10)


do_CV <- "CV"

for (i in 1:length(tasks)) {
  task_name <- tasks[i]
  tab$task[tab$task == task_name] <- renamed_tasks[i]
}

tab$LLM[tab$LLM == "GenePT_1536"] <- "OpenAI"

chosen_tasks <- c("Task 5")
LLM_levels <- c("OpenAI", "GIST-small-Embedding-v0",
                "NoInstruct-small-Embedding-v0",
                "stella-base-en-v2",
                "e5-small-v2",
                "GIST-all-MiniLM-L6-v2",
                "gte-small",
                "bge-small-en-v1.5",
                "MedEmbed-small-v0.1",
                "gte-tiny",
                "e5-small")


tab_sub <- tab
tab_sub <- tab_sub[tab_sub$task %in% chosen_tasks, ]
tab_sub <- tab_sub[tab_sub$LLM %in% LLM_levels, ]


##### Plot results for AUC-------
df_long <- tab_sub %>%
  pivot_longer(
    cols = c(AUC_mean, AUC_se
             # AUC_mean, AUC_se 
             # Precision_mean, Precision_se,
             # Recall_mean, Recall_se
             # F1_mean, F1_se
    ),
    names_to = c("Metric", "Type"),
    names_sep = "_",
    values_to = "Value"
  ) %>%
  pivot_wider(
    names_from = Type,
    values_from = Value
  )

df_long$LLM <- factor(df_long$LLM, levels = LLM_levels)
df_long$LLM <- factor(df_long$LLM, levels = rev(levels(df_long$LLM)))

df_ranked <- df_long %>%
  group_by(model, task) %>%
  mutate(rank = rank(-mean, ties.method = "average")) %>%
  ungroup()

# df_ranked <- df_ranked %>%
#   mutate(model = recode(model,
#                         "LR" = "Logistic regression",
#                         "RF" = "Random forest"))

df_ranked <- df_ranked %>%
  mutate(label = sprintf("%.3f\n(±%.3f)", mean, se))


pdf(paste0("./res/2025_0620_Plots/AUC/Fig7_", do_CV, ".pdf"), width = 6, height = 6)
ggplot(df_ranked, aes(x = task, y = LLM, fill = rank)) +
  geom_tile(color = "white", width = 0.95, height = 0.95) +
  geom_text(
    aes(label = label),
    size = 3.5,
    family = "mono",
    lineheight = 1.1,
    hjust = 0.5, vjust = 0.5
  ) +
  scale_fill_gradient2(
    low = "#FF6961", high = "#84B6F4", mid = "white",
    midpoint = 6,
    name = "Rank",
    guide = guide_colorbar(barwidth = 1.2, barheight = 8)
  ) +
  facet_grid(. ~ model) +
  labs(
    title = "",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    # plot.title = element_text(size = 14, face = "bold", hjust = 0.5, margin = margin(b = 10)),
    strip.text = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 13),
    axis.text.y = element_text(size = 12),
    panel.grid = element_blank(),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    legend.position = "right"
  )
dev.off()


##### Plot results for Precision-------
df_long <- tab_sub %>%
  pivot_longer(
    cols = c(Precision_mean, Precision_se
             # AUC_mean, AUC_se
             # Precision_mean, Precision_se
             # Recall_mean, Recall_se
             # F1_mean, F1_se
    ),
    names_to = c("Metric", "Type"),
    names_sep = "_",
    values_to = "Value"
  ) %>%
  pivot_wider(
    names_from = Type,
    values_from = Value
  )

df_long$LLM <- factor(df_long$LLM, levels = LLM_levels)
df_long$LLM <- factor(df_long$LLM, levels = rev(levels(df_long$LLM)))

df_ranked <- df_long %>%
  group_by(model, task) %>%
  mutate(rank = rank(-mean, ties.method = "average")) %>%
  ungroup()

# df_ranked <- df_ranked %>%
#   mutate(model = recode(model,
#                         "LR" = "Logistic regression",
#                         "RF" = "Random forest"))

df_ranked <- df_ranked %>%
  mutate(label = sprintf("%.3f\n(±%.3f)", mean, se))


pdf(paste0("./res/2025_0620_Plots/Precision/Fig7_", do_CV, ".pdf"), width = 6, height = 6)
ggplot(df_ranked, aes(x = task, y = LLM, fill = rank)) +
  geom_tile(color = "white", width = 0.95, height = 0.95) +
  geom_text(
    aes(label = label),
    size = 3.5,
    family = "mono",
    lineheight = 1.1,
    hjust = 0.5, vjust = 0.5
  ) +
  scale_fill_gradient2(
    low = "#FF6961", high = "#84B6F4", mid = "white",
    midpoint = 6,
    name = "Rank",
    guide = guide_colorbar(barwidth = 1.2, barheight = 8)
  ) +
  facet_grid(. ~ model) +
  labs(
    title = "",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    # plot.title = element_text(size = 14, face = "bold", hjust = 0.5, margin = margin(b = 10)),
    strip.text = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 13),
    axis.text.y = element_text(size = 12),
    panel.grid = element_blank(),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    legend.position = "right"
  )
dev.off()


##### Plot results for Recall-------
df_long <- tab_sub %>%
  pivot_longer(
    cols = c(Recall_mean, Recall_se
             # AUC_mean, AUC_se
             # Precision_mean, Precision_se
             # Recall_mean, Recall_se
             # F1_mean, F1_se
    ),
    names_to = c("Metric", "Type"),
    names_sep = "_",
    values_to = "Value"
  ) %>%
  pivot_wider(
    names_from = Type,
    values_from = Value
  )

df_long$LLM <- factor(df_long$LLM, levels = LLM_levels)
df_long$LLM <- factor(df_long$LLM, levels = rev(levels(df_long$LLM)))

df_ranked <- df_long %>%
  group_by(model, task) %>%
  mutate(rank = rank(-mean, ties.method = "average")) %>%
  ungroup()

# df_ranked <- df_ranked %>%
#   mutate(model = recode(model,
#                         "LR" = "Logistic regression",
#                         "RF" = "Random forest"))

df_ranked <- df_ranked %>%
  mutate(label = sprintf("%.3f\n(±%.3f)", mean, se))


pdf(paste0("./res/2025_0620_Plots/Recall/Fig7_", do_CV, ".pdf"), width = 6, height = 6)
ggplot(df_ranked, aes(x = task, y = LLM, fill = rank)) +
  geom_tile(color = "white", width = 0.95, height = 0.95) +
  geom_text(
    aes(label = label),
    size = 3.5,
    family = "mono",
    lineheight = 1.1,
    hjust = 0.5, vjust = 0.5
  ) +
  scale_fill_gradient2(
    low = "#FF6961", high = "#84B6F4", mid = "white",
    midpoint = 6,
    name = "Rank",
    guide = guide_colorbar(barwidth = 1.2, barheight = 8)
  ) +
  facet_grid(. ~ model) +
  labs(
    title = "",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    # plot.title = element_text(size = 14, face = "bold", hjust = 0.5, margin = margin(b = 10)),
    strip.text = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 13),
    axis.text.y = element_text(size = 12),
    panel.grid = element_blank(),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    legend.position = "right"
  )
dev.off()


##### Plot results for F1-------
df_long <- tab_sub %>%
  pivot_longer(
    cols = c(F1_mean, F1_se
             # AUC_mean, AUC_se
             # Precision_mean, Precision_se
             # Recall_mean, Recall_se
             # F1_mean, F1_se
    ),
    names_to = c("Metric", "Type"),
    names_sep = "_",
    values_to = "Value"
  ) %>%
  pivot_wider(
    names_from = Type,
    values_from = Value
  )

df_long$LLM <- factor(df_long$LLM, levels = LLM_levels)
df_long$LLM <- factor(df_long$LLM, levels = rev(levels(df_long$LLM)))

df_ranked <- df_long %>%
  group_by(model, task) %>%
  mutate(rank = rank(-mean, ties.method = "average")) %>%
  ungroup()

# df_ranked <- df_ranked %>%
#   mutate(model = recode(model,
#                         "LR" = "Logistic regression",
#                         "RF" = "Random forest"))

df_ranked <- df_ranked %>%
  mutate(label = sprintf("%.3f\n(±%.3f)", mean, se))


pdf(paste0("./res/2025_0620_Plots/F1/Fig7_", do_CV, ".pdf"), width = 6, height = 6)
ggplot(df_ranked, aes(x = task, y = LLM, fill = rank)) +
  geom_tile(color = "white", width = 0.95, height = 0.95) +
  geom_text(
    aes(label = label),
    size = 3.5,
    family = "mono",
    lineheight = 1.1,
    hjust = 0.5, vjust = 0.5
  ) +
  scale_fill_gradient2(
    low = "#FF6961", high = "#84B6F4", mid = "white",
    midpoint = 6,
    name = "Rank",
    guide = guide_colorbar(barwidth = 1.2, barheight = 8)
  ) +
  facet_grid(. ~ model) +
  labs(
    title = "",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    # plot.title = element_text(size = 14, face = "bold", hjust = 0.5, margin = margin(b = 10)),
    strip.text = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 13),
    axis.text.y = element_text(size = 12),
    panel.grid = element_blank(),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    legend.position = "right"
  )
dev.off()