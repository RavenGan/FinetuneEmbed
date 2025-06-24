rm(list = ls())
set.seed(7)

library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)

tasks <- c("long_vs_shortTF", "DosageSensitivity", "MethylationState/bivalent_vs_lys4",
           "MethylationState/bivalent_vs_no_methyl", "Multi_class")
renamed_tasks <- c("Task 1", "Task 2", "Task 3",
                   "Task 4", "Task 5")

tab <- read.csv("./res/2025_0603_All_Num_Res/NoPCA_NoCV_text_embedding_NoTruncation_final.csv")
tab$AUC_se <- tab$AUC_sd / sqrt(10)
tab$Precision_se <- tab$Precision_sd / sqrt(10)
tab$Recall_se <- tab$Recall_sd / sqrt(10)
tab$F1_se <- tab$F1_sd / sqrt(10)


do_CV <- "NoCV"

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

df_ranked <- df_ranked %>%
  mutate(label = sprintf("%.3f\n(±%.3f)", mean, se))

# Split by model
df_lr <- df_ranked %>% filter(model == "LR")
df_rf <- df_ranked %>% filter(model == "RF")


# Set LLM factor levels by avg AUC within each model
df_lr <- df_lr %>%
  group_by(LLM) %>%
  mutate(avg_auc = mean(mean, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(LLM = factor(LLM, levels = unique(LLM[order(avg_auc)])))

# Random forest: reverse order
df_rf <- df_rf %>%
  group_by(LLM) %>%
  mutate(avg_auc = mean(mean, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(LLM = factor(LLM, levels = unique(LLM[order(avg_auc)])))

p_lr <- ggplot(df_lr, aes(x = task, y = LLM, fill = rank)) +
  geom_tile(color = "white", width = 0.95, height = 0.95) +
  geom_text(aes(label = label), size = 3.5, family = "mono", hjust = 0.5) +
  scale_fill_gradient2(low = "#FF6961", high = "#84B6F4", mid = "white", midpoint = 5.5,
                       name = "Rank") +
  scale_x_discrete(position = "top") + 
  labs(title = "LR", x = NULL, y = NULL) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),  # centered, bold
        strip.text = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        panel.grid = element_blank())

p_rf <- ggplot(df_rf, aes(x = task, y = LLM, fill = rank)) +
  geom_tile(color = "white", width = 0.95, height = 0.95) +
  geom_text(aes(label = label), size = 3.5, family = "mono", hjust = 0.5) +
  scale_fill_gradient2(low = "#FF6961", high = "#84B6F4", mid = "white", midpoint = 5.5,
                       name = "Rank") +
  scale_x_discrete(position = "top") +
  labs(title = "RF", x = NULL, y = NULL) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),  # centered, bold
        strip.text = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        panel.grid = element_blank())

# Combine the two plots side by side
combined_plot <- p_lr + p_rf + plot_layout(nrow = 1, guides = "collect")
pdf(paste0("./res/2025_0624_Plots/AUC/Fig7_", do_CV, ".pdf"), width = 8.5, height = 6)
print(combined_plot)
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

# Split by model
df_lr <- df_ranked %>% filter(model == "LR")
df_rf <- df_ranked %>% filter(model == "RF")


# Set LLM factor levels by avg AUC within each model
df_lr <- df_lr %>%
  group_by(LLM) %>%
  mutate(avg_auc = mean(mean, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(LLM = factor(LLM, levels = unique(LLM[order(avg_auc)])))

# Random forest: reverse order
df_rf <- df_rf %>%
  group_by(LLM) %>%
  mutate(avg_auc = mean(mean, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(LLM = factor(LLM, levels = unique(LLM[order(avg_auc)])))

p_lr <- ggplot(df_lr, aes(x = task, y = LLM, fill = rank)) +
  geom_tile(color = "white", width = 0.95, height = 0.95) +
  geom_text(aes(label = label), size = 3.5, family = "mono", hjust = 0.5) +
  scale_fill_gradient2(low = "#FF6961", high = "#84B6F4", mid = "white", midpoint = 5.5,
                       name = "Rank") +
  scale_x_discrete(position = "top") + 
  labs(title = "LR", x = NULL, y = NULL) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),  # centered, bold
        strip.text = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        panel.grid = element_blank())

p_rf <- ggplot(df_rf, aes(x = task, y = LLM, fill = rank)) +
  geom_tile(color = "white", width = 0.95, height = 0.95) +
  geom_text(aes(label = label), size = 3.5, family = "mono", hjust = 0.5) +
  scale_fill_gradient2(low = "#FF6961", high = "#84B6F4", mid = "white", midpoint = 5.5,
                       name = "Rank") +
  scale_x_discrete(position = "top") +
  labs(title = "RF", x = NULL, y = NULL) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),  # centered, bold
        strip.text = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        panel.grid = element_blank())

# Combine the two plots side by side
combined_plot <- p_lr + p_rf + plot_layout(nrow = 1, guides = "collect")
pdf(paste0("./res/2025_0624_Plots/Precision/Fig7_", do_CV, ".pdf"), width = 8.5, height = 6)
print(combined_plot)
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

# Split by model
df_lr <- df_ranked %>% filter(model == "LR")
df_rf <- df_ranked %>% filter(model == "RF")


# Set LLM factor levels by avg AUC within each model
df_lr <- df_lr %>%
  group_by(LLM) %>%
  mutate(avg_auc = mean(mean, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(LLM = factor(LLM, levels = unique(LLM[order(avg_auc)])))

# Random forest: reverse order
df_rf <- df_rf %>%
  group_by(LLM) %>%
  mutate(avg_auc = mean(mean, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(LLM = factor(LLM, levels = unique(LLM[order(avg_auc)])))

p_lr <- ggplot(df_lr, aes(x = task, y = LLM, fill = rank)) +
  geom_tile(color = "white", width = 0.95, height = 0.95) +
  geom_text(aes(label = label), size = 3.5, family = "mono", hjust = 0.5) +
  scale_fill_gradient2(low = "#FF6961", high = "#84B6F4", mid = "white", midpoint = 5.5,
                       name = "Rank") +
  scale_x_discrete(position = "top") + 
  labs(title = "LR", x = NULL, y = NULL) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),  # centered, bold
        strip.text = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        panel.grid = element_blank())

p_rf <- ggplot(df_rf, aes(x = task, y = LLM, fill = rank)) +
  geom_tile(color = "white", width = 0.95, height = 0.95) +
  geom_text(aes(label = label), size = 3.5, family = "mono", hjust = 0.5) +
  scale_fill_gradient2(low = "#FF6961", high = "#84B6F4", mid = "white", midpoint = 5.5,
                       name = "Rank") +
  scale_x_discrete(position = "top") +
  labs(title = "RF", x = NULL, y = NULL) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),  # centered, bold
        strip.text = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        panel.grid = element_blank())

# Combine the two plots side by side
combined_plot <- p_lr + p_rf + plot_layout(nrow = 1, guides = "collect")
pdf(paste0("./res/2025_0624_Plots/Recall/Fig7_", do_CV, ".pdf"), width = 8.5, height = 6)
print(combined_plot)
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

# Split by model
df_lr <- df_ranked %>% filter(model == "LR")
df_rf <- df_ranked %>% filter(model == "RF")


# Set LLM factor levels by avg AUC within each model
df_lr <- df_lr %>%
  group_by(LLM) %>%
  mutate(avg_auc = mean(mean, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(LLM = factor(LLM, levels = unique(LLM[order(avg_auc)])))

# Random forest: reverse order
df_rf <- df_rf %>%
  group_by(LLM) %>%
  mutate(avg_auc = mean(mean, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(LLM = factor(LLM, levels = unique(LLM[order(avg_auc)])))

p_lr <- ggplot(df_lr, aes(x = task, y = LLM, fill = rank)) +
  geom_tile(color = "white", width = 0.95, height = 0.95) +
  geom_text(aes(label = label), size = 3.5, family = "mono", hjust = 0.5) +
  scale_fill_gradient2(low = "#FF6961", high = "#84B6F4", mid = "white", midpoint = 5.5,
                       name = "Rank") +
  scale_x_discrete(position = "top") + 
  labs(title = "LR", x = NULL, y = NULL) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),  # centered, bold
        strip.text = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        panel.grid = element_blank())

p_rf <- ggplot(df_rf, aes(x = task, y = LLM, fill = rank)) +
  geom_tile(color = "white", width = 0.95, height = 0.95) +
  geom_text(aes(label = label), size = 3.5, family = "mono", hjust = 0.5) +
  scale_fill_gradient2(low = "#FF6961", high = "#84B6F4", mid = "white", midpoint = 5.5,
                       name = "Rank") +
  scale_x_discrete(position = "top") +
  labs(title = "RF", x = NULL, y = NULL) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),  # centered, bold
        strip.text = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        panel.grid = element_blank())

# Combine the two plots side by side
combined_plot <- p_lr + p_rf + plot_layout(nrow = 1, guides = "collect")
pdf(paste0("./res/2025_0624_Plots/F1/Fig7_", do_CV, ".pdf"), width = 8.5, height = 6)
print(combined_plot)
dev.off()
