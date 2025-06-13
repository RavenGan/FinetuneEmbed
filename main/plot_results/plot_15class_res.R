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

colors <- c("#1B9E77", # AUC
            "#D95F02", # Precision
            "#7570B3", # Recall
            "#E6AB02"  # F1
)


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
  mutate(model = recode(model,
                        "LR" = "Logistic regression",
                        "RF" = "Random forest"))

pdf(paste0("./res/2025_0613_Plots/AUC/Fig7_", do_CV, ".pdf"), width = 7, height = 7)
ggplot(df_ranked, aes(x = mean, y = LLM)) +
  geom_point(size = 2, color = "#1B9E77") +
  geom_errorbarh(aes(xmin = mean - se, xmax = mean + se),
                 height = 0.2, color = "#1B9E77") +
  geom_text(aes(label = paste0(rank)),
            hjust = 0, vjust = -0.4, color = "#1B9E77") +
  facet_grid(rows = vars(model), cols = vars(task), scales = "free_x") +
  theme_minimal(base_size = 12) +
  labs(x = "", y = NULL) +
  theme(
    strip.placement.y = "left",                      # Move row labels to the left
    strip.text.y.left = element_text(size = 13, face = "bold", angle = 0),  # Format row labels
    strip.text = element_text(face = "bold", size = 13),
    axis.text.y = element_text(size = 12),
    axis.text.x = element_text(size = 12),
    legend.position = "none"
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

df_ranked <- df_ranked %>%
  mutate(model = recode(model,
                        "LR" = "Logistic regression",
                        "RF" = "Random forest"))

pdf(paste0("./res/2025_0613_Plots/Precision/Fig7_", do_CV, ".pdf"), width = 7, height = 7)
ggplot(df_ranked, aes(x = mean, y = LLM)) +
  geom_point(size = 2, color = "#D95F02") +
  geom_errorbarh(aes(xmin = mean - se, xmax = mean + se),
                 height = 0.2, color = "#D95F02") +
  geom_text(aes(label = paste0(rank)),
            hjust = -0.2, vjust = -0.4, color = "#D95F02") +
  facet_grid(rows = vars(model), cols = vars(task), scales = "free_x") +
  theme_minimal(base_size = 12) +
  labs(x = "", y = NULL) +
  theme(
    strip.placement.y = "left",                      # Move row labels to the left
    strip.text.y.left = element_text(size = 13, face = "bold", angle = 0),  # Format row labels
    strip.text = element_text(face = "bold", size = 13),
    axis.text.y = element_text(size = 12),
    axis.text.x = element_text(size = 12),
    legend.position = "none"
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

df_ranked <- df_ranked %>%
  mutate(model = recode(model,
                        "LR" = "Logistic regression",
                        "RF" = "Random forest"))

pdf(paste0("./res/2025_0613_Plots/Recall/Fig7_", do_CV, ".pdf"), width = 7, height = 7)
ggplot(df_ranked, aes(x = mean, y = LLM)) +
  geom_point(size = 2, color = "#7570B3") +
  geom_errorbarh(aes(xmin = mean - se, xmax = mean + se),
                 height = 0.2, color = "#7570B3") +
  geom_text(aes(label = paste0(rank)),
            hjust = -0.2, vjust = -0.4, color = "#7570B3") +
  facet_grid(rows = vars(model), cols = vars(task), scales = "free_x") +
  theme_minimal(base_size = 12) +
  labs(x = "", y = NULL) +
  theme(
    strip.placement.y = "left",                      # Move row labels to the left
    strip.text.y.left = element_text(size = 13, face = "bold", angle = 0),  # Format row labels
    strip.text = element_text(face = "bold", size = 13),
    axis.text.y = element_text(size = 12),
    axis.text.x = element_text(size = 12),
    legend.position = "none"
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

df_ranked <- df_ranked %>%
  mutate(model = recode(model,
                        "LR" = "Logistic regression",
                        "RF" = "Random forest"))

pdf(paste0("./res/2025_0613_Plots/F1/Fig7_", do_CV, ".pdf"), width = 7, height = 7)
ggplot(df_ranked, aes(x = mean, y = LLM)) +
  geom_point(size = 2, color = "#E6AB02") +
  geom_errorbarh(aes(xmin = mean - se, xmax = mean + se),
                 height = 0.2, color = "#E6AB02") +
  geom_text(aes(label = paste0(rank)),
            hjust = -0.2, vjust = -0.4, color = "#E6AB02") +
  facet_grid(rows = vars(model), cols = vars(task), scales = "free_x") +
  theme_minimal(base_size = 12) +
  labs(x = "", y = NULL) +
  theme(
    strip.placement.y = "left",                      # Move row labels to the left
    strip.text.y.left = element_text(size = 13, face = "bold", angle = 0),  # Format row labels
    strip.text = element_text(face = "bold", size = 13),
    axis.text.y = element_text(size = 12),
    axis.text.x = element_text(size = 12),
    legend.position = "none"
  )
dev.off()