rm(list = ls())
set.seed(7)

library(dplyr)
library(tidyr)
library(ggplot2)

tasks <- c("LongShortTF", "Sensitivity", "BivalentLys4",
           "BivalentNoMethyl")
renamed_tasks <- c("Task 1", "Task 2", "Task 3",
                   "Task 4")

tab <- read.csv("./res/2025_0603_All_Num_Res/finetune_Gene_Name.csv")

for (i in 1:length(tasks)) {
  task_name <- tasks[i]
  tab$task[tab$task == task_name] <- renamed_tasks[i]
}

chosen_tasks <- c("Task 1", "Task 2", "Task 3",
                  "Task 4")
LLM_levels <- c("GIST-small-Embedding-v0",
                "NoInstruct-small-Embedding-v0",
                "stella-base-en-v2",
                "e5-small-v2",
                "GIST-all-MiniLM-L6-v2",
                "gte-small",
                "bge-small-en-v1.5",
                "MedEmbed-small-v0.1",
                "gte-tiny",
                "e5-small",
                "biobert-base-cased-v1.1")


tab_sub <- tab
tab_sub <- tab_sub[tab_sub$task %in% chosen_tasks, ]


df_long <- tab_sub %>%
  pivot_longer(
    cols = c(AUC_mean, AUC_sd, Precision_mean, Precision_sd, 
             Recall_mean, Recall_sd, F1_mean, F1_sd),
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

pdf("./res/2025_0603_Plots/Fig6_finetune.pdf", width = 8, height = 6)
ggplot(df_long, aes(x = mean, y = LLM, color = Metric)) +
  geom_point(size = 2, position = position_dodge(width = 0.7)) +
  geom_errorbarh(aes(xmin = mean - sd, xmax = mean + sd),
                 height = 0.2, position = position_dodge(width = 0.7)) +
  facet_wrap(~ task, scales = "free_x", ncol = 2) +
  theme_minimal(base_size = 12) +
  scale_color_brewer(palette = "Dark2") +
  labs(x = "", y = NULL,
       title = "",
       color = "Metrics") +
  theme(strip.text = element_text(face = "bold", size = 13),
        axis.text.y = element_text(size = 10),
        legend.position = "bottom")
dev.off()

