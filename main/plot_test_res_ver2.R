rm(list = ls())
set.seed(7)

library(dplyr)
library(ggplot2)

organize_tab <- function(path2csv, remove = FALSE){
  models <- c("GenePT",
              "GIST-small-Embedding-v0",      
              "NoInstruct-small-Embedding-v0",
              "stella-base-en-v2",            
              "bge-small-en-v1.5",
              "e5-small",                     
              "GIST-all-MiniLM-L6-v2",
              "gte-small",                    
              "MedEmbed-small-v0",
              "e5-small-v2",                  
              "gte-tiny")
  tasks <- c("bivalent_vs_lys4",
             "bivalent_vs_no_methyl",
             "DosageSensitivity",
             "long_vs_shortTF")
  Tasks <- c("Bivalent versus Lys4-methylated",
             "Bivalent versus non-methylated",
             "Dosage sensitivity",
             "TF range")
  
  if (remove){
    models <- models[-1]
  }
  
  data <- read.csv(path2csv)
  data$Method <- factor(data$Method,
                        levels = models)
  data$Method <- factor(data$Method, levels = rev(levels(data$Method)))
  
  # replace the task names
  for (i in 1:length(tasks)) {
    task <- tasks[i]
    Task <- Tasks[i]
    
    data$Task[data$Task == task] <- Task
  }
  
  data$Task <- factor(data$Task, levels = c("TF range",
                                            "Dosage sensitivity",
                                            "Bivalent versus Lys4-methylated",
                                            "Bivalent versus non-methylated"))
  data$Significant <- data$pvals <= 0.05
  data$Signs <- data$mean_diff >= 0
  data$color <- "grey"
  for (i in 1:nrow(data)) {
    if (data$Significant[i] & data$Signs[i]){
      data$color[i] <- "red"
    } else if (data$Significant[i] & !data$Signs[i]){
      data$color[i] <- "blue"
    }
  }
  
  # Calculate mean values for each method
  model_mean <- data %>%
    group_by(Method) %>%
    summarize(mean_value = mean(mean_diff)) %>%
    as.data.frame()
  
  return(model_mean)
}

model_mean <- organize_tab("./res/2025_0205/Diff_results/NoTuning_LR_over_RF.csv")
p <- ggplot(model_mean, aes(x = mean_value, y = Method, fill = mean_value > 0)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("darkblue", "darkred"), guide = "none") +  # Color negative and positive differently
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +  # Centerline
  labs(x = "", y = "", title = "Mean improvement") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
        axis.text.y = element_blank())
pdf("./res/2025_0205/Diff_results_bar/NoTuning_LR_over_RF.pdf", height = 4, width = 2)
print(p)
dev.off()

model_mean <- organize_tab("./res/2025_0205/Diff_results/Tuning_LR_over_RF.csv")
p <- ggplot(model_mean, aes(x = mean_value, y = Method, fill = mean_value > 0)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("darkblue", "darkred"), guide = "none") +  # Color negative and positive differently
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +  # Centerline
  labs(x = "", y = "", title = "Mean improvement") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
        axis.text.y = element_blank())
pdf("./res/2025_0205/Diff_results_bar/Tuning_LR_over_RF.pdf", height = 4, width = 2)
print(p)
dev.off()

model_mean <- organize_tab("./res/2025_0205/Diff_results/LR_Tuning_over_NoTuning.csv")
p <- ggplot(model_mean, aes(x = mean_value, y = Method, fill = mean_value > 0)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("darkblue", "darkred"), guide = "none") +  # Color negative and positive differently
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +  # Centerline
  labs(x = "", y = "", title = "Mean improvement") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
        axis.text.y = element_blank())
pdf("./res/2025_0205/Diff_results_bar/LR_Tuning_over_NoTuning.pdf", height = 4, width = 2)
print(p)
dev.off()

model_mean <- organize_tab("./res/2025_0205/Diff_results/RF_Tuning_over_NoTuning.csv")
p <- ggplot(model_mean, aes(x = mean_value, y = Method, fill = mean_value > 0)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("darkblue", "darkred"), guide = "none") +  # Color negative and positive differently
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +  # Centerline
  labs(x = "", y = "", title = "Mean improvement") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
        axis.text.y = element_blank())
pdf("./res/2025_0205/Diff_results_bar/RF_Tuning_over_NoTuning.pdf", height = 4, width = 2)
print(p)
dev.off()

model_mean <- organize_tab("./res/2025_0205/Diff_results/Finetuning_over_LRTuning.csv")
p <- ggplot(model_mean, aes(x = mean_value, y = Method, fill = mean_value > 0)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("darkblue", "darkred"), guide = "none") +  # Color negative and positive differently
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +  # Centerline
  labs(x = "", y = "", title = "Mean improvement") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
        axis.text.y = element_blank())
pdf("./res/2025_0205/Diff_results_bar/Finetuning_over_LRTuning.pdf", height = 4, width = 2)
print(p)
dev.off()

model_mean <- organize_tab("./res/2025_0205/Diff_results/Finetuning_over_RFTuning.csv")
p <- ggplot(model_mean, aes(x = mean_value, y = Method, fill = mean_value > 0)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("darkblue", "darkred"), guide = "none") +  # Color negative and positive differently
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +  # Centerline
  labs(x = "", y = "", title = "Mean improvement") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
        axis.text.y = element_blank())
pdf("./res/2025_0205/Diff_results_bar/Finetuning_over_RFTuning.pdf", height = 4, width = 2)
print(p)
dev.off()