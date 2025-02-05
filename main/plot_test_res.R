rm(list = ls())
set.seed(7)

library(dplyr)
library(ggplot2)

obtain_plot_data <- function(path2csv, remove = FALSE){
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
  value_range <- range(data$mean_diff)
  
  p <- ggplot(data, aes(x = mean_diff, y = Method, color = color)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") + # Add dashed line
    geom_point(size = 3) +
    facet_wrap(~ Task, scales = "free_x", nrow = 1) +
    scale_x_continuous(limits = value_range) +
    theme_minimal() + 
    scale_color_identity() +  # Use colors directly from the `color` column
    theme(axis.text.x = element_text(size = 14, color = "black"), 
          axis.text.y = element_text(size = 12, color = "black"),
          strip.text = element_text(size = 12)) + 
    labs(title = "",
         x="",
         y="")
  
  return(p)
}

p <- obtain_plot_data("./res/2025_0205/Diff_results/NoTuning_LR_over_RF.csv")
pdf("./res/2025_0205/Diff_plots/NoTuning_LR_over_RF.pdf", width = 16, height = 4)
print(p)
dev.off()

p <- obtain_plot_data("./res/2025_0205/Diff_results/Tuning_LR_over_RF.csv")
pdf("./res/2025_0205/Diff_plots/Tuning_LR_over_RF.pdf", width = 16, height = 4)
print(p)
dev.off()

p <- obtain_plot_data("./res/2025_0205/Diff_results/LR_Tuning_over_NoTuning.csv")
pdf("./res/2025_0205/Diff_plots/LR_Tuning_over_NoTuning.pdf", width = 16, height = 4)
print(p)
dev.off()

p <- obtain_plot_data("./res/2025_0205/Diff_results/RF_Tuning_over_NoTuning.csv")
pdf("./res/2025_0205/Diff_plots/RF_Tuning_over_NoTuning.pdf", width = 16, height = 4)
print(p)
dev.off()

p <- obtain_plot_data("./res/2025_0205/Diff_results/Finetuning_over_LRTuning.csv")
pdf("./res/2025_0205/Diff_plots/Finetuning_over_LRTuning.pdf", width = 16, height = 4)
print(p)
dev.off()

p <- obtain_plot_data("./res/2025_0205/Diff_results/Finetuning_over_RFTuning.csv")
pdf("./res/2025_0205/Diff_plots/Finetuning_over_RFTuning.pdf", width = 16, height = 4)
print(p)
dev.off()