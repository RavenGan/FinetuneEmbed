rm(list = ls())
set.seed(7)

library(dplyr)
library(ggplot2)

obtain_plot_dat <- function(path2csv, replace = TRUE){
  data <- read.csv(path2csv)
  if (replace){
    methods <- data[, 1]
    methods[1] <- "GenePT"
  } else {
    methods <- data[, 1]
  }
  tasks <- c(rep("TF range", length(methods)),
             rep("Dosage sensitivity", length(methods)),
             rep("Bivalent versus Lys4-methylated", length(methods)),
             rep("Bivalent versus non-methylated", length(methods)))
  auc <- c(data$TF.range, 
           data$Dosage.sensitivity,
           data$Bivalent.vs..Lys4,
           data$Bivalent.vs..NonMethyl)
  sd <- c(data$s.d.,
          data$s.d..1,
          data$s.d..2,
          data$s.d..3)
  
  auc_data <- data.frame(
    Method = rep(methods, 4),
    Task = tasks,
    AUC = auc,
    SD = sd
  )
  
  auc_data$Method <- factor(auc_data$Method, levels = methods)
  auc_data$Method <- factor(auc_data$Method, levels = rev(levels(auc_data$Method)))
  auc_data$Task <- factor(auc_data$Task, levels = c("TF range",
                                                    "Dosage sensitivity",
                                                    "Bivalent versus Lys4-methylated",
                                                    "Bivalent versus non-methylated"))
  # Calculate the overall range of x-axis limits
  xmin <- min(auc_data$AUC - auc_data$SD, na.rm = TRUE)
  xmax <- max(auc_data$AUC + auc_data$SD, na.rm = TRUE)
  
  # Identify the Method with the largest AUC for each Task
  auc_data <- auc_data %>%
    group_by(Task) %>%
    mutate(is_max = ifelse(AUC == max(AUC), TRUE, FALSE))
  
  p <- ggplot(auc_data, aes(x = AUC, y = Method)) +
    geom_errorbar(aes(xmin = AUC - SD, xmax = AUC + SD), 
                  width = 0.15, 
                  position = position_dodge(width = 0.7)) +  # Error bars for SD
    geom_boxplot(aes(color = is_max),  # Color the border of the boxplot
                 outlier.shape = NA, 
                 position = position_dodge(width = 0.8)) +  # Boxplot without outliers
    facet_wrap(~ Task, scales = "free_x", nrow = 1) +  # Facet by task
    labs(x = "", y = "", title = "") +
    theme_minimal() +
    scale_x_continuous(limits = c(xmin, xmax)) + 
    theme(axis.text.x = element_text(size = 14, color = "black"), 
          axis.text.y = element_text(size = 12, color = "black"),
          strip.text = element_text(size = 12)) + 
    scale_color_manual(values = c("TRUE" = "red", "FALSE" = "black")) +  # Use red for max AUC
    guides(color = "none")  # Remove the legend for is_max
  
  return(p)
}

h <- 4
w <- 16

p <- obtain_plot_dat("./res/2024_1225/csv/MultiMod_test_811_NoTuning_LR.csv")
pdf("./res/2024_1230/NoTuning_LR.pdf", height = h, width = w)
print(p)
dev.off()

p <- obtain_plot_dat("./res/2024_1225/csv/MultiMod_test_811_NoTuning_RF.csv")
pdf("./res/2024_1230/NoTuning_RF.pdf", height = h, width = w)
print(p)
dev.off()

p <- obtain_plot_dat("./res/2024_1225/csv/MultiMod_test_811_Tuning_LR.csv")
pdf("./res/2024_1230/Tuning_LR.pdf", height = h, width = w)
print(p)
dev.off()

p <- obtain_plot_dat("./res/2024_1225/csv/MultiMod_test_811_Tuning_RF.csv")
pdf("./res/2024_1230/Tuning_RF.pdf", height = h, width = w)
print(p)
dev.off()

p <- obtain_plot_dat("./res/2024_1225/csv/MultiMod_test_811_Finetuning.csv",
                     replace = FALSE)
pdf("./res/2024_1230/Finetuning.pdf", height = h, width = w)
print(p)
dev.off()