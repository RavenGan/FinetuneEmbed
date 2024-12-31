rm(list = ls())
set.seed(7)

library(dplyr)

obtain_vals <- function(path, model, task, chosen_col){
  file_path <- paste0(path, model, "/",
                      model, "_", task, "_auc.csv")
  table <- read.csv(file_path)
  val <- table[[chosen_col]]
  
  return(val)
}

obtain_finetune_vals <- function(path, model, task, chosen_col){
  file_path <- paste0(path, model, "/", task, "_finetune_auc.csv")
  table <- read.csv(file_path)
  val <- table[[chosen_col]]
  
  return(val)
}


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

# NoTuning, LR vs. RF-------
path <- "./res/2024_1230/LRRF_num_res/"
col_1 <- "LR_test"
col_2 <- "RF_test"
res <- data.frame(Method = character(), 
                  Task = character(), 
                  mean_diff = numeric(), 
                  pvals = numeric(), stringsAsFactors = FALSE)
for (i in 1:length(models)) {
  model <- models[i]
  for (j in 1:length(tasks)) {
    task <- tasks[j]
    
    vals_1 <- obtain_vals(path, model, task, col_1)
    vals_2 <- obtain_vals(path, model, task, col_2)
    
    test_res <- t.test(vals_1, vals_2)
    
    mean_diff <- round(diff(test_res$estimate), 4)
    p_value <- round(test_res$p.value, 4)
    
    res <- rbind(res, data.frame(Method = model, Task = task, 
                                 mean_diff = mean_diff, pvals = p_value))
  }
}
write.csv(res, "./res/2024_1230/diff_results/NoTuning_LR_over_RF.csv")

# Tuning, LR vs. RF-------
path <- "./res/2024_1230/LRRF_CV_num_res/"
col_1 <- "LR_test"
col_2 <- "RF_test"
res <- data.frame(Method = character(), 
                  Task = character(), 
                  mean_diff = numeric(), 
                  pvals = numeric(), stringsAsFactors = FALSE)
for (i in 1:length(models)) {
  model <- models[i]
  for (j in 1:length(tasks)) {
    task <- tasks[j]
    
    vals_1 <- obtain_vals(path, model, task, col_1)
    vals_2 <- obtain_vals(path, model, task, col_2)
    
    test_res <- t.test(vals_1, vals_2)
    
    mean_diff <- round(diff(test_res$estimate), 4)
    p_value <- round(test_res$p.value, 4)
    
    res <- rbind(res, data.frame(Method = model, Task = task, 
                                 mean_diff = mean_diff, pvals = p_value))
  }
}
write.csv(res, "./res/2024_1230/diff_results/Tuning_LR_over_RF.csv")


# LR, Tuning vs. NoTuning-------
path_1 <- "./res/2024_1230/LRRF_CV_num_res/"
path_2 <- "./res/2024_1230/LRRF_num_res/"
col <- "LR_test"
res <- data.frame(Method = character(), 
                  Task = character(), 
                  mean_diff = numeric(), 
                  pvals = numeric(), stringsAsFactors = FALSE)
for (i in 1:length(models)) {
  model <- models[i]
  for (j in 1:length(tasks)) {
    task <- tasks[j]
    
    vals_1 <- obtain_vals(path_1, model, task, col)
    vals_2 <- obtain_vals(path_2, model, task, col)
    
    test_res <- t.test(vals_1, vals_2)
    
    mean_diff <- round(diff(test_res$estimate), 4)
    p_value <- round(test_res$p.value, 4)
    
    res <- rbind(res, data.frame(Method = model, Task = task, 
                                 mean_diff = mean_diff, pvals = p_value))
  }
}
write.csv(res, "./res/2024_1230/diff_results/LR_Tuning_over_NoTuning.csv")

# RF, Tuning vs. NoTuning-------
path_1 <- "./res/2024_1230/LRRF_CV_num_res/"
path_2 <- "./res/2024_1230/LRRF_num_res/"
col <- "RF_test"
res <- data.frame(Method = character(), 
                  Task = character(), 
                  mean_diff = numeric(), 
                  pvals = numeric(), stringsAsFactors = FALSE)
for (i in 1:length(models)) {
  model <- models[i]
  for (j in 1:length(tasks)) {
    task <- tasks[j]
    
    vals_1 <- obtain_vals(path_1, model, task, col)
    vals_2 <- obtain_vals(path_2, model, task, col)
    
    test_res <- t.test(vals_1, vals_2)
    
    mean_diff <- round(diff(test_res$estimate), 4)
    p_value <- round(test_res$p.value, 4)
    
    res <- rbind(res, data.frame(Method = model, Task = task, 
                                 mean_diff = mean_diff, pvals = p_value))
  }
}
write.csv(res, "./res/2024_1230/diff_results/RF_Tuning_over_NoTuning.csv")


# Finetuning vs. LR Tuning-----
path_1 <- "./res/2024_1230/Finetune_num_res/"
col_1 <- "test_auc"
path_2 <- "./res/2024_1230/LRRF_CV_num_res/"
col_2 <- "LR_test"
res <- data.frame(Method = character(), 
                  Task = character(), 
                  mean_diff = numeric(), 
                  pvals = numeric(), stringsAsFactors = FALSE)

models <- models[-1] # Remove the first one "GenePT"

for (i in 1:length(models)) {
  model <- models[i]
  for (j in 1:length(tasks)) {
    task <- tasks[j]
    
    vals_1 <- obtain_finetune_vals(path_1, model, task, col_1)
    vals_2 <- obtain_vals(path_2, model, task, col_2)
    
    test_res <- t.test(vals_1, vals_2)
    
    mean_diff <- round(diff(test_res$estimate), 4)
    p_value <- round(test_res$p.value, 4)
    
    res <- rbind(res, data.frame(Method = model, Task = task, 
                                 mean_diff = mean_diff, pvals = p_value))
  }
}
write.csv(res, "./res/2024_1230/diff_results/Finetuning_over_LRTuning.csv")

# Finetuning vs. RF Tuning-----
path_1 <- "./res/2024_1230/Finetune_num_res/"
col_1 <- "test_auc"
path_2 <- "./res/2024_1230/LRRF_CV_num_res/"
col_2 <- "RF_test"
res <- data.frame(Method = character(), 
                  Task = character(), 
                  mean_diff = numeric(), 
                  pvals = numeric(), stringsAsFactors = FALSE)

models <- models[-1] # Remove the first one "GenePT"

for (i in 1:length(models)) {
  model <- models[i]
  for (j in 1:length(tasks)) {
    task <- tasks[j]
    
    vals_1 <- obtain_finetune_vals(path_1, model, task, col_1)
    vals_2 <- obtain_vals(path_2, model, task, col_2)
    
    test_res <- t.test(vals_1, vals_2)
    
    mean_diff <- round(diff(test_res$estimate), 4)
    p_value <- round(test_res$p.value, 4)
    
    res <- rbind(res, data.frame(Method = model, Task = task, 
                                 mean_diff = mean_diff, pvals = p_value))
  }
}
write.csv(res, "./res/2024_1230/diff_results/Finetuning_over_RFTuning.csv")
