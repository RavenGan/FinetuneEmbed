rm(list = ls())
set.seed(7)

library(dplyr)

load_tab <- function(path2tab, remove_col){
  tab <- read.csv(path2tab)
  
  # get the row names
  rownames(tab) <- tab[, 1]
  tab <- tab[, -1]
  
  # remove specified columns
  tab <- tab[, -remove_col] %>%
    as.matrix()
  return(tab)
}

get_task_test <- function(tab){
  mean_ls <- c()
  pval_ls <- c()
  for (i in 1:ncol(tab)) {
    values <- tab[, i] # use the i-th column
    mean <- round(mean(values), 4)
    test_res <- t.test(values, mu = 0)
    # test_res <- wilcox.test(values, mu = 0)
    pval <- round(test_res$p.value, 4)
    
    mean_ls <- c(mean_ls, mean)
    pval_ls <- c(pval_ls, pval)
  }
  tab <- rbind(tab, mean_ls)
  rownames(tab)[nrow(tab)] <- "task_mean"
  tab <- rbind(tab, pval_ls)
  rownames(tab)[nrow(tab)] <- "task_pval"
  return(tab)
}

get_mod_test <- function(tab){
  mean_ls <- c()
  pval_ls <- c()
  for (i in 1:nrow(tab)) {
    values <- tab[i, ] # use the i-th row
    mean <- round(mean(values), 4)
    test_res <- t.test(values, mu = 0)
    # test_res <- wilcox.test(values, mu = 0)
    pval <- round(test_res$p.value, 4)
    
    mean_ls <- c(mean_ls, mean)
    pval_ls <- c(pval_ls, pval)
  }
  tab <- cbind(tab, mean_ls)
  colnames(tab)[ncol(tab)] <- "mod_mean"
  tab <- cbind(tab, pval_ls)
  colnames(tab)[ncol(tab)] <- "mod_pval"
  return(tab)
}



tab_save_path <- "./res/2024_1226/"

# NoTuning, compare LR and RF
sd_cols <- c(2, 4, 6, 8)
NoTuning_LR <- load_tab("./res/2024_1225/csv/MultiMod_test_811_NoTuning_LR.csv",
                        sd_cols)
NoTuning_RF <- load_tab("./res/2024_1225/csv/MultiMod_test_811_NoTuning_RF.csv",
                        sd_cols)
diff_NoTuning_LRRF <- NoTuning_LR - NoTuning_RF

diff_NoTuning_LRRF <- get_task_test(diff_NoTuning_LRRF)
diff_NoTuning_LRRF <- get_mod_test(diff_NoTuning_LRRF)

diff_NoTuning_LRRF <- round(diff_NoTuning_LRRF, 4)
write.csv(diff_NoTuning_LRRF, paste0(tab_save_path,
                                     "diff_NoTuning_LRRF.csv"))

# Tuning, compare LR and RF
sd_cols <- c(2, 4, 6, 8)
Tuning_LR <- load_tab("./res/2024_1225/csv/MultiMod_test_811_Tuning_LR.csv",
                        sd_cols)
Tuning_RF <- load_tab("./res/2024_1225/csv/MultiMod_test_811_Tuning_RF.csv",
                        sd_cols)
diff_Tuning_LRRF <- Tuning_LR - Tuning_RF

diff_Tuning_LRRF <- get_task_test(diff_Tuning_LRRF)
diff_Tuning_LRRF <- get_mod_test(diff_Tuning_LRRF)

diff_Tuning_LRRF <- round(diff_Tuning_LRRF, 4)
write.csv(diff_Tuning_LRRF, paste0(tab_save_path,
                                     "diff_Tuning_LRRF.csv"))

# LR, compare NoTuning and Tuning
diff_LR_TuningOrNot <- Tuning_LR - NoTuning_LR

diff_LR_TuningOrNot <- get_task_test(diff_LR_TuningOrNot)
diff_LR_TuningOrNot <- get_mod_test(diff_LR_TuningOrNot)

diff_LR_TuningOrNot <- round(diff_LR_TuningOrNot, 4)
write.csv(diff_LR_TuningOrNot, paste0(tab_save_path,
                                   "diff_LR_TuningOrNot.csv"))

# RF, compare NoTuning and Tuning
diff_RF_TuningOrNot <- Tuning_RF - NoTuning_RF

diff_RF_TuningOrNot <- get_task_test(diff_RF_TuningOrNot)
diff_RF_TuningOrNot <- get_mod_test(diff_RF_TuningOrNot)

diff_RF_TuningOrNot <- round(diff_RF_TuningOrNot, 4)
write.csv(diff_RF_TuningOrNot, paste0(tab_save_path,
                                      "diff_RF_TuningOrNot.csv"))

# Tuning, compare LR, RF and Finetune
Finetuning <- load_tab("./res/2024_1225/csv/MultiMod_test_811_Finetuning.csv",
                       sd_cols)
Tuning_LR_NoGenePT <- Tuning_LR[-1, ]
Tuning_RF_NoGenePT <- Tuning_RF[-1, ]

diff_Tuning_LRFinetune <- Finetuning - Tuning_LR_NoGenePT

diff_Tuning_LRFinetune <- get_task_test(diff_Tuning_LRFinetune)
diff_Tuning_LRFinetune <- get_mod_test(diff_Tuning_LRFinetune)

diff_Tuning_LRFinetune <- round(diff_Tuning_LRFinetune, 4)
write.csv(diff_Tuning_LRFinetune, paste0(tab_save_path,
                                      "diff_Tuning_LRFinetune.csv"))
diff_Tuning_RFFinetune <- Finetuning - Tuning_RF_NoGenePT

diff_Tuning_RFFinetune <- get_task_test(diff_Tuning_RFFinetune)
diff_Tuning_RFFinetune <- get_mod_test(diff_Tuning_RFFinetune)

diff_Tuning_RFFinetune <- round(diff_Tuning_RFFinetune, 4)
write.csv(diff_Tuning_RFFinetune, paste0(tab_save_path,
                                         "diff_Tuning_RFFinetune.csv"))
