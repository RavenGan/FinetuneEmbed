rm(list = ls())
set.seed(7)

library(dplyr)

tasks <- c("LongShortTF", "Sensitivity",
           "BivalentNoMethyl", "BivalentLys4")

save_mod_names <- c("biobert-base-cased-v1.1",
                  'NoInstruct-small-Embedding-v0',
                  'GIST-small-Embedding-v0',
                  'stella-base-en-v2',
                  'bge-small-en-v1.5',
                  'MedEmbed-small-v0.1',
                  'gte-small',
                  'e5-small-v2',
                  'GIST-all-MiniLM-L6-v2',
                  'e5-small',
                  'gte-tiny')

root_dir <- "./res/2025_0528_finetune_11SLLMs_4tasks/"
embedding_type <- "Gene_Text" # "Gene_Name" # "Gene_Text"

df_summary_all <- c()

for (i in 1:length(save_mod_names)){
  mod_name <- save_mod_names[i]
  for (j in 1:length(tasks)) {
    task <- tasks[j]
    
    csv_path <- paste0(root_dir, embedding_type, "/", 
                       mod_name, "/", task, ".csv")
    
    res_tab <- read.csv(csv_path)
    
    df_summary <- res_tab %>%
      summarise(
        AUC_mean = round(mean(AUC, na.rm = TRUE), 3),
        AUC_sd   = round(sd(AUC, na.rm = TRUE), 3),
        Precision_mean = round(mean(Precision, na.rm = TRUE), 3),
        Precision_sd   = round(sd(Precision, na.rm = TRUE), 3),
        Recall_mean = round(mean(Recall, na.rm = TRUE), 3),
        Recall_sd   = round(sd(Recall, na.rm = TRUE), 3),
        F1_mean = round(mean(F1, na.rm = TRUE), 3),
        F1_sd   = round(sd(F1, na.rm = TRUE), 3)
      )
    df_summary$LLM <- mod_name
    df_summary$task <- task
    
    df_summary_all <- rbind(df_summary_all, df_summary)
  }
}

save_path <- paste0("./res/2025_0601/", "finetune_", embedding_type, ".csv")
write.csv(df_summary_all, save_path, row.names = FALSE)

