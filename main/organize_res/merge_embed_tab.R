rm(list = ls())
set.seed(7)

library(dplyr)

# do_CV <- "CV" # "CV NoCV
# embedding_type <- "text_embedding" # "name_embedding" "text_embedding"
# do_truncation <- "NoTruncation"
# 
# SLLMs_tab <- read.csv(paste0("./res/2025_0601/", "NoPCA_", do_CV, "_", 
#                              embedding_type, "_", do_truncation, ".csv"))
# GenePT_tab <- read.csv(paste0("./res/2025_0601/", "NoPCA_", do_CV, "_", 
#                               embedding_type, "_", do_truncation, "_GenePT.csv"))
# 
# final_tab <- rbind(SLLMs_tab, GenePT_tab)
# 
# save_path <- paste0("./res/2025_0601/", "NoPCA_", do_CV, "_", 
#                     embedding_type, "_", do_truncation, "_final.csv")
# write.csv(final_tab, save_path, row.names = FALSE)


# Merge the embedding results for the multi class classification problem
multi_class_emb_results <- read.csv("./res/2025_0603_MultiClass/NoPCA_CV_text_embedding_NoTruncation.csv")
original_emb_results <- read.csv("./res/2025_0601/NoPCA_CV_text_embedding_NoTruncation_final.csv")

final_tab <- rbind(original_emb_results, multi_class_emb_results)
save_path <- "./res/2025_0603_All_Num_Res/NoPCA_CV_text_embedding_NoTruncation_final.csv"
write.csv(final_tab, save_path, row.names = FALSE)


multi_class_emb_results <- read.csv("./res/2025_0603_MultiClass/NoPCA_NoCV_text_embedding_NoTruncation.csv")
original_emb_results <- read.csv("./res/2025_0601/NoPCA_NoCV_text_embedding_NoTruncation_final.csv")

final_tab <- rbind(original_emb_results, multi_class_emb_results)
save_path <- "./res/2025_0603_All_Num_Res/NoPCA_NoCV_text_embedding_NoTruncation_final.csv"
write.csv(final_tab, save_path, row.names = FALSE)