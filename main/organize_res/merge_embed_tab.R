rm(list = ls())
set.seed(7)

library(dplyr)

do_CV <- "CV" # "CV NoCV
embedding_type <- "text_embedding" # "name_embedding" "text_embedding"
do_truncation <- "NoTruncation"

SLLMs_tab <- read.csv(paste0("./res/2025_0601/", "NoPCA_", do_CV, "_", 
                             embedding_type, "_", do_truncation, ".csv"))
GenePT_tab <- read.csv(paste0("./res/2025_0601/", "NoPCA_", do_CV, "_", 
                              embedding_type, "_", do_truncation, "_GenePT.csv"))

final_tab <- rbind(SLLMs_tab, GenePT_tab)

save_path <- paste0("./res/2025_0601/", "NoPCA_", do_CV, "_", 
                    embedding_type, "_", do_truncation, "_final.csv")
write.csv(final_tab, save_path, row.names = FALSE)
