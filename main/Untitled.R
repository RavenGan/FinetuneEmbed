
library(depmap)

crispr_data <- depmap_crispr()

x11()
hist(crispr_data$dependency, breaks = 100)



# Access RNAi-based dependency data (including DRIVE)
rnai_data <- depmap_rnai()
