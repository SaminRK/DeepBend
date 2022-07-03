# Usage:
# Rscript shape.r /home/sakib/playground/machine_learning/bendability/data/rl_most_1000.txt

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("DNAshapeR")

library(DNAshapeR)

args = commandArgs(trailingOnly=TRUE)

fn <- args[1]
pred <- getShape(fn)

# pred["Roll"]


