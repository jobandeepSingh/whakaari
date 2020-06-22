library(VennDiagram)

eruptions <- c("2012-8-4", "2013-10-3", "2013-8-19", 
               "2016-4-27", "2019-12-9")

# gets the eruptions from given list of file names
# to be used with relevant and feat_p-value files
get_erps <- function(filenames){
  erps = vector(mode="character", length=length(filenames))
  files_split <- strsplit(filenames, "_")
  i = 1
  for (f in files_split){
    erps[i] = strsplit(f, ".fts")[[3]][1]
    i=i+1
  }
  erps
} 

folder = "relevant_feats_venn_diagrams"
dir.create(paste("../plots/", folder, sep=""), showWarnings = FALSE)

for (erp in eruptions){
  # get file names
  rel_feats_files = Sys.glob(paste("../features/",erp,"/relevant*.fts", sep=""))
  
  
  # read in the files

  rel_feats <- lapply(rel_feats_files, read.table, sep=" ", header=TRUE)
  
  # convert
  char_rel_feats <- lapply(rel_feats, unlist, use.names=FALSE)
  
  # eruptions to use as titles in Venn Diagram
  erp_left <- get_erps(rel_feats_files)
  
  venn.diagram(x = char_rel_feats, category.names=erp_left, 
               filename = paste("../plots/", folder, "/",erp,"_relevant_features.png", sep=""),
               imagetype = "png")
}

# remove logs created when making venn diagrams
logs <- Sys.glob(paste("../plots/", folder, "/","*.log", sep=""))
file.remove(logs)


# all_feats_files = Sys.glob(paste("../features/",erp,"/feats_p-values*.fts"))
# all_feats <- lapply(all_feats_files, read.table, sep=" ", header=TRUE)

# # turn into df
# # all_feats <- lapply(all_feats, data.frame)
# get_features <- function(feats_df) {
#   feats_df <- data.frame(feats_df)
#   feats_df <- na.omit(feats_df)
#   idx <- feats_df$p_values < 0.05
#   feats_df$features[idx]
# }
# 
# feats_sig <- lapply(all_feats, get_features)
# venn.diagram(x = feats_sig, category.names=eruptions, 
#              filename = "over_0.05_features.png",
#              imagetype = "png")
