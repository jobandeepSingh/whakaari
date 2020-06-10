library(VennDiagram)

# get file names
all_feats_files = Sys.glob("../features/feats_p-values*.fts")
rel_feats_files = Sys.glob("../features/relevant*.fts")

eruptions <- c("2012-08-04", "2013-10-03", "2013-08-19", 
               "2016-04-27", "2019-12-09")

# read in the files
all_feats <- lapply(all_feats_files, read.table, sep=" ", header=TRUE)
rel_feats <- lapply(rel_feats_files, read.table, sep=" ", header=TRUE)


char_rel_feats <- lapply(rel_feats, unlist, use.names=FALSE)
venn.diagram(x = char_rel_feats, category.names=eruptions, 
             filename = "relevant_features.png",
             imagetype = "png")


# turn into df
# all_feats <- lapply(all_feats, data.frame)
get_features <- function(feats_df) {
  feats_df <- data.frame(feats_df)
  feats_df <- na.omit(feats_df)
  idx <- feats_df$p_values < 0.05
  feats_df$features[idx]
}

feats_sig <- lapply(all_feats, get_features)
venn.diagram(x = feats_sig, category.names=eruptions, 
             filename = "over_0.05_features.png",
             imagetype = "png")
