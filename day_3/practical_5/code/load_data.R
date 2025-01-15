
#### Code from Helena Crowell ####
eh <- ExperimentHub()
q <- query(eh, "MERFISH")
df <- eh[["EH7546"]]

# extract cell metadata
i <- seq_len(9)
cd <- data.frame(df[, i], row.names = 1)

# set sample identifiers
id <- grep("Bregma", names(cd))
names(cd)[id] <- "sample_id"

# rename spatial coordinates
xy <- grep("Centroid", names(cd))
xy <- names(cd)[xy] <- c("x", "y")

# simplify annotations
cd$cluster_id <- cd$Cell_class
for (. in c("Endothelial", "OD Mature", "OD Immature"))
  cd$cluster_id[grep(., cd$cluster_id)] <- .

# extract & sparsify assay data
y <- data.frame(df[, -i], row.names = df[, 1])
y <- as(t(as.matrix(y)), "dgCMatrix")

# construct SPE
(spe <- SpatialExperiment(
  assays = list(exprs  = y),
  spatialCoordsNames = xy,
  colData = cd))

# Define the directory and file paths
dir_path <- paste0(basedir, "/data")
file_path <- file.path(dir_path, "spe.rds")

# Check if the directory exists, and create it if it doesn't
if (!dir.exists(dir_path)) {
  dir.create(dir_path)
}

#save data as rds file
saveRDS(spe,  file_path)

# remove some objects to not clutter
rm(cd, df, eh, q, y, dir_path, file_path, i, id, xy); gc()
