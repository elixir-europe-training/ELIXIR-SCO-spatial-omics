pkgs <- c('SpatialExperiment', 'spatstat.geom', 'spatstat.explore', 
          'dplyr', 'ggplot2', 'patchwork', 'reshape2', 'Voyager', 
          'SpatialFeatureExperiment', 'SFEData', 'spdep', 'sf', 
          'stringr', 'tidyr','magrittr','scater','BiocStyle','here')
BiocManager::install(pkgs)
devtools::install_github("mjemons/spatialFDA")
devtools::install_github("sgunz/sosta")
