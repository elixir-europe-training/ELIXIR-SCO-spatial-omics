pkgs <- c('SpatialExperiment', 'spatstat.geom', 'spatstat.explore', 
          'dplyr', 'ggplot2', 'patchwork', 'reshape2', 'Voyager', 
          'SpatialFeatureExperiment', 'SFEData', 'spdep', 'sf', 
          'stringr', 'tidyr','magrittr','scater','BiocStyle','here', 'remotes')
BiocManager::install(pkgs)
remotes::install_github("mjemons/spatialFDA")
remotes::install_github("sgunz/sosta")
