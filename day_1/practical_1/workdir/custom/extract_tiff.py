#!/usr/bin/env python

import tifffile

# Variable 'LEVEL' determines the level to extract. It ranges from 0 (highest
# resolution) to 6 (lowest resolution) for morphology.ome.tif
LEVEL = 2

with tifffile.TiffFile('data/xenium_2.0.0_io/morphology.ome.tif') as tif:
    image = tif.series[0].levels[LEVEL].asarray()

tifffile.imwrite(
    'level_'+str(LEVEL)+'_morphology.ome.tif',
    image,
    photometric='minisblack',
    dtype='uint16',
    tile=(1024, 1024),
    compression='JPEG_2000_LOSSY',
    metadata={'axes': 'ZYX'},
)