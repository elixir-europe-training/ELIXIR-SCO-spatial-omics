#!/usr/bin/env python

import tifffile

with tifffile.TiffFile('data/xenium_2.0.0_io/morphology.ome.tif') as tif:
    for tag in tif.pages[0].tags.values():
        if tag.name == "ImageDescription":
            print(tag.name+":", tag.value)