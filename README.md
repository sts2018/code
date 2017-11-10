This is the source code and additional material for my research project.

## Code is based on the following implementation
- https://github.com/kratzert/finetune_alexnet_with_tensorflow

## Main changes are the following
- image preprocessing: see preprocess.py and manipulate.py
- batch normalization: see alexnet.py and finetune.py

## Tools directory
- download_image.py: download sources
- overlap_analysis.py: cross-matching analysis
- select.py: randomly select a number of sources

## Catalog directory
- stripe82_catalog: each line has the following fields - RA, Dec, type (FRI/fricat, FRII/friicat, or non-FR/first)

