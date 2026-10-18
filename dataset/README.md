# Dataset: HAM10000

This project uses HAM10000, a public collection of 10,015 dermoscopic images of 7,470 lesions in seven diagnostic classes.

The images themselves are not stored in this repository because the archive is 5.2 GB, far above GitHub's file size limit. Download it from either source below.

- Kaggle (the `archive.zip` used by the notebook): https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- Original release on Harvard Dataverse: https://doi.org/10.7910/DVN/DBW86T

## Files here

- `HAM10000_metadata.csv`: one row per image, with the lesion ID, image ID, diagnosis (`dx`), how the diagnosis was confirmed, age, sex and body site. The notebook uses `lesion_id` to split the data by lesion (70/15/15, seed 42), so this file is enough to reproduce the exact training, validation and test split.

## Using it with the notebook

Put `archive.zip` in Google Drive. The notebook copies it to the Colab disk, extracts it and checks that all 10,015 images are present before training.

## Citation

Tschandl, P., Rosendahl, C., Kittler, H.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data 5, 180161 (2018)
