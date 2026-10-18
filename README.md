# Automated Skin Disease Detection from Dermoscopic Images with Explainability Analysis

BSc Honours project in Computer Science, University of KwaZulu-Natal.
Student: Tashmika Pillay. Supervisor: Prof Serestina Viriri.

A lightweight hybrid model (MobileNetV3-Small backbone and a four layer Transformer encoder) for seven class skin lesion classification on HAM10000, with Grad-CAM, attention and SHAP explanations, a quantitative faithfulness test, and a Streamlit demonstration application ([live demo](https://skinapp-3q2qxhmcjkqzxh8yvpfmpw.streamlit.app/)).

## Contents

| Folder | What it holds |
|---|---|
| `notebook/` | `skin_lesion_hybrid_final_hons.ipynb`, the final notebook. Every result in the report comes from this single executed run (Google Colab, Tesla T4). |
| `paper/` | `paper.tex`, the report in Springer LNCS format. |
| `figures/` | The figures used in the report, plus other plots produced by the notebook. |
| `app/` | The Streamlit application (`app.py`), its requirements and the trained model (`hybrid_v2_best.pth`). |
| `sample_images/` | A few dermoscopic images for trying the application. |
| `dataset/` | Download links for HAM10000 and its metadata file, which defines the lesion split. The images are not stored here (5.2 GB). |
| `earlier_versions/` | Earlier notebooks and figures from before the final run, kept for reference. |

## Main results

All results are on the held out test set (1,516 images). The data are split by lesion, so no lesion appears in more than one set. Always predicting the most common class (nevus) would give an accuracy of 0.672.

### Main model at 448 pixels

| Model | Accuracy | Balanced accuracy | Macro F1 | Macro AUROC | Cohen's kappa | Parameters |
|---|---|---|---|---|---|---|
| Hybrid | 0.821 | 0.729 | 0.699 | 0.946 | 0.657 | 8.33M |
| CNN only (no Transformer) | 0.784 | 0.743 | 0.711 | 0.952 | 0.621 | 1.15M |

95% lesion bootstrap intervals for the hybrid: accuracy 0.796 to 0.844, balanced accuracy 0.677 to 0.779, macro AUROC 0.918 to 0.964.

### Comparison under an equal budget (224 pixels, mean ± standard deviation over three seeds)

| Model | Accuracy | Balanced accuracy | Macro F1 | Macro AUROC | Cohen's kappa | Parameters |
|---|---|---|---|---|---|---|
| EfficientNet-B3 | 0.814 ± 0.014 | 0.718 ± 0.014 | 0.714 ± 0.026 | 0.957 ± 0.005 | 0.651 ± 0.025 | 10.71M |
| ViT-B/16 | 0.782 ± 0.015 | 0.649 ± 0.004 | 0.661 ± 0.022 | 0.942 ± 0.007 | 0.598 ± 0.013 | 85.80M |
| CNN only | 0.773 ± 0.017 | 0.693 ± 0.016 | 0.662 ± 0.018 | 0.941 ± 0.002 | 0.592 ± 0.024 | 1.15M |
| Hybrid | 0.751 ± 0.030 | 0.682 ± 0.007 | 0.630 ± 0.019 | 0.935 ± 0.008 | 0.555 ± 0.040 | 8.27M |

EfficientNet-B3 is best on every metric. The hybrid is close to ViT-B/16 at about a tenth of its size, ahead on balanced accuracy and behind on accuracy and macro F1.

### Ablation (224 pixels, one seed, one change per row)

| Configuration | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---|---|
| Reference pipeline | 0.763 | 0.679 | 0.638 |
| No ImageNet pretraining | 0.631 | 0.535 | 0.420 |
| Sampler and weighted loss (imbalance corrected twice) | 0.575 | 0.695 | 0.521 |
| No imbalance handling | 0.808 | 0.611 | 0.636 |
| CNN only (no Transformer) | 0.786 | 0.704 | 0.673 |
| No hair removal | 0.749 | 0.690 | 0.645 |
| Aggressive hair removal | 0.685 | 0.605 | 0.510 |

Pretraining matters most, correcting for imbalance twice drops accuracy below the majority baseline, aggressive hair removal hurts, and removing the Transformer changes little.

### Efficiency (one image)

| Model | Input | Parameters | GFLOPs | GPU latency (T4) | CPU latency |
|---|---|---|---|---|---|
| Hybrid | 448 | 8.33M | 3.31 | 8.1 ms | 51 ms |
| CNN only | 448 | 1.15M | 0.52 | 5.2 ms | 19 ms |
| EfficientNet-B3 | 224 | 10.71M | 1.92 | 12.8 ms | 65 ms |
| ViT-B/16 | 224 | 85.80M | 33.70 | 19.1 ms | 482 ms |

### Explainability (faithfulness on 164 test images, area under the curve)

| Method | Deletion (lower is better) | Insertion (higher is better) |
|---|---|---|
| Grad-CAM | 0.302 | 0.694 |
| Attention | 0.296 | 0.641 |
| Centre prior | 0.311 | 0.594 |
| Random | 0.245 | 0.381 |

Grad-CAM and attention beat a simple centre prior on insertion (p < 0.0001) but not on deletion.

### Clinical reading and calibration

- Melanoma sensitivity 0.566 and specificity 0.951; 43 of 145 melanomas were classified as nevus.
- Cancer (melanoma and basal cell carcinoma) against the rest: sensitivity 0.631, specificity 0.931.
- Expected calibration error 0.053, so the predicted probabilities are close to observed frequencies.

### Few shot, zero shot and the input check

- Three way few shot accuracy on unseen rare classes: 47.4% with one example and 65.6% with ten (ImageNet features alone: 44.8% and 62.0%; chance 33.3%).
- CLIP zero shot balanced accuracy 0.278.
- As an input check in the app, CLIP accepts 98.2% of real lesion images but rejects only 52.0% of unrelated CIFAR-10 photos.

Full details, figures and discussion are in the notebook and the report.

## Running the notebook

Open the notebook in Google Colab with a GPU runtime. It expects the HAM10000 archive (`archive.zip`, available from Kaggle) in Google Drive and runs from top to bottom.

## Try the application

Live demo: https://skinapp-3q2qxhmcjkqzxh8yvpfmpw.streamlit.app/

Upload a dermoscopic image (the files in `sample_images/` work) to see the predicted class, the probability of each class and a Grad-CAM overlay.

## Running the application locally

```
cd app
pip install -r requirements.txt
streamlit run app.py
```

The application is a research demonstration and is not validated for clinical use.

## Use of AI

Generative AI tools (Claude and the Overleaf AI assistant) were used during this project, as set out in the UKZN AI declaration submitted with the report.
