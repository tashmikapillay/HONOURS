# Automated Skin Disease Detection from Dermoscopic Images with Explainability Analysis

BSc Honours project in Computer Science, University of KwaZulu-Natal.
Student: Tashmika Pillay. Supervisor: Prof Serestina Viriri.

A lightweight hybrid model (MobileNetV3-Small backbone and a four layer Transformer encoder) for seven class skin lesion classification on HAM10000, with Grad-CAM, attention and SHAP explanations, a quantitative faithfulness test, and a Streamlit demonstration application.

## Contents

| Folder | What it holds |
|---|---|
| `notebook/` | `skin_lesion_hybrid_final_hons.ipynb`, the final notebook. Every result in the report comes from this single executed run (Google Colab, Tesla T4). |
| `paper/` | `paper.tex`, the report in Springer LNCS format. |
| `figures/` | The figures used in the report, plus other plots produced by the notebook. |
| `app/` | The Streamlit application (`app.py`), its requirements and the trained model (`hybrid_v2_best.pth`). |
| `sample_images/` | A few dermoscopic images for trying the application. |
| `earlier_versions/` | Earlier notebooks and figures from before the final run, kept for reference. |

## Main results (test set, split by lesion)

| Model | Input | Accuracy | Balanced accuracy | Macro F1 | Macro AUROC | Parameters |
|---|---|---|---|---|---|---|
| Hybrid | 448 | 0.821 | 0.729 | 0.699 | 0.946 | 8.33M |
| CNN only | 448 | 0.784 | 0.743 | 0.711 | 0.952 | 1.15M |

The comparison with EfficientNet-B3 and ViT-B/16 (three seeds at 224 pixels), the ablation study and the explainability results are in the notebook and the report.

## Running the notebook

Open the notebook in Google Colab with a GPU runtime. It expects the HAM10000 archive (`archive.zip`, available from Kaggle) in Google Drive and runs from top to bottom.

## Running the application

```
cd app
pip install -r requirements.txt
streamlit run app.py
```

The application is a research demonstration and is not validated for clinical use.

## Use of AI

Generative AI tools (Claude and the Overleaf AI assistant) were used during this project, as set out in the UKZN AI declaration submitted with the report.
