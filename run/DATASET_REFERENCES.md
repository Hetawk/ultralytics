# MedDef-VISTA Dataset References

Three datasets used for training and evaluation. All pre-processed to 224×224 RGB, split train/val/test.

---

## 1. TBCR — Tuberculosis Chest X-Ray

| Field              | Value                                                                             |
| ------------------ | --------------------------------------------------------------------------------- |
| **Name**           | Tuberculosis (TB) Chest X-Ray Dataset                                             |
| **Classes**        | 2 (Normal, Tuberculosis)                                                          |
| **Total images**   | 4,200 (700 Normal + 3,500 TB)                                                     |
| **Source**         | Kaggle                                                                            |
| **URL**            | https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset |
| **Train script**   | `run/train_tbcr_final.sh`                                                         |
| **Distill script** | `run/distill_tbcr_final.sh`                                                       |

**Citation:**

```
Rahman, T., Khandakar, A., Kadir, M. A., Islam, K. R., Islam, K. F., Mazhar, R.,
Hamid, T., Islam, M. T., Kashem, S., Mahbub, Z. B., Khandakar, M. E. H., Tahir, A.,
& Chowdhury, M. E. H. (2020).
Reliable Tuberculosis Detection Using Chest X-Ray with Deep Learning, Segmentation
and Visualization. IEEE Access, 8, 191586–191601.
https://doi.org/10.1109/ACCESS.2020.3031257
```

---

## 2. Chest X-Ray — Pneumonia Detection

| Field                   | Value                                                                  |
| ----------------------- | ---------------------------------------------------------------------- |
| **Name**                | Chest X-Ray Images (Pneumonia)                                         |
| **Classes**             | 2 (NORMAL, PNEUMONIA)                                                  |
| **Total images**        | 5,856 (train 4,099 · val 878 · test 879)                               |
| **Class split (train)** | NORMAL: 1,108 · PNEUMONIA: 2,991                                       |
| **Source**              | Kaggle                                                                 |
| **URL**                 | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| **Train script**        | `run/train_chest_xray_final.sh`                                        |
| **Distill script**      | `run/distill_chest_xray_final.sh`                                      |

**Citation:**

```
Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C., Liang, H., Baxter, S. L.,
McKeown, A., Yang, G., Wu, X., Yan, F., Dong, J., Prasadha, M. K., Pei, J., Ting,
M. Y. L., Zhu, J., Li, C., Hewett, S., Dong, J., Ziyar, I., … Zhang, K. (2018).
Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.
Cell, 172(5), 1122–1131.e9.
https://doi.org/10.1016/j.cell.2018.02.010
```

---

## 3. SCISIC — Skin Cancer (ISIC, 9-class)

| Field              | Value                                                                                                                                                                    |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Name**           | Skin Cancer 9-Classes (ISIC)                                                                                                                                             |
| **Classes**        | 9 (actinic keratosis, nevus, melanoma, vascular lesion, dermatofibroma, pigmented benign keratosis, seborrheic keratosis, basal cell carcinoma, squamous cell carcinoma) |
| **Total images**   | 2,357 (train 1,882 · val 235 · test 240)                                                                                                                                 |
| **Source**         | Kaggle (ISIC archive)                                                                                                                                                    |
| **URL**            | https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic                                                                                                     |
| **Train script**   | `run/train_scisic_final.sh`                                                                                                                                              |
| **Distill script** | `run/distill_scisic_final.sh`                                                                                                                                            |

**Citation:**

```
ISIC Archive. International Skin Imaging Collaboration.
https://www.isic-archive.com

Codella, N. C. F., Gutman, D., Celebi, M. E., Helba, B., Marchetti, M. A., Dusza,
S. W., Kalloo, A., Liopyris, K., Mishra, N., Kittler, H., & Halpern, A. (2018).
Skin Lesion Analysis Toward Melanoma Detection: ISIC 2017 Challenge.
IEEE ISBI 2018, 168–172.
https://doi.org/10.1109/ISBI.2018.8363547
```

---

## Server Paths

```
/data2/enoch/ekd_coding_env/meddef_winlab/dataset/tbcr/
/data2/enoch/ekd_coding_env/meddef_winlab/dataset/chest_xray/
/data2/enoch/ekd_coding_env/meddef_winlab/dataset/scisic/

/data2/enoch/ekd_coding_env/meddef_winlab/processed_data/tbcr/
/data2/enoch/ekd_coding_env/meddef_winlab/processed_data/chest_xray/
/data2/enoch/ekd_coding_env/meddef_winlab/processed_data/scisic/
```

## Training Output Paths (server)

```
/data2/enoch/ekd_coding_env/ultralytics/runs/classify/train_tbcr_final/tbcr/
/data2/enoch/ekd_coding_env/ultralytics/runs/classify/train_chest_xray_final/chest_xray/
/data2/enoch/ekd_coding_env/ultralytics/runs/classify/train_scisic_final/scisic/
```
