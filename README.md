# Kidney-stone-detection
Nous allons en fait creer un modele capable de detecter les calculs renaux
Structure recommandé du repository :

```
kidney-stone-detection/
├── data/...                      # Dossier pour stocker les images (localement ou lien vers drive)
├── notebooks/...                 # Notebooks Jupyter d'exploration ou de test
├── scripts/                   # Scripts Python (prétraitement, utils…)
│   └── preprocess_dicom.py
│   └── preprocess_dataset.py ...
├── results/...                   # Cartes de chaleur, visualisations, logs…
├── README.md                  # Présentation du projet
├── requirements.txt           # Dépendances Python
└── .gitignore                 # Fichiers à ignorer (DICOM, outputs lourds…)
Kidney-stone-detection/
├── .gitignore
├── Kidney-stone-detection.iml
├── misc.xml
├── modules.xml
├── vcs.xml
├── data/
│   ├── Normal/                # Images de reins normaux
│   ├── stone/                 # Images de reins avec calculs rénaux
│   └── ...                    # Autres sous-dossiers dupliqués si présents
├── checkpoints/
│   └── training.log           # Fichier de log d'entraînement
├── val/
│   ├── metadata.csv
│   └── preprocess.log
├── results/
│   ├── Figure_1.png
│   ├── all_metadata.csv
│   ├── summary_metadata.csv
│   └── training_log.json
├── scripts/                   # Scripts Python
├── README.md
└── requirements.txt
