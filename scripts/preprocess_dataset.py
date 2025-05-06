# preprocess_dataset.py
"""
Pipeline de prétraitement et d'industrialisation pour détection de calculs rénaux.
- Validation des images corrompues
- Logging détaillé
- Support des extensions multiples
- Redimensionnement & conversion couleur/grayscale
- Split train/val/test avec vérification des ratios
- Augmentations optionnelles pour le train
- Export TFRecord (placeholder)
- Benchmarking et visualisation échantillons
"""
import os
import glob
import random
import time
import logging
from PIL import Image, ImageEnhance, ImageFile, UnidentifiedImageError
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ===== CONFIGURATION =====
RAW_DIRS = {
    "normal": r"data\Normal",
    "stone":  r"data\Stone"
}
OUTPUT_ROOT = r"processed"
TARGET_SIZE = (224, 224)       # (width, height)
COLOR_MODE = "RGB"             # "RGB" ou "L"
RATIOS = {"train":0.7, "val":0.15, "test":0.15}
EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")
RANDOM_SEED = 42
AUGMENTATIONS = {
    "train": {
        "flip_horizontal": True,
        "rotation_range": 15,
        "brightness_range": (0.8, 1.2)
    },
    "val": None,
    "test": None
}
# =========================

# Permet la lecture des images tronquées
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Création du répertoire OUTPUT avant toute écriture/LOG
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_ROOT, 'preprocess.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Vérification des ratios
if not abs(sum(RATIOS.values()) - 1.0) < 1e-6:
    raise ValueError(f"Les ratios doivent sommer à 1. Actuels: {RATIOS}")


def is_valid_image(path):
    """Vérifie si l'image est lisible et non corrompue"""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, IOError) as e:
        logger.warning(f"Fichier corrompu ignoré: {path} ({e})")
        return False


def apply_augmentation(img, aug_config):
    """Applique des augmentations simples à une image PIL"""
    if aug_config is None:
        return img
    # Flip horizontal
    if aug_config.get('flip_horizontal') and random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Rotation
    angle = random.uniform(-aug_config['rotation_range'], aug_config['rotation_range'])
    img = img.rotate(angle)
    # Brightness
    enh = ImageEnhance.Brightness(img)
    factor = random.uniform(*aug_config['brightness_range'])
    img = enh.enhance(factor)
    return img


def make_dirs():
    # Création des sous-répertoires train/val/test avec classes
    for split in RATIOS:
        for cls in RAW_DIRS:
            os.makedirs(os.path.join(OUTPUT_ROOT, split, cls), exist_ok=True)


def collect_paths(src_dir):
    paths = []
    for ext in EXTENSIONS:
        paths.extend(glob.glob(os.path.join(src_dir, ext)))
    return paths


def split_and_process():
    random.seed(RANDOM_SEED)
    stats = {split: 0 for split in RATIOS}
    start_time = time.time()

    for cls, src_dir in RAW_DIRS.items():
        paths = [p for p in collect_paths(src_dir) if is_valid_image(p)]
        random.shuffle(paths)
        n = len(paths)
        idx_train = int(RATIOS['train'] * n)
        idx_val   = idx_train + int(RATIOS['val'] * n)
        split_defs = {
            'train': paths[:idx_train],
            'val':   paths[idx_train:idx_val],
            'test':  paths[idx_val:]
        }

        for split, split_paths in split_defs.items():
            make_dirs()
            out_dir = os.path.join(OUTPUT_ROOT, split, cls)
            aug_conf = AUGMENTATIONS.get(split)
            for p in tqdm(split_paths, desc=f"{cls} → {split}", unit="img"):
                try:
                    img = Image.open(p).convert(COLOR_MODE).resize(TARGET_SIZE, Image.BILINEAR)
                    if split == 'train':
                        img = apply_augmentation(img, aug_conf)
                    name = os.path.splitext(os.path.basename(p))[0] + ".png"
                    dest = os.path.join(out_dir, name)
                    img.save(dest, 'PNG')
                    stats[split] += 1
                except Exception as e:
                    logger.error(f"Erreur traitement {p}: {e}")

    duration = time.time() - start_time
    logger.info(f"Traitement terminé en {duration:.1f}s | images par split: {stats}")
    return stats


def generate_metadata_csv():
    records = []
    for split in RATIOS:
        for cls in RAW_DIRS:
            dir_path = os.path.join(OUTPUT_ROOT, split, cls)
            for fname in os.listdir(dir_path):
                records.append({
                    'filepath': os.path.join(dir_path, fname),
                    'class': cls,
                    'split': split
                })
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_ROOT, 'metadata.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Metadata CSV saved to {csv_path} ({len(df)} entrées)")
    return df


def export_tfrecord(df):
    if not TF_AVAILABLE:
        logger.warning("TensorFlow non installé, TFRecord non généré.")
        return
    # Placeholder: développement selon besoin
    for split in df['split'].unique():
        writer = tf.io.TFRecordWriter(os.path.join(OUTPUT_ROOT, f"{split}.tfrecord"))
        writer.close()
    logger.info("TFRecord export done.")


def benchmark_visualize(df):
    # Affiche 3 échantillons traités par split
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(RATIOS), 3, figsize=(9, 3*len(RATIOS)))
    for i, split in enumerate(RATIOS):
        subset = df[df['split']==split].sample(3)
        for j, (_, row) in enumerate(subset.iterrows()):
            img = Image.open(row['filepath'])
            axes[i, j].imshow(img)
            axes[i, j].set_title(f"{split}/{row['class']}")
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


def main():
    logger.info("Démarrage du prétraitement")
    make_dirs()
    stats = split_and_process()
    df_meta = generate_metadata_csv()
    export_tfrecord(df_meta)
    logger.info(f"Structure finale: {stats}")
    # Visualisation échantillons
    try:
        benchmark_visualize(df_meta)
    except Exception as e:
        logger.warning(f"Visualisation impossible: {e}")

if __name__ == '__main__':
    main()
