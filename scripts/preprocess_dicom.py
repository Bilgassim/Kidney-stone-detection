# preprocess_images.py
import os
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# ======= CONFIGURATION =======
NORMAL_DIR = r"data\Normal"
STONE_DIR  = r"data\Stone"
OUTPUT_DIR = r"results"

SAMPLE_SIZE = 5  # nombre d'images à afficher par classe
# =============================

def collect_image_paths(root_dir, exts=("*.jpg", "*.jpeg")):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root_dir, ext)))
    return paths

def plot_samples(normal_paths, stone_paths):
    sample_n = min(SAMPLE_SIZE, len(normal_paths))
    sample_s = min(SAMPLE_SIZE, len(stone_paths))
    normal_sample = random.sample(normal_paths, sample_n)
    stone_sample  = random.sample(stone_paths,  sample_s)

    fig, axes = plt.subplots(2, max(sample_n, sample_s), figsize=(4*max(sample_n, sample_s), 8))
    for i, p in enumerate(normal_sample):
        img = Image.open(p).convert('L')
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f"Normal\n{os.path.basename(p)}")
        axes[0, i].axis('off')
    for i, p in enumerate(stone_sample):
        img = Image.open(p).convert('L')
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f"Stone\n{os.path.basename(p)}")
        axes[1, i].axis('off')
    plt.suptitle("Échantillon Kidney CT Slices")
    plt.tight_layout()
    plt.show()

def extract_metadata(paths, label):
    rows = []
    for p in paths:
        img = Image.open(p)
        w, h = img.size
        rows.append({
            "class": label,
            "filename": os.path.basename(p),
            "width": w,
            "height": h
        })
    return rows

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    normal_paths = collect_image_paths(NORMAL_DIR)
    stone_paths  = collect_image_paths(STONE_DIR)

    if not normal_paths and not stone_paths:
        print("❌ Aucun fichier .jpg/.jpeg trouvé. Vérifiez vos chemins.")
        return

    # 1) Visualisation
    plot_samples(normal_paths, stone_paths)

    # 2) Métadonnées
    meta_norm = extract_metadata(normal_paths, "normal")
    meta_stone= extract_metadata(stone_paths,  "stone")
    df = pd.DataFrame(meta_norm + meta_stone)

    # 3) Résumé
    summary = df.groupby("class").agg(
        count   = ("filename", "count"),
        avg_w   = ("width",    "mean"),
        avg_h   = ("height",   "mean")
    ).reset_index()

    # 4) Affichage console
    print("\n===== Résumé des dimensions =====")
    print(summary.to_string(index=False))
    print("\n===== Aperçu des 10 premières lignes =====")
    print(df.head(10).to_string(index=False))

    # 5) Sauvegarde CSV
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_metadata.csv"), index=False)
    df.to_csv(   os.path.join(OUTPUT_DIR, "all_metadata.csv"),     index=False)
    print(f"\n✅ CSV sauvegardés dans {OUTPUT_DIR}")

if __name__ == "__main__":
    main()