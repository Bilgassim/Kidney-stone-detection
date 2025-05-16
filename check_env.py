import sys

print("✅ Interpréteur Python utilisé :")
print(sys.executable)
print()

modules = {
    "torch": "torch.__version__",
    "torchvision": "torchvision.__version__",
    "numpy": "np.__version__",
    "PIL": "Image.__version__",
    "cv2": "cv2.__version__",
    "matplotlib": "plt.__version__",
    "tqdm": "tqdm.__version__",
    "sklearn": "sklearn.__version__",
    "seaborn": "sns.__version__"
}

print("🔍 Vérification des modules :\n")

for name, version_attr in modules.items():
    try:
        if name == "PIL":
            from PIL import Image
            ver = Image.__version__
        elif name == "torch":
            import torch
            ver = torch.__version__
        elif name == "torchvision":
            import torchvision
            ver = torchvision.__version__
        elif name == "numpy":
            import numpy as np
            ver = np.__version__
        elif name == "cv2":
            import cv2
            ver = cv2.__version__
        elif name == "matplotlib":
            import matplotlib.pyplot as plt
            ver = plt.__version__
        elif name == "tqdm":
            from tqdm import tqdm
            ver = tqdm.__version__
        elif name == "sklearn":
            import sklearn
            ver = sklearn.__version__
        elif name == "seaborn":
            import seaborn as sns
            ver = sns.__version__
        print(f"✅ {name:12} : OK (v{ver})")
    except Exception as e:
        print(f"❌ {name:12} : Erreur - {e}")

print("\n✔️ Environnement vérifié.")
