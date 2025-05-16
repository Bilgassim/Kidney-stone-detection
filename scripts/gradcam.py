import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pydicom import dcmread  # 🔹 1. Support DICOM ajouté

class KidneyStoneDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def __del__(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, target_class=None):
        outputs = self.model(x)
        if target_class is None:
            target_class = (torch.sigmoid(outputs) > 0.5).float().item()
        self.model.zero_grad()
        outputs.backward(retain_graph=True)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3], keepdim=True)
        self.activations *= pooled_gradients
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0)

        max_val = torch.max(heatmap)
        if max_val > 0:
            # 🔹 5. Normalisation améliorée
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap.numpy()

def apply_gradcam(model, img_path, output_dir='results/images_dicom'):
    try:
        # 🔹 6. Vérification du fichier
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Fichier introuvable : {img_path}")

        stone_dir = os.path.join(output_dir, 'stone')
        normal_dir = os.path.join(output_dir, 'normal')
        os.makedirs(stone_dir, exist_ok=True)
        os.makedirs(normal_dir, exist_ok=True)

        # 🔹 2. Lecture DICOM si besoin
        if img_path.lower().endswith('.dcm'):
            ds = dcmread(img_path)
            img_array = ds.pixel_array.astype(float)
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255
            img = Image.fromarray(img_array.astype(np.uint8)).convert('RGB')
        else:
            img = Image.open(img_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)

        target_layer = model.features[4]
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(img_tensor)

        heatmap = cv2.resize(heatmap, img.size)
        heatmap = cv2.GaussianBlur(heatmap, (9, 9), 0)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)

        img_array = np.array(img)
        superimposed_img = cv2.addWeighted(img_array, 0.7, heatmap, 0.3, 0)

        with torch.no_grad():
            output = model(img_tensor)
            raw_score = torch.sigmoid(output).item()

            # 🔹 3. Seuil + 🔹 4. Confiance cohérente
            if raw_score > 0.7:
                pred_class = 'stone'
                pred_prob = raw_score
            else:
                pred_class = 'normal'
                pred_prob = raw_score

        target_dir = stone_dir if pred_class == 'stone' else normal_dir
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(target_dir, f'{base_name}_conf_{pred_prob:.2f}.png')
        cv2.imwrite(output_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

        # Texte sur image
        cv2.putText(superimposed_img, f"{pred_class} ({pred_prob:.1%})", (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(superimposed_img, f"{pred_class} ({pred_prob:.1%})", (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        print(f"Résultat {os.path.basename(img_path)}:")
        print(f"- Classe: {pred_class}")
        print(f"- Confiance: {pred_prob:.2%}")
        print(f"- Chemin: {output_path}\n")

        return {
            'image_path': img_path,
            'prediction': pred_class,
            'confidence': pred_prob,
            'output_path': output_path
        }

    except Exception as e:
        print(f"\nERREUR sur {img_path}: {str(e)}")
        return None

def batch_gradcam(model, img_dir, output_dir='results/images_dicom'):
    img_paths = []
    valid_exts = ('.png', '.jpg', '.jpeg', '.dcm')

    for root, _, files in os.walk(img_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                img_paths.append(os.path.join(root, f))

    if not img_paths:
        print("Aucune image valide trouvée.")
        return []

    print(f"\nDébut du traitement de {len(img_paths)} images...")

    results = []
    for img_path in tqdm(sorted(img_paths), desc="Traitement", unit="img"):
        result = apply_gradcam(model, img_path, output_dir)
        if result:
            results.append(result)

    if results:
        report_path = os.path.join(output_dir, f'report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv')
        pd.DataFrame(results).to_csv(report_path, index=False)
        print(f"\nRapport généré: {report_path}")

        # 🔹 8. Statistiques
        print("\nStatistiques:")
        print(f"- Images traitées: {len(results)}")
        print(f"- Calculs détectés: {sum(1 for x in results if x['prediction'] == 'stone')}")
        print(f"- Confiance moyenne: {np.mean([x['confidence'] for x in results]):.1%}")

    return results

def main():
    # 🔹 7. Optimisation CPU
    torch.set_num_threads(4)
    os.environ['OMP_NUM_THREADS'] = '4'
    torch.backends.cudnn.benchmark = True

    MODEL_PATH = './models/best_model.pth'
    device = torch.device('cpu')

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")

        print("\nChargement du modèle...")
        model = KidneyStoneDetector().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        print("\n=== Analyse d'images médicales ===")
        print("1. Image unique")
        print("2. Dossier complet")
        print("3. Quitter")

        while True:
            choice = input("\nChoix (1-3): ").strip()

            if choice == '1':
                img_path = input("Chemin de l'image: ").strip()
                if os.path.exists(img_path):
                    apply_gradcam(model, img_path)
                else:
                    print("Fichier introuvable.")
            elif choice == '2':
                img_dir = input("Chemin du dossier: ").strip()
                if os.path.isdir(img_dir):
                    batch_gradcam(model, img_dir)
                else:
                    print("Dossier invalide.")
            elif choice == '3':
                print("Arrêt du programme.")
                break
            else:
                print("Choix invalide.")

    except Exception as e:
        print(f"\nERREUR: {str(e)}")

if __name__ == '__main__':
    main()