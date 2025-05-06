import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import time
import os
import numpy as np
from multiprocessing import freeze_support

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

def main():
    freeze_support()  # Pour Windows/CPU

    # Détection de device (CPU ou GPU futur)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")

    # Config
    DATA_DIR = "./processed"
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    IMG_SIZE = 64
    PATIENCE = 5

    # Augmentations améliorées
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), val_transform)

    # DataLoaders optimisés CPU
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

    # Modèle
    model = KidneyStoneDetector().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    best_val_acc = 0.0
    patience_counter = 0

    # Dossier pour sauvegarder
    os.makedirs("./models", exist_ok=True)

    # Entraînement
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()

        val_acc = val_correct / len(val_dataset)
        print(f"[Epoch {epoch+1}] Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%}")

        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), './models/best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping à l'epoch {epoch+1}")
            break

    # Chargement du meilleur
    model.load_state_dict(torch.load('./models/best_model.pth', map_location=device))

    # Test
    print("\nÉvaluation sur le test set...")
    plot_confusion_matrix(model, test_loader, device)

def plot_confusion_matrix(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Évaluation"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Binariser à 0.5
    bin_preds = (np.array(all_preds) > 0.5).astype(int)

    # Métriques
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
    cm = confusion_matrix(all_labels, bin_preds)
    report = classification_report(all_labels, bin_preds, target_names=["Normal", "Stone"])
    auc = roc_auc_score(all_labels, all_preds)

    print("\n=== Rapport de classification ===")
    print(report)
    print(f"AUC-ROC: {auc:.4f}")
    print("\nMatrice de confusion :")
    print(cm)

    # Visualisation
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=["Normal", "Stone"], 
                    yticklabels=["Normal", "Stone"])
        plt.xlabel('Prédictions')
        plt.ylabel('Véritables étiquettes')
        plt.title('Matrice de Confusion')
        plt.savefig('./confusion_matrix.png', bbox_inches='tight')
        plt.show()

    except ImportError:
        print("\nPour la visualisation, installez seaborn et matplotlib :")
        print("pip install seaborn matplotlib")

if __name__ == '__main__':
    main()