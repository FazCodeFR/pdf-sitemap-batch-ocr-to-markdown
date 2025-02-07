# PDF Sitemap Batch OCR to Markdown

Ce projet permet de **télécharger**, **analyser** et **convertir en Markdown** des fichiers **PDF** référencés dans un **sitemap XML**. Il utilise l'**OCR** pour extraire du texte lorsque cela est nécessaire.

## 🚀 Fonctionnalités

- 📥 **Téléchargement automatique** des PDF depuis un **sitemap XML**  
- 🔎 **Détection des nouveaux fichiers** et de ceux modifiés  
- 🔄 **Conversion OCR** pour les PDF scannés  
- 📝 **Export en Markdown** avec indication de la source  
- ⚡ **Traitement en parallèle** pour optimiser la vitesse  

## 🛠️ Installation

### 1️⃣ Prérequis

- Python 3.8+  
- CUDA (pour utiliser l'OCR sur GPU, sinon CPU sera utilisé)  

### 2️⃣ Installation des dépendances  

Exécuter la commande suivante dans le terminal :  

pip install -r requirements.txt  

## 📂 Configuration

Crée un fichier **.env** à la racine du projet et ajoute ces variables :  

SITEMAP_URL="https://exemple.com/sitemap.xml"  
LOCAL_SITEMAP_FILE="sitemap_local.xml"  
DOWNLOAD_FOLDER="downloads"  
MARKDOWN_FOLDER="markdowns"  

## ▶️ Utilisation

Lancer le script avec :  

python converter.py  

Le programme va :  

1. **Télécharger le sitemap**  
2. **Détecter les nouveaux fichiers PDF** ou ceux mis à jour  
3. **Télécharger les PDFs concernés**  
4. **Convertir les PDFs en Markdown avec OCR**  
5. **Enregistrer les résultats**  

Les fichiers Markdown seront enregistrés dans le dossier **markdowns/**.

## ⚙️ Options avancées

Ce projet utilise `multiprocessing` pour améliorer la vitesse.  
Si ton GPU a peu de mémoire, ajuste le nombre de processus dans `converter.py` (paramètre `NUM_WORKERS`).  

## 📜 Licence

Projet open-source sous licence **MIT**.

---
💡 **Développé par Fazcode**  
