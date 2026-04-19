import requests
import json
import re
from dotenv import load_dotenv
import os
from urllib.parse import unquote

# Charger les variables depuis .env
load_dotenv()

# Configuration
CHATBOT_ID = os.getenv("CHATBOT_ID")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
BASE_URL = os.getenv("BASE_URL")
PDF_FOLDER = "pdf-temp"
MD_FOLDER = "markdown"

# Headers pour l'authentification
HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

def get_sources():
    """Récupère toutes les sources du chatbot"""
    response = requests.get(f"{BASE_URL}/chatbot/{CHATBOT_ID}/sources", headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    print(f"❌ Erreur {response.status_code} : {response.text}")
    return None

def extract_wpdmdl(url):
    """Extrait l'ID wpdmdl stable d'une URL (ignore le filename &ind=)"""
    match = re.search(r"wpdmdl=(\d+)", url or "")
    return match.group(1) if match else None

def find_sources_by_wpdmdl(sources, wpdmdl_id):
    """Retourne TOUTES les sources matchant ce wpdmdl (gère les doublons existants)"""
    if not wpdmdl_id:
        return []
    return [s for s in sources if extract_wpdmdl(s.get("url", "")) == wpdmdl_id]

def delete_source(source_id):
    """Supprime une source spécifique"""
    response = requests.delete(f"{BASE_URL}/sources/{source_id}", headers=HEADERS)
    if response.status_code in [200, 404]:  # On continue même si la source est introuvable
        print(f"✅ Source supprimée ou introuvable : {source_id}")
        return True
    print(f"❌ Erreur {response.status_code} : {response.text}")
    return False

def get_clean_filename(url):
    """Extrait et nettoie le nom de fichier depuis l'URL (décode URL + strip préfixe wpdm_)"""
    raw_filename = url.split("&ind=")[-1]
    raw_filename = unquote(raw_filename)
    return re.sub(r"^\d+wpdm_", "", raw_filename)


def read_markdown_content(pdf_url):
    """Lit le contenu du fichier markdown correspondant au PDF"""
    pdf_name = get_clean_filename(pdf_url)
    md_path = os.path.join(MD_FOLDER, pdf_name.replace(".pdf", ".md"))
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as file:
            return file.read()
    print(f"⚠️ Fichier Markdown introuvable : {md_path}")
    return ""


def create_source(url, markdown_content):
    """Ajoute une nouvelle source avec l'URL et le contenu Markdown"""
    payload = {"url": url, "content": markdown_content}
    print(f"🚀 Envoi du JSON : {json.dumps(payload, indent=2)}")
    response = requests.post(f"{BASE_URL}/chatbot/{CHATBOT_ID}/sources", headers=HEADERS, json=payload)
    print(f"📩 Réponse ({response.status_code}): {response.text}")
    if response.status_code == 200:
        print(f"✅ Source ajoutée : {url}")
        return True
    print(f"❌ Erreur {response.status_code} : {response.text}")
    return False

def verify_source_added(wpdmdl_id):
    """Vérifie que la nouvelle source a bien été ajoutée (match par wpdmdl)"""
    sources = get_sources()
    if sources and find_sources_by_wpdmdl(sources, wpdmdl_id):
        print("✅ Vérification réussie : la source est bien présente")
    else:
        print("❌ Vérification échouée : la source n'a pas été ajoutée.")

def main(pdf_url):
    wpdmdl_id = extract_wpdmdl(pdf_url)
    if not wpdmdl_id:
        return print(f"❌ Impossible d'extraire wpdmdl de {pdf_url}")

    sources = get_sources()
    if sources is None:
        return print("❌ Impossible de récupérer les sources.")

    existing = find_sources_by_wpdmdl(sources, wpdmdl_id)
    for src in existing:
        source_id = src["id"]
        print(f"🔍 Source existante trouvée (wpdmdl={wpdmdl_id}) : {source_id}")
        if not delete_source(source_id):
            return print(f"❌ Échec de suppression de la source {source_id}")

    markdown_content = read_markdown_content(pdf_url)
    if create_source(pdf_url, markdown_content):
        verify_source_added(wpdmdl_id)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <pdf_url>")
    else:
        main(sys.argv[1])
