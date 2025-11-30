import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import os
import re
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
import torch
import time
import logging
from datetime import datetime
from ftplib import FTP
import psutil
import gc
import subprocess
import json
from urllib.parse import urlparse


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs.log", mode="a"),
        logging.StreamHandler()
    ]
)

start_time = time.time()
torch.cuda.is_available = lambda: False

# Charger les variables depuis .env
load_dotenv()

SITEMAP_URL = os.getenv("SITEMAP_URL")
LOCAL_SITEMAP_FILE = os.getenv("LOCAL_SITEMAP_FILE")
DOWNLOAD_FOLDER = os.getenv("DOWNLOAD_FOLDER")
MARKDOWN_FOLDER = os.getenv("MARKDOWN_FOLDER")
FTP_HOST = os.getenv("FTP_HOST")
FTP_USER = os.getenv("FTP_USER")
FTP_PASS = os.getenv("FTP_PASS")
FTP_DIR = "/markdown"
FAILED_PDF_LOG = "failed_pdfs.txt"
PROCESSED_PDF_LOG = "processed_pdfs.json"  # Nouveau fichier pour tracker les PDFs traités
CHATBOT_ID = os.getenv("CHATBOT_ID")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
BASE_URL = os.getenv("BASE_URL")

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
    logging.error(f"Erreur {response.status_code} : {response.text}")
    return None

def find_source_by_keyword(sources, keyword):
    """Recherche une source contenant un mot-clé dans son URL"""
    return next((s for s in sources if keyword in s.get("url", "")), None)

def delete_source(source_id):
    """Supprime une source spécifique"""
    response = requests.delete(f"{BASE_URL}/sources/{source_id}", headers=HEADERS)
    if response.status_code in [200, 404]:
        logging.info(f"Source supprimée ou introuvable : {source_id}")
        return True
    logging.error(f"Erreur {response.status_code} : {response.text}")
    return False

def read_markdown_content(pdf_url):
    """Lit le contenu du fichier markdown correspondant au PDF"""
    pdf_name = pdf_url.split("&ind=")[-1]
    md_path = os.path.join(MARKDOWN_FOLDER, pdf_name.replace(".pdf", ".md"))
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as file:
            return file.read()
    logging.warning(f"Fichier Markdown introuvable : {md_path}")
    return ""

def create_source(url, markdown_content):
    """Ajoute une nouvelle source avec l'URL et le contenu Markdown"""
    payload = {"url": url, "content": markdown_content}
    response = requests.post(f"{BASE_URL}/chatbot/{CHATBOT_ID}/sources", headers=HEADERS, json=payload)
    logging.info(f"Réponse ({response.status_code})")
    if response.status_code == 200:
        logging.info(f"Source ajoutée : {url}")
        return True
    logging.error(f"Erreur {response.status_code} : {response.text}")
    return False

def verify_source_added(keyword):
    """Vérifie que la nouvelle source a bien été ajoutée"""
    sources = get_sources()
    if sources and find_source_by_keyword(sources, keyword):
        logging.info("Vérification réussie : la source est bien présente")
    else:
        logging.error("Vérification échouée : la source n'a pas été ajoutée.")

def process_chatbot_source(pdf_url):
    pdf_name = pdf_url.split("&ind=")[-1]
    sources = get_sources()
    if not sources:
        return logging.error("Impossible de récupérer les sources.")

    source_to_reset = find_source_by_keyword(sources, pdf_name)
    if source_to_reset:
        source_id = source_to_reset["id"]
        logging.info(f"Source trouvée : {source_id}")
        if delete_source(source_id):
            markdown_content = read_markdown_content(pdf_url)
            if create_source(pdf_url, markdown_content):
                verify_source_added(pdf_name)
    else:
        logging.warning("Aucune source trouvée, ajout d'une nouvelle source.")
        markdown_content = read_markdown_content(pdf_url)
        if create_source(pdf_url, markdown_content):
            verify_source_added(pdf_name)

def suspendInstance():
    try:
        result = subprocess.run(["python", "suspendInstance.py"], check=True, capture_output=True, text=True)
        logging.info(f"Script suspendInstance exécuté avec succès : {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Erreur lors de l'exécution de suspendInstance.py : {e.stderr}")
        suspendInstance()

def check_memory_usage():
    mem = psutil.virtual_memory()
    if mem.percent > 80:
        logging.warning("Alerte Memoire tres haute risque d arret du programme!")
        upload_to_ftp("logs.log")

def upload_to_ftp(file_path):
    try:
        with FTP(FTP_HOST) as ftp:
            ftp.login(FTP_USER, FTP_PASS)
            ftp.cwd(FTP_DIR)
            with open(file_path, "rb") as f:
                ftp.storbinary(f"STOR {os.path.basename(file_path)}", f)
            logging.info(f"Upload réussi : {file_path} -> {FTP_DIR}")
    except Exception as e:
        logging.error(f"Échec de l'upload FTP : {e}")

# ============= NOUVELLES FONCTIONS POUR LE TRACKING =============

def load_processed_pdfs():
    """Charge le dictionnaire des PDFs déjà traités avec leur date de traitement"""
    if os.path.exists(PROCESSED_PDF_LOG):
        try:
            with open(PROCESSED_PDF_LOG, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning("Fichier processed_pdfs.json corrompu, création d'un nouveau")
            return {}
    return {}

def save_processed_pdf(url, date):
    """Enregistre un PDF comme traité avec sa date"""
    processed = load_processed_pdfs()
    processed[url] = {
        "date": date,
        "processed_at": datetime.now().isoformat()
    }
    with open(PROCESSED_PDF_LOG, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    logging.info(f"PDF enregistré comme traité : {url}")

def is_pdf_already_processed(url, current_date):
    """Vérifie si un PDF a déjà été traité avec cette date"""
    processed = load_processed_pdfs()
    if url in processed:
        # Vérifier si la date est la même
        if processed[url].get("date") == current_date:
            return True
    return False

# ================================================================

if not all([SITEMAP_URL, LOCAL_SITEMAP_FILE, DOWNLOAD_FOLDER, MARKDOWN_FOLDER, FTP_HOST, FTP_USER, FTP_PASS]):
    logging.error("Certaines variables d'environnement sont manquantes.")
    upload_to_ftp("logs.log")
    suspendInstance()
    raise ValueError("Certaines variables d'environnement sont manquantes.")

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(MARKDOWN_FOLDER, exist_ok=True)

def download_sitemap():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(SITEMAP_URL, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        logging.error(f"Erreur lors du téléchargement du sitemap. {response.status_code}")
        return None

def parse_sitemap(xml_content):
    root = ET.fromstring(xml_content)
    namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    pdfs = {}

    for url in root.findall(f"{namespace}url"):
        loc = url.find(f"{namespace}loc").text
        lastmod = url.find(f"{namespace}lastmod").text if url.find(f"{namespace}lastmod") is not None else ""
        pdfs[loc] = lastmod

    return pdfs

def save_sitemap(xml_content):
    with open(LOCAL_SITEMAP_FILE, "w", encoding="utf-8") as f:
        f.write(xml_content)

def load_local_sitemap():
    if not os.path.exists(LOCAL_SITEMAP_FILE):
        return {}

    with open(LOCAL_SITEMAP_FILE, "r", encoding="utf-8") as f:
        return parse_sitemap(f.read())

def compare_sitemaps(old_pdfs, new_pdfs):
    added = {url: date for url, date in new_pdfs.items() if url not in old_pdfs}
    changed = {url: date for url, date in new_pdfs.items() if url in old_pdfs and old_pdfs[url] != date}
    return added, changed

def download_pdf(url):
    raw_filename = url.split("&ind=")[-1]
    clean_filename = re.sub(r"^\d+wpdm_", "", raw_filename)
    filename = os.path.join(DOWNLOAD_FOLDER, clean_filename)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        logging.info(f"Téléchargé : {filename}")
        return filename
    else:
        logging.error(f"Erreur lors du téléchargement de {url}")
        return None


def convert_pdf_to_markdown(pdf_path, source_url):
    config = {
        "output_format": "markdown",
        "languages": "fr",
        "disable_image_extraction": True,
        "max_tasks_per_worker": 1
    }

    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer()
    )

    rendered = converter(pdf_path)
    text, _, _ = text_from_rendered(rendered)

    raw_filename = source_url.split("&ind=")[-1]
    clean_title = raw_filename.replace("-", " ").replace(".pdf", "")

    md_filename = os.path.join(MARKDOWN_FOLDER, os.path.basename(pdf_path).replace(".pdf", ".md"))
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(text)
        f.write(f"\n\n---\n\n**Source :** [{clean_title}]({source_url})")

    logging.info(f"Converti en Markdown : {md_filename}")
    upload_to_ftp(md_filename)
    process_chatbot_source(source_url) 
    torch.cuda.empty_cache()
    gc.collect()


def process_pdf(url, date):
    logging.info(f"Traitement du PDF : {url} (Ajouté/Modifié le {date})")
    pdf_path = download_pdf(url)
    if pdf_path:
        try:
            convert_pdf_to_markdown(pdf_path, url)
            save_processed_pdf(url, date)  # Enregistrer comme traité
            check_memory_usage()
        except Exception as e:
            logging.error(f"Erreur lors du traitement du PDF {url}: {e}")
            with open(FAILED_PDF_LOG, "a", encoding="utf-8") as f:
                f.write(f"{url} - {e}\n")
    else:
        logging.error(f"Erreur de téléchargement du PDF {url}")
        with open(FAILED_PDF_LOG, "a", encoding="utf-8") as f:
            f.write(f"{url} - Erreur de téléchargement\n")

def load_failed_pdfs():
    if os.path.exists(FAILED_PDF_LOG):
        with open(FAILED_PDF_LOG, "r", encoding="utf-8") as f:
            failed_urls = [line.split(" - ")[0] for line in f.readlines()]
        return set(failed_urls)
    return set()

def main():
    logging.info("--- DÉMARRAGE DU SCRIPT ---")
    new_sitemap_content = download_sitemap()
    if not new_sitemap_content:
        return

    new_pdfs = parse_sitemap(new_sitemap_content)
    old_pdfs = load_local_sitemap()

    added, changed = compare_sitemaps(old_pdfs, new_pdfs)
    
    # Charger les PDFs échoués et déjà traités
    failed_pdfs = load_failed_pdfs()
    
    # Filtrer les PDFs à traiter
    to_process = {}
    for url, date in {**added, **changed}.items():
        # Exclure les PDFs échoués
        if url in failed_pdfs:
            continue
        # Exclure les PDFs déjà traités avec la même date
        if is_pdf_already_processed(url, date):
            logging.info(f"PDF déjà traité, ignoré : {url}")
            continue
        to_process[url] = date
    
    total_pdfs = len(to_process)
    logging.info(f"{total_pdfs} PDF(s) vont être traités")

    processed_count = 0
    for url, date in to_process.items():
        try:
            process_pdf(url, date)
            processed_count += 1
            logging.info(f"Progression : {processed_count}/{total_pdfs} PDFs traités")
            time.sleep(5)
        except Exception as e:
            logging.error(f"Erreur lors du traitement du PDF {url}: {e}")

    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Temps total d'exécution : {execution_time:.2f} secondes")
    logging.info(f"PDFs traités dans cette session : {processed_count}/{total_pdfs}")
    
    # Sauvegarder le sitemap seulement si tous les PDFs ont été traités
    if processed_count == total_pdfs:
        save_sitemap(new_sitemap_content)
        logging.info("Sitemap mis à jour - Tous les PDFs ont été traités")
    else:
        logging.warning(f"Sitemap non mis à jour - {total_pdfs - processed_count} PDFs restants")
    
    logging.info("---")
    upload_to_ftp("logs.log")
    upload_to_ftp(PROCESSED_PDF_LOG)  # Upload du fichier de tracking
    suspendInstance()

if __name__ == "__main__":
    main()