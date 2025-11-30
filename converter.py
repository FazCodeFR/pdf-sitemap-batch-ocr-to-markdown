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
from datetime import datetime, timedelta
from ftplib import FTP
import psutil
import gc
import subprocess
import json


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
# torch.cuda.is_available = lambda: False  # Commenté - à activer seulement si nécessaire

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
FAILED_PDF_LOG = "failed_pdfs.json"  # Changé en JSON pour plus de flexibilité
PROCESSED_PDF_LOG = "processed_pdfs.json"
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
        return True
    else:
        logging.error("Vérification échouée : la source n'a pas été ajoutée.")
        return False

def process_chatbot_source(pdf_url):
    """Gère l'ajout/mise à jour de la source dans le chatbot"""
    pdf_name = pdf_url.split("&ind=")[-1]
    sources = get_sources()
    if not sources:
        raise Exception("Impossible de récupérer les sources.")

    source_to_reset = find_source_by_keyword(sources, pdf_name)
    if source_to_reset:
        source_id = source_to_reset["id"]
        logging.info(f"Source trouvée : {source_id}")
        if not delete_source(source_id):
            raise Exception(f"Échec de suppression de la source {source_id}")
    
    markdown_content = read_markdown_content(pdf_url)
    if not markdown_content:
        raise Exception(f"Contenu Markdown vide pour {pdf_url}")
    
    if not create_source(pdf_url, markdown_content):
        raise Exception(f"Échec de création de la source pour {pdf_url}")
    
    if not verify_source_added(pdf_name):
        raise Exception(f"Source non vérifiée pour {pdf_url}")

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
        raise

# ============= FONCTIONS DE TRACKING =============

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
        if processed[url].get("date") == current_date:
            return True
    return False

def load_failed_pdfs():
    """Charge les PDFs échoués avec leur date d'échec"""
    if os.path.exists(FAILED_PDF_LOG):
        try:
            with open(FAILED_PDF_LOG, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning("Fichier failed_pdfs.json corrompu")
            return {}
    return {}

def save_failed_pdf(url, error_msg):
    """Enregistre un PDF échoué"""
    failed = load_failed_pdfs()
    failed[url] = {
        "error": str(error_msg),
        "failed_at": datetime.now().isoformat(),
        "retry_count": failed.get(url, {}).get("retry_count", 0) + 1
    }
    with open(FAILED_PDF_LOG, "w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)

def should_retry_failed_pdf(url, max_retries=3, retry_after_days=7):
    """Vérifie si un PDF échoué devrait être réessayé"""
    failed = load_failed_pdfs()
    if url not in failed:
        return True
    
    retry_count = failed[url].get("retry_count", 0)
    if retry_count >= max_retries:
        return False
    
    failed_at = datetime.fromisoformat(failed[url]["failed_at"])
    if datetime.now() - failed_at > timedelta(days=retry_after_days):
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
        raise Exception(f"Erreur HTTP {response.status_code} lors du téléchargement")


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
    
    # Upload FTP et intégration chatbot avec gestion d'erreur
    try:
        upload_to_ftp(md_filename)
        process_chatbot_source(source_url)
    except Exception as e:
        logging.error(f"Erreur lors de l'intégration : {e}")
        raise
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def process_pdf(url, date):
    logging.info(f"Traitement du PDF : {url} (Date: {date})")
    pdf_path = None
    
    try:
        pdf_path = download_pdf(url)
        convert_pdf_to_markdown(pdf_path, url)
        save_processed_pdf(url, date)  # Sauvegarder APRÈS succès complet
        check_memory_usage()
        logging.info(f"✅ PDF traité avec succès : {url}")
        
    except Exception as e:
        error_msg = f"Erreur lors du traitement du PDF {url}: {e}"
        logging.error(error_msg)
        save_failed_pdf(url, error_msg)
        raise
    
    finally:
        # Nettoyage du fichier temporaire
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logging.info(f"Fichier temporaire supprimé : {pdf_path}")
            except Exception as e:
                logging.warning(f"Impossible de supprimer {pdf_path}: {e}")


def main():
    logging.info("--- DÉMARRAGE DU SCRIPT ---")
    new_sitemap_content = download_sitemap()
    if not new_sitemap_content:
        return

    new_pdfs = parse_sitemap(new_sitemap_content)
    old_pdfs = load_local_sitemap()

    added, changed = compare_sitemaps(old_pdfs, new_pdfs)
    
    # Filtrer les PDFs à traiter
    to_process = {}
    for url, date in {**added, **changed}.items():
        # Vérifier si déjà traité avec la même date
        if is_pdf_already_processed(url, date):
            logging.info(f"PDF déjà traité, ignoré : {url}")
            continue
        
        # Vérifier si échec récent et ne pas retry
        if not should_retry_failed_pdf(url):
            logging.info(f"PDF en échec multiple, ignoré : {url}")
            continue
        
        to_process[url] = date
    
    total_pdfs = len(to_process)
    logging.info(f"{total_pdfs} PDF(s) vont être traités")

    processed_count = 0
    failed_count = 0
    
    for url, date in to_process.items():
        try:
            process_pdf(url, date)
            processed_count += 1
            logging.info(f"Progression : {processed_count}/{total_pdfs} PDFs traités")
            time.sleep(5)
        except Exception as e:
            failed_count += 1
            logging.error(f"Échec du traitement : {e}")

    end_time = time.time()
    execution_time = end_time - start_time
    
    # Toujours sauvegarder le sitemap (le tracking JSON gère les PDFs)
    save_sitemap(new_sitemap_content)
    
    logging.info(f"Temps total d'exécution : {execution_time:.2f} secondes")
    logging.info(f"PDFs traités : {processed_count}/{total_pdfs}")
    if failed_count > 0:
        logging.warning(f"PDFs en échec : {failed_count} (voir {FAILED_PDF_LOG})")
    
    logging.info("--- FIN DU SCRIPT ---")
    upload_to_ftp("logs.log")
    upload_to_ftp(PROCESSED_PDF_LOG)
    upload_to_ftp(FAILED_PDF_LOG)
    suspendInstance()

if __name__ == "__main__":
    main()