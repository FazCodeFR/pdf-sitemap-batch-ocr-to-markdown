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
import sys
import fcntl
import atexit


# ============================================
# CONFIGURATION DU LOGGING
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs.log", mode="a"),
        logging.StreamHandler()
    ]
)

start_time = time.time()

# Charger les variables depuis .env
load_dotenv()

# ============================================
# VARIABLES D'ENVIRONNEMENT
# ============================================

SITEMAP_URL = os.getenv("SITEMAP_URL")
LOCAL_SITEMAP_FILE = os.getenv("LOCAL_SITEMAP_FILE")
DOWNLOAD_FOLDER = os.getenv("DOWNLOAD_FOLDER")
MARKDOWN_FOLDER = os.getenv("MARKDOWN_FOLDER")
FTP_HOST = os.getenv("FTP_HOST")
FTP_USER = os.getenv("FTP_USER")
FTP_PASS = os.getenv("FTP_PASS")
FTP_DIR = "/markdown"
FAILED_PDF_LOG = "failed_pdfs.json"
PROCESSED_PDF_LOG = "processed_pdfs.json"
CHATBOT_ID = os.getenv("CHATBOT_ID")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
BASE_URL = os.getenv("BASE_URL")
LOCK_FILE = "/tmp/converter.lock"

# Timeouts (en secondes)
HTTP_TIMEOUT = 60
FTP_TIMEOUT = 120

HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# Variable globale pour le converter (réutilisation)
_converter = None
_lock_fd = None


# ============================================
# GESTION DU LOCK (éviter exécutions concurrentes)
# ============================================

def acquire_lock():
    """Acquiert un lock exclusif pour éviter les exécutions concurrentes"""
    global _lock_fd
    try:
        _lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd.write(str(os.getpid()))
        _lock_fd.flush()
        logging.info("Lock acquis avec succès")
        return True
    except (IOError, OSError) as e:
        logging.error(f"Impossible d'acquérir le lock - une autre instance tourne ? {e}")
        return False


def release_lock():
    """Libère le lock"""
    global _lock_fd
    if _lock_fd:
        try:
            fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_UN)
            _lock_fd.close()
            os.remove(LOCK_FILE)
            logging.info("Lock libéré")
        except Exception as e:
            logging.warning(f"Erreur lors de la libération du lock: {e}")


# ============================================
# UTILITAIRES
# ============================================

def get_clean_filename(url):
    """Extrait et nettoie le nom de fichier depuis l'URL (fonction centralisée)"""
    raw_filename = url.split("&ind=")[-1]
    # Decode URL encoding si présent
    from urllib.parse import unquote
    raw_filename = unquote(raw_filename)
    # Supprime le préfixe numérique wpdm_ si présent
    return re.sub(r"^\d+wpdm_", "", raw_filename)


def get_markdown_path(pdf_url):
    """Retourne le chemin du fichier markdown pour un PDF donné"""
    clean_filename = get_clean_filename(pdf_url)
    return os.path.join(MARKDOWN_FOLDER, clean_filename.replace(".pdf", ".md"))


def suspendInstance():
    """Suspend l'instance OVH"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["python", "suspendInstance.py"], 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=60
            )
            logging.info(f"Script suspendInstance exécuté avec succès : {result.stdout}")
            return
        except subprocess.TimeoutExpired:
            logging.error(f"Timeout lors de l'exécution de suspendInstance.py (tentative {attempt + 1}/{max_retries})")
        except subprocess.CalledProcessError as e:
            logging.error(f"Erreur lors de l'exécution de suspendInstance.py (tentative {attempt + 1}/{max_retries}): {e.stderr}")
        
        if attempt < max_retries - 1:
            time.sleep(10)
    
    logging.critical("Impossible de suspendre l'instance après plusieurs tentatives")


def check_memory_usage():
    """Vérifie l'utilisation mémoire et alerte si critique"""
    mem = psutil.virtual_memory()
    gpu_mem_used = 0
    
    if torch.cuda.is_available():
        gpu_mem_used = torch.cuda.memory_allocated() / 1024**3  # En GB
        
    logging.info(f"Mémoire RAM: {mem.percent}% | GPU: {gpu_mem_used:.2f} GB")
    
    if mem.percent > 85:
        logging.warning("⚠️ ALERTE: Mémoire RAM très haute (>85%) - risque d'arrêt!")
        # Tenter de libérer de la mémoire
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        upload_to_ftp("logs.log")
        return False
    
    return True


def cleanup_gpu_memory():
    """Nettoie la mémoire GPU de manière agressive"""
    global _converter
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    
    # Réinitialiser le converter si la mémoire est trop utilisée
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        if mem_allocated > 2.0:  # Plus de 2GB utilisés
            logging.warning(f"Mémoire GPU élevée ({mem_allocated:.2f}GB), réinitialisation du converter")
            _converter = None
            gc.collect()
            torch.cuda.empty_cache()


# ============================================
# FONCTIONS FTP
# ============================================

def upload_to_ftp(file_path, max_retries=3):
    """Upload un fichier vers le serveur FTP avec retry"""
    if not os.path.exists(file_path):
        logging.warning(f"Fichier inexistant, upload ignoré: {file_path}")
        return False
    
    for attempt in range(max_retries):
        try:
            with FTP(FTP_HOST, timeout=FTP_TIMEOUT) as ftp:
                ftp.login(FTP_USER, FTP_PASS)
                ftp.cwd(FTP_DIR)
                with open(file_path, "rb") as f:
                    ftp.storbinary(f"STOR {os.path.basename(file_path)}", f)
                logging.info(f"Upload FTP réussi : {file_path} -> {FTP_DIR}")
                return True
        except Exception as e:
            logging.error(f"Échec upload FTP (tentative {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    
    logging.error(f"Upload FTP définitivement échoué pour {file_path}")
    return False


# ============================================
# FONCTIONS CHATBOT API
# ============================================

def get_sources():
    """Récupère toutes les sources du chatbot"""
    try:
        response = requests.get(
            f"{BASE_URL}/chatbot/{CHATBOT_ID}/sources", 
            headers=HEADERS,
            timeout=HTTP_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        logging.error(f"Erreur API sources {response.status_code}: {response.text}")
    except requests.exceptions.Timeout:
        logging.error("Timeout lors de la récupération des sources")
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur réseau lors de la récupération des sources: {e}")
    return None


def find_source_by_keyword(sources, keyword):
    """Recherche une source contenant un mot-clé dans son URL"""
    if not sources:
        return None
    return next((s for s in sources if keyword in s.get("url", "")), None)


def delete_source(source_id):
    """Supprime une source spécifique"""
    try:
        response = requests.delete(
            f"{BASE_URL}/sources/{source_id}", 
            headers=HEADERS,
            timeout=HTTP_TIMEOUT
        )
        if response.status_code in [200, 204, 404]:
            logging.info(f"Source supprimée ou introuvable : {source_id}")
            return True
        logging.error(f"Erreur suppression source {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur réseau lors de la suppression: {e}")
    return False


def read_markdown_content(pdf_url):
    """Lit le contenu du fichier markdown correspondant au PDF"""
    md_path = get_markdown_path(pdf_url)
    
    if os.path.exists(md_path):
        try:
            with open(md_path, "r", encoding="utf-8") as file:
                content = file.read()
                if content.strip():
                    return content
                logging.warning(f"Fichier Markdown vide: {md_path}")
        except Exception as e:
            logging.error(f"Erreur lecture Markdown {md_path}: {e}")
    else:
        logging.warning(f"Fichier Markdown introuvable: {md_path}")
    
    return ""


def create_source(url, markdown_content):
    """Ajoute une nouvelle source avec l'URL et le contenu Markdown"""
    try:
        payload = {"url": url, "content": markdown_content}
        response = requests.post(
            f"{BASE_URL}/chatbot/{CHATBOT_ID}/sources", 
            headers=HEADERS, 
            json=payload,
            timeout=HTTP_TIMEOUT
        )
        
        if response.status_code in [200, 201]:
            logging.info(f"Source ajoutée : {url}")
            return True
        logging.error(f"Erreur création source {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur réseau lors de la création: {e}")
    return False


def verify_source_added(keyword, max_retries=3):
    """Vérifie que la nouvelle source a bien été ajoutée"""
    for attempt in range(max_retries):
        time.sleep(2)  # Attendre que l'API se synchronise
        sources = get_sources()
        if sources and find_source_by_keyword(sources, keyword):
            logging.info("Vérification réussie : la source est bien présente")
            return True
        if attempt < max_retries - 1:
            logging.info(f"Vérification en cours... (tentative {attempt + 2}/{max_retries})")
    
    logging.error("Vérification échouée : la source n'a pas été ajoutée")
    return False


def process_chatbot_source(pdf_url):
    """Gère l'ajout/mise à jour de la source dans le chatbot"""
    clean_filename = get_clean_filename(pdf_url)
    
    sources = get_sources()
    if sources is None:
        raise Exception("Impossible de récupérer les sources du chatbot")

    # Chercher si une source existe déjà
    source_to_reset = find_source_by_keyword(sources, clean_filename)
    if source_to_reset:
        source_id = source_to_reset["id"]
        logging.info(f"Source existante trouvée : {source_id}")
        if not delete_source(source_id):
            logging.warning(f"Échec suppression source {source_id}, tentative de création quand même")
    
    # Lire le contenu markdown
    markdown_content = read_markdown_content(pdf_url)
    if not markdown_content:
        raise Exception(f"Contenu Markdown vide ou inexistant pour {pdf_url}")
    
    # Créer la nouvelle source
    if not create_source(pdf_url, markdown_content):
        raise Exception(f"Échec de création de la source pour {pdf_url}")
    
    # Vérifier l'ajout
    if not verify_source_added(clean_filename):
        raise Exception(f"Source non vérifiée pour {pdf_url}")
    
    logging.info(f"✅ Source chatbot mise à jour pour {clean_filename}")


# ============================================
# FONCTIONS DE TRACKING (JSON)
# ============================================

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
        "processed_at": datetime.now().isoformat(),
        "filename": get_clean_filename(url)
    }
    with open(PROCESSED_PDF_LOG, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    logging.info(f"PDF enregistré comme traité : {get_clean_filename(url)}")


def is_pdf_already_processed(url, current_date):
    """Vérifie si un PDF a déjà été traité avec cette date"""
    processed = load_processed_pdfs()
    if url in processed:
        return processed[url].get("date") == current_date
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
    retry_count = failed.get(url, {}).get("retry_count", 0) + 1
    failed[url] = {
        "error": str(error_msg)[:500],  # Limiter la taille de l'erreur
        "failed_at": datetime.now().isoformat(),
        "retry_count": retry_count,
        "filename": get_clean_filename(url)
    }
    with open(FAILED_PDF_LOG, "w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)
    logging.info(f"PDF enregistré comme échoué (tentative {retry_count}): {get_clean_filename(url)}")


def remove_from_failed(url):
    """Retire un PDF de la liste des échecs après succès"""
    failed = load_failed_pdfs()
    if url in failed:
        del failed[url]
        with open(FAILED_PDF_LOG, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2, ensure_ascii=False)
        logging.info(f"PDF retiré de la liste des échecs : {get_clean_filename(url)}")


def should_retry_failed_pdf(url, max_retries=3, retry_after_days=7):
    """Vérifie si un PDF échoué devrait être réessayé"""
    failed = load_failed_pdfs()
    if url not in failed:
        return True
    
    retry_count = failed[url].get("retry_count", 0)
    if retry_count >= max_retries:
        logging.debug(f"PDF {get_clean_filename(url)} ignoré: max retries atteint ({retry_count})")
        return False
    
    try:
        failed_at = datetime.fromisoformat(failed[url]["failed_at"])
        if datetime.now() - failed_at > timedelta(days=retry_after_days):
            logging.info(f"PDF {get_clean_filename(url)} éligible pour retry (dernier échec > {retry_after_days} jours)")
            return True
    except (KeyError, ValueError):
        return True
    
    return False


# ============================================
# FONCTIONS SITEMAP
# ============================================

def download_sitemap():
    """Télécharge le sitemap depuis l'URL configurée"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        response = requests.get(SITEMAP_URL, headers=headers, timeout=HTTP_TIMEOUT)
        if response.status_code == 200:
            logging.info(f"Sitemap téléchargé ({len(response.text)} caractères)")
            return response.text
        else:
            logging.error(f"Erreur téléchargement sitemap: HTTP {response.status_code}")
    except requests.exceptions.Timeout:
        logging.error("Timeout lors du téléchargement du sitemap")
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur réseau sitemap: {e}")
    return None


def parse_sitemap(xml_content):
    """Parse le contenu XML du sitemap et extrait les PDFs"""
    try:
        root = ET.fromstring(xml_content)
        namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
        pdfs = {}

        for url in root.findall(f"{namespace}url"):
            loc_elem = url.find(f"{namespace}loc")
            lastmod_elem = url.find(f"{namespace}lastmod")
            
            if loc_elem is not None and loc_elem.text:
                loc = loc_elem.text
                lastmod = lastmod_elem.text if lastmod_elem is not None else ""
                pdfs[loc] = lastmod

        logging.info(f"Sitemap parsé: {len(pdfs)} URLs trouvées")
        return pdfs
    except ET.ParseError as e:
        logging.error(f"Erreur parsing XML sitemap: {e}")
        return {}


def save_sitemap(xml_content):
    """Sauvegarde le sitemap localement"""
    try:
        with open(LOCAL_SITEMAP_FILE, "w", encoding="utf-8") as f:
            f.write(xml_content)
        logging.info(f"Sitemap sauvegardé: {LOCAL_SITEMAP_FILE}")
    except Exception as e:
        logging.error(f"Erreur sauvegarde sitemap: {e}")


def load_local_sitemap():
    """Charge le sitemap local s'il existe"""
    if not os.path.exists(LOCAL_SITEMAP_FILE):
        logging.info("Pas de sitemap local existant")
        return {}

    try:
        with open(LOCAL_SITEMAP_FILE, "r", encoding="utf-8") as f:
            return parse_sitemap(f.read())
    except Exception as e:
        logging.error(f"Erreur lecture sitemap local: {e}")
        return {}


def compare_sitemaps(old_pdfs, new_pdfs):
    """Compare deux sitemaps et retourne les changements"""
    added = {url: date for url, date in new_pdfs.items() if url not in old_pdfs}
    changed = {url: date for url, date in new_pdfs.items() if url in old_pdfs and old_pdfs[url] != date}
    removed = {url: date for url, date in old_pdfs.items() if url not in new_pdfs}
    
    logging.info(f"Comparaison sitemap: {len(added)} ajoutés, {len(changed)} modifiés, {len(removed)} supprimés")
    
    return added, changed, removed


# ============================================
# FONCTIONS PDF
# ============================================

def download_pdf(url):
    """Télécharge un PDF depuis l'URL"""
    clean_filename = get_clean_filename(url)
    filepath = os.path.join(DOWNLOAD_FOLDER, clean_filename)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=HTTP_TIMEOUT)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
            
            file_size = os.path.getsize(filepath)
            logging.info(f"Téléchargé: {clean_filename} ({file_size / 1024:.1f} KB)")
            
            # Vérifier que le fichier n'est pas vide
            if file_size < 100:
                raise Exception(f"Fichier PDF trop petit ({file_size} bytes), probablement corrompu")
            
            return filepath
        else:
            raise Exception(f"HTTP {response.status_code}")
            
    except requests.exceptions.Timeout:
        raise Exception("Timeout lors du téléchargement")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Erreur réseau: {e}")


def get_converter():
    """Retourne ou crée le converter PDF (singleton pour réutilisation)"""
    global _converter
    
    if _converter is None:
        logging.info("Initialisation du converter Marker...")
        config = {
            "output_format": "markdown",
            "languages": "fr",
            "disable_image_extraction": True,
            "max_tasks_per_worker": 1
        }
        
        config_parser = ConfigParser(config)
        _converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer()
        )
        logging.info("Converter initialisé")
    
    return _converter


def convert_pdf_to_markdown(pdf_path, source_url):
    """Convertit un PDF en Markdown"""
    clean_filename = get_clean_filename(source_url)
    md_filename = os.path.join(MARKDOWN_FOLDER, clean_filename.replace(".pdf", ".md"))
    
    try:
        converter = get_converter()
        rendered = converter(pdf_path)
        text, _, _ = text_from_rendered(rendered)
        
        # Vérifier que la conversion a produit du contenu
        if not text or len(text.strip()) < 50:
            raise Exception("Conversion produit un contenu vide ou trop court")
        
        # Créer le titre propre
        clean_title = clean_filename.replace("-", " ").replace(".pdf", "")
        
        # Écrire le fichier markdown
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(text)
            f.write(f"\n\n---\n\n**Source :** [{clean_title}]({source_url})")
        
        logging.info(f"Converti en Markdown: {clean_filename} ({len(text)} caractères)")
        
        # Upload FTP
        if not upload_to_ftp(md_filename):
            logging.warning("Upload FTP échoué, mais on continue")
        
        # Intégration chatbot
        process_chatbot_source(source_url)
        
    except Exception as e:
        logging.error(f"Erreur conversion {clean_filename}: {e}")
        raise
    finally:
        cleanup_gpu_memory()


def process_pdf(url, date):
    """Traite un PDF complet: téléchargement, conversion, upload"""
    clean_filename = get_clean_filename(url)
    logging.info(f"{'='*50}")
    logging.info(f"Traitement: {clean_filename}")
    logging.info(f"Date sitemap: {date}")
    logging.info(f"URL: {url}")
    
    pdf_path = None
    
    try:
        # Vérifier la mémoire avant de commencer
        if not check_memory_usage():
            raise Exception("Mémoire insuffisante pour traiter ce PDF")
        
        # Télécharger le PDF
        pdf_path = download_pdf(url)
        
        # Convertir en Markdown
        convert_pdf_to_markdown(pdf_path, url)
        
        # Marquer comme traité
        save_processed_pdf(url, date)
        
        # Retirer de la liste des échecs si présent
        remove_from_failed(url)
        
        logging.info(f"✅ SUCCESS: {clean_filename}")
        return True
        
    except Exception as e:
        error_msg = f"Erreur traitement {clean_filename}: {e}"
        logging.error(f"❌ FAILED: {error_msg}")
        save_failed_pdf(url, str(e))
        return False
    
    finally:
        # Nettoyer le fichier PDF temporaire
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logging.debug(f"Fichier temporaire supprimé: {pdf_path}")
            except Exception as e:
                logging.warning(f"Impossible de supprimer {pdf_path}: {e}")


def handle_removed_pdfs(removed_urls):
    """Gère les PDFs supprimés du sitemap (optionnel: supprimer les sources)"""
    if not removed_urls:
        return
    
    logging.info(f"Traitement de {len(removed_urls)} PDF(s) supprimé(s) du sitemap")
    
    sources = get_sources()
    if not sources:
        logging.warning("Impossible de récupérer les sources pour nettoyer les PDFs supprimés")
        return
    
    for url in removed_urls:
        clean_filename = get_clean_filename(url)
        source = find_source_by_keyword(sources, clean_filename)
        
        if source:
            logging.info(f"Suppression de la source pour PDF retiré: {clean_filename}")
            delete_source(source["id"])
        
        # Retirer du tracking
        processed = load_processed_pdfs()
        if url in processed:
            del processed[url]
            with open(PROCESSED_PDF_LOG, "w", encoding="utf-8") as f:
                json.dump(processed, f, indent=2, ensure_ascii=False)


# ============================================
# FONCTION PRINCIPALE
# ============================================

def main():
    logging.info("=" * 60)
    logging.info("DÉMARRAGE DU SCRIPT PDF CONVERTER")
    logging.info("=" * 60)
    
    # Acquérir le lock
    if not acquire_lock():
        logging.error("Une autre instance est déjà en cours d'exécution. Arrêt.")
        sys.exit(1)
    
    # Enregistrer la libération du lock à la fin
    atexit.register(release_lock)
    
    # Vérifier les variables d'environnement
    required_vars = [SITEMAP_URL, LOCAL_SITEMAP_FILE, DOWNLOAD_FOLDER, MARKDOWN_FOLDER, 
                     FTP_HOST, FTP_USER, FTP_PASS, CHATBOT_ID, BEARER_TOKEN, BASE_URL]
    if not all(required_vars):
        logging.error("Variables d'environnement manquantes!")
        upload_to_ftp("logs.log")
        suspendInstance()
        sys.exit(1)
    
    # Créer les dossiers si nécessaire
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(MARKDOWN_FOLDER, exist_ok=True)
    
    # Télécharger le nouveau sitemap
    new_sitemap_content = download_sitemap()
    if not new_sitemap_content:
        logging.error("Impossible de télécharger le sitemap")
        upload_to_ftp("logs.log")
        suspendInstance()
        sys.exit(1)
    
    # Parser les sitemaps
    new_pdfs = parse_sitemap(new_sitemap_content)
    old_pdfs = load_local_sitemap()
    
    # Comparer
    added, changed, removed = compare_sitemaps(old_pdfs, new_pdfs)
    
    # Gérer les PDFs supprimés
    handle_removed_pdfs(removed)
    
    # Filtrer les PDFs à traiter
    to_process = {}
    for url, date in {**added, **changed}.items():
        # Vérifier si déjà traité avec la même date
        if is_pdf_already_processed(url, date):
            logging.debug(f"Déjà traité, ignoré: {get_clean_filename(url)}")
            continue
        
        # Vérifier si en échec multiple
        if not should_retry_failed_pdf(url):
            logging.debug(f"Échec multiple, ignoré: {get_clean_filename(url)}")
            continue
        
        to_process[url] = date
    
    total_pdfs = len(to_process)
    logging.info(f"{'='*50}")
    logging.info(f"PDFs à traiter: {total_pdfs}")
    logging.info(f"{'='*50}")
    
    if total_pdfs == 0:
        logging.info("Aucun PDF à traiter")
    else:
        processed_count = 0
        failed_count = 0
        
        for idx, (url, date) in enumerate(to_process.items(), 1):
            logging.info(f"\n[{idx}/{total_pdfs}] Début traitement...")
            
            success = process_pdf(url, date)
            
            if success:
                processed_count += 1
            else:
                failed_count += 1
            
            logging.info(f"Progression: {processed_count} réussis, {failed_count} échoués sur {idx} traités")
            
            # Pause entre les PDFs pour éviter la surcharge
            if idx < total_pdfs:
                time.sleep(5)
        
        logging.info(f"\n{'='*50}")
        logging.info(f"RÉSUMÉ: {processed_count}/{total_pdfs} PDFs traités avec succès")
        if failed_count > 0:
            logging.warning(f"⚠️ {failed_count} PDF(s) en échec (voir {FAILED_PDF_LOG})")
    
    # Sauvegarder le sitemap
    save_sitemap(new_sitemap_content)
    
    # Stats finales
    end_time = time.time()
    execution_time = end_time - start_time
    
    logging.info(f"\n{'='*60}")
    logging.info(f"FIN DU SCRIPT")
    logging.info(f"Temps d'exécution: {execution_time:.2f} secondes ({execution_time/60:.1f} minutes)")
    logging.info(f"{'='*60}")
    
    # Upload des logs et fichiers de tracking
    upload_to_ftp("logs.log")
    if os.path.exists(PROCESSED_PDF_LOG):
        upload_to_ftp(PROCESSED_PDF_LOG)
    if os.path.exists(FAILED_PDF_LOG):
        upload_to_ftp(FAILED_PDF_LOG)
    
    # Suspendre l'instance
    suspendInstance()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Interruption utilisateur (Ctrl+C)")
        release_lock()
        sys.exit(0)
    except Exception as e:
        logging.critical(f"Erreur fatale non gérée: {e}", exc_info=True)
        upload_to_ftp("logs.log")
        release_lock()
        suspendInstance()
        sys.exit(1)
