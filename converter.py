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
import multiprocessing
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(filename='logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

start_time = time.time()
torch.cuda.is_available = lambda: False  # Force l'utilisation du CPU

# Charger les variables depuis .env
load_dotenv()

SITEMAP_URL = os.getenv("SITEMAP_URL")
LOCAL_SITEMAP_FILE = os.getenv("LOCAL_SITEMAP_FILE")
DOWNLOAD_FOLDER = os.getenv("DOWNLOAD_FOLDER")
MARKDOWN_FOLDER = os.getenv("MARKDOWN_FOLDER")

if not all([SITEMAP_URL, LOCAL_SITEMAP_FILE, DOWNLOAD_FOLDER, MARKDOWN_FOLDER]):
    logging.error("Certaines variables d'environnement sont manquantes.")
    raise ValueError("Certaines variables d'environnement sont manquantes.")

# Création des dossiers nécessaires
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(MARKDOWN_FOLDER, exist_ok=True)

def download_sitemap():
    response = requests.get(SITEMAP_URL)
    if response.status_code == 200:
        return response.text
    else:
        logging.error("Erreur lors du téléchargement du sitemap.")
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
    # Extraire le nom du fichier depuis l'URL
    raw_filename = url.split("&ind=")[-1]

    # Supprimer la partie numérique aléatoire avant "wpdm_"
    clean_filename = re.sub(r"^\d+wpdm_", "", raw_filename)

    filename = os.path.join(DOWNLOAD_FOLDER, clean_filename)

    response = requests.get(url, stream=True)
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
    # Configuration avec les options nécessaires
    config = {
        "output_format": "markdown",          # Format de sortie
        "languages": "fr",                # Langue
        "disable_image_extraction": True, # Désactivation de l'extraction d'images
        # "force_ocr": True,                # Forcer l'OCR
    }

    # Génération du parser de configuration
    config_parser = ConfigParser(config)

    # Création de l'objet PdfConverter avec la configuration et le modèle
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer()
    )

    # Conversion du fichier PDF
    rendered = converter(pdf_path)

    # Extraction du texte rendu
    text, _, _ = text_from_rendered(rendered)

    # Création du fichier Markdown
    md_filename = os.path.join(MARKDOWN_FOLDER, os.path.basename(pdf_path).replace(".pdf", ".md"))
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(text)
        f.write(f"\n\n---\n\n**Source :** [{source_url}]({source_url})")

    logging.info(f"Converti en Markdown : {md_filename}")
    torch.cuda.empty_cache()

def process_pdf(url, date):
    logging.info(f"Téléchargement et conversion : {url} (Ajouté/Modifié le {date})")
    pdf_path = download_pdf(url)
    if pdf_path:
        convert_pdf_to_markdown(pdf_path, url)

def main():
    new_sitemap_content = download_sitemap()
    if not new_sitemap_content:
        return

    new_pdfs = parse_sitemap(new_sitemap_content)
    old_pdfs = load_local_sitemap()

    added, changed = compare_sitemaps(old_pdfs, new_pdfs)

    # Afficher le nombre de PDFs ajoutés ou modifiés
    total_pdfs = len(added) + len(changed)
    logging.info(f"{total_pdfs} PDF(s) vont être traités (Ajouté(s)/Modifié(s))")

    # Utilisation de multiprocessing pour exécuter les tâches en parallèle
    processes = []
    for url, date in {**added, **changed}.items():
        p = multiprocessing.Process(target=process_pdf, args=(url, date))
        p.start()
        processes.append(p)

        # Limiter le nombre de processus actifs pour éviter une surcharge mémoire
        if len(processes) >= 2:  # Ajuste cette valeur selon ta mémoire GPU/CPU
            for proc in processes:
                proc.join()  # Attendre la fin des processus actifs
            processes = []

    # Assurer la fin des derniers processus restants
    for proc in processes:
        proc.join()

    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Temps total d'exécution : {execution_time:.2f} secondes")
    save_sitemap(new_sitemap_content)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Meilleure compatibilité entre OS
    main()