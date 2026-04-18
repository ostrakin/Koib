# -*- coding: utf-8 -*-
"""
KOIB RAG - Preprocessing Module

Часть 1 системы RAG для обработки документации КОИБ:
извлечение текста, OCR, рисунков, определение моделей.

Этот модуль является рефакторированной версией оригинального скрипта
koib_rag_preprocessing.py с улучшенной структурой и поддержкой
переменных окружения.
"""

import os
import re
import json
import csv
import io
import time
import datetime
import hashlib
from pathlib import Path
from collections import defaultdict
import logging
from typing import List, Dict, Any, Tuple, Optional

import fitz               # pymupdf
from docx import Document as DocxDocument
from PIL import Image
import pytesseract
import numpy as np
from tqdm import tqdm

# Импорт утилит из локального модуля
try:
    from .utils import (
        clean_text, text_hash, normalize_model_key,
        detect_model_in_text, detect_model_from_filename,
        find_figure_caption, ensure_dirs, get_docs_dir, get_output_dir,
        KOIB_MODEL_PATTERNS, MODEL_DISPLAY_NAMES, FIGURE_CAPTION_PATTERNS
    )
except ImportError:
    from utils import (
        clean_text, text_hash, normalize_model_key,
        detect_model_in_text, detect_model_from_filename,
        find_figure_caption, ensure_dirs, get_docs_dir, get_output_dir,
        KOIB_MODEL_PATTERNS, MODEL_DISPLAY_NAMES, FIGURE_CAPTION_PATTERNS
    )

# EasyOCR (ленивая загрузка)
_easyocr_reader = None


def get_easyocr_reader():
    """Получить или создать экземпляр EasyOCR reader."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['ru', 'en'], gpu=True, verbose=False)
    return _easyocr_reader


# Параметры обработки (могут быть переопределены через переменные окружения)
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "50"))
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "80"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "80"))
SCREENSHOT_AREA_THRESHOLD = float(os.getenv("SCREENSHOT_AREA_THRESHOLD", "0.80"))
TEXT_DENSITY_THRESHOLD = float(os.getenv("TEXT_DENSITY_THRESHOLD", "0.35"))


def ocr_image(image_pil, lang='rus+eng') -> str:
    """
    Выполнить OCR на изображении.
    
    Args:
        image_pil: PIL Image
        lang: Языки для распознавания
        
    Returns:
        Распознанный текст
    """
    if image_pil is None:
        return ""
    
    # pytesseract
    try:
        text_tess = pytesseract.image_to_string(image_pil, lang=lang, config='--psm 6')
        text_tess = clean_text(text_tess)
        if len(text_tess) >= 30:
            return text_tess
    except Exception:
        text_tess = ""
    
    # easyocr fallback
    try:
        reader = get_easyocr_reader()
        img_np = np.array(image_pil)
        results = reader.readtext(img_np, paragraph=True, detail=0)
        text_easy = clean_text('\n'.join(results))
        if len(text_easy) >= 20:
            return text_easy
    except Exception:
        text_easy = ""
    
    return text_tess if len(text_tess) >= len(text_easy) else text_easy


def detect_scanned_page(page, min_text_chars: int = 50) -> bool:
    """
    Определить, является ли страница отсканированной.
    
    Args:
        page: Страница PDF
        min_text_chars: Минимальное количество символов текста
        
    Returns:
        True если страница отсканированная
    """
    try:
        text = page.get_text("text").strip()
        images = page.get_images(full=True)
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        
        if len(text) < min_text_chars:
            for img_info in images:
                try:
                    xref = img_info[0]
                    base_image = page.parent.extract_image(xref)
                    if not base_image:
                        continue
                    img_bytes = base_image["image"]
                    img = Image.open(io.BytesIO(img_bytes))
                    img_area = img.width * img.height
                    if img_area / page_area > SCREENSHOT_AREA_THRESHOLD:
                        return True
                except Exception:
                    continue
            return True
        
        if images and len(text) < min_text_chars * 3:
            return True
        
        return False
    except Exception:
        return False


def extract_text_from_pdf(pdf_path: Path) -> Tuple[List[Dict], List[Dict], int]:
    """
    Извлечь текст, изображения и выполнить OCR для PDF файла.
    
    Args:
        pdf_path: Путь к PDF файлу
        
    Returns:
        Кортеж (текстовые блоки, изображения, количество OCR страниц)
    """
    text_blocks = []
    figures = []
    ocr_count = 0
    
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            # Извлечение текста
            text = page.get_text("text").strip()
            
            # Проверка на отсканированную страницу
            is_scanned = detect_scanned_page(page)
            
            if is_scanned:
                # OCR для отсканированных страниц
                pix = page.get_pixmap(matrix=fitz.Matrix(OCR_DPI/72, OCR_DPI/72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = ocr_image(img)
                if len(ocr_text) >= OCR_MIN_TEXT_CHARS:
                    text = ocr_text
                    ocr_count += 1
            
            if text:
                caption = find_figure_caption(text)
                text_blocks.append({
                    "file": str(pdf_path),
                    "page": page_num + 1,
                    "text": text,
                    "caption": caption,
                    "hash": text_hash(text)
                })
            
            # Извлечение изображений
            images = page.get_images(full=True)
            for img_info in images:
                try:
                    xref = img_info[0]
                    base_image = page.parent.extract_image(xref)
                    if not base_image:
                        continue
                    
                    img_bytes = base_image["image"]
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    if img.width < MIN_IMAGE_WIDTH or img.height < MIN_IMAGE_HEIGHT:
                        continue
                    
                    # Сохранение изображения
                    img_hash = hashlib.md5(img_bytes).hexdigest()[:12]
                    img_filename = f"figure_{pdf_path.stem}_p{page_num+1}_{img_hash}.{base_image['ext']}"
                    img_path = FIGURES_DIR / img_filename
                    img.save(img_path)
                    
                    # Поиск подписи
                    rect = fitz.Rect(img_info[1:5])
                    nearby_text = page.get_text("text", clip=rect.expand(50, 50, 50, 50))
                    caption = find_figure_caption(nearby_text)
                    
                    figures.append({
                        "file": str(pdf_path),
                        "page": page_num + 1,
                        "image_path": str(img_path),
                        "caption": caption,
                        "width": img.width,
                        "height": img.height
                    })
                except Exception:
                    continue
        
        doc.close()
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_path}: {e}")
    
    return text_blocks, figures, ocr_count


def extract_text_from_docx(docx_path: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Извлечь текст и изображения из DOCX файла.
    
    Args:
        docx_path: Путь к DOCX файлу
        
    Returns:
        Кортеж (текстовые блоки, изображения)
    """
    text_blocks = []
    figures = []
    
    try:
        doc = DocxDocument(docx_path)
        
        # Извлечение текста
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        
        if full_text:
            text = '\n'.join(full_text)
            text_blocks.append({
                "file": str(docx_path),
                "page": 0,
                "text": text,
                "caption": "",
                "hash": text_hash(text)
            })
        
        # Извлечение изображений
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    img_bytes = rel.target_part.blob
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    if img.width < MIN_IMAGE_WIDTH or img.height < MIN_IMAGE_HEIGHT:
                        continue
                    
                    img_hash = hashlib.md5(img_bytes).hexdigest()[:12]
                    ext = rel.target_ref.split('.')[-1] if '.' in rel.target_ref else 'png'
                    img_filename = f"figure_{docx_path.stem}_{img_hash}.{ext}"
                    img_path = FIGURES_DIR / img_filename
                    img.save(img_path)
                    
                    figures.append({
                        "file": str(docx_path),
                        "page": 0,
                        "image_path": str(img_path),
                        "caption": "",
                        "width": img.width,
                        "height": img.height
                    })
                except Exception:
                    continue
    
    except Exception as e:
        logging.error(f"Error processing DOCX {docx_path}: {e}")
    
    return text_blocks, figures


def generate_source_models(docs_dir: Path, metadata_dir: Path) -> Dict[str, str]:
    """
    Сгенерировать файл source_models.json с моделями для каждого файла.
    
    Args:
        docs_dir: Директория с документами
        metadata_dir: Директория для метаданных
        
    Returns:
        Словарь {имя_файла: модель}
    """
    source_models = {}
    
    for file_path in docs_dir.glob("*"):
        if file_path.suffix.lower() not in ['.pdf', '.docx']:
            continue
        
        filename = file_path.name
        model = detect_model_from_filename(filename)
        source_models[filename] = model
    
    # Сохранение
    sm_path = metadata_dir / "source_models.json"
    with open(sm_path, 'w', encoding='utf-8') as f:
        json.dump(source_models, f, ensure_ascii=False, indent=2)
    
    print(f"✅ source_models.json generated: {len(source_models)} files")
    return source_models


class KoibPreprocessingPipeline:
    """
    Основной пайплайн предобработки документов КОИБ.
    """
    
    def __init__(self, docs_dir: Optional[Path] = None, 
                 output_dir: Optional[Path] = None,
                 source_models: Optional[Dict[str, str]] = None):
        """
        Инициализировать пайплайн.
        
        Args:
            docs_dir: Директория с документами
            output_dir: Выходная директория
            source_models: Словарь моделей файлов
        """
        self.docs_dir = docs_dir or get_docs_dir()
        self.output_dir = output_dir or get_output_dir()
        
        # Поддиректории
        self.classified_dir = self.output_dir / "classified"
        self.ocr_results_dir = self.output_dir / "ocr_results"
        self.figures_dir = self.output_dir / "figures"
        self.metadata_dir = self.output_dir / "metadata"
        self.logs_dir = self.output_dir / "logs"
        
        # Создание директорий
        ensure_dirs(self.output_dir, self.classified_dir, self.ocr_results_dir,
                   self.figures_dir, self.metadata_dir, self.logs_dir)
        
        # Глобальные переменные для совместимости
        global DOCS_DIR, OUTPUT_DIR, CLASSIFIED_DIR, OCR_RESULTS_DIR
        global FIGURES_DIR, METADATA_DIR, LOGS_DIR
        
        DOCS_DIR = self.docs_dir
        OUTPUT_DIR = self.output_dir
        CLASSIFIED_DIR = self.classified_dir
        OCR_RESULTS_DIR = self.ocr_results_dir
        FIGURES_DIR = self.figures_dir
        METADATA_DIR = self.metadata_dir
        LOGS_DIR = self.logs_dir
        
        # Данные
        self.source_models = source_models or {}
        self.text_blocks = []
        self.figures_index = []
        self.processing_log = []
        self.total_files = 0
        self.total_ocr_pages = 0
        self.classification_report = {}
    
    def process_all(self) -> List[Dict]:
        """
        Обработать все документы в директории.
        
        Returns:
            Список текстовых блоков
        """
        print(f"\n📂 Processing documents from: {self.docs_dir}")
        
        files = list(self.docs_dir.glob("*.pdf")) + list(self.docs_dir.glob("*.docx"))
        self.total_files = len(files)
        
        if not files:
            print("⚠️ No PDF or DOCX files found!")
            return []
        
        for file_path in tqdm(files, desc="Processing files"):
            self._process_file(file_path)
        
        self._build_classification_report()
        return self.text_blocks
    
    def _process_file(self, file_path: Path) -> None:
        """Обработать один файл."""
        start_time = time.time()
        filename = file_path.name
        suffix = file_path.suffix.lower()
        
        # Определение модели
        model = self.source_models.get(filename, detect_model_from_filename(filename))
        
        if suffix == '.pdf':
            text_blocks, figures, ocr_count = extract_text_from_pdf(file_path)
            file_type = 'pdf'
        elif suffix == '.docx':
            text_blocks, figures = extract_text_from_docx(file_path)
            ocr_count = 0
            file_type = 'docx'
        else:
            return
        
        self.text_blocks.extend(text_blocks)
        self.figures_index.extend(figures)
        self.total_ocr_pages += ocr_count
        
        elapsed = time.time() - start_time
        
        self.processing_log.append({
            "file": filename,
            "type": file_type,
            "model": model,
            "text_blocks": len(text_blocks),
            "figures": len(figures),
            "ocr_pages": ocr_count,
            "time_sec": round(elapsed, 2)
        })
    
    def _build_classification_report(self) -> None:
        """Построить отчет классификации."""
        report = {
            "total_files": self.total_files,
            "by_type": defaultdict(int),
            "by_model": defaultdict(int),
            "by_model_type": defaultdict(int),
            "total_text_blocks": len(self.text_blocks),
            "total_figures": len(self.figures_index),
            "total_ocr_pages": self.total_ocr_pages,
            "files": [],
        }
        
        for entry in self.processing_log:
            report["by_type"][entry["type"]] += 1
            report["by_model"][entry["model"]] += 1
            report["by_model_type"][f"{entry['model']}_{entry['type']}"] += 1
            report["files"].append(entry)
        
        self.classification_report = dict(report)
    
    def save_artifacts(self) -> None:
        """Сохранить артефакты."""
        print("\n--- Saving artifacts ---")
        
        # text_blocks.json
        blocks_path = self.metadata_dir / "text_blocks.json"
        with open(blocks_path, 'w', encoding='utf-8') as f:
            json.dump(self.text_blocks, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Text blocks saved: {blocks_path} ({len(self.text_blocks)} blocks)")
        
        # figures_index.json
        figures_path = self.metadata_dir / "figures_index.json"
        with open(figures_path, 'w', encoding='utf-8') as f:
            json.dump(self.figures_index, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Figures index saved: {figures_path} ({len(self.figures_index)} figures)")
        
        # processing_log.json
        log_path = self.logs_dir / "processing_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.processing_log, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Processing log saved: {log_path}")
        
        # classification_report.csv
        csv_path = self.classified_dir / "classification_report.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["File", "Type", "Model", "Text Blocks", "Figures", "OCR Pages", "Time (s)"])
            for entry in self.processing_log:
                writer.writerow([entry["file"], entry["type"], entry["model"],
                                entry["text_blocks"], entry["figures"],
                                entry["ocr_pages"], entry["time_sec"]])
        print(f"  ✅ Classification report saved: {csv_path}")
    
    def print_summary(self) -> None:
        """Вывести сводку обработки."""
        cr = self.classification_report
        print("\n" + "="*70)
        print("📊 PREPROCESSING SUMMARY")
        print("="*70)
        print(f"  Total files:          {cr.get('total_files', 0)}")
        print(f"  By type: PDF={cr['by_type'].get('pdf', 0)} DOCX={cr['by_type'].get('docx', 0)}")
        print(f"  By model: {dict(cr.get('by_model', {}))}")
        print(f"  Text blocks:          {cr.get('total_text_blocks', 0)}")
        print(f"  Figures extracted:    {cr.get('total_figures', 0)}")
        print(f"  OCR pages/images:     {cr.get('total_ocr_pages', 0)}")
        print("="*70)


def main():
    """Точка входа для запуска предобработки."""
    print("╔" + "═"*68 + "╗")
    print("║" + "  KOIB RAG v4 – PART 1: PREPROCESSING".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    start_time = time.time()
    
    # Получение директорий
    docs_dir = get_docs_dir()
    output_dir = get_output_dir()
    metadata_dir = output_dir / "metadata"
    
    # Генерация source_models.json (если ещё нет)
    sm_path = metadata_dir / "source_models.json"
    if sm_path.exists():
        print(f"\n📂 Loading existing source_models.json from {sm_path}")
        with open(sm_path, 'r', encoding='utf-8') as f:
            source_models = json.load(f)
    else:
        print("\n📂 Generating source_models.json...")
        ensure_dirs(metadata_dir)
        source_models = generate_source_models(docs_dir, metadata_dir)
    
    # Запуск пайплайна предобработки
    pipeline = KoibPreprocessingPipeline(docs_dir, output_dir, source_models)
    pipeline.process_all()
    pipeline.save_artifacts()
    pipeline.print_summary()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total preprocessing time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("✅ PART 1 FINISHED. Now you can run PART 2 (index building).")


if __name__ == "__main__":
    main()
