# -*- coding: utf-8 -*-
"""
KOIB RAG - Index Building Module

Часть 2 системы RAG: чанкирование текстов, построение FAISS индекса
с multilingual эмбеддингами.

Этот модуль является рефакторированной версией оригинального скрипта
koib_rag_index_building.py с улучшенной структурой и поддержкой
переменных окружения.
"""

import os
import json
import re
import time
import datetime
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Импорт утилит из локального модуля
try:
    from .utils import (
        clean_text, text_hash, normalize_model_key,
        get_output_dir, MODEL_DISPLAY_NAMES
    )
except ImportError:
    from utils import (
        clean_text, text_hash, normalize_model_key,
        get_output_dir, MODEL_DISPLAY_NAMES
    )

# Константы (могут быть переопределены через переменные окружения)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "320"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "120"))

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", 
    "intfloat/multilingual-e5-large"
)
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


class KoibIndexBuilder:
    """
    Класс для построения векторного индекса на основе обработанных текстов.
    Загружает text_blocks.json, созданный в Part 1, разбивает на чанки
    и строит FAISS индекс.
    """
    
    def __init__(self, metadata_dir: Optional[Path] = None,
                 figures_index_path: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        """
        Инициализировать билдер индекса.
        
        Args:
            metadata_dir: Директория с метаданными из Part 1
            figures_index_path: Путь к figures_index.json
            output_dir: Выходная директория для FAISS индекса
        """
        self.output_dir = output_dir or get_output_dir()
        self.metadata_dir = metadata_dir or (self.output_dir / "metadata")
        self.faiss_index_dir = self.output_dir / "faiss_index"
        self.logs_dir = self.output_dir / "logs"
        
        # Создание директорий
        self.faiss_index_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.figures_index = []
        if figures_index_path and figures_index_path.exists():
            with open(figures_index_path, 'r', encoding='utf-8') as f:
                self.figures_index = json.load(f)
        
        self.text_blocks = []
        self.chunks = []
        self.vectorstore = None
    
    def load_text_blocks(self) -> List[Dict]:
        """
        Загрузить text_blocks.json, созданный в Part 1.
        
        Returns:
            Список текстовых блоков
            
        Raises:
            FileNotFoundError: Если файл не найден
        """
        blocks_path = self.metadata_dir / "text_blocks.json"
        if not blocks_path.exists():
            raise FileNotFoundError(
                f"❌ text_blocks.json not found at {blocks_path}. "
                "Run Part 1 first."
            )
        
        with open(blocks_path, 'r', encoding='utf-8') as f:
            self.text_blocks = json.load(f)
        
        print(f"✅ Loaded {len(self.text_blocks)} text blocks from {blocks_path}")
        return self.text_blocks
    
    def build_chunks(self) -> List[Document]:
        """
        Создать чанки из текстовых блоков, группируя по (model, source, page).
        
        Returns:
            Список документов LangChain
        """
        if not self.text_blocks:
            self.load_text_blocks()
        
        # Группировка блоков
        groups = defaultdict(list)
        for block in self.text_blocks:
            key = (
                block.get("model", "unknown"),
                block.get("source", ""),
                block.get("page", 0)
            )
            groups[key].append(block)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        all_chunks = []
        seen_hashes = set()
        
        for (model, source, page), blocks in groups.items():
            # Сортировка: сначала заголовки, потом текст, таблицы, OCR
            type_order = {"heading": 0, "text": 1, "table": 2, "ocr_text": 3}
            blocks.sort(key=lambda b: type_order.get(b.get("block_type", "text"), 1))
            
            # Объединение текста блоков
            combined = "\n\n".join([b.get("text", "") for b in blocks if b.get("text")])
            combined = clean_text(combined)
            if not combined or len(combined) < MIN_CHUNK_LEN:
                continue
            
            # Разбивка на чанки
            split_texts = splitter.split_text(combined)
            
            # Сбор метаданных: заголовки и подписи
            all_headings = []
            all_captions = []
            for b in blocks:
                if b.get("headings"):
                    all_headings.extend(b["headings"])
                if b.get("block_type") == "heading":
                    all_headings.append(b["text"])
                if b.get("caption"):
                    all_captions.append(b["caption"])
            
            # Проверка наличия рисунков на этой странице/файле
            relevant_figs = [
                f for f in self.figures_index
                if f.get("source") == source and f.get("page") == page 
                and f.get("model") == model
            ]
            has_figures = len(relevant_figs) > 0
            
            for chunk_text in split_texts:
                chunk_text = chunk_text.strip()
                if len(chunk_text) < MIN_CHUNK_LEN:
                    continue
                
                # Дедупликация
                h = text_hash(chunk_text)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "model": model,
                        "source": source,
                        "page": page,
                        "chunk_type": "mixed",
                        "has_figures": has_figures,
                        "captions": "; ".join(all_captions[:3]),
                        "headings": "; ".join(all_headings[:5]),
                    }
                )
                all_chunks.append(doc)
        
        self.chunks = all_chunks
        print(f"✅ Created {len(self.chunks)} chunks from {len(self.text_blocks)} blocks")
        return self.chunks
    
    def save_chunks(self) -> None:
        """Сохранить чанки в metadata/chunks.json."""
        chunks_path = self.metadata_dir / "chunks.json"
        chunks_data = []
        for doc in self.chunks:
            chunks_data.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
            })
        
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Chunks saved to {chunks_path} ({len(self.chunks)} chunks)")
    
    def build_faiss_index(self) -> FAISS:
        """
        Построить FAISS индекс на основе чанков.
        
        Returns:
            Векторное хранилище FAISS
            
        Raises:
            ValueError: Если нет чанков для индексирования
        """
        if not self.chunks:
            self.build_chunks()
        if not self.chunks:
            raise ValueError("No chunks to index. Check text_blocks.json.")
        
        print(f"\n🚀 Building FAISS index with model: {EMBEDDING_MODEL_NAME}")
        
        # Определение устройства (CUDA если доступно)
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        
        print(f"   Using device: {device}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": device},
        )
        
        # Добавляем префикс passage: для каждого текста
        texts = [PASSAGE_PREFIX + doc.page_content for doc in self.chunks]
        metadatas = [doc.metadata for doc in self.chunks]
        
        print(f"   Creating FAISS index from {len(texts)} texts...")
        self.vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        
        # Сохранение индекса на диск
        index_path = self.faiss_index_dir / "koib_index"
        self.vectorstore.save_local(str(index_path))
        print(f"✅ FAISS index saved to {index_path}")
        
        # Сохранение метаинформации об индексе
        meta_path = self.faiss_index_dir / "index_meta.json"
        meta = {
            "model_name": EMBEDDING_MODEL_NAME,
            "num_chunks": len(self.chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "passage_prefix": PASSAGE_PREFIX,
            "created": datetime.datetime.now().isoformat(),
        }
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Index metadata saved to {meta_path}")
        return self.vectorstore


def main():
    """Точка входа для запуска построения индекса."""
    print("╔" + "═"*68 + "╗")
    print("║" + "  KOIB RAG v4 – PART 2: INDEX BUILDING".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    start_time = time.time()
    
    # Получение директорий
    output_dir = get_output_dir()
    metadata_dir = output_dir / "metadata"
    
    # Проверка существования директории метаданных
    if not metadata_dir.exists():
        print(f"❌ Metadata directory {metadata_dir} not found. Run Part 1 first.")
        return
    
    # Загрузка figures_index.json
    figures_path = metadata_dir / "figures_index.json"
    if not figures_path.exists():
        print("⚠️ figures_index.json not found. Proceeding without figure metadata.")
        figures_path = None
    
    builder = KoibIndexBuilder(metadata_dir, figures_path, output_dir)
    
    # Шаг 1: загрузить text_blocks.json
    try:
        builder.load_text_blocks()
    except FileNotFoundError as e:
        print(e)
        return
    
    # Шаг 2: построить чанки
    chunks = builder.build_chunks()
    if not chunks:
        print("❌ No chunks created. Check your text_blocks.json content.")
        return
    
    # Шаг 3: сохранить chunks.json
    builder.save_chunks()
    
    # Шаг 4: построить FAISS индекс
    try:
        builder.build_faiss_index()
    except Exception as e:
        print(f"❌ FAISS building failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total indexing time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("✅ PART 2 FINISHED. Now you can run PART 3 (query engine).")


if __name__ == "__main__":
    main()
