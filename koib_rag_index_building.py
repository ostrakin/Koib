# -*- coding: utf-8 -*-
"""PART 2 – KOIB RAG Index Building (chunking + FAISS)"""

# ============================================================================
# 1. Установка зависимостей (только для Colab)
# ============================================================================
!pip uninstall -q -y langchain langchain-core langchain-community langchain-text-splitters langchain-huggingface 2>/dev/null || true
!pip install -q langchain-core langchain langchain-text-splitters langchain-community langchain-huggingface faiss-cpu sentence-transformers

import os
import json
import re
import time
import datetime
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

# LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================================
# 2. Константы (должны совпадать с Part 1)
# ============================================================================
OUTPUT_DIR = Path("/content/drive/MyDrive/Koib/koib_rag_GLM1")
METADATA_DIR = OUTPUT_DIR / "metadata"
FAISS_INDEX_DIR = OUTPUT_DIR / "faiss_index"
LOGS_DIR = OUTPUT_DIR / "logs"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 320
MIN_CHUNK_LEN = 120

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

MODEL_DISPLAY_NAMES = {
    "koib2010": "КОИБ-2010",
    "koib2017a": "КОИБ-2017А",
    "koib2017b": "КОИБ-2017Б",
    "unknown": "Неизвестная модель",
}

# Создаём папки, если их нет
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 3. Утилиты (копия необходимых функций из Part 1)
# ============================================================================
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\x20-\x7E\u0400-\u04FF\u2116\n\r\t]', '', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()

def text_hash(text):
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()[:12]

def normalize_model_key(key):
    key = str(key).strip().lower()
    return key if key in ["koib2010", "koib2017a", "koib2017b"] else "unknown"

# ============================================================================
# 4. Класс для построения индекса (загружает text_blocks.json)
# ============================================================================
class KoibIndexBuilder:
    def __init__(self, metadata_dir: Path, figures_index_path: Path = None):
        self.metadata_dir = metadata_dir
        self.figures_index = []
        if figures_index_path and figures_index_path.exists():
            with open(figures_index_path, 'r', encoding='utf-8') as f:
                self.figures_index = json.load(f)
        self.text_blocks = []
        self.chunks = []
        self.vectorstore = None

    def load_text_blocks(self):
        """Загружает text_blocks.json, созданный в Part 1."""
        blocks_path = self.metadata_dir / "text_blocks.json"
        if not blocks_path.exists():
            raise FileNotFoundError(f"❌ text_blocks.json not found at {blocks_path}. Run Part 1 first.")
        with open(blocks_path, 'r', encoding='utf-8') as f:
            self.text_blocks = json.load(f)
        print(f"✅ Loaded {len(self.text_blocks)} text blocks from {blocks_path}")
        return self.text_blocks

    def build_chunks(self) -> List[Document]:
        """Создаёт чанки из текстовых блоков, группируя по (model, source, page)."""
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
                if f.get("source") == source and f.get("page") == page and f.get("model") == model
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

    def save_chunks(self):
        """Сохраняет чанки в metadata/chunks.json для последующего использования."""
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

    def build_faiss_index(self):
        """Строит FAISS индекс на основе чанков."""
        if not self.chunks:
            self.build_chunks()
        if not self.chunks:
            raise ValueError("No chunks to index. Check text_blocks.json and chunking parameters.")

        print(f"\n🚀 Building FAISS index with model: {EMBEDDING_MODEL_NAME}")
        # Определяем устройство (CUDA если доступно)
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

        # Сохраняем индекс на диск
        index_path = FAISS_INDEX_DIR / "koib_index"
        self.vectorstore.save_local(str(index_path))
        print(f"✅ FAISS index saved to {index_path}")

        # Сохраняем метаинформацию об индексе
        meta_path = FAISS_INDEX_DIR / "index_meta.json"
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

# ============================================================================
# 5. MAIN – запуск построения индекса
# ============================================================================
def main():
    print("╔" + "═"*68 + "╗")
    print("║" + "  KOIB RAG v4 – PART 2: INDEX BUILDING".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    start_time = time.time()

    # Проверяем, что директории существуют
    if not METADATA_DIR.exists():
        print(f"❌ Metadata directory {METADATA_DIR} not found. Run Part 1 first.")
        return

    # Загружаем figures_index.json (нужен для определения наличия рисунков в чанках)
    figures_path = METADATA_DIR / "figures_index.json"
    if not figures_path.exists():
        print("⚠️ figures_index.json not found. Proceeding without figure metadata.")
        figures_path = None

    builder = KoibIndexBuilder(METADATA_DIR, figures_path)

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