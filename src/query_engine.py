# -*- coding: utf-8 -*-
"""
KOIB RAG - Query Engine Module

Часть 3 системы RAG: поисковый движок, интерактивные запросы,
интеграция с LLM и VK ботом.

Этот модуль является рефакторированной версией оригинального скрипта
koib_rag_query_engine.py с улучшенной структурой и поддержкой
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
from typing import List, Dict, Any, Tuple, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Импорт утилит из локального модуля
try:
    from .utils import (
        clean_text, text_hash, normalize_model_key,
        get_output_dir, MODEL_DISPLAY_NAMES, KOIB_MODEL_PATTERNS
    )
except ImportError:
    from utils import (
        clean_text, text_hash, normalize_model_key,
        get_output_dir, MODEL_DISPLAY_NAMES, KOIB_MODEL_PATTERNS
    )

# Константы (могут быть переопределены через переменные окружения)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", 
    "intfloat/multilingual-e5-large"
)
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
FAISS_SEARCH_K = int(os.getenv("FAISS_SEARCH_K", "5"))


class KoibQueryEngine:
    """
    Поисковый движок для системы RAG КОИБ.
    Загружает FAISS индекс и метаданные, предоставляет методы
    для поиска и генерации контекста для LLM.
    """
    
    def __init__(self, faiss_index_path: Optional[Path] = None,
                 figures_index_path: Optional[Path] = None,
                 chunks_path: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        """
        Инициализировать query engine.
        
        Args:
            faiss_index_path: Путь к FAISS индексу
            figures_index_path: Путь к figures_index.json
            chunks_path: Путь к chunks.json
            output_dir: Выходная директория
        """
        self.output_dir = output_dir or get_output_dir()
        self.metadata_dir = self.output_dir / "metadata"
        self.faiss_index_dir = self.output_dir / "faiss_index"
        self.logs_dir = self.output_dir / "logs"
        self.figures_dir = self.output_dir / "figures"
        
        self.vectorstore = None
        self.embeddings = None
        self.figures_index = []
        self.chunks_data = []
        
        # Пути по умолчанию
        if faiss_index_path is None:
            faiss_index_path = self.faiss_index_dir / "koib_index"
        if figures_index_path is None:
            figures_index_path = self.metadata_dir / "figures_index.json"
        if chunks_path is None:
            chunks_path = self.metadata_dir / "chunks.json"
        
        # Загрузка FAISS индекса
        try:
            device = "cuda" if self._has_cuda() else "cpu"
            print(f"Loading embedding model {EMBEDDING_MODEL_NAME} on {device}...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                encode_kwargs={"normalize_embeddings": True},
                model_kwargs={"device": device},
            )
            
            if faiss_index_path.exists():
                # Примечание: allow_dangerous_deserialization=True требуется FAISS
                # для загрузки сериализованных данных. Убедитесь, что индекс
                # получен из доверенного источника.
                self.vectorstore = FAISS.load_local(
                    str(faiss_index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"✅ FAISS index loaded from {faiss_index_path}")
            else:
                print(f"⚠️ FAISS index not found at {faiss_index_path}")
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
        
        # Загрузка индекса рисунков
        try:
            if figures_index_path.exists():
                with open(figures_index_path, 'r', encoding='utf-8') as f:
                    self.figures_index = json.load(f)
                print(f"✅ Figures index loaded: {len(self.figures_index)} figures")
        except Exception as e:
            print(f"⚠️ Could not load figures index: {e}")
        
        # Загрузка чанков (опционально)
        try:
            if chunks_path.exists():
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self.chunks_data = json.load(f)
                print(f"✅ Chunks loaded: {len(self.chunks_data)} chunks")
        except Exception:
            pass
    
    def _has_cuda(self) -> bool:
        """Проверить доступность CUDA."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _search(self, query: str, model_filter: str = "", 
                k: int = FAISS_SEARCH_K) -> List[Tuple[Document, float]]:
        """
        Выполнить поиск в векторном индексе.
        
        Args:
            query: Поисковый запрос
            model_filter: Фильтр по модели КОИБ
            k: Количество результатов
            
        Returns:
            Список кортежей (документ, оценка релевантности)
        """
        if self.vectorstore is None:
            print("❌ FAISS index not loaded!")
            return []
        
        query_text = QUERY_PREFIX + query
        results = self.vectorstore.similarity_search_with_score(query_text, k=k*3)
        
        if model_filter:
            model_filter = normalize_model_key(model_filter)
            results = [
                (doc, score) for doc, score in results
                if doc.metadata.get("model") == model_filter
            ][:k]
        else:
            results = results[:k]
        
        return results
    
    def ask(self, query: str, koib_model: str = "", 
            k: int = FAISS_SEARCH_K) -> Tuple[str, List[Dict], List[Dict]]:
        """
        Выполнить поисковый запрос и вернуть контекст, документы и рисунки.
        
        Args:
            query: Поисковый запрос
            koib_model: Модель КОИБ для фильтрации
            k: Количество результатов
            
        Returns:
            Кортеж (context_text, relevant_docs, relevant_figures)
        """
        results = self._search(query, koib_model, k)
        if not results:
            return "", [], []
        
        context_parts = []
        relevant_docs = []
        sources_used = set()
        
        for doc, score in results:
            meta = doc.metadata
            source = meta.get("source", "unknown")
            page = meta.get("page", "?")
            model = meta.get("model", "unknown")
            has_figs = meta.get("has_figures", False)
            headings = meta.get("headings", "")
            captions = meta.get("captions", "")
            
            source_key = f"{source}_p{page}"
            if source_key in sources_used:
                continue
            sources_used.add(source_key)
            
            model_display = MODEL_DISPLAY_NAMES.get(model, model)
            header = f"--- [{model_display}] {source}, стр. {page}"
            if headings:
                header += f" | {headings}"
            if has_figs:
                header += " [🖼️ есть рисунки]"
            if captions:
                header += f"\n    Картинки: {captions}"
            
            context_parts.append(f"{header}\n{doc.page_content}")
            relevant_docs.append({
                "source": source,
                "page": page,
                "model": model,
                "score": round(float(score), 4),
                "text": (doc.page_content[:300] + "...") 
                       if len(doc.page_content) > 300 else doc.page_content,
                "has_figures": has_figs,
            })
        
        # Поиск релевантных рисунков
        relevant_figures = self._find_figures(query, koib_model, sources_used)
        
        context_text = "\n\n".join(context_parts)
        return context_text, relevant_docs, relevant_figures
    
    def _find_figures(self, query: str, koib_model: str, 
                      sources_used: set) -> List[Dict]:
        """
        Найти релевантные рисунки по запросу.
        
        Args:
            query: Поисковый запрос
            koib_model: Модель КОИБ для фильтрации
            sources_used: Использованные источники
            
        Returns:
            Список релевантных рисунков
        """
        if not self.figures_index:
            return []
        
        model_filter = normalize_model_key(koib_model) if koib_model else ""
        query_words = set(re.findall(r'\w+', query.lower()))
        stop_words = {
            'как', 'что', 'где', 'когда', 'почему', 'для', 'это', 
            'на', 'в', 'с', 'и', 'или', 'не', 'по', 'к', 'от', 'до', 
            'при', 'об', 'о'
        }
        query_words -= stop_words
        
        scored = []
        for fig in self.figures_index:
            if model_filter and fig.get("model") != model_filter:
                continue
            if sources_used:
                fig_source_key = f"{fig.get('source', '')}_p{fig.get('page', 0)}"
                if fig_source_key not in sources_used:
                    continue
            
            caption = (fig.get("caption") or "").lower()
            surrounding = (fig.get("surrounding_text") or "").lower()
            combined = caption + " " + surrounding
            fig_words = set(re.findall(r'\w+', combined))
            overlap = query_words & fig_words
            score = len(overlap) / max(len(query_words), 1)
            
            if score > 0.1 or not query_words:
                scored.append((fig, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [fig for fig, _ in scored[:5]]
    
    def ask_with_llm_context(self, query: str, koib_model: str = "", 
                             k: int = FAISS_SEARCH_K) -> str:
        """
        Сгенерировать форматированный контекст для передачи в LLM.
        
        Args:
            query: Поисковый запрос
            koib_model: Модель КОИБ для фильтрации
            k: Количество результатов
            
        Returns:
            Форматированный контекст для LLM
        """
        context_text, docs, figures = self.ask(query, koib_model, k)
        if not context_text:
            return ("Контекст не найден. Нет релевантных документов "
                   "по данному запросу.")
        
        model_display = (
            MODEL_DISPLAY_NAMES.get(normalize_model_key(koib_model), "Все модели")
            if koib_model else "Все модели"
        )
        
        parts = [
            f"КОНТЕКСТ ИЗ ТЕХНИЧЕСКОЙ ДОКУМЕНТАЦИИ КОИБ ({model_display}):",
            "",
            context_text,
        ]
        
        if figures:
            parts.append("")
            parts.append("РЕЛЕВАНТНЫЕ РИСУНКИ И СХЕМЫ:")
            for fig in figures:
                fig_model = MODEL_DISPLAY_NAMES.get(
                    fig.get("model", ""), fig.get("model", "")
                )
                caption = fig.get("caption", "без подписи")
                source = fig.get("source", "")
                parts.append(f"  - [{fig_model}] {source}: {caption}")
        
        parts.extend([
            "",
            "ИНСТРУКЦИЯ:",
            "На основе приведенного контекста из технической документации КОИБ, "
            f"ответь на вопрос:",
            f"«{query}»",
            "Если ответ не содержится в контексте, явно укажи это.",
            "При ответе ссылайся на конкретный документ и страницу.",
        ])
        
        return "\n".join(parts)


def main():
    """Точка входа для запуска query engine."""
    print("╔" + "═"*68 + "╗")
    print("║" + "  KOIB RAG v4 – PART 3: QUERY ENGINE".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    output_dir = get_output_dir()
    engine = KoibQueryEngine(output_dir=output_dir)
    
    if engine.vectorstore is None:
        print("❌ Failed to initialize query engine. "
              "Check that Part 2 was run successfully.")
        return
    
    # Автоматический тест
    print("\n🧪 Автоматический тестовый запрос...")
    _, docs, _ = engine.ask("Как включить КОИБ-2010?", k=3)
    if docs:
        print(f"✅ Тест успешен: найдено {len(docs)} результатов")
    else:
        print("⚠️ Тестовый запрос не вернул результатов")
    
    print("\n✅ Query engine initialized successfully.")
    print("   Use engine.ask(query, koib_model) for searching.")


if __name__ == "__main__":
    main()
