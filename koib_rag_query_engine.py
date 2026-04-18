# -*- coding: utf-8 -*-
"""PART 3 – KOIB RAG Query Engine & Interaction"""

# ============================================================================
# 1. Установка зависимостей (для Colab)
# ============================================================================
!pip install -q langchain-core langchain langchain-community faiss-cpu sentence-transformers
!pip install -q vk-api  # опционально, для VK бота

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

# ============================================================================
# 2. Константы (должны совпадать с Part 1 и Part 2)
# ============================================================================
OUTPUT_DIR = Path("/content/drive/MyDrive/Koib/koib_rag_GLM1")
METADATA_DIR = OUTPUT_DIR / "metadata"
FAISS_INDEX_DIR = OUTPUT_DIR / "faiss_index"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
FAISS_SEARCH_K = 5

MODEL_DISPLAY_NAMES = {
    "koib2010": "КОИБ-2010",
    "koib2017a": "КОИБ-2017А",
    "koib2017b": "КОИБ-2017Б",
    "unknown": "Неизвестная модель",
}

KOIB_MODEL_PATTERNS = {
    "koib2010": [r"КОИБ[-\s]?2010", r"КОИБ\s*2010", r"0912054"],
    "koib2017a": [r"КОИБ[-\s]?2017\s*[АA]", r"КОИБ[-\s]?2017А"],
    "koib2017b": [r"КОИБ[-\s]?2017\s*[БB]", r"КОИБ[-\s]?2017Б"],
}

# ============================================================================
# 3. Утилиты (необходимые для работы)
# ============================================================================
def normalize_model_key(key):
    key = str(key).strip().lower()
    return key if key in KOIB_MODEL_PATTERNS else "unknown"

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

# ============================================================================
# 4. Класс Query Engine (загружает FAISS и метаданные)
# ============================================================================
class KoibQueryEngine:
    def __init__(self, faiss_index_path: Optional[Path] = None, 
                 figures_index_path: Optional[Path] = None,
                 chunks_path: Optional[Path] = None):
        self.vectorstore = None
        self.embeddings = None
        self.figures_index = []
        self.chunks_data = []  # для справки

        # Пути по умолчанию
        if faiss_index_path is None:
            faiss_index_path = FAISS_INDEX_DIR / "koib_index"
        if figures_index_path is None:
            figures_index_path = METADATA_DIR / "figures_index.json"
        if chunks_path is None:
            chunks_path = METADATA_DIR / "chunks.json"

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

    def _has_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _search(self, query: str, model_filter: str = "", k: int = FAISS_SEARCH_K):
        if self.vectorstore is None:
            print("❌ FAISS index not loaded!")
            return []
        query_text = QUERY_PREFIX + query
        results = self.vectorstore.similarity_search_with_score(query_text, k=k*3)
        if model_filter:
            model_filter = normalize_model_key(model_filter)
            results = [(doc, score) for doc, score in results 
                       if doc.metadata.get("model") == model_filter][:k]
        else:
            results = results[:k]
        return results

    def ask(self, query: str, koib_model: str = "", k: int = FAISS_SEARCH_K):
        """Возвращает (context_text, relevant_docs, relevant_figures)"""
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
                "text": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "has_figures": has_figs,
            })

        # Поиск релевантных рисунков
        relevant_figures = self._find_figures(query, koib_model, sources_used)

        context_text = "\n\n".join(context_parts)
        return context_text, relevant_docs, relevant_figures

    def _find_figures(self, query: str, koib_model: str, sources_used: set):
        if not self.figures_index:
            return []
        model_filter = normalize_model_key(koib_model) if koib_model else ""
        query_words = set(re.findall(r'\w+', query.lower()))
        stop_words = {'как', 'что', 'где', 'когда', 'почему', 'для', 'это', 'на', 'в', 'с', 'и', 'или', 'не', 'по', 'к', 'от', 'до', 'при', 'об', 'о'}
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

    def ask_with_llm_context(self, query: str, koib_model: str = "", k: int = FAISS_SEARCH_K):
        """Генерирует форматированный контекст для передачи в LLM (GigaChat и др.)"""
        context_text, docs, figures = self.ask(query, koib_model, k)
        if not context_text:
            return "Контекст не найден. Нет релевантных документов по данному запросу."

        model_display = MODEL_DISPLAY_NAMES.get(normalize_model_key(koib_model), "Все модели") if koib_model else "Все модели"
        parts = [
            f"КОНТЕКСТ ИЗ ТЕХНИЧЕСКОЙ ДОКУМЕНТАЦИИ КОИБ ({model_display}):",
            "",
            context_text,
        ]
        if figures:
            parts.append("")
            parts.append("РЕЛЕВАНТНЫЕ РИСУНКИ И СХЕМЫ:")
            for fig in figures:
                fig_model = MODEL_DISPLAY_NAMES.get(fig.get("model", ""), fig.get("model", ""))
                caption = fig.get("caption", "без подписи")
                source = fig.get("source", "")
                parts.append(f"  - [{fig_model}] {source}: {caption}")
        parts.extend([
            "",
            "ИНСТРУКЦИЯ:",
            f"На основе приведенного контекста из технической документации КОИБ, ответь на вопрос:",
            f"«{query}»",
            "Если ответ не содержится в контексте, явно укажи это.",
            "При ответе ссылайся на конкретный документ и страницу.",
        ])
        return "\n".join(parts)

# ============================================================================
# 5. Интерактивный режим
# ============================================================================
def interactive_query(engine: KoibQueryEngine):
    if engine is None or engine.vectorstore is None:
        print("❌ Query engine not initialized. Run part 3 first.")
        return
    print("\n" + "="*60)
    print("🎯 КОИБ RAG — Интерактивный запрос")
    print("="*60)
    print("Выберите модель:")
    print("  1 = КОИБ-2010")
    print("  2 = КОИБ-2017А")
    print("  3 = КОИБ-2017Б")
    print("  4 = Все модели")
    print("  0 = Выход")
    model_map = {"1": "koib2010", "2": "koib2017a", "3": "koib2017b", "4": ""}
    while True:
        choice = input("\n📝 Модель (1-4, 0=выход): ").strip()
        if choice == "0":
            print("👋 Выход.")
            break
        koib_model = model_map.get(choice, "")
        if choice not in model_map:
            print("⚠️ Неверный выбор, используем все модели.")
            koib_model = ""
        model_display = MODEL_DISPLAY_NAMES.get(koib_model, "Все модели") if koib_model else "Все модели"
        print(f"\n🔍 Режим: {model_display}")
        while True:
            query = input("\n❓ Вопрос (или пустая строка для смены модели): ").strip()
            if not query:
                break
            start = time.time()
            context, docs, figures = engine.ask(query, koib_model, k=5)
            elapsed = time.time() - start
            if not context:
                print("  😔 По вашему запросу ничего не найдено.")
                continue
            print(f"\n📋 КОНТЕКСТ ({elapsed:.2f}s, {len(docs)} результатов):\n")
            print("-"*60)
            for i, d in enumerate(docs):
                print(f"\n[{i+1}] {MODEL_DISPLAY_NAMES.get(d['model'], d['model'])} | {d['source']} стр.{d['page']} | score={d['score']}")
                print(f"    {d['text']}")
                if d.get('has_figures'):
                    print("    🖼️ [есть связанные рисунки]")
            if figures:
                print(f"\n🖼️ Найдено рисунков: {len(figures)}")
                for fig in figures:
                    caption = fig.get('caption', 'без подписи')
                    path = fig.get('fig_path', '')
                    print(f"  📎 {fig.get('source', '')}: {caption}")
                    if path:
                        print(f"     Путь: {path}")
            llm_ctx = engine.ask_with_llm_context(query, koib_model)
            print(f"\n🤖 LLM-КОНТЕКСТ (для копирования в LLM):")
            print("="*60)
            print(llm_ctx[:2000])
            if len(llm_ctx) > 2000:
                print(f"... (обрезано, всего {len(llm_ctx)} символов)")
            print("="*60)

# ============================================================================
# 6. Тесты
# ============================================================================
def run_tests(engine: KoibQueryEngine):
    if engine is None or engine.vectorstore is None:
        print("❌ Query engine not available for testing.")
        return
    print("\n" + "="*70)
    print("🧪 RUNNING VALIDATION TESTS")
    print("="*70)
    test_queries = [
        ("Как включить сканирующее устройство КОИБ-2010?", "koib2010"),
        ("Режим переносного голосования КОИБ-2017А", "koib2017a"),
        ("Подключение принтера к КОИБ-2017Б", "koib2017b"),
        ("Как распечатать протокол?", ""),
        ("Что делать если бюллетень не распознается?", ""),
        ("Калибровка сканера КОИБ-2010 при ошибке чтения", "koib2010"),
        ("Схема подключения КОИБ-2017А к сети питания", "koib2017a"),
    ]
    results = []
    passed = 0
    for i, (query, model) in enumerate(test_queries, 1):
        model_display = MODEL_DISPLAY_NAMES.get(model, "Все модели") if model else "Все модели"
        print(f"\n--- Тест {i}/{len(test_queries)} ---")
        print(f"  ❓ {query}")
        print(f"  🏷️ Модель: {model_display}")
        start = time.time()
        _, docs, figures = engine.ask(query, model, k=3)
        elapsed = time.time() - start
        if docs:
            passed += 1
            status = "✅ PASS"
            print(f"  {status} — {len(docs)} результатов за {elapsed:.2f}s")
            for j, d in enumerate(docs):
                print(f"    [{j+1}] score={d['score']} | {d['source']} стр.{d['page']}")
                print(f"        {d['text'][:150]}...")
            if figures:
                print(f"    🖼️ Рисунков: {len(figures)}")
        else:
            status = "❌ FAIL"
            print(f"  {status} — нет результатов за {elapsed:.2f}s")
        results.append({
            "query": query,
            "model": model,
            "status": status,
            "num_results": len(docs),
            "num_figures": len(figures),
            "time_sec": round(elapsed, 3),
        })
    print(f"\n{'='*70}")
    print(f"📊 TEST RESULTS: {passed}/{len(test_queries)} passed")
    print(f"{'='*70}")
    test_path = LOGS_DIR / "test_results.json"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Test results saved to {test_path}")
    return results

# ============================================================================
# 7. Быстрый поиск
# ============================================================================
def quick_search(query: str, model: str = "", k: int = 3):
    """Функция для быстрого поиска из одной строки в Colab"""
    if 'engine' not in globals() or engine is None:
        print("❌ Engine not initialized. Run part 3 first.")
        return
    context, docs, figures = engine.ask(query, model, k=k)
    if not docs:
        print("No results found.")
        return
    print(f"Found {len(docs)} results:\n")
    for i, d in enumerate(docs):
        model_name = MODEL_DISPLAY_NAMES.get(d['model'], d['model'])
        print(f"[{i+1}] {model_name} | {d['source']} стр.{d['page']} (score={d['score']})")
        print(f"    {d['text'][:300]}...\n")
    if figures:
        print(f"Figures: {len(figures)}")
        for f in figures:
            print(f"  - {f.get('caption', 'no caption')} ({f.get('source', '')})")
    return context, docs, figures

# ============================================================================
# 8. Экспорт конфигурации для GigaChat
# ============================================================================
def export_for_gigachat(output_path: Optional[Path] = None):
    if output_path is None:
        output_path = METADATA_DIR / "gigachat_config.json"
    config = {
        "description": "KOIB RAG v4 — Configuration for GigaChat integration",
        "created": datetime.datetime.now().isoformat(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "faiss_index_path": str(FAISS_INDEX_DIR / "koib_index"),
        "figures_index_path": str(METADATA_DIR / "figures_index.json"),
        "chunks_path": str(METADATA_DIR / "chunks.json"),
        "passage_prefix": PASSAGE_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "models": list(KOIB_MODEL_PATTERNS.keys()),
        "api_template": {
            "gigachat_base_url": "https://gigachat.devices.sberbank.ru/api/v1",
            "auth_url": "https://ngw.devices.sberbank.ru/api/v2/oauth",
            "model": "GigaChat",
            "temperature": 0.3,
            "max_tokens": 1024,
        },
        "example_queries": [
            "Как включить КОИБ-2010?",
            "Режим переносного голосования КОИБ-2017А",
            "Подключение принтера к КОИБ-2017Б",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"✅ GigaChat config exported to {output_path}")
    return config

# ============================================================================
# 9. Скачивание индекса в ZIP
# ============================================================================
def download_index(output_zip: Optional[Path] = None):
    import zipfile
    if output_zip is None:
        output_zip = OUTPUT_DIR / "koib_rag_index.zip"
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder in [FAISS_INDEX_DIR, METADATA_DIR]:
            if folder.exists():
                for file_path in folder.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(OUTPUT_DIR)
                        zf.write(file_path, arcname)
    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"✅ Index archive created: {output_zip} ({size_mb:.1f} MB)")
    return output_zip

# ============================================================================
# 10. Статистика (на основе сохранённых файлов)
# ============================================================================
def print_statistics():
    print("\n╔" + "═"*68 + "╗")
    print("║" + "  КОИБ RAG v4 — СТАТИСТИКА".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    # Загрузка метаданных
    blocks_path = METADATA_DIR / "text_blocks.json"
    chunks_path = METADATA_DIR / "chunks.json"
    figures_path = METADATA_DIR / "figures_index.json"
    log_path = LOGS_DIR / "processing_log.json"

    if blocks_path.exists():
        with open(blocks_path, 'r') as f:
            blocks = json.load(f)
        print(f"\n📄 Текстовых блоков: {len(blocks)}")
        by_model = defaultdict(int)
        for b in blocks:
            by_model[b.get("model", "unknown")] += 1
        for m, cnt in by_model.items():
            print(f"   {MODEL_DISPLAY_NAMES.get(m, m)}: {cnt}")
    if chunks_path.exists():
        with open(chunks_path, 'r') as f:
            chunks = json.load(f)
        print(f"\n📦 Чанков: {len(chunks)}")
        lens = [len(c["text"]) for c in chunks]
        if lens:
            print(f"   Средняя длина: {sum(lens)//len(lens)} символов")
    if figures_path.exists():
        with open(figures_path, 'r') as f:
            figs = json.load(f)
        print(f"\n🖼️ Рисунков: {len(figs)}")
        with_caption = sum(1 for f in figs if f.get("caption"))
        print(f"   С подписями: {with_caption}")
    if log_path.exists():
        with open(log_path, 'r') as f:
            logs = json.load(f)
        total_time = sum(e["time_sec"] for e in logs)
        print(f"\n⏱️ Общее время обработки: {total_time:.1f}s")

    # Информация о FAISS индексе
    meta_path = FAISS_INDEX_DIR / "index_meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print(f"\n🔍 FAISS индекс: {meta.get('num_chunks', '?')} чанков")
        print(f"   Модель эмбеддингов: {meta.get('model_name', '?')}")
        print(f"   Создан: {meta.get('created', '?')}")
    print("\n" + "="*70)

# ============================================================================
# 11. VK Bot (опционально, требует настройки токенов)
# ============================================================================
def create_vk_bot(engine: KoibQueryEngine):
    try:
        import vk_api
        from vk_api.longpoll import VkLongPoll, VkEventType
    except ImportError:
        print("❌ vk-api not installed. Run: !pip install vk-api")
        return None

    vk_token = os.environ.get("VK_GROUP_TOKEN", "")
    if not vk_token:
        print("⚠️ VK_GROUP_TOKEN not set. Bot will not work.")
        return None

    class KoibVkBot:
        def __init__(self, token, engine):
            self.token = token
            self.engine = engine
            self.vk = vk_api.VkApi(token=token)
            self.upload = vk_api.VkUpload(self.vk)
            print("✅ VK Bot initialized")

        def _detect_model_from_message(self, text):
            text_lower = text.lower()
            for model_key, patterns in KOIB_MODEL_PATTERNS.items():
                for pat in patterns:
                    if re.search(pat, text_lower, re.IGNORECASE):
                        return model_key
            return ""

        def _send_text(self, peer_id, text):
            try:
                self.vk.get_api().messages.send(
                    peer_id=peer_id,
                    message=text,
                    random_id=int(time.time()*1000),
                )
            except Exception as e:
                print(f"VK send error: {e}")

        def _send_photo(self, peer_id, photo_path):
            if not Path(photo_path).exists():
                return
            try:
                upload_url = self.upload.photo_messages(photo_path)[0]
                self.vk.get_api().messages.send(
                    peer_id=peer_id,
                    attachment=f"photo{upload_url['owner_id']}_{upload_url['id']}",
                    random_id=int(time.time()*1000),
                )
            except Exception as e:
                print(f"VK photo error: {e}")

        def handle_message(self, text, peer_id):
            if not text.strip():
                self._send_text(peer_id, "Пожалуйста, задайте вопрос о КОИБ.")
                return
            koib_model = self._detect_model_from_message(text)
            context, docs, figures = self.engine.ask(text, koib_model, k=5)
            if not docs:
                self._send_text(peer_id, "По вашему запросу ничего не найдено в документации КОИБ.")
                return
            response = f"Результаты поиска ({len(docs)} фрагментов):\n\n"
            for i, d in enumerate(docs[:3]):
                model_name = MODEL_DISPLAY_NAMES.get(d['model'], d['model'])
                response += f"{i+1}. {model_name} — {d['source']}, стр.{d['page']}\n{d['text'][:200]}...\n\n"
            if figures:
                response += "🖼️ Найдены рисунки:\n"
                for fig in figures[:2]:
                    response += f"  • {fig.get('caption', 'Рисунок')}\n"
                    if fig.get('fig_path'):
                        self._send_photo(peer_id, fig['fig_path'])
            self._send_text(peer_id, response[:4000])

        def run_longpoll(self):
            longpoll = VkLongPoll(self.vk)
            print("🤖 VK Bot listening...")
            for event in longpoll.listen():
                if event.type == VkEventType.MESSAGE_NEW and event.to_me:
                    self.handle_message(event.text, event.peer_id)

    return KoibVkBot(vk_token, engine)

# ============================================================================
# 12. MAIN – инициализация и запуск
# ============================================================================
def main():
    print("╔" + "═"*68 + "╗")
    print("║" + "  KOIB RAG v4 – PART 3: QUERY ENGINE".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    global engine
    engine = KoibQueryEngine()
    if engine.vectorstore is None:
        print("❌ Failed to initialize query engine. Check that Part 2 was run successfully.")
        return
    # Автоматический тест
    print("\n🧪 Автоматический тестовый запрос...")
    _, docs, _ = engine.ask("Как включить КОИБ-2010?", k=3)
    if docs:
        print(f"✅ Тест успешен: найдено {len(docs)} результатов")
    else:
        print("⚠️ Тестовый запрос не вернул результатов")
    # Запуск интерактивного режима
    interactive_query(engine)

if __name__ == "__main__":
    main()