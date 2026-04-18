## 📂 KOIB RAG - Система поиска по документации КОИБ

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

RAG (Retrieval-Augmented Generation) система для обработки и поиска по технической документации КОИБ (Комплексы Обработки Избирательных Бюллетеней).

### Особенности

- **Поддержка моделей**: КОИБ-2010, КОИБ-2017А, КОИБ-2017Б
- **Обработка форматов**: PDF, DOCX
- **OCR**: Распознавание текста для сканированных документов (pytesseract + EasyOCR)
- **Векторный поиск**: FAISS с multilingual эмбеддингами
- **Извлечение рисунков**: Сохранение изображений с подписями и контекстом

---

## 📑 Содержание

| № | Модуль | Файл | Описание |
|---|--------|------|----------|
| 1 | **Preprocessing** | [`src/preprocessing.py`](src/preprocessing.py) | Извлечение текста, OCR, рисунков, определение моделей |
| 2 | **Index Building** | [`src/index_building.py`](src/index_building.py) | Чанкирование текстов, построение FAISS индекса |
| 3 | **Query Engine** | [`src/query_engine.py`](src/query_engine.py) | Поисковый движок, интерактивные запросы |

---

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

Для работы OCR также требуется установить Tesseract:

```bash
# Ubuntu/Debian
sudo apt-get install -y tesseract-ocr tesseract-ocr-rus

# macOS
brew install tesseract tesseract-lang
```

### Настройка окружения

Скопируйте `.env.example` в `.env` и настройте пути:

```bash
cp .env.example .env
```

```bash
# Пути к данным
KOIB_DOCS_DIR=./data/docs
KOIB_OUTPUT_DIR=./output
```

### Запуск

#### Часть 1: Предобработка документов

```bash
python -m src.preprocessing
```

#### Часть 2: Построение индекса

```bash
python -m src.index_building
```

#### Часть 3: Поисковый движок

```bash
python -m src.query_engine
```

---

## 📁 Структура проекта

```
Koib/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Часть 1: Предобработка
│   ├── index_building.py     # Часть 2: Построение индекса
│   ├── query_engine.py       # Часть 3: Поисковый движок
│   └── utils.py              # Общие утилиты
├── notebooks/                 # Jupyter ноутбуки для демонстрации
├── tests/                     # Юнит-тесты
├── data/
│   └── docs/                  # Входные документы (PDF/DOCX)
├── output/                    # Выходные данные (игнорируется git)
│   ├── classified/
│   ├── ocr_results/
│   ├── figures/
│   ├── metadata/
│   │   ├── text_blocks.json
│   │   ├── figures_index.json
│   │   └── chunks.json
│   ├── faiss_index/
│   │   └── koiss_index/
│   └── logs/
├── .env.example               # Пример переменных окружения
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Конфигурация

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `KOIB_DOCS_DIR` | Путь к директории с документами | `./data/docs` |
| `KOIB_OUTPUT_DIR` | Путь к выходной директории | `./output` |
| `OCR_DPI` | DPI для OCR | `300` |
| `CHUNK_SIZE` | Размер чанка для разбиения | `2000` |
| `CHUNK_OVERLAP` | Перекрытие чанков | `320` |
| `EMBEDDING_MODEL_NAME` | Модель эмбеддингов | `intfloat/multilingual-e5-large` |
| `FAISS_SEARCH_K` | Количество результатов поиска | `5` |

---

## 🧪 Тестирование

```bash
python -m pytest tests/
```

---

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE) для деталей.

---

## 🤝 Вклад

Приветствуются issue и pull requests!

---

## 📞 Контакты

По вопросам и предложениям создавайте issue в репозитории.
    ├── figures/             # Извлеченные рисунки (по моделям)
    ├── metadata/            # Метаданные (JSON файлы)
    └── logs/                # Логи обработки
```

### Выходные файлы

| Файл | Описание |
|------|----------|
| `metadata/text_blocks.json` | Извлеченные текстовые блоки с метаданными |
| `metadata/figures_index.json` | Индекс извлеченных рисунков |
| `metadata/source_models.json` | Определение моделей для каждого файла |
| `classified/classification_report.csv` | Отчет классификации по файлам |
| `logs/processing_log.json` | Детальный лог обработки |

### Требования

```bash
# Системные зависимости
apt-get install -qq tesseract-ocr tesseract-ocr-rus -y

# Python пакеты
pip install pymupdf python-docx Pillow numpy tqdm pytesseract easyocr sentence-transformers
```

### Параметры настройки

```python
OCR_DPI = 300                    # DPI для OCR сканированных страниц
OCR_MIN_TEXT_CHARS = 50          # Минимум символов для считывания страницы текстом
MIN_IMAGE_WIDTH = 80             # Минимальная ширина изображения
MIN_IMAGE_HEIGHT = 80            # Минимальная высота изображения
SCREENSHOT_AREA_THRESHOLD = 0.80 # Порог для отсечения скриншотов целых страниц
TEXT_DENSITY_THRESHOLD = 0.35    # Порог плотности текста для фильтрации
```

### Следующие шаги

После успешного завершения Part 1:
- Проверьте выходные данные в директории `koib_rag_GLM1/`
- Запустите **Part 2** для построения индексов векторного поиска
- Используйте извлеченные данные для обучения RAG системы

---

## 📑 KOIB RAG Index Building (Part 2)

### Описание

Часть 2 системы RAG (Retrieval-Augmented Generation) для обработки документации КОИБ. Скрипт загружает результаты предобработки из Part 1, выполняет чанкирование текстов и строит FAISS индекс для быстрого семантического поиска с использованием multilingual эмбеддингов.

### Возможности

- **Загрузка данных**: Автоматическая загрузка `text_blocks.json` и `figures_index.json` из Part 1
- **Умное чанкирование**: Группировка по (model, source, page) с сохранением контекста
- **Дедупликация**: Удаление дублирующихся чанков по хешу
- **Метаданные**: Сохранение заголовков, подписей к рисункам, информации о наличии изображений
- **FAISS индекс**: Построение векторного индекса для быстрого семантического поиска
- **Multilingual эмбеддинги**: Использование модели `intfloat/multilingual-e5-large` для русскоязычных документов
- **Сохранение результатов**: Экспорт чанков в JSON и индекса на диск

### Рабочий процесс

1. **Загрузка текстовых блоков** из `metadata/text_blocks.json`
2. **Группировка блоков** по модели, источнику и странице
3. **Чанкирование** с помощью `RecursiveCharacterTextSplitter`
4. **Обогащение метаданными** (заголовки, подписи, наличие рисунков)
5. **Построение FAISS индекса** с multilingual эмбеддингами
6. **Сохранение** индекса и метаданных

### Структура выходных данных

Дополняет структуру Part 1:

```
Koib/
└── koib_rag_GLM1/
    ├── metadata/
    │   ├── text_blocks.json      # Из Part 1
    │   ├── figures_index.json    # Из Part 1
    │   └── chunks.json           # Новые чанки с метаданными
    └── faiss_index/
        ├── koib_index/           # FAISS индекс (vectorstore)
        └── index_meta.json       # Метаданные индекса
```

### Выходные файлы Part 2

| Файл | Описание |
|------|----------|
| `metadata/chunks.json` | Чанки с метаданными (текст, модель, источник, страница, заголовки, подписи) |
| `faiss_index/koib_index/` | FAISS векторный индекс для семантического поиска |
| `faiss_index/index_meta.json` | Метаданные индекса (модель эмбеддингов, количество чанков, дата создания) |

### Параметры чанкинга

```python
CHUNK_SIZE = 2000           # Максимальный размер чанка в символах
CHUNK_OVERLAP = 320         # Перекрытие между чанками
MIN_CHUNK_LEN = 120         # Минимальная длина чанка
```

### Модель эмбеддингов

```python
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
PASSAGE_PREFIX = "passage: "  # Префикс для индексации
QUERY_PREFIX = "query: "      # Префикс для поисковых запросов
```

Модель `multilingual-e5-large` поддерживает более 100 языков, включая русский, и показывает отличные результаты для семантического поиска.

### Требования

```bash
# Python пакеты (дополнительно к Part 1)
pip install langchain-core langchain langchain-text-splitters langchain-community langchain-huggingface faiss-cpu sentence-transformers
```

### Пример использования индекса

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Загрузка индекса
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = FAISS.load_local(
    "/content/drive/MyDrive/Koib/koib_rag_GLM1/faiss_index/koib_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Поиск похожих чанков
query = "query: как работает КОИБ-2017?"
results = vectorstore.similarity_search(query, k=5)

for doc in results:
    print(f"Источник: {doc.metadata['source']}, Страница: {doc.metadata['page']}")
    print(f"Текст: {doc.page_content[:200]}...\n")
```

### Следующие шаги

После успешного завершения Part 2:
- Индекс готов для использования в RAG-системе
- Запустите **Part 3** (Query Engine) для реализации поискового движка
- Интегрируйте с LLM (например, GLM или GPT) для генерации ответов

---

## 📑 KOIB RAG Query Engine (Part 3)

### Описание

Часть 3 системы RAG (Retrieval-Augmented Generation) для обработки документации КОИБ. Скрипт загружает FAISS индекс и метаданные из Part 2, реализует поисковый движок для семантического поиска, интерактивный режим запросов, интеграцию с LLM (GigaChat и др.) и опционально VK бота для взаимодействия через социальные сети.

### Возможности

- **Загрузка индекса**: Автоматическая загрузка FAISS индекса и метаданных из Part 2
- **Семантический поиск**: Поиск релевантных документов по запросу с использованием multilingual эмбеддингов
- **Фильтрация по моделям**: Возможность поиска по конкретным моделям КОИБ (2010, 2017А, 2017Б)
- **Интерактивный режим**: Удобный CLI для тестирования запросов с выбором модели
- **Интеграция с LLM**: Генерация форматированного контекста для передачи в LLM (GigaChat, GLM, GPT)
- **Поиск рисунков**: Автоматический поиск релевантных изображений по запросу
- **VK бот**: Опциональная интеграция с ВКонтакте для ответов на вопросы пользователей
- **Тестирование**: Встроенные тестовые запросы для валидации работы системы
- **Экспорт конфигурации**: Готовая конфигурация для интеграции с GigaChat API
- **Статистика**: Подробная статистика по всем компонентам системы

### Рабочий процесс

1. **Загрузка компонентов**:
   - FAISS индекс из `faiss_index/koib_index/`
   - Индекс рисунков из `metadata/figures_index.json`
   - Чанки из `metadata/chunks.json` (опционально)

2. **Обработка запроса**:
   - Добавление префикса `query:` к тексту запроса
   - Семантический поиск в FAISS индексе
   - Фильтрация результатов по модели (если указана)
   - Извлечение метаданных (источник, страница, заголовки, подписи)

3. **Поиск рисунков**:
   - Анализ ключевых слов запроса
   - Поиск релевантных изображений по подписям и контексту
   - Ранжирование по степени соответствия

4. **Формирование ответа**:
   - Группировка результатов по источникам
   - Форматирование контекста для человека
   - Генерация промпта для LLM с инструкциями

5. **Интерактивное взаимодействие**:
   - Выбор модели КОИБ через меню
   - Множественные запросы в рамках одной сессии
   - Отображение времени поиска и количества результатов

### Структура выходных данных

Дополняет структуру Part 1 и Part 2:

```
Koib/
└── koib_rag_GLM1/
    ├── metadata/
    │   ├── text_blocks.json      # Из Part 1
    │   ├── figures_index.json    # Из Part 1
    │   ├── chunks.json           # Из Part 2
    │   └── gigachat_config.json  # Новая: конфигурация для GigaChat
    ├── faiss_index/
    │   ├── koib_index/           # Из Part 2
    │   └── index_meta.json       # Из Part 2
    └── logs/
        ├── processing_log.json   # Из Part 1
        └── test_results.json     # Новый: результаты тестов
```

### Выходные файлы Part 3

| Файл | Описание |
|------|----------|
| `metadata/gigachat_config.json` | Конфигурация для интеграции с GigaChat API (URL, модель, параметры) |
| `logs/test_results.json` | Результаты автоматического тестирования системы |
| `koib_rag_index.zip` | Архив с индексом для скачивания и переноса |

### Основные функции

#### KoibQueryEngine

Главный класс для работы с RAG-системой:

```python
engine = KoibQueryEngine(
    faiss_index_path=None,        # Путь к FAISS индексу (по умолчанию из констант)
    figures_index_path=None,      # Путь к индексу рисунков
    chunks_path=None              # Путь к чанкам (опционально)
)
```

**Методы:**

- `ask(query, koib_model="", k=5)` — поиск релевантных документов и рисунков
- `ask_with_llm_context(query, koib_model="")` — генерация форматированного контекста для LLM
- `_search(query, model_filter="", k=5)` — низкоуровневый поиск в FAISS

#### Интерактивный режим

```python
interactive_query(engine)
```

Запускает интерактивный CLI с выбором модели и множественными запросами.

#### Тестирование

```python
run_tests(engine)
```

Запускает набор тестовых запросов для валидации работы системы.

#### Быстрый поиск

```python
quick_search("Как включить КОИБ-2010?", model="koib2010", k=3)
```

Однострочная функция для быстрого поиска из Colab-ячейки.

#### Экспорт для GigaChat

```python
export_for_gigachat()
```

Создает конфигурационный файл с параметрами для интеграции с GigaChat API.

#### VK бот

```python
bot = create_vk_bot(engine)
if bot:
    bot.run_longpoll()
```

Создает и запускает VK бота для обработки сообщений пользователей (требуется токен группы).

### Примеры использования

#### Базовый поиск

```python
from pathlib import Path

# Инициализация движка
engine = KoibQueryEngine()

# Поиск по всем моделям
context, docs, figures = engine.ask("Как распечатать протокол?", k=5)

for doc in docs:
    print(f"{doc['source']} стр.{doc['page']} (score={doc['score']})")
    print(f"{doc['text'][:200]}...\n")
```

#### Поиск с фильтром по модели

```python
# Поиск только для КОИБ-2010
context, docs, figures = engine.ask(
    "Как откалибровать сканер?",
    koib_model="koib2010",
    k=3
)
```

#### Генерация контекста для LLM

```python
llm_prompt = engine.ask_with_llm_context(
    "Что делать если бюллетень не распознается?",
    koib_model="koib2017a"
)

# Отправка в LLM (пример для GigaChat)
import requests

response = requests.post(
    "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {YOUR_TOKEN}"},
    json={
        "model": "GigaChat",
        "messages": [{"role": "user", "content": llm_prompt}],
        "temperature": 0.3,
        "max_tokens": 1024
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

#### Интерактивный режим

```python
# Запуск интерактивного CLI
interactive_query(engine)

# Меню выбора модели:
# 1 = КОИБ-2010
# 2 = КОИБ-2017А
# 3 = КОИБ-2017Б
# 4 = Все модели
# 0 = Выход
```

#### VK бот

```python
import os
os.environ["VK_GROUP_TOKEN"] = "your_token_here"

bot = create_vk_bot(engine)
if bot:
    print("Запуск VK бота...")
    bot.run_longpoll()
```

### Параметры настройки

```python
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"  # Модель эмбеддингов
PASSAGE_PREFIX = "passage: "      # Префикс для индексации
QUERY_PREFIX = "query: "          # Префикс для поисковых запросов
FAISS_SEARCH_K = 5                # Количество результатов по умолчанию
```

### Модели КОИБ

Система поддерживает фильтрацию по трем моделям:

| Ключ | Название | Паттерны |
|------|----------|----------|
| `koib2010` | КОИБ-2010 | `КОИБ-2010`, `0912054`, `PRINT_KOIB2010` |
| `koib2017a` | КОИБ-2017А | `КОИБ-2017А`, `17404049.5013009` |
| `koib2017b` | КОИБ-2017Б | `КОИБ-2017Б`, `БАВУ.201119`, `0912053` |

### Требования

```bash
# Python пакеты (дополнительно к Part 1 и Part 2)
pip install vk-api  # опционально, для VK бота
```

### Статистика системы

Функция `print_statistics()` выводит подробную информацию:

- Количество текстовых блоков по моделям
- Количество чанков и средняя длина
- Количество рисунков (с подписями и без)
- Общее время обработки из Part 1
- Информация о FAISS индексе (модель, количество чанков, дата создания)

### Интеграция с GigaChat

Файл `gigachat_config.json` содержит:

- URL API GigaChat
- Параметры аутентификации
- Название модели и параметры генерации
- Примеры запросов для тестирования

Пример использования:

```python
config = export_for_gigachat()

# Загрузка конфигурации
with open("metadata/gigachat_config.json") as f:
    config = json.load(f)

api_url = config["api_template"]["gigachat_base_url"]
model = config["api_template"]["model"]
temperature = config["api_template"]["temperature"]
```

### Скачивание индекса

Функция `download_index()` создает ZIP-архив со всеми файлами индекса:

```python
zip_path = download_index()
# Результат: /content/drive/MyDrive/Koib/koib_rag_GLM1/koib_rag_index.zip
```

Удобно для переноса индекса на другой сервер или локальную машину.

### Следующие шаги

После успешного завершения Part 3:

- Система готова к продакшн-использованию
- Интегрируйте с вашим предпочтительным LLM (GigaChat, YandexGPT, GPT-4)
- Настройте VK бота для поддержки пользователей
- Используйте статистику для мониторинга и оптимизации
- Расширьте функциональность своими сценариями использования

---

🧑‍💻 Автор  
GitHub: [github.com/ostrakin](https://github.com/ostrakin)
