# 🚀 Система детекции и классификации людей в различных позах

Комплексная система для обнаружения и классификации людей в различных позах и действиях с акцентом на обнаружение нарушителей. Включает несколько архитектур нейронных сетей для достижения максимальной точности и скорости.

## 📋 Содержание

- [Описание проекта](#описание-проекта)
- [Быстрый старт](#быстрый-старт)
- [Установка](#установка)
- [Структура проекта](#структура-проекта)
- [Архитектуры моделей](#архитектуры-моделей)
- [Команды и использование](#команды-и-использование)
- [Примеры использования](#примеры-использования)
- [Анализ результатов](#анализ-результатов)
- [Устранение неполадок](#устранение-неполадок)
- [Требования к системе](#требования-к-системе)

## 🎯 Описание проекта

Проект **[SecurityVision](https://github.com/LewyWho/SecurityVision)** содержит набор данных для обнаружения и классификации людей в различных позах:

### Классы поз:

- **creeping** - ползание
- **crawling** - ползание на четвереньках
- **stooping** - наклон
- **climbing** - лазание
- **other** - другие действия

### Категории:

- **intruder** - нарушители
- **pedestrian** - пешеходы

---

**Источник датасета:** [Intruder-Thermal-Dataset (GitHub)](https://github.com/thuan-researcher/Intruder-Thermal-Dataset)

## ⚡ Быстрый старт

### 1. Клонирование и установка

```bash
git clone https://github.com/LewyWho/SecurityVision
cd SecurityVision
pip install -r requirements.txt
```

### 2. Обучение детекционной модели

```bash
python train.py --model yolo --epochs 10 --batch_size 16 --image_size 512
```

### 3. Тестирование скорости

```bash
python benchmark_models.py --models yolo ssd efficientdet
```

### 4. Инференс с визуализацией

```bash
python inference.py --model_path models/yolo/best.pth --input test_images/
```

## 📦 Установка

### Требования

- Python 3.8+
- PyTorch 1.12+
- CUDA (опционально, для ускорения на GPU)

### Пошаговая установка

1. **Клонирование репозитория:**

```bash
git clone https://github.com/LewyWho/SecurityVision
cd SecurityVision-Classifier
```

2. **Создание виртуального окружения:**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Установка зависимостей:**

```bash
pip install -r requirements.txt
```

4. **Проверка установки:**

```bash
python check_compatibility.py
```

## 📁 Структура проекта

```
SecurityVision-Classifier/
├── 📂 Data/                           # Данные
│   ├── 📂 train/                      # Обучающие изображения
│   ├── 📂 test/                       # Тестовые изображения
│   ├── 📄 train.json                  # Аннотации для обучения (COCO формат)
│   └── 📄 test.json                   # Аннотации для тестирования (COCO формат)
│
├── 🧠 models.py                       # Архитектуры нейронных сетей
├── 📊 dataset.py                      # Класс для загрузки данных
├── 🎓 train.py                        # Скрипт обучения
├── 📈 evaluate.py                     # Скрипт оценки моделей
├── 🔍 inference.py                    # Полный инференс с визуализацией
├── 📊 benchmark_models.py             # Бенчмарк скорости моделей
├── 🛠️ train_robust.py                # Надежное обучение с обработкой ошибок
├── 🔧 check_compatibility.py          # Проверка совместимости
├── 📋 requirements.txt                # Зависимости
├── 📄 training_results_*.json         # Результаты обучения (точность, метрики)
├── 📄 model_benchmark_*.json          # Результаты бенчмарка моделей (скорость, FPS)
└── 📖 README.md                       # Документация
│
├── 📂 models/                         # Обученные модели
│   ├── 📂 yolo/
│   │   ├── best.pth
│   │   ├── final.pth
│   │   └── training_curves.png
│   ├── 📂 efficientdet/
│   │   ├── best.pth
│   │   └── training_curves.png
│   └── ...
│
├── 📂 evaluation_results/             # Результаты оценки
│   ├── 📂 yolo/
│   │   ├── evaluation_report.json   # JSON-отчёт с метриками
│   │   ├── class_metrics.png        # График метрик по классам
│   │   ├── confusion_matrix.png     # Матрица ошибок
│   │   └── prediction_analysis.png  # Анализ предсказаний
│   ├── 📂 efficientdet/
│   │   ├── evaluation_report.json
│   │   ├── class_metrics.png
│   │   ├── confusion_matrix.png
│   │   └── prediction_analysis.png
│   └── ...
│
└── 📂 inference_images/              # Результаты инференса
    ├── inference_images_yolo/
    │   ├── detected_*.JPG         # Изображения с детекцией (YOLO)
    │   └── summary_report.json    # Краткий отчёт по инференсу
    ├── inference_images_efficientdet/
    │   ├── detected_*.JPG         # Изображения с детекцией (EfficientDet)
    │   └── summary_report.json    # Краткий отчёт по инференсу
    └── ...
```

---

## 📊 Результаты и метрики

### 📈 Результаты обучения

- Все результаты обучения сохраняются в файлы `training_results_*.json`.
- Пример (YOLO):
  - Лучшая точность на валидации: **71.8%**
  - Финальная точность на обучении: **100.0%**
  - Финальная точность на валидации: **71.0%**
- Пример (EfficientDet):
  - Лучшая точность на валидации: **71.8%**
  - Финальная точность на обучении: **99.95%**
  - Финальная точность на валидации: **70.8%**

### ⚡ Бенчмарк моделей

- Все результаты бенчмарка сохраняются в файлы `model_benchmark_*.json`.
- | Пример (CPU, image_size=512): | Модель | FPS  | Среднее время инференса (сек) | Параметры | Размер модели (MB) |
  | ----------------------------- | ------ | ---- | ----------------------------- | --------- | ------------------ |
  | efficientdet                  | 5.01   | 0.20 | 7.4M                          | 28.2      |
  | yolo                          | 4.13   | 0.24 | 10.1M                         | 38.5      |
  | ssd                           | 0.89   | 1.13 | 24.1M                         | 92.1      |

### 🏆 Метрики оценки (evaluation_results)

- Для каждой модели сохраняется подробный отчет `evaluation_report.json`.
- Пример (YOLO):
  - Общая точность (accuracy): **71.8%**
  - Precision (macro avg): **0.73**
  - Recall (macro avg): **0.73**
  - F1-score (macro avg): **0.73**
- Пример (EfficientDet):
  - Общая точность (accuracy): **71.8%**
  - Precision (macro avg): **0.73**
  - Recall (macro avg): **0.73**
  - F1-score (macro avg): **0.73**
- Для каждого класса доступны precision, recall, f1-score, confusion matrix и графики.

### 🔍 Результаты инференса

- Для каждой модели сохраняется summary в `summary_report.json`.
- Пример (YOLO):
  - Всего изображений: **20**
  - Всего детекций: **20**
  - Распределение по классам (YOLO):
    - creeping: 2 (avg_conf: 0.99998)
    - crawling: 6 (avg_conf: 0.99346)
    - stooping: 3 (avg_conf: 0.99996)
    - climbing: 5 (avg_conf: 0.99185)
    - other: 4 (avg_conf: 0.99999)
- Пример (EfficientDet):
  - creeping: 2 (avg_conf: 0.99999)
  - crawling: 5 (avg_conf: 0.99999)
  - stooping: 3 (avg_conf: 0.99999)
  - climbing: 6 (avg_conf: 0.98321)
  - other: 4 (avg_conf: 0.99999)

### 📄 Описание файлов результатов

- `training_results_*.json` — точность и метрики обучения по эпохам и моделям.
- `model_benchmark_*.json` — скорость инференса, FPS, параметры моделей.
- `evaluation_report.json` — метрики (accuracy, precision, recall, f1-score, confusion matrix) по классам и в среднем.
- `summary_report.json` — статистика инференса: количество детекций, распределение по классам, средняя уверенность.
- `class_metrics.png`, `confusion_matrix.png`, `prediction_analysis.png` — графики анализа результатов.

---

## 🧠 Архитектуры моделей

### 🔍 Детекционные модели

| Модель         | Размер изображения | Скорость (CPU) | Параметры | Применение        |
| -------------- | ------------------ | -------------- | --------- | ----------------- |
| `efficientdet` | 512×512            | 5.0 FPS        | 7.4M      | Точная детекция   |
| `yolo`         | 512×512            | 4.1 FPS        | 10.1M     | Детекция объектов |
| `ssd`          | 512×512            | 0.9 FPS        | 24.1M     | Детекция объектов |

## 🎮 Команды и использование

### 🎓 Обучение моделей

#### Обучение одной модели

```bash
python train.py --model yolo --epochs 50 --batch_size 16 --lr 0.001
```

#### Обучение всех моделей

```bash
python train.py --model all --epochs 30 --batch_size 16
```

#### Обучение детекционных моделей

```bash
# YOLO - быстрая детекция
python train.py --model yolo --epochs 30 --batch_size 16 --image_size 512

# SSD - сбалансированная модель
python train.py --model ssd --epochs 25 --batch_size 16 --image_size 512

# EfficientDet - точная детекция
python train.py --model efficientdet --epochs 20 --batch_size 16 --image_size 512
```

**Параметры обучения:**

- `--model`: Выбор модели (`yolo`, `efficientdet`, `ssd`, `all`)
- `--epochs`: Количество эпох (по умолчанию: 50)
- `--batch_size`: Размер батча (по умолчанию: 16)
- `--lr`: Learning rate (по умолчанию: 0.001)
- `--image_size`: Размер изображения (по умолчанию: 512)

### 📊 Оценка моделей

#### Оценка одной модели

```bash
python evaluate.py --model_path models/yolo/best.pth --batch_size 16
```

#### Оценка всех моделей

```bash
python evaluate.py --evaluate_all --batch_size 16
```

**Параметры оценки:**

- `--model_path`: Путь к обученной модели
- `--batch_size`: Размер батча (по умолчанию: 16)
- `--image_size`: Размер изображения (по умолчанию: 512)
- `--evaluate_all`: Оценить все найденные модели

### 🔍 Инференс с визуализацией

#### Детекция на одном изображении

```bash
python inference.py --model_path models/yolo/best.pth --input image.jpg
```

#### Детекция на директории изображений

```bash
python inference.py --model_path models/yolo/best.pth --input test_images/ --batch_size 16
```

## 📈 Анализ результатов

### Метрики оценки

Система предоставляет следующие метрики:

- **mAP** - средняя точность детекции
- **Precision** - точность детекции по каждому классу
- **Recall** - полнота детекции по каждому классу
- **F1-Score** - гармоническое среднее precision и recall
- **IoU** - Intersection over Union для bounding boxes

### Визуализация

Автоматически генерируются графики:

- Кривые обучения (loss и accuracy)
- Примеры детекции с bounding boxes
- Метрики по классам
- Анализ предсказаний

### Структура результатов

После выполнения команд создаются директории:

```
models/
├── yolo/
│   ├── best.pth          # Лучшая модель
│   ├── final.pth         # Финальная модель
│   └── training_curves.png # Графики обучения
├── efficientdet/
│   ├── best.pth
│   └── training_curves.png
└── ...

evaluation_results/
├── yolo/
│   ├── evaluation_report.json   # JSON-отчёт с метриками
│   ├── class_metrics.png        # График метрик по классам
│   ├── confusion_matrix.png     # Матрица ошибок (confusion matrix)
│   └── prediction_analysis.png  # Анализ предсказаний
├── efficientdet/
│   ├── evaluation_report.json
│   ├── class_metrics.png
│   ├── confusion_matrix.png
│   └── prediction_analysis.png
└── ...

inference_images/
├── inference_images_yolo/
│   ├── detected_*.JPG         # Изображения с детекцией (YOLO)
│   └── summary_report.json    # Краткий отчёт по инференсу
├── inference_images_efficientdet/
│   ├── detected_*.JPG         # Изображения с детекцией (EfficientDet)
│   └── summary_report.json    # Краткий отчёт по инференсу
└── ...
```

**Описание файлов в evaluation_results/**

- `evaluation_report.json` — подробный JSON-отчёт с метриками (mAP, precision, recall и др.)
- `class_metrics.png` — график метрик по классам
- `confusion_matrix.png` — матрица ошибок (confusion matrix)
- `prediction_analysis.png` — анализ предсказаний (график ошибок/уверенности)

**Описание файлов в inference_images/**

- `detected_*.JPG` — изображения с наложенными bbox, классами и confidence
- `summary_report.json` — краткий отчёт по инференсу (количество объектов, распределение классов и др.)

## 🔧 Устранение неполадок

### Проблемы совместимости версий

Если возникают ошибки с torchvision API:

```bash
# Проверьте совместимость версий
python check_compatibility.py

# Обновите torch и torchvision
pip install torch>=1.12.0 torchvision>=0.13.0

# Для Google Colab
!pip install torch>=1.12.0 torchvision>=0.13.0
```

**Ошибки и решения:**

1. **`Conv2dNormActivation` ошибки:**

   - Система автоматически использует упрощенную версию RetinaNet
   - Проверьте версию torchvision: `python -c "import torchvision; print(torchvision.__version__)"`

2. **API изменения в torchvision:**

   - Новые версии используют `weights='DEFAULT'` вместо `pretrained=True`
   - Система автоматически адаптируется к разным версиям

### Проблемы с памятью

Если возникают ошибки CUDA out of memory:

```bash
# Уменьшите размер батча
python train.py --batch_size 8

# Уменьшите размер изображения
python train.py --image_size 256

# Используйте более легкие модели
python train.py --model yolo --image_size 256
```

### Проблемы с загрузкой данных

Убедитесь, что структура данных соответствует ожидаемой:

```
Data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── test/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── train.json
└── test.json
```

### Медленное обучение

Для ускорения обучения:

1. Используйте GPU
2. Увеличьте количество workers в DataLoader
3. Используйте более простые архитектуры (YOLO, SSD)
4. Уменьшите размер изображения

### Частые ошибки

1. **"CUDA out of memory":**

   ```bash
   python train.py --batch_size 4 --image_size 256
   ```

2. **"Module not found":**

   ```bash
   pip install -r requirements.txt
   python check_compatibility.py
   ```

3. **"Data not found":**

   ```bash
   # Проверьте структуру данных
   ls -la Data/
   ```

## 💻 Требования к системе

### Минимальные требования

- Python 3.8+
- PyTorch 1.9+
- 8GB RAM
- 10GB свободного места

### Рекомендуемые требования

- Python 3.9+
- PyTorch 1.12+
- CUDA 11.0+ (для GPU)
- 16GB RAM
- GPU с 4GB+ VRAM
- 20GB свободного места

### Производительность

| Устройство          | YOLO    | SSD     | EfficientDet |
| ------------------- | ------- | ------- | ------------ |
| CPU (Intel i7)      | 4.1 FPS | 0.9 FPS | 5.0 FPS      |
| GPU (RTX 3080)      | ~30 FPS | ~20 FPS | ~15 FPS      |
| Mobile (Snapdragon) | ~2 FPS  | ~1 FPS  | N/A          |

## 📄 Лицензия

Этот проект предназначен для исследовательских целей.

## 🤝 Контакты

По вопросам использования системы обращайтесь к разработчикам проекта.

## 📋 Полный список команд (CTRL+C → CTRL+V)

### 🔧 Установка и настройка

```bash
# Клонирование репозитория
git clone https://github.com/LewyWho/SecurityVision-Classifier
cd SecurityVision-Classifier

# Создание виртуального окружения (Windows)
python -m venv venv
venv\Scripts\activate

# Создание виртуального окружения (Linux/Mac)
python -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Проверка совместимости
python check_compatibility.py
```

### 🎓 Обучение моделей

```bash
# Обучение YOLO модели (быстрая детекция)
python train.py --model yolo --epochs 30 --batch_size 16 --image_size 512

# Обучение SSD модели (сбалансированная)
python train.py --model ssd --epochs 25 --batch_size 16 --image_size 512

# Обучение EfficientDet (точная детекция)
python train.py --model efficientdet --epochs 20 --batch_size 16 --image_size 512

# Обучение всех моделей сразу
python train.py --model all --epochs 15 --batch_size 16 --image_size 512

# Быстрое тестирование (1 эпоха)
python train.py --model yolo --epochs 1 --batch_size 4 --image_size 256
```

### 📊 Оценка моделей

```bash
# Оценка одной модели
python evaluate.py --model_path models/yolo/best.pth --batch_size 16

# Оценка всех найденных моделей
python evaluate.py --evaluate_all --batch_size 16

# Оценка с кастомным размером изображения
python evaluate.py --model_path models/yolo/best.pth --image_size 256 --batch_size 8
```

### 🔍 Инференс и детекция

```bash
# Детекция на одном изображении
python inference.py --model_path models/yolo/best.pth --input image.jpg

# Детекция на директории изображений
python inference.py --model_path models/yolo/best.pth --input test_images/ --batch_size 16

# Детекция с кастомным порогом уверенности
python inference.py --model_path models/yolo/best.pth --input test_images/ --confidence_threshold 0.7

# Детекция с сохранением результатов
python inference.py --model_path models/yolo/best.pth --input test_images/ --save_results
```

### ⚡ Бенчмарк скорости

```bash
# Тестирование всех детекционных моделей
python benchmark_models.py --models yolo ssd efficientdet

# Тестирование только быстрых моделей
python benchmark_models.py --models yolo ssd

# Тестирование с кастомным размером изображения
python benchmark_models.py --models yolo ssd --image_size 256

# Быстрое тестирование (10 прогонов)
python benchmark_models.py --models yolo ssd --num_runs 10
```

### 🛠️ Надежное обучение

```bash
# Надежное обучение с проверкой совместимости
python train_robust.py --models yolo ssd --epochs 10 --check_compatibility

# Надежное обучение всех моделей
python train_robust.py --models yolo efficientdet ssd --epochs 15

# Надежное обучение с кастомными параметрами
python train_robust.py --models yolo --epochs 20 --batch_size 8 --image_size 256
```

### 🔧 Утилиты и проверки

```bash
# Проверка совместимости версий
python check_compatibility.py

# Проверка структуры данных
ls -la Data/
ls -la Data/train/
ls -la Data/test/

# Проверка аннотаций
python -c "import json; data=json.load(open('Data/train.json')); print('Categories:', data.get('categories', 'Not found')); print('Annotations count:', len(data.get('annotations', [])))"

# Проверка доступных моделей
python -c "from models import get_model; print('Доступные модели:'); models = ['yolo', 'efficientdet', 'faster_rcnn', 'retinanet', 'ssd']; [print(f'- {model}') for model in models]"
```

### 🐛 Решение проблем

```bash
# Проблема с памятью - уменьшить размер батча
python train.py --model yolo --batch_size 4 --image_size 256

# Проблема с памятью - уменьшить размер изображения
python train.py --model yolo --image_size 128 --batch_size 8

# Переустановка зависимостей
pip uninstall torch torchvision
pip install torch>=1.12.0 torchvision>=0.13.0

# Очистка кэша PyTorch
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
```

### 📁 Управление файлами

```bash
# Создание папки для тестовых изображений
mkdir test_images

# Копирование изображений для тестирования
copy "Data\test\*.jpg" "test_images\"

# Просмотр структуры проекта
tree /f

# Очистка результатов
rmdir /s models
rmdir /s evaluation_results
rmdir /s inference_images
```

### 🚀 Быстрые сценарии

```bash
# Сценарий 1: Быстрое тестирование системы
python check_compatibility.py
python train.py --model yolo --epochs 1 --batch_size 4 --image_size 256
python benchmark_models.py --models yolo --num_runs 10

# Сценарий 2: Полное обучение и оценка
python train.py --model all --epochs 20 --batch_size 16 --image_size 512
python evaluate.py --evaluate_all --batch_size 16
python benchmark_models.py --models yolo ssd efficientdet faster_rcnn retinanet

# Сценарий 3: Продакшн обучение
python train_robust.py --models yolo ssd --epochs 50 --batch_size 16 --image_size 512 --check_compatibility
python evaluate.py --model_path models/yolo/best.pth --batch_size 16
python inference.py --model_path models/yolo/best.pth --input test_images/ --save_results
```

### 📊 Анализ результатов

```bash
# Просмотр результатов обучения
dir models\yolo\

# Просмотр результатов оценки
dir evaluation_results\yolo\

# Просмотр результатов инференса
dir inference_images\

# Просмотр графиков обучения
start models\yolo\training_curves.png
```
