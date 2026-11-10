# Распознавание жестов рук (HaGRID + MobileNetV2)

Классификация 10 статических жестов рук по изображению. Модель — **MobileNetV2** (transfer learning), точность на тесте **80.8%**. Включает приложение реального времени с веб-камеры.

## Результаты

| | Значение |
|---|---|
| Модель | MobileNetV2, предобучена на ImageNet |
| Классы (10) | dislike, fist, like, mute, ok, palm, peace, rock, stop, three |
| Обучающая выборка | 5 000 изображений (500 × 10 жестов, сбалансировано) |
| Accuracy (val / test) | 87.6% / **80.8%** |
| Скорость инференса | реальное время (веб-камера, ~30 FPS) |
| GPU для обучения | ~12 минут на RTX 4060 (8 ГБ) |

## Структура репозитория

```
├── README.md           — этот файл
├── report.md           — отчёт о проделанной работе
├── requirements.txt    — зависимости Python
├── config.json         — конфигурации экспериментов и метрики
├── download_data.py    — формирование выборки из HaGRID (5000 фото)
├── train.py            — обучение MobileNetV2
├── app.py              — приложение: веб-камера / фото
├── best_model.pth      — веса обученной модели (~14 МБ)
└── screenshots/        — примеры работы приложения
```

## Датасет

- Исходный датасет: [HaGRID — HAnd Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid) (WACV 2024, лицензия CC BY-SA 4.0)
- Использованная версия: [HaGRIDv2 512px на Hugging Face](https://huggingface.co/datasets/testdummyvt/hagRIDv2_512px)
- Скрипт `download_data.py` сам скачивает нужные изображения (≈1 ГБ) и раскладывает их по папкам: `data/hagrid_5000/{train,val,test}/<жест>/*.jpg` — 4000/500/500 фото. В репозиторий данные не входят.

## Установка

1. Python 3.10–3.13: https://www.python.org/downloads/ (при установке отметить **Add python.exe to PATH**).
2. Зависимости:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # с GPU (NVIDIA)
   # или: pip install torch torchvision   # только CPU (обучение будет медленным)
   pip install -r requirements.txt
   ```
3. Проверка GPU: `python -c "import torch; print(torch.cuda.is_available())"` → должно быть `True`.

## Запуск (3 шага)

```bash
# 1. Выгрузка данных: 5000 фото HaGRID (~1 ГБ, 5–15 минут)
python download_data.py

# 2. Обучение MobileNetV2 (15 эпох, ~12 минут на GPU)
python train.py

# 3. Приложение
python app.py                 # веб-камера в реальном времени (Q / Esc — выход)
python app.py photo.jpg       # или предсказание для одного фото
```

Результат обучения: `best_model.pth` (веса) и `classes.json` (порядок классов). Веса уже приложены в репозитории — шаги 1–2 можно пропустить и сразу запустить `app.py`.

## Конфигурация экспериментов

Все гиперпараметры и результаты двух экспериментов — в [`config.json`](config.json):
базовый прогон (10 эпох) и итоговый (+ RandomAffine аугментация, 15 эпох). Метрики по классам — в [`report.md`](report.md).

## Примечания

- Жесты `ok`, `peace`, `three` распознаются хуже остальных (58–66%) — они визуально похожи (различаются числом видимых пальцев). Подробнее — в разделе анализа ошибок отчёта.
- Приложение ожидает руку в центре кадра, как в фото HaGRID (рука занимает заметную часть кадра, нейтральный фон).
