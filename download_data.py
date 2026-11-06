# -*- coding: utf-8 -*-
"""
Шаг 1. Выгрузка сбалансированной выборки из HaGRID (512px) с Hugging Face.
10 жестов x 500 фото = 5000 изображений.
Папки: data/hagrid_5000/{train,val,test}/<жест>/*.jpg
Запуск: python download_data.py

Как работает (данные упорядочены по классам, parquet колоночный):
  1) в каждом файле читаем ТОЛЬКО колонку label (быстро, почти без скачивания),
  2) бинарным поиском находим файл с началом каждого жеста (10 потоков),
  3) скачиваем 1-2 целых parquet-файла на жест и сохраняем 500 фото.
"""
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # xet-CDN иногда сбоит — качаем по обычному HTTP

import fsspec
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

REPO = "testdummyvt/hagRIDv2_512px"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/refs%2Fconvert%2Fparquet/default/train/{{name}}.parquet"
CLASSES = ["dislike", "fist", "like", "mute", "ok",
           "palm", "peace", "rock", "stop", "three"]  # 10 жестов
PER_CLASS = 500          # всего на класс
N_TRAIN, N_VAL = 400, 50  # остальное (50) — тест
OUT = Path(__file__).parent / "data" / "hagrid_5000"

_cache = {}


def labels_of(i):
    """Номера классов в файле i — читаем только колонку label (диапазонные запросы)."""
    if i not in _cache:
        with fsspec.open(URL.format(name=f"{i:04d}"), "rb").open() as f:
            _cache[i] = pq.read_table(f, columns=["label"])["label"].to_pylist()
    return _cache[i]


def first_file_with(target):
    """Первый файл, где встречается класс target (файлы упорядочены по классам)."""
    lo, hi = 0, N_FILES
    while lo < hi:
        mid = (lo + hi) // 2
        if max(labels_of(mid)) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


N_FILES = 0


def main():
    for split in ("train", "val", "test"):
        for c in CLASSES:
            (OUT / split / c).mkdir(parents=True, exist_ok=True)

    global N_FILES
    print("Список файлов...")
    N_FILES = len([f for f in HfApi().list_repo_files(
        REPO, repo_type="dataset", revision="refs/convert/parquet")
        if f.startswith("default/train/") and f.endswith(".parquet")])
    names = load_names()
    print(f"Файлов: {N_FILES}, классов: {len(names)}")

    print("Ищу файлы с нужными жестами (параллельный бинарный поиск)...")
    pos_file = OUT / "_pos.json"
    if pos_file.exists():
        pos = json.load(open(pos_file))
        print(f"Взял из кэша: {pos}")
    else:
        t0 = time.time()
        with ThreadPoolExecutor(len(CLASSES)) as ex:
            starts = list(ex.map(lambda c: first_file_with(names.index(c)), CLASSES))
        pos = dict(zip(CLASSES, starts))
        json.dump(pos, open(pos_file, "w"))
        print(f"Найдено за {time.time()-t0:.0f} сек: {pos}")

    print("Скачиваю и сохраняю фото...")
    for c in sorted(CLASSES, key=names.index):
        t, fi, need = names.index(c), pos[c], PER_CLASS
        while need > 0 and fi < N_FILES:
            local = download_with_retry(fi)
            table = pq.read_table(local, columns=["image", "label"])
            for cell, lab in zip(table["image"].to_pylist(),
                                 table["label"].to_pylist()):
                if lab != t:
                    continue
                i = PER_CLASS - need
                split = ("train" if i < N_TRAIN else
                         "val" if i < N_TRAIN + N_VAL else "test")
                Image.open(io.BytesIO(cell["bytes"])).convert("RGB").save(
                    OUT / split / c / f"{c}_{i:04d}.jpg", quality=90)
                need -= 1
                if need == 0:
                    break
            fi += 1
        if need:
            raise SystemExit(f"Не добрали фото для {c}: не хватает {need}")
        print(f"  {c}: 500 фото готово")

    n = sum(len(list(p.glob("*.jpg"))) for p in OUT.rglob("*") if p.is_dir())
    print(f"Готово! Всего {n} фото в {OUT}")


def load_names():
    from datasets import load_dataset
    return load_dataset(REPO, split="train", streaming=True).features["label"].names


def download_with_retry(fi, tries=6):
    """Скачивание parquet-файла с повторами при сбоях сети."""
    for n in range(1, tries + 1):
        try:
            return hf_hub_download(REPO, f"default/train/{fi:04d}.parquet",
                                   repo_type="dataset",
                                   revision="refs/convert/parquet")
        except Exception as e:
            wait = 5 * n
            print(f"  файл {fi:04d}: сбой ({str(e)[:80]}), жду {wait} сек и пробую снова")
            time.sleep(wait)
    raise SystemExit(f"Не удалось скачать файл {fi:04d} после {tries} попыток")


if __name__ == "__main__":
    main()
