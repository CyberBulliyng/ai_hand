# -*- coding: utf-8 -*-
"""
Шаг 2. Обучение MobileNetV2 (transfer learning) на 10 жестах.
Запуск: python train.py
Результат: best_model.pth (веса) + classes.json (порядок классов).
"""
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA = Path(__file__).parent / "data" / "hagrid_5000"
BATCH = 32
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # повороты и сдвиги
    transforms.RandomHorizontalFlip(),                 # жест левой/правой рукой
    transforms.ColorJitter(0.2, 0.2, 0.2),             # вариации света
    transforms.ToTensor(), NORM,
])
EVAL_TF = transforms.Compose([transforms.Resize((224, 224)),
                              transforms.ToTensor(), NORM])


def loaders():
    tr = datasets.ImageFolder(DATA / "train", TRAIN_TF)
    va = datasets.ImageFolder(DATA / "val", EVAL_TF)
    te = datasets.ImageFolder(DATA / "test", EVAL_TF)
    common = dict(num_workers=0, pin_memory=True)
    return (DataLoader(tr, BATCH, shuffle=True, **common),
            DataLoader(va, BATCH, **common),
            DataLoader(te, BATCH, **common), tr.classes)


def run_epoch(model, loader, criterion, opt=None):
    train = opt is not None
    model.train(train)
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main():
    if not DATA.exists():
        raise SystemExit("Нет папки с данными! Сначала запусти download_data.py")
    print(f"Устройство: {DEVICE}")

    tr_loader, va_loader, te_loader, classes = loaders()
    print(f"Классы ({len(classes)}): {classes}")
    print(f"Фото: train={len(tr_loader.dataset)}, val={len(va_loader.dataset)}, "
          f"test={len(te_loader.dataset)}")

    # MobileNetV2 с весами ImageNet, меняем последний слой под 10 классов
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc, t0 = 0.0, time.time()
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, tr_loader, criterion, opt)
        va_loss, va_acc = run_epoch(model, va_loader, criterion)
        sched.step()
        tag = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), Path(__file__).parent / "best_model.pth")
            json.dump(classes, open(Path(__file__).parent / "classes.json", "w"))
            tag = "  <- лучшая, сохранена"
        print(f"Эпоха {ep:02d}/{EPOCHS} | train: loss={tr_loss:.3f} acc={tr_acc:.3f} | "
              f"val: loss={va_loss:.3f} acc={va_acc:.3f}{tag}")

    # итог на тесте
    model.load_state_dict(torch.load(Path(__file__).parent / "best_model.pth",
                                     weights_only=True))
    _, te_acc = run_epoch(model, te_loader, criterion)
    print(f"\nИТОГ: точность на тесте = {te_acc:.3f} "
          f"({time.time()-t0:.0f} сек обучения)")

    # точность по каждому жесту
    model.eval()
    per = {c: [0, 0] for c in classes}
    with torch.no_grad():
        for x, y in te_loader:
            pred = model(x.to(DEVICE)).argmax(1).cpu()
            for t, p in zip(y, pred):
                per[classes[t]][1] += 1
                if t == p:
                    per[classes[t]][0] += 1
    for c, (ok, n) in per.items():
        print(f"  {c:10s}: {ok}/{n} ({ok/n:.2f})")


if __name__ == "__main__":
    main()
