# -*- coding: utf-8 -*-
"""
Шаг 3. Приложение: распознавание жеста.
- python app.py              -> веб-камера в реальном времени (Q или Esc — выход)
- python app.py photo.jpg    -> предсказание для одного файла
"""
import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF = transforms.Compose([transforms.Resize((224, 224)),
                         transforms.ToTensor(), NORM])
RU = {"dislike": "дизлайк", "fist": "кулак", "like": "лайк", "mute": "мьют",
      "ok": "ок", "palm": "ладонь", "peace": "мир (V)", "rock": "рок",
      "stop": "стоп", "three": "три"}


def load_model():
    classes = json.load(open(ROOT / "classes.json"))
    model = models.mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(torch.load(ROOT / "best_model.pth", map_location=DEVICE,
                                     weights_only=True))
    model.to(DEVICE).eval()
    return model, classes


@torch.no_grad()
def predict(model, classes, img: Image.Image):
    logits = model(TF(img).unsqueeze(0).to(DEVICE))
    probs = torch.softmax(logits, 1)[0]
    i = probs.argmax().item()
    return classes[i], probs[i].item()


def predict_file(path):
    model, classes = load_model()
    cls, p = predict(model, classes, Image.open(path).convert("RGB"))
    print(f"Жест: {RU.get(cls, cls)} (уверенность {p:.0%})")


def run_webcam():
    model, classes = load_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Веб-камера недоступна")
    print("Камера включена. Покажи жест в центр кадра. Q или Esc — выход.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cls, p = predict(model, classes, img)
        label = f"{RU.get(cls, cls)}: {p:.0%}"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)
        cv2.imshow("Gesture", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_file(sys.argv[1]) if len(sys.argv) > 1 else run_webcam()
