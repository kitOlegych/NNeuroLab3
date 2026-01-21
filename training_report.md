# Отчет обучения EfficientNet-B0 с ранней остановкой

## Результаты
- **Всего эпох:** 17
- **Лучшая точность:** 0.873 (87.3%)
- **Финальная точность:** 0.860 (86.0%)
- **Цель ≥87%:** ДОСТИГНУТА ✅

## Сохраненные файлы
1. `checkpoint_best.tar` - лучшая модель (accuracy=0.873)
2. `checkpoint_last.tar` - последняя модель
3. `training_with_early_stop.png` - график обучения

## Параметры ранней остановки
- **Patience:** 3 эпохи
- **Условие остановки:** 3 эпохи без улучшения accuracy > 0.001
- **Максимум эпох:** 20

## Как использовать чекпоинт
```python
import torch
from model import EfficientNetB0

# Загрузка чекпоинта
checkpoint = torch.load('checkpoint_best.tar')

# Восстановление модели
model = EfficientNetB0(num_classes=10)
model.load_state_dict(checkpoint['model_state_dict'])

# Восстановление оптимизатора
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#print(f"Эпоха: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.3f}