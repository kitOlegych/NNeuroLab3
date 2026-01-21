import torch, torchvision, time, onnx, onnxruntime, numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import platform
import psutil  # нужно установить: pip install psutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
markdown_content = f"""
# Информация о системе

## Характеристики CPU
| Параметр | Значение |
|----------|----------|
| Процессор | {platform.processor()} |
| Архитектура | {platform.machine()} |
| Память RAM | {psutil.virtual_memory().total / (1024**3):.2f} GB |
| Потоков CPU | {psutil.cpu_count(logical=True)} |

## PyTorch информация
| Параметр | Значение |
|----------|----------|
| Версия PyTorch | {torch.__version__} |
| Доступен CUDA | {torch.cuda.is_available()} |
| Кол-во GPU | {torch.cuda.device_count() if torch.cuda.is_available() else 0} |

"""

print(markdown_content)

# Сохраняем в файл
with open('docs/system_info.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)

