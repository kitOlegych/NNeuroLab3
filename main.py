import torch
import torchvision
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

os.makedirs('docs', exist_ok=True)

# ========== НАСТРОЙКА ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Создаем EfficientNet-B0
model = torchvision.models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
model = model.to(device)
model.eval()

print(f"Model: EfficientNet-B0")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ========== ИЗМЕРЕНИЕ LATENCY ==========
print(f"\nMeasuring latency for 100 runs...")

# Создаем тестовый тензор (CIFAR-10 размер)
dummy_input = torch.randn(1, 3, 32, 32).to(device)

# Warm-up
print("  Warm-up (10 runs)...")
for _ in range(10):
    with torch.no_grad():
        _ = model(dummy_input)

# Измеряем latency
latencies = []
print("  Measuring (100 runs)...")
for i in tqdm(range(100), desc="Progress", ncols=60):
    start_time = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_input)
    end_time = time.perf_counter()
    latencies.append((end_time - start_time) * 1000)  # в миллисекунды

# ========== СТАТИСТИКА ==========
latencies_np = np.array(latencies)
stats = {
    'mean': np.mean(latencies_np),
    'median': np.median(latencies_np),
    'std': np.std(latencies_np),
    'min': np.min(latencies_np),
    'max': np.max(latencies_np),
    'p95': np.percentile(latencies_np, 95),
}

print(f"\nRESULTS:")
print(f"  Mean latency:   {stats['mean']:.2f} ± {stats['std']:.2f} ms")
print(f"  Median latency: {stats['median']:.2f} ms")
print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}] ms")
print(f"  P95 latency:    {stats['p95']:.2f} ms")
print(f"  Throughput:     {1000/stats['mean']:.0f} FPS")

# ========== ГРАФИК ==========
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# График 1: Гистограмма
axes[0].hist(latencies_np, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                label=f'Mean: {stats["mean"]:.2f}ms')
axes[0].axvline(stats['median'], color='green', linestyle=':', linewidth=2,
                label=f'Median: {stats["median"]:.2f}ms')
axes[0].set_xlabel('Latency (ms)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Latency Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# График 2: По прогонам
axes[1].plot(latencies, 'b-', alpha=0.6, linewidth=1)
axes[1].axhline(y=stats['mean'], color='r', linestyle='--', label='Mean')
axes[1].fill_between(range(100),
                     stats['mean'] - stats['std'],
                     stats['mean'] + stats['std'],
                     alpha=0.2, color='gray', label='±1σ')
axes[1].set_xlabel('Run #')
axes[1].set_ylabel('Latency (ms)')
axes[1].set_title('Latency over 100 runs')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'EfficientNet-B0 Latency Measurement | Device: {device}',
             fontsize=12, fontweight='bold')
plt.tight_layout()

# Сохраняем график
plt.savefig('docs/latency.png', dpi=120, bbox_inches='tight')

with open('docs/latency_results.txt', 'w') as f:
    f.write(f"""EfficientNet-B0 Latency Results
================================
Device: {device}
Model parameters: {sum(p.numel() for p in model.parameters()):,}
Input shape: 1x3x32x32
Number of runs: 100

Latency Statistics (ms):
  Mean:    {stats['mean']:.2f} ± {stats['std']:.2f}
  Median:  {stats['median']:.2f}
  Min:     {stats['min']:.2f}
  Max:     {stats['max']:.2f}
  P95:     {stats['p95']:.2f}

Performance:
  FPS:     {1000/stats['mean']:.0f}
  ms/frame: {stats['mean']:.1f}
""")

print(f"docs/latency_results.txt")