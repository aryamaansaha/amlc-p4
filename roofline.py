import numpy as np
import matplotlib.pyplot as plt

gflops_peak = 5500  # Theoretical peak performance in GFLOP/s (FP32)
memory_bandwidth = 192  # Memory bandwidth in GB/s

def roofline(x):
    return np.minimum(gflops_peak, memory_bandwidth * x)

x = np.logspace(-2, 4, 1000)
y = roofline(x)

benchmark = {
    "ViT": {
        "flops": 9683.18,
        "bytes_gb": 2.9814,
        "time_s": 5.728
    },
    "ResNet18": {
        "flops": 570.467,
        "bytes_gb": 0.091951,
        "time_s": 777.151/1000
    },
    "SqueezeNet": {
        "flops": 18.06,
        "bytes_gb": 0.050127,
        "time_s": 96.469/1000
    }
}

# Calculate arithmetic intensity and throughput
for name, data in benchmark.items():
    data["ai"] = data["flops"] / (data["bytes_gb"] / data["time_s"])  # FLOP/Byte
    data["throughput"] = data["flops"] / data["time_s"]  # GFLOP/s

plt.figure(figsize=(9, 6))
plt.loglog(x, y, label="Roofline", color='navy', linewidth=2)

colors = {"ViT": "red", "ResNet18": "green", "SqueezeNet": "orange"}

for name, data in benchmark.items():
    plt.scatter(
        data["ai"], data["throughput"],
        label=f"{name} (AI={data['ai']:.1f}, {data['throughput']:.0f} GFLOP/s)",
        color=colors[name],
        s=120,
        alpha=0.85,
        edgecolors='black',
        linewidth=0.75
    )

# Memory-compute intersection point
ai_roof = gflops_peak / memory_bandwidth
plt.axvline(x=ai_roof, color='gray', linestyle='dashed', linewidth=2, label=f"AI={ai_roof:.2f} FLOP/Byte")

plt.xlabel("Arithmetic Intensity (FLOP/Byte)")
plt.ylabel("Performance (GFLOP/s)")
plt.title("Roofline Model on NVIDIA P4 GPU")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("roofline_p4.png", dpi=300)
plt.show()