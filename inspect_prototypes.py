import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load latest prototype geometry
with open("YOLOX_outputs/tde_yolox_s/telemetry/prototype_geometry.jsonl", "r") as f:
    epochs = [json.loads(line) for line in f]

latest = epochs[-1]
epoch_num = latest["epoch"]
p3_matrix = np.array(latest["levels"]["level_0_stride_8"]["inter_class_similarity_matrix"])
intra_sim = latest["levels"]["level_0_stride_8"]["intra_class_subcentroid_similarity"]

class_names = ["Ped", "Rider", "Car", "Truck", "Bus", "Train", "Moto", "Bike", "Light"]

plt.figure(figsize=(10, 8))
sns.heatmap(p3_matrix, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names, cmap="coolwarm")
plt.title(f"Level P3 Class-Prototype Cosine Similarity (Epoch {epoch_num})")
plt.savefig(f"prototype_matrix_epoch_{epoch_num}.png", dpi=300)

print(f"Epoch {epoch_num} Prototype Diagnostics:")
for c, sim in zip(class_names, intra_sim):
    status = "OK" if sim < 0.70 else "WARNING: COLLAPSING"
    print(f"  Class {c:<6} Intra-Centroid Sim: {sim:.3f} [{status}]")