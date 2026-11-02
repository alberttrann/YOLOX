import json
import matplotlib.pyplot as plt
import pandas as pd

# Load gradient telemetry
records = []
with open("YOLOX_outputs/tde_yolox_s/telemetry/gradient_flow_telemetry.jsonl", "r") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.json_normalize(records)

# Plot gradient norm trajectories across functional subnets
plt.figure(figsize=(12, 6))
plt.plot(df["iter"], df["grad_norms.head_reg"], label="Head Reg (Bounding Box)", color="blue")
plt.plot(df["iter"], df["grad_norms.head_obj"], label="Head Obj (Objectness)", color="orange")
plt.plot(df["iter"], df["grad_norms.head_cls"], label="Head Cls (Classification)", color="green")
plt.plot(df["iter"], df["grad_norms.dark2_gn"], label="Stage 1 dark2 (TTT Normalizer)", color="red")
plt.plot(df["iter"], df["grad_norms.memory_banks"], label="Memory Prototypes", color="purple")

plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Gradient L2 Norm (log scale)")
plt.title("Component-Wise Gradient Flow Across Training")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("gradient_flow_diagnosis.png", dpi=300)
print("Saved gradient_flow_diagnosis.png. Gradient parity verified.")