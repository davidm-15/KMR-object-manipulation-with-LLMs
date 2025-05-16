import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set your path here
BASE_DIR = "vision_models_comparison"

# Containers for collected data
records = []
model_names = ["dino", "owlvit2", "QwenDino72"]
# Load data from folders
for model_name in model_names:
    model_path = os.path.join(BASE_DIR, model_name)
    if not os.path.isdir(model_path):
        continue

    all_files = os.listdir(model_path)
    all_files.sort()
    for filename in all_files:
        if filename.startswith("metrics_") and filename.endswith(".json"):
            filepath = os.path.join(model_path, filename)
            with open(filepath, "r") as f:
                data = json.load(f)["metrics"]
            
            prompt_id = int(filename.split("_")[1].split(".")[0])
            
            records.append({
                "model": model_name,
                "prompt": f"Prompt {prompt_id}",
                "f1_score": data["f1_score"],
                "precision": data["precision"],
                "recall": data["recall"],
                "accuracy": data["accuracy"],
                "true_positives": data["true_positives"],
                "true_negatives": data["true_negatives"],
                "false_positives": data["false_positives"],
                "false_negatives": data["false_negatives"],
            })

# Convert to DataFrame
df = pd.DataFrame(records)

# Set plot style
sns.set(style="whitegrid")
palette = sns.color_palette("Set2")

# Plotting function
def plot_metric(metric_name):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="prompt",
        y=metric_name,
        hue="model",
        palette=palette,
        ci="sd"
    )
    plt.title(f"{metric_name.capitalize()} per Model and Prompt")
    plt.ylabel(metric_name.capitalize())
    plt.xlabel("Prompt Variant")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"images/comparison/{metric_name}_comparison.png")

# Plot key metrics
plot_metric("f1_score")
plot_metric("precision")
plot_metric("recall")
