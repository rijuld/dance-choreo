from utils import datasets
from utils.config import default_config

# Define mappings
time_effort_map = {
    1: "Sustained",
    2: "Neutral Time",
    3: "Sudden/Quick",
    4: "Unknown Time"
}

space_effort_map = {
    1: "Indirect",
    2: "Neutral Space",
    3: "Direct",
    4: "Unknown Space"
}

# Helper: Create config per effort
def get_config(effort):
    cfg = {k: getattr(default_config, k) for k in dir(default_config) if not k.startswith("__") and not callable(getattr(default_config, k))}
    cfg["effort"] = effort
    return cfg

# Load both datasets
config_time = get_config("time")
config_space = get_config("space")

# Load with same strategy (choose one)
if config_time["train_ratio"] and config_time["train_lab_frac"]:
    _, labels_train_time, *_ = datasets.get_model_data(config_time)
    _, labels_train_space, *_ = datasets.get_model_data(config_space)
else:
    _, labels_train_time, *_ = datasets.get_model_specific_data(config_time)
    _, labels_train_space, *_ = datasets.get_model_specific_data(config_space)

# Combine labels
combined_text_labels = []

for batch_time, batch_space in zip(labels_train_time, labels_train_space):
    for time_label, space_label in zip(batch_time, batch_space):
        time_label_idx = int(time_label.item())
        space_label_idx = int(space_label.item())
        label_str = f"{time_effort_map[time_label_idx]} and {space_effort_map[space_label_idx]}"
        combined_text_labels.append(label_str)
    break  # only first batch for demo

# Result
print("\n=== Combined Text Labels (time and space) ===")
for label in combined_text_labels[:100]:  # preview first 10
    print("-", label)

# If needed as dictionary
effort_text_labels = {
    "time_and_space": combined_text_labels
}
