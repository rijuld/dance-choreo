from utils import datasets
from utils.config import default_config


config = {
    "run_name": default_config.run_name,
    "load_from_checkpoint": default_config.load_from_checkpoint,
    "epochs": default_config.epochs,
    "learning_rate": default_config.learning_rate,
    "batch_size": default_config.batch_size,
    "seq_len": default_config.seq_len,
    "with_clip": default_config.with_clip,
    "input_dim": default_config.input_dim,
    "kl_weight": default_config.kl_weight,
    "neg_slope": default_config.neg_slope,
    "n_layers": default_config.n_layers,
    "h_dim": default_config.h_dim,
    "latent_dim": default_config.latent_dim,
    "neg_slope_classif": default_config.neg_slope_classif,
    "n_layers_classif": default_config.n_layers_classif,
    "h_dim_classif": default_config.h_dim_classif,
    "label_dim": default_config.label_dim,
    "device": default_config.device,
    "effort": default_config.effort,
    "fraction_label": default_config.fraction_label,
    "train_ratio": default_config.train_ratio,
    "train_lab_frac": default_config.train_lab_frac,
    "shuffle_data": default_config.shuffle_data,
}

# Priority 1: use train_ratio + train_lab_frac if defined
if config["train_ratio"] is not None and config["train_lab_frac"] is not None:
    (
        labelled_data_train,
        labels_train,
        unlabelled_data_train,
        labelled_data_valid,
        labels_valid,
        labelled_data_test,
        labels_test,
        unlabelled_data_test,
    ) = datasets.get_model_data(config)

# Priority 2: else use fraction_label if defined
elif config["fraction_label"] is not None:
    (
        labelled_data_train,
        labels_train,
        unlabelled_data_train,
        labelled_data_valid,
        labels_valid,
        labelled_data_test,
        labels_test,
        unlabelled_data_test,
    ) = datasets.get_model_specific_data(config)

# print information about the dataloaders
print("labelled_data_train batch size: ", labelled_data_train.batch_size)
print("labels_train batch size: ", labels_train.batch_size)
print("unlabelled_data_train batch size: ", unlabelled_data_train.batch_size)
print("labelled_data_valid batch size: ", labelled_data_valid.batch_size)
print("labels_valid batch size: ", labels_valid.batch_size)
print("labelled_data_test batch size: ", labelled_data_test.batch_size)
print("labels_test batch size: ", labels_test.batch_size)
print("unlabelled_data_test batch size: ", unlabelled_data_test.batch_size)

print("\n=== Dataset Shapes ===\n")

# Access the dataset objects to get their shapes
print("labelled_data_train dataset shape:", labelled_data_train.dataset.shape)
print("labels_train dataset shape:", labels_train.dataset.shape)
print("unlabelled_data_train dataset shape:", unlabelled_data_train.dataset.shape)
print("labelled_data_valid dataset shape:", labelled_data_valid.dataset.shape)
print("labels_valid dataset shape:", labels_valid.dataset.shape)
print("labelled_data_test dataset shape:", labelled_data_test.dataset.shape)
print("labels_test dataset shape:", labels_test.dataset.shape)
print("unlabelled_data_test dataset shape:", unlabelled_data_test.dataset.shape)

# Print additional information about the data structure
print("\n=== Data Structure Information ===\n")
print(f"Sequence length: {config['seq_len']}")
print(f"Input dimension: {config['input_dim']}")
print(f"Label dimension: {config['label_dim']}")

# Print a sample from each dataset to understand the data format
print("\n=== Sample Data ===\n")
print("Sample from labelled_data_train (first batch):")
for batch in labelled_data_train:
    print(f"  Batch shape: {batch.shape}")
    break

print("Sample from labels_train (first batch):")
for batch in labels_train:
    print(f"  Batch shape: {batch.shape}")
    break
