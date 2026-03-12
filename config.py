PROJECT_NAME = "adaptive-emotion-aware-virtual-interaction-system"

DATA_CONFIG = {
    "dataset_root": "data",
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "num_classes": 7,
}

TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "adam",
    "scheduler": "cosine",
}

MODEL_CONFIGS = {
    "mn_xception": {
        "display_name": "MN-Xception",
        "input_size": 64,
        "pretrained": True,
        "dropout": 0.5,
    },
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "input_size": 224,
        "pretrained": True,
        "dropout": 0.4,
    },
    "resnet18": {
        "display_name": "ResNet-18",
        "input_size": 224,
        "pretrained": True,
        "dropout": 0.3,
    },
    "hsemotion": {
        "display_name": "HSEmotion",
        "input_size": 224,
        "pretrained": True,
        "dropout": 0.5,
    },
}
