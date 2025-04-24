import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and test a model for object detection."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="hw3-data-release",
        help="Path to the dataset root.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results."
    )
    parser.add_argument(
        "--batch_size", type=int, default=3, help="Batch size for data loaders."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr_max", type=float, default=2e-4, help="Maximum learning rate."
    )
    parser.add_argument(
        "--lr_min", type=float, default=5e-6, help="Minimum learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of classes (including background).",
    )
    parser.add_argument(
        "--hidden_layer", type=int, default=256, help="Size of the hidden layer."
    )
    parser.add_argument(
        "--trainable_backbone_layers",
        type=int,
        default=3,
        help="Number of trainable backbone layers.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Score threshold for predictions.",
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.5,
        help="NMS threshold for predictions.",
    )
    return parser.parse_args()
