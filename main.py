import torch
from model import get_model
from datasets import create_data_loaders
from train import train_model
from test import test_and_save_results
from argsparse import parse_args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = create_data_loaders(
        data_root=args.data_root, batch_size=args.batch_size
    )

    model = get_model(
        num_classes=args.num_classes,
        hidden_layer=args.hidden_layer,
        trainable_backbone_layers=args.trainable_backbone_layers,
    )

    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        num_epochs=args.num_epochs,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=args.output_dir,
    )

    test_and_save_results(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
