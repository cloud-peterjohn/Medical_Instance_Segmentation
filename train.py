import torch
import os
from tqdm import tqdm
import os
from torchvision.ops import nms, box_iou


def evaluate_model(model, data_loader, device, score_threshold=0.5, nms_threshold=0.5):
    model.eval()
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validate"):

            images = list(image.to(device) for image in images)
            targets = [
                {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)}
                for t in targets
            ]

            predictions = model(images)

            for pred, target in zip(predictions, targets):
                if len(pred["boxes"]) > 0 and len(target["boxes"]) > 0:
                    keep = pred["scores"] > score_threshold
                    if keep.any():
                        nms_keep = nms(
                            pred["boxes"][keep], pred["scores"][keep], nms_threshold
                        )
                        idx = torch.where(keep)[0][nms_keep]
                        keep = torch.zeros_like(keep)
                        keep[idx] = True
                    pred_boxes = pred["boxes"][keep]
                    pred_labels = pred["labels"][keep]

                    if len(pred_boxes) == 0 or len(target["boxes"]) == 0:
                        num_total += len(target["boxes"])
                        continue

                    ious = box_iou(pred_boxes, target["boxes"])

                    max_ious, max_idxs = ious.max(dim=0)

                    for i, (max_iou, max_idx) in enumerate(zip(max_ious, max_idxs)):
                        if (
                            max_iou > 0.5
                            and pred_labels[max_idx] == target["labels"][i]
                        ):
                            num_correct += 1

                num_total += len(target["boxes"])

    accuracy = num_correct / max(1, num_total)
    return accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    score_threshold,
    nms_threshold,
    num_epochs=10,
    lr_max=2e-4,
    lr_min=5e-6,
    weight_decay=1e-4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    output_dir="results",
):
    model = model.to(device)
    print(f"Training Model: {model.__class__.__name__} on device: {device}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_max, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr_min
    )
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0.0
        print(f"\n>>> Training Epoch: {epoch+1}/{num_epochs}")

        scaler = torch.amp.GradScaler("cuda")
        progress_bar = tqdm(train_loader, desc="Train")
        for images, targets in progress_bar:

            images = list(image.to(device) for image in images)
            targets = [
                {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)}
                for t in targets
            ]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += losses.item()
            progress_bar.set_postfix({"train loss": f"{losses.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        print(f"Training Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_train_loss:.4f}")

        val_score = evaluate_model(
            model, val_loader, device, score_threshold, nms_threshold
        )
        print(f"Validation accuracy: {val_score:.4f}")

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr:.6f}")

        checkpoint_path = os.path.join(
            output_dir, f"epoch{epoch+1}_acc{val_score:.4f}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    return model
