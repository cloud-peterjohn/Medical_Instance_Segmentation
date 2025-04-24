from torchvision.ops import nms
import torch
import os
import json
from tqdm import tqdm
import os
import json
from torchvision.ops import nms
from utils import format_predictions


def test_and_save_results(
    model,
    test_loader,
    device,
    score_threshold=0.5,
    nms_threshold=0.5,
    output_dir="results",
):
    model.eval()
    all_results = []

    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Test"):
            images = [img.to(device) for img in images]
            predictions = model(images)

            processed_preds = []
            for pred in predictions:
                if "scores" in pred and len(pred["scores"]) > 0:
                    keep = pred["scores"] > score_threshold
                    if keep.any():
                        nms_keep = nms(
                            pred["boxes"][keep], pred["scores"][keep], nms_threshold
                        )
                        idx = torch.where(keep)[0][nms_keep]
                        keep = torch.zeros_like(keep)
                        keep[idx] = True
                    pred = {
                        "boxes": pred["boxes"][keep],
                        "scores": pred["scores"][keep],
                        "labels": pred["labels"][keep],
                        "masks": pred["masks"][keep],
                    }
                processed_preds.append(pred)

            results = format_predictions(processed_preds, image_ids)
            all_results.extend(results)

    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, "test_results.json")
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Test results saved to {output_json}")
