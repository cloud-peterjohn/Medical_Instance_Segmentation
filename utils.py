import numpy as np
import skimage.io as sio
from pycocotools import mask as mask_utils


def decode_maskobj(mask_obj):
    return mask_utils.decode(mask_obj)


def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def read_maskfile(filepath):
    mask_array = sio.imread(filepath)
    return mask_array


def format_predictions(predictions, image_ids):
    formatted_results = []

    for pred, image_id in zip(predictions, image_ids):
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        masks = pred["masks"].cpu().numpy()

        for i in range(len(boxes)):
            mask = masks[i, 0] > 0.5
            rle = encode_mask(mask)
            result = {
                "image_id": image_id,
                "bbox": boxes[i].tolist(),
                "score": float(scores[i]),
                "category_id": int(labels[i]),
                "segmentation": rle,
            }
            formatted_results.append(result)

    return formatted_results
