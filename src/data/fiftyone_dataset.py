from typing import List
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as func
import cv2


class FiftyOneDataset:
    def __init__(self, dataset_name: str) -> None:
        self.dataset = fo.load_dataset(dataset_name)
        self.classes = None

    def load_zoo_dataset(self):
        self.dataset = foz.load_zoo_dataset("coco-2017", split="validation")

    def load_image_as_tensor(self, sample: fo.Sample) -> torch.Tensor:
        image = Image.open(sample.filepath)
        image = func.to_tensor(image)
        return image

    def load_image_as_numpy(self, sample: fo.Sample) -> np.ndarray:
        return cv2.imread(sample.filepath)

    def evaluate(self, model) -> None:
        self.classes = self.dataset.default_classes
        with fo.ProgressBar() as pb:
            for sample in pb(self.dataset):
                image = self.load_image_as_numpy(sample)

                # Inference
                results = model([image])

                labels = results[:, 5].astype(int)
                scores = results[:, 4]
                boxes = results[:, :4]

                self.add_predictions_to_sample(labels, scores, boxes, sample)

    def add_predictions_to_sample(
        self, labels, scores, boxes, sample: fo.Sample
    ) -> None:
        h, w = sample.size
        detections = []

        for label, score, box in zip(labels, scores, boxes):
            # convert to tlwh in relative coordinates
            x1, y1, x2, y2 = box
            rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
            detections.append(
                fo.Detection(
                    label=self.classes[label],
                    bounding_box=rel_box,
                    confidence=score,
                )
            )
        # save predictions to dataset
        sample["predictions"] = fo.Detections(detections=detections)
        sample.save()
