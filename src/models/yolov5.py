import torch


class YoloV5Hub:
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    def forward(self, img):
        # Inference
        results = self.model(img)
        return self.get_bounding_boxes_numpy(results)

    def get_bounding_boxes_numpy(self, results):
        return results.xyxy[0].numpy()
