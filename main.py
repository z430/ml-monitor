from src.models.yolov5 import YoloV5Hub
from src.data.fiftyone_dataset import FiftyOneDataset

model = YoloV5Hub()
dataset = FiftyOneDataset("coco-2017-validation")
dataset.evaluate(model.model)
