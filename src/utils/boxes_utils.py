def xyxy_to_xywh_norm(boxes, img_size):
    boxes = boxes.copy()
    boxes[:, 0] = boxes[:, 0] / img_size[0]
    boxes[:, 1] = boxes[:, 1] / img_size[1]
    boxes[:, 2] = boxes[:, 2] / img_size[0]
    boxes[:, 3] = boxes[:, 3] / img_size[1]
    return boxes
