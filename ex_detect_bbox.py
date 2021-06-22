# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import numpy as np

# get image
#!wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
#im = cv2.imread("./input.jpg")
im = cv2.imdecode(np.fromfile('/home/cronos/kiwamecvm/Jose.jpg', dtype=np.uint8), cv2.IMREAD_UNCHANGED)

# Create config
cfg = get_cfg()
cfg.merge_from_file("detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"

# Create predictor
predictor = DefaultPredictor(cfg)

# Make prediction
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("img", v.get_image()[:, :, ::-1])
cv2.waitKey(0)
