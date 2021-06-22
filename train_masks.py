import os
import numpy as np
import json
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


 

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

import itertools
import cv2
import random


# write a function that loads the dataset into detectron2's standard format
def get_food_dicts(img_dir):
   
    json_file = os.path.join(img_dir, "json_masks.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)["image"]
       
        #print(imgs_anns)

    dataset_dicts = []
    print("Loading Dataset!")
    for v in imgs_anns:
        record = {}
#        dim = (width,height)
         

        filename = os.path.join(img_dir, v["-name"])

        try:
            height,width = cv2.imread(filename).shape[:2]
#            print(height)
            
        except:
            continue

        if int(height)>int(width):
            continue
        record["image_id"]=filename
        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
#     print(filename) 
        annos = v["polygon"]

        
        objs = []
        for   anno in annos:
            
            if  isinstance(anno,dict):

                arr = anno["-points"].split(";")
            
                px =[]
                py =[]
                for s in arr:
                    px.append(float(s.split(",")[0]))
                    py.append(float(s.split(",")[1]))
   
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = list(itertools.chain.from_iterable(poly))
            
    
                obj = {
                
                        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": 0,
                        "iscrowd": 0
                    }
   
                objs.append(obj)
                record["annotations"] = objs
                    
                dataset_dicts.append(record)
            if isinstance(anno,str):
                if "point" in anno :
                    arr = annos["-points"].split(";")
                
                    px =[]
                    py =[]
                    for s in arr:
                        px.append(float(s.split(",")[0]))
                        py.append(float(s.split(",")[1]))
       
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                    poly = list(itertools.chain.from_iterable(poly))
                
        
                    obj = {
                    
                            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": [poly],
                            "category_id": 0,
                            "iscrowd": 0
                        }
       
                    objs.append(obj)
                    record["annotations"] = objs
                        
                    dataset_dicts.append(record)
                     
              
              
    return dataset_dicts


for d in ["train_test", "val_test"]:
    DatasetCatalog.register("foodSeg/" + d, lambda d=d: get_food_dicts("./foodSeg/" + d))
    MetadataCatalog.get("foodSeg/" + d).set(thing_classes=["foodSeg"])
    MetadataCatalog.get("foodSeg/" + d).evaluator_type = "coco"

food_metadata = MetadataCatalog.get("foodSeg/train_test")

cfg = get_cfg()
cfg.merge_from_file("detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.OUTPUT_DIR = "output5"
cfg.DATASETS.TRAIN = ("foodSeg/train_test",)
cfg.TEST.EVAL_PERIOD = 100
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS =  "model_data/stage2_35.pth"
  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.000000025
cfg.SOLVER.MAX_ITER = 100   # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# load weights
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# Set training data-set path
cfg.DATASETS.TEST = ("foodSeg/val_test", )

# Create predictor (model for inference)
predictor = DefaultPredictor(cfg)

#Call the COCO Evaluator function and pass the Validation Dataset
evaluator = COCOEvaluator("foodSeg/val_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "foodSeg/val_test")

#Use the created predicted model in the previous step
inference_on_dataset(trainer.model, val_loader, evaluator)

dataset_dicts = get_food_dicts("foodSeg/val_test")
for d in random.sample(dataset_dicts, 30):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=food_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Img", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)
