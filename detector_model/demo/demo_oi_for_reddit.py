# coding=utf-8
# Copyleft 2019 Project LXRT

import argparse
import base64
import csv
import json
import math
import os
import random
import sys
import time
csv.field_size_limit(sys.maxsize)

# import some common libraries
import cv2
import numpy as np
import torch
import tqdm

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.data import MetadataCatalog

D2_ROOT = os.path.dirname(os.path.dirname(detectron2.__file__)) # Root of detectron2
# DATA_ROOT = os.getenv('COCO_IMG_ROOT', '/ssd-playpen/data/mscoco/images/')
# COCO_ROOT = '/data/mscoco'
# VG_ROOT = '/data/vg'
LXRT_ROOT = '/data/lxmert'
OI_ROOT = '/data/data_for_vokenization/open_images/images'
OI_PATH_FILE = '/home/projects/vokenization/data/vokenization/images/open_images.path'
# data_split = 'test' #train valid test


DATA_ROOT = '/data/vg/images'
MIN_BOXES = 36
MAX_BOXES = 36

imgset2fname = {
    'coco_train': 'mscoco_train.json',
    'coco_nominival': 'mscoco_nominival.json',
    'coco_minival': 'mscoco_minival.json',
    'vg_nococo': 'vgnococo.json',
    'cc_train': 'training.tsv',
    'cc_valid': 'validation.tsv',
}

parser = argparse.ArgumentParser()
parser.add_argument('--img-set', default='open_images_490k', help='coco_train, coco_nominival, coco_minival, vg_nococo') # useless param
parser.add_argument('--data_split', default='valid', help='voken split of the reddit data, eg., train, valid, test') # useless param
parser.add_argument('--split', default='train2014', help='train2014, val2014') # useless param
parser.add_argument('--batchsize', default=6, type=int, help='batch_size')
parser.add_argument('--model', default='res5', type=str, help='options: "res4", "res5"; features come from)')
parser.add_argument('--weight', default='vg', type=str, 
        help='option: mask, obj, vg. mask:mask_rcnn on COCO, obj: faster_rcnn on COCO, vg: faster_rcnn on Visual Genome')

args = parser.parse_args()

from torchvision.ops import nms
from detectron2.structures import Boxes, Instances

data_path = 'data/genome/1600-400-20'

VOKEN_ROOT = '/data/xmatching_model/clsmlp_sent-level_b1024_embed512_maxlen75_resnext_bert/vokens_reddit1M_oi/'+ args.data_split +'.pkl.txt.open_images.494159'

vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())
        
vg_attrs = []
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for object in f.readlines():
        vg_attrs.append(object.split(',')[0].lower().strip())


MetadataCatalog.get("vg").thing_classes = vg_classes
MetadataCatalog.get("vg").attr_classes = vg_attrs

def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    
    # Select max scores
    max_scores, max_classes = scores.max(1)       # R x C --> R
    num_objs = boxes.size(0)
    boxes = boxes.view(-1, 4)
    idxs = torch.arange(num_objs).cuda() * num_bbox_reg_classes + max_classes
    max_boxes = boxes[idxs]     # Select max boxes according to the max scores.
    
    # Apply NMS
    keep = nms(max_boxes, max_scores, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores = max_boxes[keep], max_scores[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = max_classes[keep]
    
    return result, keep

def doit(detector, raw_images):
    with torch.no_grad():
        # Preprocessing
        inputs = []
        for raw_image in raw_images:
            image = detector.transform_gen.get_transform(raw_image).apply_image(raw_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs.append({"image": image, "height": raw_image.shape[0], "width": raw_image.shape[1]})
        images = detector.model.preprocess_image(inputs)
        
        # Run Backbone Res1-Res4
        features = detector.model.backbone(images.tensor)
        
        # Generate proposals with RPN
        proposals, _ = detector.model.proposal_generator(images, features, None)
        
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in detector.model.roi_heads.in_features]
        box_features = detector.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # (sum_proposals, 2048), pooled to 1x1
        
        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = detector.model.roi_heads.box_predictor(feature_pooled)
        rcnn_outputs = FastRCNNOutputs(
            detector.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            detector.model.roi_heads.smooth_l1_beta,
        )
        
        # Fixed-number NMS
        instances_list, ids_list = [], []
        probs_list = rcnn_outputs.predict_probs()
        boxes_list = rcnn_outputs.predict_boxes()

        # attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        # max_attr_prob, max_attr_label = attr_prob.max(-1)

        for probs, boxes, image_size in zip(probs_list, boxes_list, images.image_sizes):
            for nms_thresh in np.arange(0.3, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image_size, 
                    score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=MAX_BOXES
                )
                if len(ids) >= MIN_BOXES:
                    break
            instances_list.append(instances)
            ids_list.append(ids)
        
        # Post processing for features
        features_list = feature_pooled.split(rcnn_outputs.num_preds_per_image) # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
        roi_features_list = []
        for ids, features in zip(ids_list, features_list):
            roi_features_list.append(features[ids].detach())
        
        # Post processing for bounding boxes (rescale to raw_image)
        raw_instances_list = []
        for instances, input_per_image, image_size in zip(
                instances_list, inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                raw_instances = detector_postprocess(instances, height, width)
                raw_instances_list.append(raw_instances)
        
        return raw_instances_list, roi_features_list


def dump_features(writer, detector, pathXid):
    img_paths, img_ids = zip(*pathXid)
    imgs = [cv2.imread(img_path) for img_path in img_paths]
    instances_list, features_list = doit(detector, imgs)

    for img, img_id, instances, features in zip(imgs, img_ids, instances_list, features_list):
        instances = instances.to('cpu')
        features = features.to('cpu')

        num_objects = len(instances)

        item = {
            # "img_id": img_id,
            "objects_class": [MetadataCatalog.get("vg").thing_classes[i] for i in instances.pred_classes],
            "objects_id": base64.b64encode(instances.pred_classes.numpy()).decode(),  # int64
            "objects_conf": base64.b64encode(instances.scores.numpy()).decode(),  # float32
            # "attrs_id": base64.b64encode(np.zeros(num_objects, np.int64)).decode(),  # int64
            # "attrs_conf": base64.b64encode(np.zeros(num_objects, np.float32)).decode(),  # float32
            # "attrs_id": base64.b64encode(instances.attr_classes.numpy()).decode(),  # int64
            # "attrs_conf": base64.b64encode(instances.attr_scores.numpy()).decode(),  # float32
            # "num_boxes": num_objects,
            "boxes": base64.b64encode(instances.pred_boxes.tensor.numpy()).decode(),  # float32
            "features": base64.b64encode(features.numpy()).decode(),  # float32
            "img_h": img.shape[0],
            "img_w": img.shape[1]
            # "attrs_class": [MetadataCatalog.get("vg").attr_classes[i] for i in instances.attr_classes]
        }

        # json_item = json.dumps(item)

        content = {"img_id": img_id,
                    "json_content": item
                    }

        writer.writerow(content)
    

# FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
            #   "attrs_id", "attrs_conf", "num_boxes", "boxes", "features","objects_class"]
FIELDNAMES = ["img_id", "json_content"]
def extract_feat(outfile, detector, pathXid):
    # Check existing images in tsv file.
    wanted_ids = set([image_id[1] for image_id in pathXid])
    found_ids = set()
    if os.path.exists(outfile):
        with open(outfile, 'r') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                found_ids.add(item['img_id'])
    missing = wanted_ids - found_ids
    
    # Extract features for missing images.
    missing_pathXid = list(filter(lambda x:x[1] in missing, pathXid))
    with open(outfile, 'a') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
        for start in tqdm.tqdm(range(0, len(pathXid), args.batchsize)):
            pathXid_trunk = pathXid[start: start + args.batchsize]
            dump_features(writer, detector, pathXid_trunk)
            """
            try:
                dump_features(writer, detector, pathXid_trunk)
            except Exception as e:
                print(e)
                break
            """


def load_image_ids(img_root, img_set='vg_nococo'):
    """images in the same directory are in the same split"""
    fname = os.path.join(LXRT_ROOT, imgset2fname[img_set])
    pathXid = []
    # img_root = os.path.join(img_root, split_dir)

    with open(fname) as f:
        data = json.load(f)
        for datum in data:
            gt_img_name = datum['img_id'] + '.jpg'
            
            pathXid.append(
                (
                    os.path.join(img_root, gt_img_name),
                    datum['img_id']))
    print("Load the %d (img, sent) pairs for image set %s from %s" % (
        len(pathXid), img_set, fname))

    return pathXid

# load from the datadir file, which is faster
def load_open_images_ids(imageDir_file):
    # imageDir = '/data/data_for_vokenization/open_images/train'
    pathXid = []

    with open(imageDir_file,'r') as fi:
        for i in fi:
            image_path = i.strip()
            image_id = os.path.splitext(os.path.basename(image_path))[0]

            pathXid.append(
                (
                    image_path,
                    image_id
            ))
    print("Load  %d imgs from %s " % (
        len(pathXid), imageDir_file))

    return pathXid

def load_open_images_ids_fromVokens(imageDir_file):
    # imageDir = '/data/data_for_vokenization/open_images/train'
    pathXid = []

    top1_images_path = [line.decode().rstrip('\n').split('\t')[0] for line in open(imageDir_file,'rb')]

    top1_images_path = list(set(top1_images_path)) # no repeate

    for top1 in top1_images_path:
        image_path = top1
        image_path = os.path.join(OI_ROOT, image_path)
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        pathXid.append(
            (
                image_path,
                image_id
        ))
    print("Load  %d imgs from %s " % (
        len(pathXid), imageDir_file))

    return pathXid

# load from the system datadir, which is slow
# def load_open_images_ids(imageDir):
#     # imageDir = '/data/data_for_vokenization/open_images/train'
#     pathXid = []

#     for dirpath, dirnames, filenames in os.walk(imageDir):
#         for filename in filenames:
#             gt_img_name = filename
#             image_id = os.path.splitext(filename)[0]
#             pathXid.append(
#                 (
#                     os.path.join(imageDir, gt_img_name),
#                     image_id
#             ))
#     print("Load  %d imgs from %s " % (
#         len(pathXid), imageDir))

#     return pathXid

def build_model():
    # Build model and load weights.
    if args.weight == 'mask':
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(
            D2_ROOT, "configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        print("Load the Mask RCNN weight for ResNet101, pretrained on MS COCO segmentation. ")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl"
    elif args.weight == 'obj':
        print("Load the Faster RCNN weight for ResNet101, pretrained on MS COCO detection.")
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(
            D2_ROOT, "configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl"
    elif args.weight == 'vg':
        cfg = get_cfg() # Renew the cfg file
        cfg.merge_from_file(os.path.join(
            D2_ROOT, "configs/VG-Detection/faster_rcnn_R_101_C4_caffemaxpool.yaml"))
            # D2_ROOT, "configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")) # w/ attr
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        cfg.INPUT.MIN_SIZE_TEST = 600
        cfg.INPUT.MAX_SIZE_TEST = 1000
        cfg.MODEL.RPN.NMS_THRESH = 0.7
        # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
        cfg.MODEL.WEIGHTS = "/home/.torch/fvcore_cache/models/faster_rcnn_from_caffe.pkl"
        # cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
        # cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl" # w/ attr
    else:
        assert False, "no this weight"
    detector = DefaultPredictor(cfg)
    return detector

if __name__ == "__main__":
    # pathXid = load_image_ids(DATA_ROOT, args.img_set)     # Get paths and ids
    # pathXid = load_open_images_ids(OI_PATH_FILE)     # Get paths and ids
    pathXid = load_open_images_ids_fromVokens(VOKEN_ROOT)     # Get paths and ids
    
    print(pathXid[:5])

    #tmp for debug
    # pathXid = pathXid[:5]
    
    detector = build_model()
    # extract_feat('/data/data_for_Oscar/vgnococo_reddit/%s_d2obj36_batch_noAttr.tsv' % args.img_set, detector, pathXid)
    # extract_feat('/data/%s_forRead_d2obj36_batch_noAttr.tsv' % args.img_set, detector, pathXid)
    extract_feat('/data/%s_%s_d2obj36_batch_noAttr.tsv' % (args.img_set, args.data_split), detector, pathXid)
