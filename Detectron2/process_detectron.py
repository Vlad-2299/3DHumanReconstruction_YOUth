#!pip install opencv-python
#!pip install opencv-contrib-python-headless
#!pip install gcc

#dist = distutils.core.run_setup("./setup.py")
#!pip install {' '.join([f"'{x}'" for x in dist.install_requires])}

import sys, os, distutils.core
import torch, detectron2
import torch
# Some basic setup:
# Setup Detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

# import some common libraries
import numpy as np
import time
import tqdm
import os, json, random
import cv2
import copy
import math

# import some common Detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultTrainer
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageChops
import math
import copy


def calc_point_overlap(key_a, key_b):
    #If value close to 0, then keypoints belong to the same person
    assert len(key_a) == len(key_b)
    mean = 0
    for k_a, k_b in zip(key_a, key_b):
        mean += abs(k_a - k_b)
    return mean/len(key_a)


def calc_box_overlap(box_detectron_, box_list_alphapose_):
    '''
    NOTE
    box_detectron: box values are exact pixel coordinates
    box_alphapose: down right corner (idx 2 & 3) are the exact pixel value minus the top left values (idx 0 & 1)
    '''
    box_detectron = copy.deepcopy(box_detectron_)
    box_list_alphapose = copy.deepcopy(box_list_alphapose_)
    #Set box_detectron values to the same format as box_list_alphapose
    box_detectron[2] = abs(box_detectron[2] - box_detectron[0])
    box_detectron[3] = abs(box_detectron[3] - box_detectron[1])

    overlap_a = calc_point_overlap(box_detectron, box_list_alphapose[0])
    overlap_b = calc_point_overlap(box_detectron, box_list_alphapose[1])

    idx = 0 if overlap_a < overlap_b else 1
    best_overlap = overlap_a if overlap_a < overlap_b else overlap_b

    return idx, best_overlap


def get_best_box_overlap(view_det_, view_alpha_):
    #Set box_detectron values to the same format as box_list_alphapose
    view_det_box_copy = copy.deepcopy(view_det_)
    view_alpha_box_copy = copy.deepcopy(view_alpha_)
    first_idx_overlap_vals = []
    second_idx_overlap_vals = []
    for frame_det_box, frame_alpha_box in zip(view_det_box_copy, view_alpha_box_copy):
        if len(frame_det_box) in range(1,3) and len(frame_alpha_box) == 2:
            
            for i in range(len(frame_det_box)):
                frame_det_box[i][2] = abs(frame_det_box[i][2] - frame_det_box[i][0])
                frame_det_box[i][3] = abs(frame_det_box[i][3] - frame_det_box[i][1])

                if len(frame_det_box) == 2:
                    overlap = calc_point_overlap(frame_det_box[i], frame_alpha_box[i])
                    if i == 0:
                        first_idx_overlap_vals.append(overlap)
                    else:
                        second_idx_overlap_vals.append(overlap)

                elif len(frame_det_box) == 1:
                    overlap_a = calc_point_overlap(frame_det_box[i], frame_alpha_box[0])
                    overlap_b = calc_point_overlap(frame_det_box[i], frame_alpha_box[1])

                    idx = 0 if overlap_a < overlap_b else 1
                    best_overlap = overlap_a if overlap_a < overlap_b else overlap_b

                    if idx == 0:
                        first_idx_overlap_vals.append(best_overlap)
                        second_idx_overlap_vals.append(sys.maxsize)
                    else:
                        first_idx_overlap_vals.append(sys.maxsize)
                        second_idx_overlap_vals.append(best_overlap)
        else:
            first_idx_overlap_vals.append(sys.maxsize)
            second_idx_overlap_vals.append(sys.maxsize)

    return first_idx_overlap_vals, second_idx_overlap_vals


def get_list_best_idx_box_overlap(detectron_box_list, alpha_box_list):
    best_idx_cmp_hist = []
    for view_det, view_alpha in zip(detectron_box_list, alpha_box_list):
        best_idx_frame = [None, None]
        f, s = get_best_box_overlap(view_det, view_alpha)
        best_idx_frame[0] = (np.argmin(f))
        best_idx_frame[1] = (np.argmin(s))
        best_idx_cmp_hist.append(best_idx_frame)
    return best_idx_cmp_hist


def get_det_best_idx(det_boxes, alpha_boxes):
    '''
    det_boxes: List of detectron2 instance boxes
    alpha_boxes: pair of alphapose boxes

    RETURN; detectron_alpha_map_idx: index number of best detections in det_boxes
    '''
    detectron_alpha_map = {
    0: sys.maxsize,
    1: sys.maxsize
    }

    detectron_alpha_map_idx = [None, None]

    for curr_idx, det_b in enumerate(det_boxes):
        best_idx, best_overlap = calc_box_overlap(det_b, alpha_boxes)
        #print(best_idx, best_overlap)
        if detectron_alpha_map[best_idx] > best_overlap:
            detectron_alpha_map[best_idx] = best_overlap
            detectron_alpha_map_idx[best_idx] = curr_idx
    #print(detectron_alpha_map_idx)
    return detectron_alpha_map_idx


def calc_pair_box_overlap(box_list_detectron_, box_list_alphapose_):
    '''
    NOTE
    box_detectron: box values are exact pixel coordinates
    box_alphapose: down right corner (idx 2 & 3) are the exact pixel value minus the top left values (idx 0 & 1)
    '''
    assert len(box_list_detectron_) == len(box_list_alphapose_) == 2, f'This method takes the detection boxes of both persons in a single frame!' \
                                                               f'Given detections are of length {len(box_list_detectron_)} from detectron and {len(box_list_alphapose_)} from AlphaPose'
    detectron_alpha_index = [-1, -1]
    detectron_alpha_vals = [-1, -1]

    box_list_detectron = copy.deepcopy(box_list_detectron_)
    box_list_alphapose = copy.deepcopy(box_list_alphapose_)


    for idx in range(len(box_list_detectron)):
        # Set box_detectron values to the same format as box_list_alphapose
        box_list_detectron[idx][2] = abs(box_list_detectron[idx][2] - box_list_detectron[idx][0])
        box_list_detectron[idx][3] = abs(box_list_detectron[idx][3] - box_list_detectron[idx][1])

        overlap_a = calc_point_overlap(box_list_detectron[idx], box_list_alphapose[0])
        overlap_b = calc_point_overlap(box_list_detectron[idx], box_list_alphapose[1])

        detectron_alpha_index[idx] = 0 if overlap_a < overlap_b else 1
        detectron_alpha_vals[idx] = overlap_a if overlap_a < overlap_b else overlap_b

    if len(detectron_alpha_index) != len(set(detectron_alpha_index)):
        print(detectron_alpha_vals)

    return detectron_alpha_index


def get_det_masks(out_inst, in_img, alpha_box):
    #assert len(out_inst['instances']) >= 2, f'Frame contains less than 2 detections!'

    colored_mask_list = [] * 2
    det_box_list = []
    assert len(alpha_box) == 2, f'Given AlphaPose instances should have 2 detections!! {len(alpha_box)} were given!'

    search_idx = get_det_best_idx(copy.deepcopy(out_inst['instances'].pred_boxes.tensor.cpu()), alpha_box)
    
    for i in search_idx:  # if there are more detections, we get the 2 with highest score
        if i != None:
            mask = copy.deepcopy(out_inst['instances'].pred_masks.cpu().numpy()[i])
            box = copy.deepcopy(out_inst['instances'].pred_boxes.tensor.cpu().numpy()[i])

            mask_h = int(math.ceil(box[3] - box[1]))
            mask_w = int(math.ceil(box[2] - box[0]))

            temp_mask = np.zeros((mask_h, mask_w))
            for h_id in range(int(box[1]), int(box[3])):
                for w_id in range(int(box[0]), int(box[2])):
                    temp_mask[h_id - int(box[1])][w_id - int(box[0])] = mask[h_id][w_id]

            temp_mask_fill = np.zeros((mask_h, mask_w, 3))

            for h_id, h_bw in enumerate(temp_mask):
                for w_id, w_bw in enumerate(h_bw):
                    if w_bw == 0:
                        temp_mask_fill[h_id][w_id] = [0, 0, 0]
                    else:
                        temp_mask_fill[h_id, w_id] = in_img[int(h_id + box[1]), int(w_id + box[0])] / 255
            #
            det_box_list.append(copy.deepcopy(out_inst['instances'].pred_boxes.tensor.cpu().numpy()[i]))
            colored_mask_list.append(temp_mask_fill)

    return colored_mask_list, det_box_list


def get_info():
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("Detectron2:", detectron2.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))


def get_predictor():
    # Initialization of Detectron2
    cfg = get_cfg()
    cfg.merge_from_file('configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    # cfg.DATASETS.TRAIN = ("my_dataset_coco_person_640", )
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = 'Detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 350
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Uncoment if you want to fine-tune the model on the selected cfg.DATASETS.TRAIN data
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the fine-tunned model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    return predictor, cfg


def compare_hist(hist1, hist2):
    # 1: *Chi-square* -> 0 means same colors
    return cv2.compareHist(hist1, hist2, 1)


def get_view_image_list(path):
    img_list = []
    for png in os.listdir(path):
        if png.endswith('.png'):
            img = cv2.imread(os.path.join(path, png))
            img_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    return img_list


def run_detect_on_img_list(predictor, view_path):
    # for img in img_list:
    detection_view_list = []

    for img in view_path:
        detection_view_list.append(predictor(img))

    return detection_view_list


def run_detect_and_viz_on_img_list(predictor, cfg, view_path):
    # for img in img_list:
    detection_view_list = []
    visualization_view_list = []

    for img in view_path:
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        visualization = v.draw_instance_predictions(outputs["instances"].to('cpu'))
        visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
        detection_view_list.append(outputs)
        visualization_view_list.append(visualization)

    return detection_view_list, visualization_view_list


def calc_color_hist(image):
    image = image.astype(np.float32)
    image = image * 255

    #image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    chans = cv2.split(image)
    #h,s,v = cv2.split(image)
    #mask = (v == 0) + (v == 100) + (s == 0)
    #mask = np.logical_not(mask)


    #colors = ('r', 'g', 'b')

    #plt.figure()
    #mask = np.zeros(image.shape[:2]).astype(np.float32)

    #print(np.shape(mask))

    hist = cv2. calcHist(chans, [0, 0, 0], None, [16, 16, 16], [1, 256, 1, 256, 1, 256])
    hist_norm = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist_norm


def compare_pair_hist_views(comp_with_pair_hist, comp_to_pair_hist, comp_with_frame_id, comp_to_frame_id):
    '''
    comp_with_pair_hist: Histogram Pair -> will always be view 0
    comp_to_pair_hist: Histogram pair of another view
    comp_to_frame_id: Best frames to compare to 
    comp_with_frame_id: Best frames to compare with 

    Retrun: Ture if idx need to be swapped
    '''
    n_people = 2
    n_tries = 2 #Repeat comparing with diff idx

    #If idx 0 smaller than idx 1, then we know that the indexes are correct
    idx_chi_val = [0] * n_people

    for iter in range(n_tries):
        chi_comul = 0
        for pers_idx in range(n_people):
            if iter == 0:
                cmp_to = pers_idx 
            else:
                cmp_to = 1 if pers_idx == 0 else 0
            chi = compare_hist(comp_with_pair_hist[comp_with_frame_id[pers_idx]][pers_idx], comp_to_pair_hist[comp_to_frame_id[cmp_to]][cmp_to])
            chi_comul += chi
        idx_chi_val[iter] = chi_comul
    return False if idx_chi_val[0] < idx_chi_val[1] else True





