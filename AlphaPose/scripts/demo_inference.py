"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time


import numpy as np
import torch
from tqdm import tqdm
import natsort

#path_to_file = os.path.join(os.getcwd(), 'AlphaPose')
#print(f'Path to File: {path_to_file}')
ARGS_CONTAINER = {
    #'cfg': '',
    #'checkpoint': '',
    'cfg': 'configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml',
    'checkpoint': 'pretrained_models/halpe26_fast_res50_256x192.pth',
    'det_dir': '',
    'det_view': '',
    'sp': False,
    'detector': 'yolox',
    'detfile': '',
    'indir': '',
    'inputpath': '',
    'outdir': '',
    'outputpath': '',
    'inputlist': '',
    'list': '',
    'inputimg': '',
    'save_img': False, #
    'showbox': False, #
    'vis': False,
    'save_video': False,
    'vis_fast': False,
    'profile': False,
    'format': '',
    'min_box_area': 0,
    'detbatch': 5,
    'posebatch': 64,
    'eval': False,
    'gpus': '0',
    'device': 'gpu',
    'qsize': 924,
    'flip': False,
    'debug': False,
    'pose_track': True,
    'pose_flow': False
}

ARGS_CONTAINER['gpus'] = [int(i) for i in ARGS_CONTAINER['gpus'].split(',')] if torch.cuda.device_count() >= 1 else [-1]
ARGS_CONTAINER['device'] = torch.device("cuda:" + str(ARGS_CONTAINER['gpus'][0]) if ARGS_CONTAINER['gpus'][0] >= 0 else "cpu")
ARGS_CONTAINER['detbatch'] = ARGS_CONTAINER['detbatch'] * len(ARGS_CONTAINER['gpus'])
ARGS_CONTAINER['posebatch'] = ARGS_CONTAINER['posebatch'] * len(ARGS_CONTAINER['gpus'])
ARGS_CONTAINER['tracking'] = ARGS_CONTAINER['pose_track']

if platform.system() == 'Windows':
    ARGS_CONTAINER["sp"] = True


#current_script_dir = os.path.join(abs_path, 'AlphaPose')
#ARGS_CONTAINER['cfg'] = os.path.normpath(
#    os.path.join(current_script_dir, 'configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'))
#ARGS_CONTAINER['checkpoint'] = os.path.normpath(
#    os.path.join(current_script_dir, 'pretrained_models/halpe26_fast_res50_256x192.pth'))

#print(ARGS_CONTAINER['cfg'])
#print(ARGS_CONTAINER['checkpoint'])

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter


cfg = update_config(ARGS_CONTAINER['cfg'])


def check_input():
    # for detection results
    if len(ARGS_CONTAINER['detfile']):
        if os.path.isfile(ARGS_CONTAINER['detfile']):
            detfile = ARGS_CONTAINER['detfile']
            return 'detfile', detfile
        else:
            raise IOError('Error: --detfile must refer to a detection json file, not directory.')

    # for images
    if len(ARGS_CONTAINER["inputpath"]) or len(ARGS_CONTAINER["inputlist"]) or len(ARGS_CONTAINER["inputimg"]):
        inputpath = ARGS_CONTAINER["inputpath"]
        inputlist = ARGS_CONTAINER["inputlist"]
        inputimg = ARGS_CONTAINER["inputimg"]

        if len(inputlist):
            im_names = open(inputlist, 'r').readlines()
        elif len(inputpath) and inputpath != '/':
            for root, dirs, files in os.walk(inputpath):
                if files:
                    im_names = files
            im_names = natsort.natsorted(im_names)
        elif len(inputimg):
            ARGS_CONTAINER[inputpath] = os.path.split(inputimg)[0]
            im_names = [os.path.split(inputimg)[1]]
        return 'image', im_names

    else:
        raise NotImplementedError


def print_finish_info():
    print('===========================> Finish Model Running.')
    if (ARGS_CONTAINER["save_img"] or ARGS_CONTAINER["save_video"]) and not ARGS_CONTAINER["vis_fast"]:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


def loop():
    n = 0
    while True:
        yield n
        n += 1


def main_process_inference(in_file, out_file, viz_out_img):
#if __name__ == "__main__":
    ARGS_CONTAINER['indir'] = in_file
    ARGS_CONTAINER['inputpath'] = ARGS_CONTAINER['indir']
    ARGS_CONTAINER['outdir'] = out_file
    ARGS_CONTAINER['outputpath'] = ARGS_CONTAINER['outdir']
    ARGS_CONTAINER['save_img'] = viz_out_img
    ARGS_CONTAINER['showbox'] = viz_out_img


    ARGS_CONTAINER["det_dir"] = os.path.basename(os.path.normpath(in_file))

    mode, input_source = check_input()

    # Load detection loader
    det_loader = DetectionLoader(input_source, get_detector(ARGS_CONTAINER), cfg, ARGS_CONTAINER, batchSize=ARGS_CONTAINER["detbatch"], mode=mode, queueSize=ARGS_CONTAINER["qsize"])
    det_worker = det_loader.start()

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (ARGS_CONTAINER["checkpoint"],))
    pose_model.load_state_dict(torch.load(ARGS_CONTAINER["checkpoint"], map_location=ARGS_CONTAINER["device"]))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    if ARGS_CONTAINER["pose_track"]:
        #print('Initializing Tracker')
        tracker = Tracker(tcfg, ARGS_CONTAINER)
    if len(ARGS_CONTAINER["gpus"]) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=ARGS_CONTAINER["gpus"]).to(ARGS_CONTAINER["device"])
    else:
        pose_model.to(ARGS_CONTAINER["device"])
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    queueSize = 2 if mode == 'webcam' else ARGS_CONTAINER["qsize"]
    if ARGS_CONTAINER["save_video"] and mode != 'image':
        from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
        if mode == 'video':
            video_save_opt['savepath'] = os.path.join(ARGS_CONTAINER["outdir"], 'AlphaPose_' + os.path.basename(input_source))
        else:
            video_save_opt['savepath'] = os.path.join(ARGS_CONTAINER["outdir"], 'AlphaPose_webcam' + str(input_source) + '.mp4')
        video_save_opt.update(det_loader.videoinfo)
        writer = DataWriter(cfg, ARGS_CONTAINER, save_video=True, video_save_opt=video_save_opt, queueSize=queueSize).start()
    else:
        writer = DataWriter(cfg, ARGS_CONTAINER, save_video=False, queueSize=queueSize).start()

    if mode == 'webcam':
        print('Starting webcam demo, press Ctrl + C to terminate...')
        sys.stdout.flush()
        im_names_desc = tqdm(loop())
    else:
        data_len = det_loader.length
        print(f"-- Processing a total of {data_len} frames --")
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = ARGS_CONTAINER["posebatch"]
    if ARGS_CONTAINER["flip"]:
        batchSize = int(batchSize / 2)
    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                if ARGS_CONTAINER["profile"]:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                # Pose Estimation
                inps = inps.to(ARGS_CONTAINER["device"])
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    if ARGS_CONTAINER["flip"]:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = pose_model(inps_j)
                    if ARGS_CONTAINER["flip"]:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)
                if ARGS_CONTAINER["profile"]:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
                if ARGS_CONTAINER["pose_track"]:
                    boxes,scores,ids,hm,cropped_boxes = track(tracker,ARGS_CONTAINER,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
                    #print('Len: boxes,scores,ids ', len(boxes), len(scores), len(ids),)
                    #print('Boxes ', boxes )
                    #print('scores ', scores )
                    #print('ids ', ids )
                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                if ARGS_CONTAINER["profile"]:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if ARGS_CONTAINER["profile"]:
                # TQDM
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
        print_finish_info()
        while(writer.running()):
            time.sleep(1)
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
        writer.stop()
        det_loader.stop()
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        print_finish_info()
        # Thread won't be killed when press Ctrl+C
        if ARGS_CONTAINER["sp"]:
            det_loader.terminate()
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            det_loader.terminate()
            writer.terminate()
            writer.clear_queues()
            det_loader.clear_queues()

