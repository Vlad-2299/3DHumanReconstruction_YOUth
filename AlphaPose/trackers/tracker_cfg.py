from easydict import EasyDict as edict
cfg = edict()
cfg.nid = 2
cfg.arch = "osnet_ain" # "osnet" or "res50-fc512"
cfg.loadmodel = "trackers/weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
cfg.frame_rate =  999
cfg.track_buffer = 999
cfg.conf_thres = 0.6
cfg.nms_thres = 0.5
cfg.iou_thres = 0.6
