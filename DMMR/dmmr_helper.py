import yaml
import torch
import json
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
import pandas as pd
from DMMR.core.utils.module_utils import load_camera_para
import os
import copy
import shutil
import numpy as np

halpe_skeleton = { 0:  "Nose", 1:  "LEye", 2:  "REye", 3:  "LEar", 4:  "REar", 5:  "LShlder", 6:  "RShlder", 7:  "LElbow", 8:  "RElbow", 9:  "LWrist",
    10: "RWrist", 11: "LHip", 12: "RHip", 13: "LKnee", 14: "RKnee", 15: "LAnkle", 16: "RAnkle", 17:  "Head", 18:  "Neck", 19:  "Hip",
    20: "LBToe", 21: "RBToe", 22: "LSToe", 23: "RSToe", 24: "LHeel", 25: "RHeel"
}


def get_all_view_info(padded_drop_info, final):
    initial = 0
    all_views_batches = []
    prev_range = [0, 0]

    for item_idx, curr_item in enumerate(padded_drop_info):
            [[curr_key, curr_range]] = curr_item.items()

            if prev_range[1] + 1 < curr_range[0]:
                    if initial < curr_range[0]:
                            all_view_range = [initial, curr_range[0]]
                            initial = curr_range[1] + 1
                            all_views_batches.append(all_view_range)
                    if item_idx + 1 == len(padded_drop_info) and curr_range[1] != final:
                            all_view_range = [curr_range[1], final]
                            all_views_batches.append(all_view_range)
            else:
                    initial = curr_range[1] + 1

            prev_range = curr_range
    return all_views_batches

def save_json(folder, file, temp):
    with open(f'{folder}/{file}_keypoints.json', 'w') as f:
        json.dump(temp, f)
    f.close()


def split_json_frames(keypoints_view, to_folder):

    #to_folder = create_folder(path_json)

    keypoint_string = '''
    {
        "version":1.1,
        "people":[
        {"pose_keypoints_2d":[]
        },
        {"pose_keypoints_2d":[]
        }
        ]
    }
    '''

    for frame in keypoints_view:
        json_string = json.loads(keypoint_string)

        for i_d, det in enumerate(frame):
            frame_name = det['image_id'].split('.png')[0]
            json_string['people'][i_d]['pose_keypoints_2d'] = copy.deepcopy(det['keypoints'])
        save_json(to_folder, frame_name, json_string)





def split_json_frames_ablation(keypoints_view, to_folder):
    #Abalation Study!
    #to_folder = create_folder(path_json)

    keypoint_string = '''
    {
        "version":1.1,
        "people":[
        {"pose_keypoints_2d":[]
        }
        ]
    }
    '''

    for frame in keypoints_view:
        json_string = json.loads(keypoint_string)

        pers_id = 1
        #for i_d, det in enumerate(frame):
        frame_name = frame[pers_id]['image_id'].split('.png')[0]
        json_string['people'][0]['pose_keypoints_2d'] = copy.deepcopy(frame[pers_id]['keypoints'])
        save_json(to_folder, frame_name, json_string)



def create_populate_dmmr_data(DMMR_PATH, DMMR_CAMPARAMS_PATH, DMMR_DATA_PATH, DMMR_KEYPOINTS_PATH, DMMR_IMAGES_PATH,
                               ap_data_view_subdir, correct_keypoint, padded_drop_info, video_name, dmmr_config):
    dmmr_camparams_file = os.path.join(DMMR_CAMPARAMS_PATH, 'camparams.txt')
    data_folders = ['camparams', 'images', 'keypoints']
    data_folders_path = []
    camera_list = [0, 1, 2, 3]
    extrinsic_params, intrinsic_params = load_camera_para(dmmr_camparams_file)

    #Delete everything inside data_folders!
    for folder in data_folders:
        folder_path = os.path.join(DMMR_DATA_PATH, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            os.mkdir(folder_path)
            data_folders_path.append(folder_path)

    frame_path_list = [] #[view][frame_path]
    for view_idx, view_img_path in enumerate(ap_data_view_subdir):
        n_frames = 0
        view_frame_path = []
        for img_batch in view_img_path: ################
            batch_frames = os.listdir(img_batch)
            for frame in batch_frames:
                n_frames += 1
                frame_path = os.path.join(img_batch, frame)
                view_frame_path.append(frame_path)
        frame_path_list.append(view_frame_path) 

    all_views_batches = get_all_view_info(padded_drop_info, n_frames)

    if all_views_batches:

        for data_folder_path in data_folders:
            curr_initial_frame = 0


            for item_idx, curr_range in enumerate(all_views_batches):
                data_video_batch = os.path.join(DMMR_DATA_PATH, data_folder_path, f'{video_name}_{curr_range[0]+1}-{curr_range[1]}_All')
                os.mkdir(data_video_batch)
                
                if data_folder_path == 'camparams':
                    #f = open(os.path.join(data_video_batches, "camparams.txt"), "w")

                    with open(os.path.join(data_video_batch, "camparams.txt"), "w") as params_file:
                        for n_cam in camera_list:
                            try:
                                params_file.write(f'{str(n_cam)}\n')
                                for cam_inrins in intrinsic_params[n_cam]:
                                    for intris_value in cam_inrins:
                                        params_file.write(f'{str(intris_value)} ')
                                    params_file.write(f'\n')
                                params_file.write('0 0 \n' )
                                for cam_extrins in extrinsic_params[n_cam][:-1]:
                                    for extrins_values in cam_extrins:
                                        params_file.write(f'{str(extrins_values)} ')
                                    params_file.write(f'\n')
                                params_file.write(f'\n')
                                
                            except:
                                print('unable to write')
                                params_file.close()
                    params_file.close()

                #[{2: [10, 20]}, {0: [26, 37]}, {0: [38, 40]}, {1: [45, 50]}]
                elif data_folder_path == 'images':
                    for n_cam in camera_list:
                        camera_batch = os.path.join(data_video_batch, 'Camera'+str(n_cam).zfill(2))
                        os.mkdir(camera_batch)
                        for img_batch_after in frame_path_list[n_cam][curr_range[0]:curr_range[1]]:
                            shutil.copy(img_batch_after, camera_batch)

                elif data_folder_path == 'keypoints':
                    for n_cam in camera_list:
                        camera_batch = os.path.join(data_video_batch, 'Camera'+str(n_cam).zfill(2))
                        os.mkdir(camera_batch)
                        split_json_frames(correct_keypoint[n_cam][curr_range[0]:curr_range[1]], camera_batch)

        #################################################################################
            for item_idx, curr_item in enumerate(padded_drop_info):
                [[curr_key, curr_range]] = curr_item.items()
                data_video_batch = os.path.join(DMMR_DATA_PATH, data_folder_path, f'{video_name}_{curr_range[0]+1}-{curr_range[1]}_N{curr_key}')
                os.mkdir(data_video_batch)

                not_ignored_cams = copy.copy(camera_list)
                not_ignored_cams.remove(curr_key)
                
                if data_folder_path == 'camparams':
                    #f = open(os.path.join(data_video_batches, "camparams.txt"), "w")

                    with open(os.path.join(data_video_batch, "camparams.txt"), "w") as params_file:
                        for n_cam in not_ignored_cams:
                            try:
                                params_file.write(f'{str(n_cam)}\n')
                                for cam_inrins in intrinsic_params[n_cam]:
                                    for intris_value in cam_inrins:
                                        params_file.write(f'{str(intris_value)} ')
                                    params_file.write(f'\n')
                                params_file.write('0 0 \n' )
                                for cam_extrins in extrinsic_params[n_cam][:-1]:
                                    for extrins_values in cam_extrins:
                                        params_file.write(f'{str(extrins_values)} ')
                                    params_file.write(f'\n')
                                params_file.write(f'\n')
                                
                            except:
                                print('unable to write')
                                params_file.close()
                    params_file.close()

                #[{2: [10, 20]}, {0: [26, 37]}, {0: [38, 40]}, {1: [45, 50]}]
                elif data_folder_path == 'images':
                    for n_cam in not_ignored_cams:
                        camera_batch = os.path.join(data_video_batch, 'Camera'+str(n_cam).zfill(2))
                        os.mkdir(camera_batch)
                        for img_batch_after in frame_path_list[n_cam][curr_range[0]:curr_range[1]]:
                            shutil.copy(img_batch_after, camera_batch)

                elif data_folder_path == 'keypoints':
                    for n_cam in not_ignored_cams:
                        camera_batch = os.path.join(data_video_batch, 'Camera'+str(n_cam).zfill(2))
                        os.mkdir(camera_batch)
                        split_json_frames(correct_keypoint[n_cam][curr_range[0]:curr_range[1]], camera_batch)
    else:
        #Create camparams.txt 
        #Create DMMR camparams folder
        dmmr_camparam_video_path = os.path.join(DMMR_DATA_PATH, data_folders[0], video_name)
        if not os.path.exists(dmmr_camparam_video_path):
            os.mkdir(dmmr_camparam_video_path)
        else:
            shutil.rmtree(dmmr_camparam_video_path)
            os.mkdir(dmmr_camparam_video_path)
        with open(os.path.join(dmmr_camparam_video_path, "camparams.txt"), "w") as params_file:
            for n_cam in camera_list:
                try:
                    params_file.write(f'{str(n_cam)}\n')
                    for cam_inrins in intrinsic_params[n_cam]:
                        for intris_value in cam_inrins:
                            params_file.write(f'{str(intris_value)} ')
                        params_file.write(f'\n')
                    params_file.write('0 0 \n' )
                    for cam_extrins in extrinsic_params[n_cam][:-1]:
                        for extrins_values in cam_extrins:
                            params_file.write(f'{str(extrins_values)} ')
                        params_file.write(f'\n')
                    params_file.write(f'\n')
                    
                except:
                    print('unable to write')
                    params_file.close()
        params_file.close()



        #Create DMMR keypoint folders
        dmmr_key_video_path = os.path.join(DMMR_KEYPOINTS_PATH, video_name)
        if not os.path.exists(dmmr_key_video_path):
            os.mkdir(dmmr_key_video_path)
        else:
            shutil.rmtree(dmmr_key_video_path)
            os.mkdir(dmmr_key_video_path)

        for view_idx, view_keypoints in enumerate(correct_keypoint):
            cam_path = os.path.join(dmmr_key_video_path, 'Camera'+str(view_idx).zfill(2))
            if not os.path.exists(cam_path):
                os.mkdir(cam_path)
            else:
                shutil.rmtree(cam_path)
                os.mkdir(cam_path)
            split_json_frames(view_keypoints, cam_path)
            #split_json_frames_single(view_keypoints, cam_path)
        
        #Create DMMR image folders
        dmmr_img_video_path = os.path.join(DMMR_IMAGES_PATH, video_name)
        if not os.path.exists(dmmr_img_video_path):
            os.mkdir(dmmr_img_video_path)
        else:
            shutil.rmtree(dmmr_img_video_path)
            os.mkdir(dmmr_img_video_path)

        for view_idx, view_img_path in enumerate(ap_data_view_subdir):
            n_frames = 0
            cam_path = os.path.join(dmmr_img_video_path, 'Camera'+str(view_idx).zfill(2))
            if not os.path.exists(cam_path):
                os.mkdir(cam_path)
            else:
                shutil.rmtree(cam_path)
                os.mkdir(cam_path)
                
            for img_batch in view_img_path:
                batch_frames = os.listdir(img_batch)
                for frame in batch_frames:
                    n_frames += 1
                    shutil.copy(os.path.join(img_batch, frame), cam_path)


    #Update DMMR config file
    with open(os.path.join(DMMR_PATH, 'cfg_files', 'fit_smpl.yaml'), 'r') as file:
        try:
            data = yaml.safe_load(file)
            data['frames'] = n_frames
            data['num_people'] = dmmr_config["num_people"]
            data['opt_cam'] = dmmr_config["opt_cam"]
            data['save_images'] = dmmr_config["save_images"]
            data['visualize'] = dmmr_config["visualize"]
            data['scale_child'] = dmmr_config["scale_child"]
        except yaml.YAMLError as exception:
            print(exception)
    file.close()
    with open(os.path.join(DMMR_PATH, 'cfg_files', 'fit_smpl.yaml'), 'w') as f:
            yaml.dump(data, f)
    f.close()



def remove_noisy_frames(DMMR_IMAGES_PATH, correct_keypoint, initial_frame_ignore, last_frame_ignore):
    _, folder, _ = next(os.walk(DMMR_IMAGES_PATH))
    vid_folder_path = os.path.join(DMMR_IMAGES_PATH, folder[0])
    _, cam_folder, _ = next(os.walk(vid_folder_path))

    for v_id, cam in enumerate(cam_folder):
        cam_file = os.path.join(vid_folder_path, cam)
        if initial_frame_ignore > 0:
            _, _, frames = next(os.walk(cam_file))
            if len(frames) != len(correct_keypoint[v_id]):
                for frame in frames[:initial_frame_ignore]:
                    os.remove(os.path.join(cam_file, frame))

        if last_frame_ignore < -1:
            _, _, frames = next(os.walk(cam_file))
            if len(frames) != len(correct_keypoint[v_id]):
                for frame in frames[last_frame_ignore:]:
                    os.remove(os.path.join(cam_file, frame))



def plot_pair_run_reproj(float_list1, float_list2):
    x = np.arange(len(float_list1))
    y1 = np.array(float_list1)
    y2 = np.array(float_list2)

    plt.figure(figsize=(14, 4))
    plt.plot(x, y1, label='Phase One of Camera Calibration & 3D Human Reconstruction', alpha=0.7)
    plt.plot(x, y2, label='Phase Two of Camera Calibration & 3D Human Reconstruction', alpha=0.7)
    plt.xlabel('Frames')
    plt.ylabel('Re-Projection Error')
    plt.title('B33718: Re-Projection Error Differences Over Different Reconstruction Phases')
    plt.legend()
    plt.show()


def get_view_comul_conf_err_per_key(out_parent, out_child):
    comul_reproject_child = [] 
    comul_reproject_parent = [] 
    comul_conf_child = [] 
    comul_conf_parent = []

    for view_idx, (view_det_parent, view_det_culd) in enumerate(zip(out_parent, out_child)):#view
        reproject_child = [0] * len(out_child[0][0]['pred_reproj']) 
        reproject_parent = [0] * len(out_parent[0][0]['pred_reproj'])
        conf_child = [0] * len(out_child[0][0]['alpha_joints'])
        conf_parent = [0] * len(out_parent[0][0]['alpha_joints'])
        for det_parent, det_child in zip(view_det_parent, view_det_culd):#frame
            for key_idx, (conf_parent_val, conf_child_val, reproj_parent_val, reproj_child_val) in enumerate(zip(det_parent['alpha_joints'], det_child['alpha_joints'], det_parent['pred_reproj'], det_child['pred_reproj'])):
                reproject_child[key_idx] += reproj_child_val[-1]
                reproject_parent[key_idx] += reproj_parent_val[-1]
                conf_child[key_idx] += conf_child_val[-1]
                conf_parent[key_idx] += conf_parent_val[-1]

        comul_reproject_child.append(list(map(lambda x: x / len(out_child[0]), reproject_child)))
        comul_reproject_parent.append(list(map(lambda x: x / len(out_parent[0]), reproject_parent)))
        comul_conf_child.append(list(map(lambda x: x / len(out_child[0]), conf_child)))
        comul_conf_parent.append(list(map(lambda x: x / len(out_parent[0]), conf_parent)))

    return comul_reproject_child, comul_reproject_parent, comul_conf_child, comul_conf_parent

def print_comul_view_and_key(comul_list):
    key_comul_reproj = list(map(lambda x: x / len(comul_list), np.sum(comul_list, axis=0)))
    view_comul_reproj = list(map(lambda x: x / len(comul_list[0]), np.sum(comul_list, axis=1)))

    for key_idx, val in enumerate(key_comul_reproj):
        print(f'{halpe_skeleton[key_idx]}: {val}')
    print('----')
    for view_idx, val in enumerate(view_comul_reproj):
        print(f'View {view_idx}: {val}')
    print()
    return key_comul_reproj, view_comul_reproj



def plot_pearson_corr_coef(var1, var2, lbl1, lbl2, max_y):
    variable1 = np.array(var1)
    variable2 = np.array(var2)

    corr_matrix = np.corrcoef(variable1, variable2)
    corr_coeff = corr_matrix[0, 1]

    plt.scatter(variable1, variable2)
    plt.xlabel(lbl1)
    plt.ylabel(lbl2)

    m, b = np.polyfit(variable1, variable2, 1)
    plt.plot(variable1, m * variable1 + b, color='red')
    plt.xlim([0, max_y])
    plt.ylim([0, 1])

    plt.title(f'Pearson correlation coefficient: {corr_coeff:.2f}')

    #%matplotlib inline
    plt.show()




def plot_allviews_pearson_corr_coef(comul_reproj, comul_conf, lbl1, lbl2, max_y):
    assert len(comul_reproj) == 4, f'Passed wrong comulative reprojection error variable!'
    assert len(comul_conf) == 4, f'Passed wrong comulative keypoint confidence variable!'

    r1 = np.array(comul_reproj[0])
    r2 = np.array(comul_reproj[1])
    r3 = np.array(comul_reproj[2])
    r4 = np.array(comul_reproj[3])

    c1 = np.array(comul_conf[0])
    c2 = np.array(comul_conf[1])
    c3 = np.array(comul_conf[2])
    c4 = np.array(comul_conf[3])


    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    corr_matrix = np.corrcoef(r1, c1)
    corr_coeff = corr_matrix[0, 1]
    axes[0, 0].scatter(r1, c1)
    axes[0, 0].set_xlabel(lbl1)
    axes[0, 0].set_ylabel(lbl2)
    m, b = np.polyfit(r1, c1, 1)
    axes[0, 0].plot(r1, m * r1 + b, color='red')
    axes[0, 0].set_title(f'View 0 - Pearson correlation coeff: {corr_coeff:.2f}')
    axes[0, 0].set_xlim([0, max_y])
    axes[0, 0].set_ylim([0, 1])

    corr_matrix = np.corrcoef(r2, c2)
    corr_coeff = corr_matrix[0, 1]
    axes[0, 1].scatter(r2, c2)
    axes[0, 1].set_xlabel(lbl1)
    axes[0, 1].set_ylabel(lbl2)
    m, b = np.polyfit(r2, c2, 1)
    axes[0, 1].plot(r2, m * r2 + b, color='red')
    axes[0, 1].set_title(f'View 1 - Pearson correlation coeff: {corr_coeff:.2f}')
    axes[0, 1].set_xlim([0, max_y])
    axes[0, 1].set_ylim([0, 1])

    corr_matrix = np.corrcoef(r3, c3)
    corr_coeff = corr_matrix[0, 1]
    axes[1, 0].scatter(r3, c3)
    axes[1, 0].set_xlabel(lbl1)
    axes[1, 0].set_ylabel(lbl2)
    m, b = np.polyfit(r3, c3, 1)
    axes[1, 0].plot(r3, m * r3 + b, color='red')
    axes[1, 0].set_title(f'View 2 - Pearson correlation coeff: {corr_coeff:.2f}')
    axes[1, 0].set_xlim([0, max_y])
    axes[1, 0].set_ylim([0, 1])

    corr_matrix = np.corrcoef(r4, c4)
    corr_coeff = corr_matrix[0, 1]
    axes[1, 1].scatter(r4, c4)
    axes[1, 1].set_xlabel(lbl1)
    axes[1, 1].set_ylabel(lbl2)
    m, b = np.polyfit(r4, c4, 1)
    axes[1, 1].plot(r4, m * r4 + b, color='red')
    axes[1, 1].set_title(f'View 3 - Pearson correlation coeff: {corr_coeff:.2f}')
    axes[1, 1].set_xlim([0, max_y])
    axes[1, 1].set_ylim([0, 1])

    # Adjust spacing between subplots
    plt.tight_layout()

    #%matplotlib inline
    plt.show()


def plot_comul_hist(comul_value, y_label, max_y):
    labels = list(halpe_skeleton.values())

    bins_ = np.arange(len(comul_value) + 1) -0.5
    plt.hist(range(len(comul_value)), bins=bins_, weights=comul_value, facecolor='orange', edgecolor='gray', alpha=0.7)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylim([0, max_y])
    plt.title(f'{y_label} - Avg:{round(sum(comul_value)/len(comul_value), 3)}')


    #%matplotlib inline
    plt.show()


def plot_allviews_histogram(comul_value, y_label, max_y, title_top):
    assert len(comul_value) == 4, f'Passed wrong variable!'
    
    labels = list(halpe_skeleton.values())
    avg_val_list = []

    v1 = comul_value[0]
    v2 = comul_value[1]
    v3 = comul_value[2]
    v4 = comul_value[3]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    fig.suptitle(title_top)

    bins_ = np.arange(len(v1) + 1) -0.5
    axes[0, 0].hist(range(len(v1)), bins=bins_, weights=v1, facecolor='red', edgecolor='gray', alpha=0.7)
    axes[0, 0].set_xticks(range(len(labels)))
    axes[0, 0].set_xticklabels(labels, rotation=90)
    axes[0, 0].yaxis.set_ticks_position('both')
    axes[0, 0].xaxis.set_ticks_position('both')
    axes[0, 0].tick_params(axis="y", direction="in")
    axes[0, 0].tick_params(axis="x", direction="in")
    avg = round(sum(v1)/len(v1), 3)
    avg_val_list.append(avg)
    axes[0, 0].set_title(f'View 0 - Avg: {avg}')
    axes[0, 0].set_ylabel(y_label)
    axes[0, 0].set_ylim([0, max_y])

    bins_ = np.arange(len(v2) + 1) -0.5
    axes[0, 1].hist(range(len(v2)), bins=bins_, weights=v2, facecolor='yellow', edgecolor='gray', alpha=0.7)
    axes[0, 1].set_xticks(range(len(labels)))
    axes[0, 1].set_xticklabels(labels, rotation=90)
    axes[0, 1].yaxis.set_ticks_position('both')
    axes[0, 1].xaxis.set_ticks_position('both')
    axes[0, 1].tick_params(axis="y", direction="in")
    axes[0, 1].tick_params(axis="x", direction="in")
    avg = round(sum(v2)/len(v2), 3)
    avg_val_list.append(avg)
    axes[0, 1].set_title(f'View 1 - Avg: {avg}')
    axes[0, 1].set_ylabel(y_label)
    axes[0, 1].set_ylim([0, max_y])

    bins_ = np.arange(len(v3) + 1) -0.5
    axes[1, 0].hist(range(len(v3)), bins=bins_, weights=v3, facecolor='blue', edgecolor='gray', alpha=0.7)
    axes[1, 0].set_xticks(range(len(labels)))
    axes[1, 0].set_xticklabels(labels, rotation=90)
    axes[1, 0].yaxis.set_ticks_position('both')
    axes[1, 0].xaxis.set_ticks_position('both')
    axes[1, 0].tick_params(axis="y", direction="in")
    axes[1, 0].tick_params(axis="x", direction="in")
    avg = round(sum(v3)/len(v3), 3)
    avg_val_list.append(avg)
    axes[1, 0].set_title(f'View 2 - Avg: {avg}')
    axes[1, 0].set_ylabel(y_label)
    axes[1, 0].set_ylim([0, max_y])

    bins_ = np.arange(len(v4) + 1) -0.5
    axes[1, 1].hist(range(len(v4)), bins=bins_, weights=v4, facecolor='green', edgecolor='gray', alpha=0.7)
    axes[1, 1].set_xticks(range(len(labels)))
    axes[1, 1].set_xticklabels(labels, rotation=90)
    axes[1, 1].yaxis.set_ticks_position('both')
    axes[1, 1].xaxis.set_ticks_position('both')
    axes[1, 1].tick_params(axis="y", direction="in")
    axes[1, 1].tick_params(axis="x", direction="in")
    avg = round(sum(v4)/len(v4), 3)
    avg_val_list.append(avg)
    axes[1, 1].set_title(f'View 3 - Avg: {avg}')
    axes[1, 1].set_ylabel(y_label)
    axes[1, 1].set_ylim([0, max_y])

    # Adjust spacing between subplots
    plt.tight_layout()

    #%matplotlib inline
    plt.show()
    #std_dev = np.std(avg_val_list)
    #print("Standard Deviation:", std_dev)


def get_standard_dev(var_list):
    assert len(var_list) == 4, f'This method takes a four dimentional list, according to four views'

    avg_val_list = []
    for view_id, val in enumerate(var_list):
        print(f'View {view_id} std: {np.std(val)}')
        avg_val_list.append(round(sum(val)/len(val), 3))
    return np.std(avg_val_list)


def visualize_view_miss_frames(title, missing_frames, frame_lim):

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    data = np.ones(frame_lim)
    bar_head = ["View 0", "View 1", "View 2", "View 3"]
    view_color = ['red', 'yellow', 'blue', 'green']
    for idx_v, y in enumerate(bar_head):
        view_data = copy.copy(data)
        if missing_frames[idx_v]:
            flat_list = list(np.concatenate(missing_frames[idx_v]).flat)
            for f in flat_list:
                if f < len(data):
                    view_data[f] = 2


        colors = [view_color[idx_v] if value == 1 else "black" for value in view_data]
        for i, (c, value) in enumerate(zip(colors, view_data)):
            edgecolor_ = None if value == 1 else 'gray'
            #alpah_ = 1 if value == 1 else 1
            alpah_ = 1
            linewidth_ = 1 if value == 2 else .1
            ax.barh(y, value, height=1, color=c, linewidth=1 , align="edge", left=i, edgecolor=edgecolor_, alpha=alpah_)

    ax.set_ylim(0, 4)
    ax.set_xlabel("Frames")
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(bar_head)
    ax.set_title(title)