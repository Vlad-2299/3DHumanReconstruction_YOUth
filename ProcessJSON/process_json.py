import json
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
import math


def get_json_files_in_det_path(path):
    keypoint_json_list = []

    for json_ in os.listdir(path):
        if json_.endswith('.json'):
            keypoint_json_list.append(json_)

    print(f'{len(keypoint_json_list)} JSON files to be processed: {keypoint_json_list}')
    return keypoint_json_list



def calc_keypoint_overlap(key_a, key_b):
    #If value close to 0, then keypoints belong to the same person
    assert len(key_a) == len(key_b)
    mean = 0
    for k_a, k_b in zip(key_a, key_b):
        mean += abs(k_a - k_b)
    return mean/len(key_a)


def plot_detections(det_list):
    for det in det_list:
        plt.plot(det)
    plt.show()


def write_csv(keys, det_list):
    filename = 'detections.csv'

    df = pd.DataFrame()
    for i in range(len(det_list)):
        df[keys[i]] = det_list[i]
    print(df.to_string())
    df.to_csv(filename, index=False)


def compute_detections(keys):
    '''
     keypoint_list: List that will hold all the keypoint detections per each view  [view][frame][keypoints]
     ensure that:   -keypoint_list[view][frame][keypoints][0] corresponds to exclusively one of the agents
                    -keypoint_list[view][frame][keypoints][1] corresponds to exclusively one of the agents
                    -view indexation consistency^^
    '''
    keypoint_list = []
    box_list = []

    for key in keys:
        detect_keypoint_list = []  # [frame][keypoints]
        alph_box_list = [None, None]
        frame_count = 0

        with open(key) as f:
            k = json.load(f)
        prev = k[0]['image_id'].split('.png')[0]
        last = k[-1]['image_id'].split('.png')[0]
        frame = []
        alph_box_list

        for id, det in enumerate(k):
            curr = det['image_id'].split('.png')[0]
            if curr == prev:
                frame.append(det)
                if curr == last:
                    if id == len(k) - 1:  # last element
                        detect_keypoint_list.append(frame.copy())
                        frame.clear()
                        frame_count += 1
            elif curr != prev:
                detect_keypoint_list.append(frame.copy())
                frame.clear()
                frame_count += 1
                frame.append(det)
            prev = curr
        keypoint_list.append(detect_keypoint_list)

    return keypoint_list

def get_detections_of_view(json_list):
    '''
     json_list:     Input that contains the path to the JSON files that contain the detections of a single view [batch]
     keypoint_list: List that will hold all the keypoint detections for a single view [frame][keypoints]

    '''
    detect_keypoint_list = []  # [frame][keypoints]
    frame_count = 0
    view_box_list = []
    n_detections = 0

    for key in json_list:
        with open(key) as f:
            k = json.load(f)
        prev = k[0]['image_id'].split('.png')[0]
        last = k[-1]['image_id'].split('.png')[0]
        frame = []
        n_detections += len(k)
        for id, det in enumerate(k):
            curr = det['image_id'].split('.png')[0]
            if curr == prev:
                frame.append(det)
                if curr == last:
                    if id == len(k) - 1:  # last element
                        detect_keypoint_list.append(frame.copy())
                        frame.clear()
                        frame_count += 1
            elif curr != prev:
                detect_keypoint_list.append(frame.copy())
                frame.clear()
                frame_count += 1
                frame.append(det)

                if curr == last and id == len(k) - 1: #Last frame only has 1 detection
                    frame.clear()
                    frame.append(det)
                    detect_keypoint_list.append(frame.copy())
                    frame.clear()
                    frame_count += 1

            prev = curr
    n_processed_detections = 0
    for frame_det in detect_keypoint_list:
        n_processed_detections += len(frame_det)

    assert n_processed_detections == n_detections, f'Number of processed detections is not the same as the number of detections returned by AlphaPose'
    return detect_keypoint_list


def get_erronous_det(det_per_view, missing):
    '''
    det_per_view: list of lists that has the detection count for each frame
    missing: Boolean that chooses if we count the frames where detections are missing (TRUE), or the frames where there are excessive detecions (FALSE)
    '''
    miss_det = []
    for view in det_per_view:
        miss_det_view = []
        for idx, n_det in enumerate(view):
            if missing:
                if len(n_det) < 2:
                    miss_det_view.append(idx)
            else:
                if len(n_det) > 2:
                    miss_det_view.append(idx)
        miss_det.append(miss_det_view)
    return miss_det


def get_missing_detection_ranges(all_detections):
    miss_det = get_erronous_det(all_detections, True)

    miss_det_range = []

    for view in range(len(miss_det)):
        view_ranges = []
        interpolation_range = []
        for frame in miss_det[view]:
            if not interpolation_range:
                interpolation_range.append(frame)
                if frame == miss_det[view][-1]:
                    view_ranges.append(interpolation_range.copy())
            elif interpolation_range[-1] + 1 == frame:
                interpolation_range.append(frame)
                if frame == miss_det[view][-1]:
                    # Last frame in miss det is of the same range of interpool frames
                    view_ranges.append(interpolation_range.copy())
                    interpolation_range.clear()
            elif interpolation_range[-1] + 1 != frame:
                view_ranges.append(interpolation_range.copy())
                interpolation_range.clear()
                interpolation_range.append(frame)
                if frame == miss_det[view][-1]:
                    view_ranges.append(interpolation_range.copy())
        miss_det_range.append(view_ranges.copy())

    return miss_det_range


def identify_keypoint_index(current_det, prev_det):
    '''
    Mehtod that identifies the keypoints similarity in current frame, based on keypoints of previous/next frame
    0: current_det is more similar to the detection prev_det[0]
    1: current_det is more similar to the detection prev_det[1]
    '''
    assert len(current_det) == 1
    assert len(prev_det) == 2

    overlap_one = calc_keypoint_overlap(current_det[0]['keypoints'], prev_det[0]['keypoints'])
    overlap_two = calc_keypoint_overlap(current_det[0]['keypoints'], prev_det[1]['keypoints'])
    #print('Overlap with last index 0, ', overlap_one)
    #print('Overlap with last index 1, ', overlap_two)
    if overlap_one < overlap_two:
        return 0
    else:
        return 1
    

def get_drop_view_info(missing_ranges, max_consec):
    drop_view_in_frames = {
    0: [],
    1: [],
    2: [],
    3: []
    }

    for person_id in range(len(missing_ranges)):
        for view in range(len(missing_ranges[person_id])):
            #if len(missing_ranges[person_id][view]) > MAX_CONSC_MISS:
            if missing_ranges[person_id][view]:
                for consc in missing_ranges[person_id][view]:
                    if len(consc) > max_consec:
                        drop_view_in_frames[view].append([consc[0], consc[-1]])
    return drop_view_in_frames


def is_range_buffer_empty(drop_range_drop_buff):
    is_empty = True

    for range_items in drop_range_drop_buff.items():
        if range_items[1]:
            is_empty = False
    return is_empty


def order_drop_view_info(drop_view_in_frames, frame_seq_len):
    drop_view_in_frames_comp = copy.deepcopy(drop_view_in_frames)

    ignore_view_order = []
    item_aux = [0, 0]
    while not is_range_buffer_empty(drop_view_in_frames_comp):
        initial = frame_seq_len
        item = None
        
        for view_idx, view_ranges in enumerate(drop_view_in_frames_comp.items()): #[key_view][[range_list]]
            if view_ranges[1]:
                for range_idx, range_ in enumerate(view_ranges[1]):
                    if range_[0] < initial:
                        initial = range_[0]
                        item_aux = [view_idx, range_]
                        item = {view_idx: range_}
                    break

        drop_view_in_frames_comp[item_aux[0]].remove(item_aux[1])
        #ignore_view_order.insert(len(ignore_view_order), item)
        ignore_view_order.append(item)
    
    return ignore_view_order

def padd_drop_view_info(ordered_drop_info):
    padded_ordered_drop_info = []
    ordered_drop_info_copy = copy.deepcopy(ordered_drop_info)

    last_frame_buff = 0

    for drop_idx, dict_item in enumerate(ordered_drop_info_copy):
        assert len(dict_item) == 1, 'Check (order_drop_view_info) for error origin!'
        
        [[key, range]] = dict_item.items()

        if drop_idx > 0:
            [[prev_key, prev_range]] = ordered_drop_info_copy[drop_idx - 1].items()
            if prev_range[1] > range[0]: #If last missing frame of the prev view drop overlaps with initial frames of the next view drop, pad the next
                range[0] = prev_range[1] + 1
            assert prev_range[1] < range[0], 'Too many missing frames for reconstruction to work!'
        padded_ordered_drop_info.append({key: range})
    
    return padded_ordered_drop_info


def get_box_perim_(box_list):
    box_perimeter = []
    for frame_box in box_list:
            box_perimeter.append(frame_box[2] + frame_box[3])
    return box_perimeter

def correct_excess_det(ex_det, key_view_list):
    corrected_list = copy.deepcopy(key_view_list)

    assert len(ex_det) == len(corrected_list)

    for view, frame_list in enumerate(ex_det):
        for frame in frame_list:
            #print('--------')
            #print('Frame: ', frame)
            #print('View: ', view)

            remove_by_worst_score = False
            match_idx = True

            if len(corrected_list[view][frame - 1]) == 2:
                # Remove detection with least overlap
                list_cand_overlap = []
                idx_prev_det = []
                for prev_det in corrected_list[view][frame - 1]:
                    # For loop gets the overlap between prev. detections (2) and current detections (>2)
                    idx_cand = []
                    overlap_cand = []
                    #print('Prev idx: ', prev_det['idx'])
                    idx_prev_det.append(prev_det['idx'])
                    for candidate in corrected_list[view][frame]:
                        overlap = calc_keypoint_overlap(prev_det['keypoints'], candidate['keypoints'])
                        overlap_cand.append(overlap)
                        idx_cand.append(candidate['idx'])
                    #print('Candidate idx ', idx_cand)
                    list_cand_overlap.append(overlap_cand.copy())

                assert len(list_cand_overlap) == 2

                curr_best_cand_idx = []
                for cand_result_list in list_cand_overlap:
                    # Gets the index of the current detection that best overlaps the prev. frame detection
                    #print(cand_result_list)
                    curr_best_cand_idx.append(np.argmin(cand_result_list))

                #print(curr_best_cand_idx)
                #print(corrected_list[view][frame][curr_best_cand_idx[0]]['idx'],
                     # corrected_list[view][frame][curr_best_cand_idx[1]]['idx'])

                if len(curr_best_cand_idx) == len(set(curr_best_cand_idx)):
                    # If candidate detection is not the same
                    if list_cand_overlap[0][curr_best_cand_idx[0]] <= 16 and list_cand_overlap[1][curr_best_cand_idx[1]] <= 16:
                        corrected_list[view][frame].clear()
                        corrected_list[view][frame].append(key_view_list[view][frame][curr_best_cand_idx[0]])
                        corrected_list[view][frame].append(key_view_list[view][frame][curr_best_cand_idx[1]])
                        match_idx = False
                if match_idx:
                    candidate_idx = []
                    for prev_det in corrected_list[view][frame - 1]:
                        for candidate in key_view_list[view][frame]:
                            if prev_det['idx'] == candidate['idx']:
                                candidate_idx.append(candidate)
                    if len(candidate_idx) == 2:
                        corrected_list[view][frame].clear()
                        corrected_list[view][frame] = copy.deepcopy(candidate_idx)
                    else:
                        remove_by_worst_score = True
            else:
                #print('Prev frame does not have 2 detections')
                remove_by_worst_score = True

            if remove_by_worst_score:
                #print(f'Removing by score in frame {frame}, view {view}')
                # Remove detection with least score
                box_list = []
                for det_box in key_view_list[view][frame]:
                    box_list.append(det_box['box'])

                det_score_list = get_box_perim_(box_list)
                sorted_lst = sorted(range(len(det_score_list)), key=lambda i: det_score_list[i], reverse=True)
                det_idx_sort = sorted_lst[:2]



                
                
                corrected_list[view][frame].clear()
                #print(first_conf_det['idx'], second_conf_det['idx'])
                corrected_list[view][frame].append(key_view_list[view][frame][det_idx_sort[0]])
                corrected_list[view][frame].append(key_view_list[view][frame][det_idx_sort[1]])

            assert len(corrected_list[view][frame]) == 2
            assert None not in corrected_list[view][frame]
    return corrected_list


def correct_excess_det_old(ex_det, key_view_list):
    corrected_list = copy.deepcopy(key_view_list)

    assert len(ex_det) == len(corrected_list)

    for view, frame_list in enumerate(ex_det):
        for frame in frame_list:
            #print('--------')
            #print('Frame: ', frame)
            #print('View: ', view)

            remove_by_worst_score = False
            match_idx = True

            if len(corrected_list[view][frame - 1]) == 2:
                # Remove detection with least overlap
                list_cand_overlap = []
                idx_prev_det = []
                for prev_det in corrected_list[view][frame - 1]:
                    # For loop gets the overlap between prev. detections (2) and current detections (>2)
                    idx_cand = []
                    overlap_cand = []
                    #print('Prev idx: ', prev_det['idx'])
                    idx_prev_det.append(prev_det['idx'])
                    for candidate in corrected_list[view][frame]:
                        overlap = calc_keypoint_overlap(prev_det['keypoints'], candidate['keypoints'])
                        overlap_cand.append(overlap)
                        idx_cand.append(candidate['idx'])
                    #print('Candidate idx ', idx_cand)
                    list_cand_overlap.append(overlap_cand.copy())

                assert len(list_cand_overlap) == 2

                curr_best_cand_idx = []
                for cand_result_list in list_cand_overlap:
                    # Gets the index of the current detection that best overlaps the prev. frame detection
                    #print(cand_result_list)
                    curr_best_cand_idx.append(np.argmin(cand_result_list))

                #print(curr_best_cand_idx)
                #print(corrected_list[view][frame][curr_best_cand_idx[0]]['idx'],
                     # corrected_list[view][frame][curr_best_cand_idx[1]]['idx'])

                if len(curr_best_cand_idx) == len(set(curr_best_cand_idx)):
                    # If candidate detection is not the same
                    if list_cand_overlap[0][curr_best_cand_idx[0]] <= 16 and list_cand_overlap[1][curr_best_cand_idx[1]] <= 16:
                        corrected_list[view][frame].clear()
                        corrected_list[view][frame].append(key_view_list[view][frame][curr_best_cand_idx[0]])
                        corrected_list[view][frame].append(key_view_list[view][frame][curr_best_cand_idx[1]])
                        match_idx = False
                if match_idx:
                    candidate_idx = []
                    for prev_det in corrected_list[view][frame - 1]:
                        for candidate in key_view_list[view][frame]:
                            if prev_det['idx'] == candidate['idx']:
                                candidate_idx.append(candidate)
                    if len(candidate_idx) == 2:
                        corrected_list[view][frame].clear()
                        corrected_list[view][frame] = copy.deepcopy(candidate_idx)
                    else:
                        remove_by_worst_score = True
            else:
                #print('Prev frame does not have 2 detections')
                remove_by_worst_score = True

            if remove_by_worst_score:
                #print(f'Removing by score in frame {frame}, view {view}')
                # Remove detection with least score
                det_score_list = []
                for d in key_view_list[view][frame]:
                    det_score_list.append(d['score'])
                sorted_lst = sorted(range(len(det_score_list)), key=lambda i: det_score_list[i], reverse=True)
                det_idx_sort = sorted_lst[:2]
                
                corrected_list[view][frame].clear()
                #print(first_conf_det['idx'], second_conf_det['idx'])
                corrected_list[view][frame].append(key_view_list[view][frame][det_idx_sort[0]])
                corrected_list[view][frame].append(key_view_list[view][frame][det_idx_sort[1]])

            assert len(corrected_list[view][frame]) == 2
            assert None not in corrected_list[view][frame]
    return corrected_list



def identify_keypoint_index_with_paird_dets(curr_det_list, prev_det_list):
    assert len(curr_det_list) == len(prev_det_list) == 2, f'This method takes a pair of current deections and a pair of prev detections'
    #print('curr')
    #print(curr_det_list[0]['idx'])
    #print(curr_det_list[1]['idx'])
    #print('prev')
    #print(prev_det_list[0]['idx'])
    #print(prev_det_list[1]['idx'])
    overlap_00 = calc_keypoint_overlap(curr_det_list[0]['keypoints'], prev_det_list[0]['keypoints'])
    overlap_01 = calc_keypoint_overlap(curr_det_list[0]['keypoints'], prev_det_list[1]['keypoints'])
    
    overlap_10 = calc_keypoint_overlap(curr_det_list[1]['keypoints'], prev_det_list[0]['keypoints'])
    overlap_11 = calc_keypoint_overlap(curr_det_list[1]['keypoints'], prev_det_list[1]['keypoints'])

    best_overlap_0 = {}
    best_overlap_1 = {}

    if overlap_00 < overlap_01:
        best_overlap_0 = {0: overlap_00}
    else:
        best_overlap_0 = {1: overlap_01}

    if overlap_10 < overlap_11:
        best_overlap_1 = {0: overlap_10}
    else:
        best_overlap_1 = {1: overlap_11}

    [[k_0, v_0]] = best_overlap_0.items()
    [[k_1, v_1]] = best_overlap_1.items()



    if k_0 != k_1:
        return [k_0, k_1]
    else:
        #print(overlap_00)
        #print(overlap_01)
        #print(overlap_10)
        #print(overlap_11)
        #print('----')
        #print(best_overlap_0)
        #print(best_overlap_1)
        if k_0 == k_1 == 0:
            if v_0 < v_1:
                return [0, 1]
            else:
                return [1, 0]
        else:
            if v_0 < v_1:
                return [1, 0]
            else:
                return [0, 1]
        

def calc_point_overlap(key_a, key_b):
    #If value close to 0, then keypoints belong to the same person
    assert len(key_a) == len(key_b)
    mean = 0
    for k_a, k_b in zip(key_a, key_b):
        mean += abs(k_a - k_b)
    return mean/len(key_a)


def calc_pair_box_overlap_alpha(curr_box_list, prev_box_list):
    '''
    NOTE
    box_alphapose: down right corner (idx 2 & 3) are the exact pixel value minus the top left values (idx 0 & 1)
    '''
    overlap_00 = calc_point_overlap(curr_box_list[0]['box'], prev_box_list[0]['box'])
    overlap_01 = calc_point_overlap(curr_box_list[0]['box'], prev_box_list[1]['box'])

    overlap_10 = calc_point_overlap(curr_box_list[1]['box'], prev_box_list[0]['box'])
    overlap_11 = calc_point_overlap(curr_box_list[1]['box'], prev_box_list[1]['box'])
    #idx = 0 if overlap_a < overlap_b else 1
    #best_overlap = overlap_a if overlap_a < overlap_b else overlap_b

    best_overlap_0 = {}
    best_overlap_1 = {}

    if overlap_00 < overlap_01:
        best_overlap_0 = {0: overlap_00}
    else:
        best_overlap_0 = {1: overlap_01}

    if overlap_10 < overlap_11:
        best_overlap_1 = {0: overlap_10}
    else:
        best_overlap_1 = {1: overlap_11}

    [[k_0, v_0]] = best_overlap_0.items()
    [[k_1, v_1]] = best_overlap_1.items()

    if k_0 != k_1:
        return [k_0, k_1]
    else:
        if v_0 < v_1:
            return [0, 1]
        else:
            return [1, 0]


def view_index_consistency(unstruct_det, v_id):
    '''
    Method that indexes the detections according to ReID and last frame

    unstruct_det: list of all the detections in a single frame PER VIEW
    '''
    indexed_det_list = []
    id_list_first = []
    id_list_second = []
    re_id_error_count = 0
    two_box_one_det_count = 0

    for i, detection in enumerate(unstruct_det):  # For each frame, get detections
        assert len(detection) <= 2,  f'Before this method, run the function that removes excess detections!'
        idx_frame_keypoint_list = [None, None]  # [agent_1, agent_2]

        #Get index of prev detection
        prev_idx = -1
        if indexed_det_list:
            if indexed_det_list[-1][0]:
                if indexed_det_list[-1][0]['idx'] == 0: #Custom made idx where there are NO detections in the frame, so we jump back to frames where there are detections

                    for frame_id, rev_frame in enumerate(reversed(indexed_det_list)):
                        if rev_frame[0] is None:
                            prev_idx = indexed_det_list.index(rev_frame)
                            break
                        else:
                            if rev_frame[0]['idx'] != 0:
                                prev_idx = indexed_det_list.index(rev_frame)
                                break

                    #has_det_idx = [i for i, e in enumerate(indexed_det_list) if e[0]['idx'] != 0]
                    #prev_idx = has_det_idx[-1]

        if len(detection) == 1 and detection[0]['idx'] == 0:
            #No detections
            idx_frame_keypoint_list[0] = detection[0]

        else:
            if len(detection) == 1:
                if not indexed_det_list:
                    #First frame only has one detection!'
                    idx_frame_keypoint_list[0] = detection[0]
                else:
                    if None not in indexed_det_list[prev_idx]:
                        #Prev det has 2 detections
                        match_id = identify_keypoint_index(detection, indexed_det_list[prev_idx])
                        idx_frame_keypoint_list[match_id] = detection[0]
                    else:
                        #Prev det has 1 detection
                        prev_none_idx = indexed_det_list[prev_idx].index(None)
                        prev_single_idx = 1 if prev_none_idx == 0 else 0
                        
                        #if detection[0]['idx'] in id_list_first and detection[0]['idx'] not in id_list_second:
                        #    idx_frame_keypoint_list[0] = detection[0]
                        #elif detection[0]['idx'] in id_list_second and detection[0]['idx'] not in id_list_first:
                        #    idx_frame_keypoint_list[1] = detection[0]
                        #else:
                        if True:
                            prev_missing_true_det_idx = None
                            for frame, det in reversed(list(enumerate(indexed_det_list))):
                                if det[prev_none_idx] != None:
                                    prev_missing_true_det_idx = frame
                            assert prev_missing_true_det_idx != None
                            
                            overlap_with_prev = calc_keypoint_overlap(indexed_det_list[prev_idx][prev_single_idx]['keypoints'],
                                                    detection[0]['keypoints'])
                            overlap_with_last_det_undetected = calc_keypoint_overlap(indexed_det_list[prev_missing_true_det_idx][prev_none_idx]['keypoints'],
                                                    detection[0]['keypoints'])
                            
                            if overlap_with_prev < overlap_with_last_det_undetected:
                                idx_frame_keypoint_list[prev_single_idx] = detection[0]
                            else:
                                idx_frame_keypoint_list[prev_none_idx] = detection[0]

            else:# len(detection) == 2:
                if not indexed_det_list:
                    #First Frame
                    idx_frame_keypoint_list[0] = detection[0]
                    idx_frame_keypoint_list[1] = detection[1]
                else:
                    #print(i, prev_idx, indexed_det_list[prev_idx])
                    unidentified_detections = []  # stores the detections that changed their idx score, compared to the last frame
                    if None not in indexed_det_list[prev_idx]:
                        idx_last_a = indexed_det_list[prev_idx][0]['idx']
                        idx_last_b = indexed_det_list[prev_idx][1]['idx']
                        
                        for d in detection:
                            if d['idx'] in id_list_first and d['idx'] not in id_list_second:
                                idx_frame_keypoint_list[0] = d
                            elif d['idx'] in id_list_second and d['idx'] not in id_list_first:
                                idx_frame_keypoint_list[1] = d
                            else:
                                unidentified_detections.append(d)
                        
                        if not unidentified_detections and None in idx_frame_keypoint_list:
                            #The same ID was assigned to different people
                            re_id_error_count += 1
                            idx_frame_keypoint_list = [None, None]
                            unidentified_detections.clear()
                            curr_diff_overlap   = calc_keypoint_overlap(detection[0]['keypoints'],detection[1]['keypoints'])
                            prev_same_overlap_0 = calc_keypoint_overlap(indexed_det_list[prev_idx][0]['keypoints'],detection[0]['keypoints'])
                            prev_same_overlap_1 = calc_keypoint_overlap(indexed_det_list[prev_idx][1]['keypoints'],detection[1]['keypoints'])
                            if curr_diff_overlap < prev_same_overlap_0 or curr_diff_overlap < prev_same_overlap_1:
                                two_box_one_det_count += 1
                                det_idx_box = calc_pair_box_overlap_alpha(detection, indexed_det_list[prev_idx])
                                idx_frame_keypoint_list[0] = detection[det_idx_box[0]]
                                idx_frame_keypoint_list[1] = detection[det_idx_box[1]]
                            else:
                                idx_frame_keypoint_list[0] = detection[det_idx[0]]
                                idx_frame_keypoint_list[1] = detection[det_idx[1]]
                            #print('Assigned: ', idx_frame_keypoint_list[0]['idx'], idx_frame_keypoint_list[1]['idx'])
                            #for d in detection:
                            #    if d['idx'] == idx_last_a:
                            #        idx_frame_keypoint_list[0] = d
                            #    elif d['idx'] == idx_last_b:
                            #        idx_frame_keypoint_list[1] = d
                            #    else:
                            #        unidentified_detections.append(d)
                    else:
                        none_idx = indexed_det_list[prev_idx].index(None)
                        single_idx = 1 if none_idx == 0 else 0

                        if indexed_det_list[prev_idx][single_idx]['idx'] != detection[0]['idx'] and indexed_det_list[prev_idx][single_idx]['idx'] != detection[1]['idx']:
                            #Get index of prev frame with 2 detections
                            if indexed_det_list:
                                for frame_id, rev_frame in enumerate(reversed(indexed_det_list)):
                                    if None not in rev_frame:
                                        prev_idx = indexed_det_list.index(rev_frame)
                                        break
                            #Now prev frame was 2 dets
                            #print(v_id, i)
                            idx_frame_keypoint_list = [None, None]
                            unidentified_detections.clear()
                            curr_diff_overlap   = calc_keypoint_overlap(detection[0]['keypoints'],detection[1]['keypoints'])
                            prev_same_overlap_0 = calc_keypoint_overlap(indexed_det_list[prev_idx][0]['keypoints'],detection[0]['keypoints'])
                            prev_same_overlap_1 = calc_keypoint_overlap(indexed_det_list[prev_idx][1]['keypoints'],detection[1]['keypoints'])
                            if curr_diff_overlap < prev_same_overlap_0 or curr_diff_overlap < prev_same_overlap_1:
                                two_box_one_det_count += 1
                                det_idx_box = calc_pair_box_overlap_alpha(detection, indexed_det_list[prev_idx])
                                idx_frame_keypoint_list[0] = detection[det_idx_box[0]]
                                idx_frame_keypoint_list[1] = detection[det_idx_box[1]]
                            else:
                                det_idx = identify_keypoint_index_with_paird_dets(detection, indexed_det_list[prev_idx])
                                idx_frame_keypoint_list[0] = detection[det_idx[0]]
                                idx_frame_keypoint_list[1] = detection[det_idx[1]]
                                
                        else:

                            curr_diff_overlap   = calc_keypoint_overlap(detection[0]['keypoints'],detection[1]['keypoints'])
                            prev_same_overlap_0 = calc_keypoint_overlap(indexed_det_list[prev_idx][single_idx]['keypoints'],detection[0]['keypoints'])
                            prev_same_overlap_1 = calc_keypoint_overlap(indexed_det_list[prev_idx][single_idx]['keypoints'],detection[1]['keypoints'])
                            
                            if curr_diff_overlap < prev_same_overlap_0 and curr_diff_overlap < prev_same_overlap_1:
                                two_box_one_det_count += 1
                                if prev_same_overlap_0 < prev_same_overlap_1:
                                    idx_frame_keypoint_list[single_idx] = detection[0]
                                    idx_frame_keypoint_list[none_idx] = detection[1]
                                else:
                                    idx_frame_keypoint_list[single_idx] = detection[1]
                                    idx_frame_keypoint_list[none_idx] = detection[0]
                            else:
                                match_id = identify_keypoint_index([indexed_det_list[prev_idx][single_idx]], detection)
                                rem_match_id = 1 if match_id == 0 else 0
                                idx_frame_keypoint_list[single_idx] = detection[match_id]
                                idx_frame_keypoint_list[none_idx] = detection[rem_match_id]
                    
                    if unidentified_detections:
                        #If here prev and curr frames have both 2 dets
                        det_idx = identify_keypoint_index_with_paird_dets(detection, indexed_det_list[prev_idx])
                        #print(f'Det idx: {det_idx}')
                        idx_frame_keypoint_list = [None, None]
                        if detection[det_idx[0]]['idx'] in id_list_second or detection[det_idx[1]]['idx'] in id_list_first:
                            two_box_one_det_count += 1
                            #Desambiguate by calculating box overlap, and not keypoints
                            #print(v_id, i, unidentified_detections)
                            det_idx_box = calc_pair_box_overlap_alpha(detection, indexed_det_list[prev_idx])
                            idx_frame_keypoint_list[0] = detection[det_idx_box[0]]
                            idx_frame_keypoint_list[1] = detection[det_idx_box[1]]
                        elif detection[det_idx[0]]['idx'] in id_list_first or detection[det_idx[1]]['idx'] in id_list_second:
                            idx_frame_keypoint_list[0] = detection[det_idx[0]]
                            idx_frame_keypoint_list[1] = detection[det_idx[1]]
                           
#
                        else:
                            curr_diff_overlap   = calc_keypoint_overlap(detection[0]['keypoints'],detection[1]['keypoints'])
                            prev_same_overlap_0 = calc_keypoint_overlap(indexed_det_list[prev_idx][0]['keypoints'],detection[0]['keypoints'])
                            prev_same_overlap_1 = calc_keypoint_overlap(indexed_det_list[prev_idx][1]['keypoints'],detection[1]['keypoints'])
                            if curr_diff_overlap < prev_same_overlap_0 or curr_diff_overlap < prev_same_overlap_1:
                                two_box_one_det_count += 1
                              
                                det_idx_box = calc_pair_box_overlap_alpha(detection, indexed_det_list[prev_idx])
                                idx_frame_keypoint_list[0] = detection[det_idx_box[0]]
                                idx_frame_keypoint_list[1] = detection[det_idx_box[1]]
                            else:
                               
                                idx_frame_keypoint_list[0] = detection[det_idx[0]]
                                idx_frame_keypoint_list[1] = detection[det_idx[1]]
                            #print('Assigned: ', idx_frame_keypoint_list[0]['idx'], idx_frame_keypoint_list[1]['idx'])
                    assert None not in idx_frame_keypoint_list, f"Error assigning the detections at frame {detection[0]['image_id']}, view {v_id}"
                  
        indexed_det_list.append(idx_frame_keypoint_list)

        #Auxiliary idx verification
        #Keeps track of all the id's assigned to one person, so those id's cannot be assigned to the other person

        if idx_frame_keypoint_list[0]:
            if idx_frame_keypoint_list[0]['idx'] > 0 and idx_frame_keypoint_list[0]['idx'] in id_list_second:
                re_id_error_count += 1
        if idx_frame_keypoint_list[1]:
            if idx_frame_keypoint_list[1]['idx'] > 0 and idx_frame_keypoint_list[1]['idx'] in id_list_first:
                re_id_error_count += 1

                
        if idx_frame_keypoint_list[0]:
            if idx_frame_keypoint_list[0]['idx'] > 0 and idx_frame_keypoint_list[0]['idx'] not in id_list_first:
                id_list_first.append(idx_frame_keypoint_list[0]['idx'])
        if idx_frame_keypoint_list[1]:
            if idx_frame_keypoint_list[1]['idx'] > 0 and idx_frame_keypoint_list[1]['idx'] not in id_list_second:
                id_list_second.append(idx_frame_keypoint_list[1]['idx'])

            
    print(f'In view {v_id}, ReID errors: {re_id_error_count}')
    print(f'In view {v_id}, AlphaPose errors: {two_box_one_det_count}')
    assert len(unstruct_det) == len(indexed_det_list)
    return indexed_det_list, re_id_error_count, two_box_one_det_count




def get_missing_frames(key_list, search_idx):
    #Finds missing detections of a single individual per view
    miss= []
    opp = 1 if search_idx == 0 else 0
    for n_frame, frame_det in enumerate(key_list):
        if frame_det[search_idx] is None:
            #miss.append(int(frame_det[opp]['image_id'].split('.png')[0]))
            miss.append(n_frame)
        elif frame_det[search_idx]['idx'] == 0:
            #miss.append(int(frame_det[search_idx]['image_id'].split('.png')[0]))
            miss.append(n_frame)
    return miss


def get_missing_frame_ranges(miss_list):
    range_list = []
    miss_det_range = []
    for idx, f_miss in enumerate(miss_list):
        if not miss_det_range:
            miss_det_range.append(f_miss)
        elif f_miss == (miss_det_range[-1] + 1):
            miss_det_range.append(f_miss)
            if f_miss == miss_list[-1]:
                # Last frame in miss det is of the same range of interpool frames
                range_list.append(miss_det_range.copy())
                miss_det_range.clear()
        else:
            range_list.append(miss_det_range.copy())
            miss_det_range.clear()
            miss_det_range.append(f_miss)

    if miss_det_range:
        range_list.append(miss_det_range.copy())

    return range_list


def get_missing_frames_names(key_list, search_idx):
    #Finds missing detections of a single individual per view
    miss= []
    opp = 1 if search_idx == 0 else 0
    for n_frame, frame_det in enumerate(key_list):
        if frame_det[search_idx] is None:
            miss.append(int(frame_det[opp]['image_id'].split('.png')[0]))
            #miss.append(n_frame)
        elif frame_det[search_idx]['idx'] == 0:
            miss.append(int(frame_det[search_idx]['image_id'].split('.png')[0]))
            #miss.append(n_frame)
    return miss


def get_missing_det_list(coorect_missing_key):
    missing_dets_list = []
    for person_id in range(2):
        pers_missing_det = []
        for view in range(len(coorect_missing_key)):
            missing_frame_list = get_missing_frames_names(coorect_missing_key[view], person_id)
            pers_miss_range = []
            if missing_frame_list:
                pers_miss_range = get_missing_frame_ranges(missing_frame_list)
            pers_missing_det.append(pers_miss_range)
        missing_dets_list.append(pers_missing_det)
    return missing_dets_list


def print_missing_info(box_perimeter, missing_dets_list):
    total_miss_pers = [0, 0]
    max_consc_miss_pers = [0, 0]
    person_name = ['None', 'None']
    for person in range(len(missing_dets_list)):
        other = 0 if person == 1 else 1
        if box_perimeter[person] > box_perimeter[other]:
            print('---Parent---')
            person_name[person] = 'Parent'
        else:
            print('---Child---')
            person_name[person] = 'Child'
        for view in range(len(missing_dets_list[person])):
            print(f'View: {view}')
            print(f'Missing in frames: {missing_dets_list[person][view]}')
            for miss_range in missing_dets_list[person][view]:
                total_miss_pers[person] += len(miss_range)
                max_consc_miss_pers[person] = len(miss_range) if max_consc_miss_pers[person] < len(miss_range) else max_consc_miss_pers[person]
        print(f'{person_name[person]} is missing in a total of {total_miss_pers[person]} frames, in a maximum of {max_consc_miss_pers[person]} consecutive frames')


def interpolate_keypoints(start_point, end_point, n_interpol):
    assert (len(start_point) == len(end_point)), f'Interpolation value lists need to be of the same size'
    interpolation_list = []

    for i in range(1, n_interpol):
        # For each missing point
        keypoints = []

        for start_, end_ in zip(start_point, end_point):
            # For each keypoint in list
            start = start_
            end = end_

            rng = abs(start - end)
            stride = rng / n_interpol
            if start > end:
                value = start - (stride * i)
            else:
                value = start + (stride * i)
            keypoints.append(value)

        interpolation_list.append(keypoints.copy())

    return interpolation_list


def interpolate_missing_detections(keypoints):
    n_people = 2
    inter_keypoints = copy.deepcopy(keypoints)

    for person_id in range(n_people):
        for view in range(len(keypoints)):
            # Get missing detections of *person_id* in *view*
            #print(f'View: {view} - Person {person_id}')
            missing_frame_list = get_missing_frames(keypoints[view], person_id)
            #print(missing_frame_list)
            if missing_frame_list:
                pers_miss_range = get_missing_frame_ranges(missing_frame_list)
                #print(f'Person {person_id}, miss: {pers_miss_range}')
                pooled_points = []
                #print(pers_miss_range)
                for range_f in pers_miss_range:
                    start = range_f[0]
                    end = range_f[-1]

                    if start == 0:
                        print('First missing frame has no previous detections for interpolation')
                    if end == len(keypoints[view]):
                        print('There are no detections past this frame')

                    try:
                        pooled_points = interpolate_keypoints(keypoints[view][start - 1][person_id]['keypoints'].copy(),
                                                              keypoints[view][end + 1][person_id]['keypoints'].copy(),
                                                              len(range_f) + 1)
                    except:
                        print(f'Range: {range_f}')
                        print(f"First frame: {keypoints[view][start][person_id]['image_id']}")
                        print(f'Person Id: {person_id}')
                        print(keypoints[view][start - 1][person_id]['keypoints'])
                        print(keypoints[view][end + 1][person_id]['keypoints'])
                        print(len(pooled_points))
                        print('Cant interpolate')
                    base_inst = copy.deepcopy(keypoints[view][start - 1][person_id])
                    base_img_name = int(base_inst['image_id'].split('.png')[0])

                    assert len(range_f) == len(pooled_points)

                    for id, miss_frame in enumerate(range_f):
                        #assert inter_keypoints[view][miss_frame][person_id] is None

                        inter_keypoints[view][miss_frame][person_id] = copy.deepcopy(base_inst)
                        frame_numb = '{0:05d}'.format(base_img_name + (id + 1)) + '.png'

                        inter_keypoints[view][miss_frame][person_id]['image_id'] = frame_numb
                        inter_keypoints[view][miss_frame][person_id]['keypoints'] = pooled_points[id].copy()
    return inter_keypoints


def get_alpha_box_list_old(keypoints, img_names):
    alpha_box_list = []
    for view_key, view_img in zip(keypoints, img_names):
        view_box_list = []
        for frame_img in view_img:
            frame_box = [None, None]
            frame_box[0] = view_key[frame_img-1][0]['box']
            frame_box[1] = view_key[frame_img-1][1]['box']
            view_box_list.append(frame_box)
        alpha_box_list.append(view_box_list)
    return alpha_box_list


def get_alpha_box_list(keypoints, img_names):
    #Change the loop order, so we iterate only once over all the keypoints
    alpha_box_list = []
    for view_key, view_img in zip(keypoints, img_names):
        view_box_list = []
        for dets in view_key:
            name = int(dets[0]['image_id'].split('.png')[0])
            if name in view_img:
                frame_box_list = []
                for det in dets:
                    frame_box_list.append(det['box'])
                view_box_list.append(frame_box_list)
        alpha_box_list.append(view_box_list)
    return alpha_box_list


def get_box_list(keypoints, f_per_f):
    box_list = []
    counter = -1
    for view_det in keypoints:
        view_det_box_list = []
        for n_frame, dets in enumerate(view_det):
            frame_det_box = [None, None]
            if f_per_f - counter == f_per_f:
                frame_det_box[0] = dets[0]['box']
                frame_det_box[1] = dets[1]['box']


def create_folder(json_path):
    # Creates folders to store frame by frame detections per view

    name = os.path.basename(os.path.normpath(json_path))
    path_split_json = os.path.join(json_path, name)

    if not os.path.exists(path_split_json):
        os.makedirs(path_split_json)
        print(f'Created folder {name} to store detections per frame')
        return path_split_json
    else:
        print(f'{name} already exists!')
        return path_split_json


def save_json(folder, file, temp):
    with open(f'{folder}/{file}_keypoints.json', 'w') as f:
        json.dump(temp, f)
    f.close()


def get_box_perim(box_list):
    box_perimeter = [0, 0]
    for view_list in box_list:
        for frame_box in view_list:
                for idx, box in enumerate(frame_box):
                    #mask_h = abs(math.ceil(box[3] - box[1]))
                    #mask_w = abs(math.ceil(box[2] - box[0]))
                    box_perimeter[idx] += (box[2] + box[3])
    return box_perimeter



def get_view_to_view_consistency(keypoint_list_, swap_view):
    keypoint_list = copy.deepcopy(keypoint_list_)
    indexes = list(np.where(swap_view)[0])
    if not indexes:
        return keypoint_list
    for sp_idx in indexes:
        view_detections = []
        for frame_idx, frame_dets in enumerate(keypoint_list[sp_idx]):
            corrected_frame_dests = []
            for det in frame_dets[::-1]:
                corrected_frame_dests.append(det)
            view_detections.append(corrected_frame_dests)
        keypoint_list[sp_idx] = view_detections

    return keypoint_list


def split_json_frames_single(keypoints_view, to_folder):

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

        frame_name = frame[0]['image_id'].split('.png')[0]
        json_string['people'][0]['pose_keypoints_2d'] = copy.deepcopy(frame[0]['keypoints'])
        save_json(to_folder, frame_name, json_string)


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


def main_json_processor(all_detections):
    '''
    Gets list of all detections, computes excess detections, removes excess, keeps index consistency, interpolates
    '''

    #json_files = get_json_files_in_det_path(det_path)
    #create_folders(json_files)
    #all_detections = compute_detections(json_files)


    #missing_detection_ranges = get_missing_detection_ranges(missing_detections)

    excess_detections = get_erronous_det(all_detections, False)

    corrected_key_view_list = correct_excess_det(excess_detections, all_detections)

    ice_keypoints_one = []
    reid_error = []
    alphapose_error = []
    for id_v, det_view in enumerate(corrected_key_view_list):
        keypoints, re_error, ap_error = view_index_consistency(det_view, id_v)
        ice_keypoints_one.append(keypoints)
        reid_error.append(re_error)
        alphapose_error.append(ap_error)

    missing_keypoints = copy.deepcopy(ice_keypoints_one)
    interpool_keypoints = interpolate_missing_detections(ice_keypoints_one)

    #ice_keypoints_two = []
    #for id_v, det_view in enumerate(interpool_keypoints):
        #ice_keypoints_two.append(view_index_consistency(det_view, id_v))

    return interpool_keypoints, missing_keypoints, reid_error, alphapose_error



def first_last_guarantee(keypoint_list):
    #Ensure that first and last frame habe both detections
    n_people = 2
    initial_frame = 0
    last_frame = -1
    search_initial = False
    search_final = False

    for view_dets_ini in keypoint_list:
        if len(view_dets_ini[initial_frame]) < 2:
            search_initial = True

    for view_dets_last in keypoint_list:
        if len(view_dets_last[last_frame]) < 2:
            search_final = True

    if not search_initial and not search_final:
        return keypoint_list, initial_frame, last_frame

    while search_initial:
        view_det_len_ini = []
        for view_id in range(len(keypoint_list)):
            view_det_len_ini.append(len(keypoint_list[view_id][initial_frame]))
        if all(i >= 2 for i in view_det_len_ini):
            search_initial = False
        else:
            initial_frame += 1

    for frame in range(len(keypoint_list[0])-1, -1, -1):
        frame_n_dets = []
        for view in range(len(keypoint_list)):
            frame_n_dets.append(len(keypoint_list[view][frame]))
        if all(i >= 2 for i in frame_n_dets):
            last_frame = frame + 1
            break
    for view_id in range(len(keypoint_list)):
        if last_frame == -1:
            keypoint_list[view_id] = keypoint_list[view_id][initial_frame:]
        else:
            keypoint_list[view_id] = keypoint_list[view_id][initial_frame:last_frame]

    for view_i in range(len(keypoint_list)):
        assert len(keypoint_list[view_i][0]) > 1, f'Initial frame of view {view_i} has {len(keypoint_list[view_i][0])} detection'
        assert len(keypoint_list[view_i][-1]) > 1, f"Final frame of view {view_i} has {len(keypoint_list[view_i][-1])} detection, frame: {keypoint_list[view_i][-1][0]['image_id']}"
    return keypoint_list, initial_frame, last_frame