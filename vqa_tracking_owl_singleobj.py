# -*- coding: utf-8 -*-
import json
import os
from typing import List, Dict, Any

import numpy as np
from torch.utils.data import Dataset, DataLoader
from trackeval.metrics.hota import HOTA

from owlv2_sam2 import Tracker_with_OwlV2

from dataloader import _get_rawvideo_dec_gqa


# Helper functions.
def load_db_json(db_file: str) -> Dict[Any, Any]:
    """Loads a JSON file as a dictionary.

    Args:
        db_file (str): Path to the JSON file.

    Returns:
        Dict: Loaded JSON data as a dictionary.

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        TypeError: If the JSON file is not formatted as a dictionary.
    """
    if not os.path.isfile(db_file):
        raise FileNotFoundError(f'No such file: {db_file}')

    with open(db_file, 'r') as f:
        db_file_dict = json.load(f)
        if not isinstance(db_file_dict, dict):
            raise TypeError('JSON file is not formatted as a dictionary.')
        return db_file_dict


# Dataset class.
class PerceptionGQADataset(Dataset):
    """Dataset class to store video items from dataset.

    Attributes:
        video_folder_path: Path to the folder containing the videos.
        task: Task type for annotations.
        split: Dataset split to load.
        task_db: List containing annotations for dataset according to
            split and task availability.
    """

    def __init__(self, db_path: Dict[str, Any], current_results_dict, 
                 video_folder_path: str, task: str, split: str) -> None:
        """Initializes the PerceptionDataset class.

        Args:
            db_path (str): Path to the annotation file.
            video_folder_path (str): Path to the folder containing the videos.
            task (str): Task type for annotations.
            split (str): Dataset split to load.
        """
        self.video_folder_path = video_folder_path
        self.task = task
        self.split = split
        self.task_db = self.load_dataset(db_path)
        self.image_processor = None
        self.video_processor = None
        self.current_results_dict = current_results_dict

    def load_dataset(self, db_path: str) -> List:
        """Loads the dataset from the annotation file and processes.

        Dict is processed according to split and task.

        Args:
            db_path (str): Path to the annotation file.

        Returns:
            List: List of database items containing annotations.
        """
        db_dict = load_db_json(db_path)
        db_list = []
        for _, val in db_dict.items():
            if val['metadata']['split'] == self.split:
                if val[self.task]:  # If video has annotations for this task
                    db_list.append(val)

        return db_list

    def __len__(self) -> int:
        """Returns the total number of videos in the dataset.

        Returns:
            int: Total number of videos.
        """
        return len(self.task_db)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns the video and annotations for a given index.

        Args:
        idx (int): Index of the video.

        Returns:
        Dict: Dictionary containing the video frames, metadata, annotations.
        """
        data_item = self.task_db[idx]
        annot = data_item[self.task]

        metadata = data_item['metadata']
        video_id = metadata['video_id']
        if video_id in self.current_results_dict:
            return {'metadata': metadata}
        video_file_path = os.path.join(self.video_folder_path, video_id) + '.mp4'
        _, _, _, tracking_frames, frames_idx = _get_rawvideo_dec_gqa(video_file_path, self.image_processor, self.video_processor, frame_resolution=224, max_frames=8, num_video_frames=16, num_context_images=8)
        # vid_frames, context_images, ssl_frames, slice_len, tracking_frames, frames_idx = _get_rawvideo_dec_gqa(video_file_path, self.image_processor, self.video_processor, self.ssl_video_processor, frame_resolution=224, max_frames=8, num_video_frames=16, num_context_images=8)

        return {'metadata': metadata,
                'grounded_question': annot,
                'object_tracking': data_item['object_tracking'],
                'tracking_frames': tracking_frames,
                'frames_idx': frames_idx
                }
                # 'ssl_frames': ssl_frames,


# @title Evaluation functions
def get_start_frame(track_arr: List[List[float]]) -> int:
    """Returns index of the first non-zero element in a track array.

    Args:
        track_arr (list): one hot vector correspoinding to annotations,
        showing which index to start tracking .

    Returns:
        int: Index of the first non-zero element in the track array.

    Raises:
        ValueError: Raises error if the length of the array is 0
        or if there is no one-hot value.
    """
    if not track_arr or np.count_nonzero(track_arr) == 0:
        raise ValueError('Track is empty or has no non-zero elements')
    return np.nonzero(track_arr)[0][0]


def get_start_info(track: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve information about the start frame of a track.

    Args:
        track (Dict): A dictionary containing information about the track.

    Returns:
        Dict[str: Any]: A dictionary with the following keys:
        'start_id': The frame ID of the start frame.
        'start_bounding_box': The bounding box coordinates of the start
            frame.
        'start_idx': The index of the start frame in the
        'bounding_boxes' list.
    """
    track_start_idx = get_start_frame(track['initial_tracking_box'])
    track_start_id = track['frame_ids'][track_start_idx]
    track_start_bb = track['bounding_boxes'][track_start_idx]

    return {'start_id': track_start_id,
            'start_bounding_box': track_start_bb,
            'start_idx': track_start_idx}


def filter_pred_boxes(pred_bb: np.ndarray, pred_fid: np.ndarray,
                      gt_fid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filter bounding boxes and frame IDs based on ground truth frame IDs.

    Args:
        pred_bb (np.ndarray): Array of predicted bounding boxes.
        pred_fid (np.ndarray): Array of frame IDs for predicted bounding boxes.
        gt_fid (np.ndarray): Array of frame IDs for ground truth bounding boxes.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered predicted bounding boxes and
            their corresponding frame IDs.
    """
    pred_idx = np.isin(pred_fid, gt_fid).nonzero()[0]
    filter_pred_bb = pred_bb[pred_idx]
    filter_pred_fid = pred_fid[pred_idx]
    if len(filter_pred_fid) != len(gt_fid):
        filter_pred_bb = np.concatenate((filter_pred_bb, np.expand_dims(pred_bb[-1], axis=0)))
        filter_pred_fid = np.concatenate((filter_pred_fid, np.expand_dims(gt_fid[-1], axis=0)))
    return filter_pred_bb, filter_pred_fid


def search_frame_ids(frame_id_dict: Dict[int, List[int]]
                    ) -> Dict[int, Any]:
    """Search for frame IDs in tracks to get per Frame ID track ID dict.

    Args:
        frame_id_dict (Dict): A dictionary containing track IDs as
        keys and lists of corresponding frame IDs as values.

    Returns:
        Dict[int, List]: A combined dictionary with frame IDs as keys and
        lists of track IDs containing each frame ID as values.
    """
    combined_id_dict = {}

    for k, v in frame_id_dict.items():
        for i in v:
            if i not in combined_id_dict:
                combined_id_dict[i] = [k]
            else:
                combined_id_dict[i].append(k)

    return combined_id_dict


def build_gt_ids(tracks: List[Dict[str, Any]], gt: bool,
                 track_ids: List[int] = None) -> Dict[int, List[int]]:
    """Build ground truth track IDs dict based on input tracks and track IDs.

    Args:
        tracks (List): A list of track
        dictionaries, each containing 'id' and 'frame_ids' as keys.
        gt (bool): A boolean indicating if ground truth track IDs are to be built.
        track_ids (List, optional): A list of track IDs to be considered.
        If None, all tracks will be considered. Defaults to None.

    Returns:
        Dict: A dictionary containing track IDs as keys and lists
        of corresponding frame IDs as values.
    """
    track_dict = {}
    if track_ids is None:
        nid = 0
        for track in tracks:
            if gt:
                start_info = get_start_info(track)
                track_dict[nid] = track['frame_ids'][start_info['start_idx']:]
            else:
                track_dict[nid] = track['frame_ids']
            nid += 1

    else:
        nid = 0
        for t in track_ids:
            track = tracks[t]
            assert track['id'] == t
            if gt:
                start_info = get_start_info(track)
                track_dict[nid] = track['frame_ids'][start_info['start_idx']:]
            else:
                track_dict[nid] = track['frame_ids']
            nid += 1

    frame_id_dict = search_frame_ids(track_dict)
    return frame_id_dict


def calculate_iou(boxes1: np.array, boxes2: np.array) -> float:
    """Calculate Intersection over Union (IoU) for two sets of bounding boxes.

    Args:
        boxes1 (np.array): Bounding boxes in the format [y2, x2, y1, x1],
        shape (n, 4)
        boxes2 (np.array): Bounding boxes in the format [y2, x2, y1, x1],
        shape (n, 4)

    Returns:
        iou (float): Intersection over Union (IoU) float value
    """
    x1_1, y1_1, x2_1, y2_1 = np.split(boxes1, 4, axis=1)
    x1_2, y1_2, x2_2, y2_2 = np.split(boxes2, 4, axis=1)

    # Find intersection coordinates
    y1_inter = np.maximum(y1_1, y1_2)
    x1_inter = np.maximum(x1_1, x1_2)
    y2_inter = np.minimum(y2_1, y2_2)
    x2_inter = np.minimum(x2_1, x2_2)

    # Calculate area of intersection
    h_inter = np.maximum(0, y2_inter - y1_inter)
    w_inter = np.maximum(0, x2_inter - x1_inter)
    area_inter = h_inter * w_inter

    # Calculate area of union
    area_boxes1 = (y2_1 - y1_1) * (x2_1 - x1_1)
    area_boxes2 = (y2_2 - y1_2) * (x2_2 - x1_2)
    union = area_boxes1 + area_boxes2 - area_inter

    return area_inter / union


def top_k_tracks(tracks: List[Dict[str, Any]], max_num: int
                 ) -> List[Dict[str, Any]]:
    """Select the top-k tracks based on their score.

    Args:
        tracks (List): A list of dictionaries representing tracks with 'score' as
        one of the keys.
        max_num (int): The maximum number of tracks to select.

    Returns:
        List: A list of the top-k tracks with the highest scores.
    """
    sorted_tracks = sorted(tracks, key=lambda d: d['score'], reverse=True)
    return sorted_tracks[0:max_num]


def run_iou(results: Dict[Any, Any], label_dict: Dict[Any, Any],
            max_num_tracks: int = 10) -> Dict[Any, Any]:
    """Run IoU calculations for ground truth and predicted tracks.

    Args:
        results (Dict]): A dictionary containing results for different videos,
        with video IDs as keys and video results as values.
        label_dict (Dict): A dictionary containing label information for different
        videos, with video IDs as keys and label data as values.
        max_num_tracks (int, optional): The maximum number of tracks to consider.
        Defaults to 10.

    Returns:
        Dict: A dictionary containing video data with IoU scores.
    """
    data = {}
    for video_id, video_results in results.items():
        fail_ids = ['video_9985', 'video_10462', 'video_117', 'video_7940', 'video_7336', 'video_4622']
        if video_id in fail_ids:
            continue
        if video_id == 'video_2431':
            break
        video_data = {}
        gt_tracks = label_dict[video_id]['object_tracking']
        gt_qs = label_dict[video_id]['grounded_question']
        pred_answers = video_results['grounded_question']

        for q in gt_qs:
            q_data = {}
            answer_gt_tracks = [gt_tracks[t] for t in q['answers']]
            answer_pred_tracks = pred_answers[str(q['id'])] # cast to string to match key
            if len(answer_pred_tracks) > max_num_tracks:
                answer_pred_tracks = top_k_tracks(answer_pred_tracks, max_num_tracks)

            gt_frame_id_dict = build_gt_ids(gt_tracks, True, q['answers'])
            pred_frame_id_dict = build_gt_ids(answer_pred_tracks, False)

            q_data['num_gt_ids'] = len(q['answers'])
            q_data['num_gt_dets'] = (
                sum([len(gt_tracks[t]['bounding_boxes']) for t in q['answers']])
            )
            q_data['gt_ids'] = [np.array(x) for x in list(gt_frame_id_dict.values())]

            q_data['num_tracker_ids'] = len(answer_pred_tracks)
            q_data['num_tracker_dets'] = (
                sum([len(t['bounding_boxes']) for t in answer_pred_tracks])
            )
            q_data['tracker_ids'] = (
                [np.array(x) for x in list(pred_frame_id_dict.values())]
            )

            sim_score_dict = {k: [] for k in list(gt_frame_id_dict.keys())}
            sim_scores = []
            for gt_track in answer_gt_tracks:
                track_sim_scores = []
                for pred_track in answer_pred_tracks:
                    start_info = get_start_info(gt_track)
                    start_idx = start_info['start_idx']
                    gt_bb = np.array(gt_track['bounding_boxes'])[start_idx:]
                    gt_fid = gt_track['frame_ids'][start_idx:]

                    # case where only one box is labelled
                    if not gt_fid:
                        continue

                    pred_bb = np.array(pred_track['bounding_boxes'])
                    pred_fid = np.array(pred_track['frame_ids'])
                    
                    # filter predicted trajectory for frame IDs where we have annotations
                    pred_bb, pred_fid = filter_pred_boxes(pred_bb, pred_fid, gt_fid)
                    # clip 0 -> 1
                    pred_bb = np.minimum(np.maximum(pred_bb, 0), 1)
                    iou = calculate_iou(gt_bb, pred_bb)
                    track_sim_scores.append(iou)
                    for frame_iou, frame_id in zip(iou, pred_fid):
                        sim_score_dict[frame_id].append(frame_iou.item())

                sim_scores.append(np.array(track_sim_scores))

            if q_data['num_tracker_dets'] == 0 or q_data['num_gt_dets'] == 0:
                sim_scores = []
            else:
                sim_scores = []
                for idx, frame_scores in enumerate(sim_score_dict.values()):
                    sim_scores.append(np.array(frame_scores).reshape(
                        (len(q_data['gt_ids'][idx]), len(q_data['tracker_ids'][idx]))))

            q_data['similarity_scores'] = sim_scores
            video_data[q['id']] = q_data

        data[video_id] = video_data
    return data


def eval_sequences(evaluator: Any, data: Dict[Any, Any]) -> Dict[str, float]:
    """Evaluate sequences using the evaluator.

    Args:
        evaluator (Any): The evaluator object to be used for evaluation.
        data (Dict): A dictionary containing the data to be evaluated, with video
        IDs as keys and video results as values.

    Returns:
        Dict: A dictionary containing the average evaluation results for the
        sequences, with keys 'HOTA', 'DetA', 'AssA', and 'LocA'.
    """
    hota = []
    deta = []
    assa = []
    loca = []
    for video_results in data.values():
        for question_results in video_results.values():
            res = evaluator.eval_sequence(question_results)
            hota.append(np.mean(res['HOTA']))
            deta.append(np.mean(res['DetA']))
            assa.append(np.mean(res['AssA']))
            loca.append(np.mean(res['LocA']))

    ave_hota = np.mean(hota)
    print('HOTA: ', ave_hota)
    ave_deta = np.mean(deta)
    print('DetA: ', ave_deta)
    ave_assa = np.mean(assa)
    print('AssA: ', ave_assa)
    ave_loca = np.mean(loca)
    print('LocA: ', ave_loca)

    return {'HOTA': ave_hota, 'DetA': ave_deta,
            'AssA': ave_assa, 'LocA': ave_loca}


def get_answer_tracks(ex_data: dict, goq_ids: List) -> List[dict]:
    """Filters and retrieves object tracks based on the given object ids.

    Args:
        ex_data (dict): The data containing object tracking information.
        goq_ids (List): The list of IDs to filter tracks.

    Returns:
        List[dict]: The filtered tracks matching the goq_ids.
    """
    goq_tracks = []
    for track in ex_data['object_tracking']:
        if track['id'] in goq_ids:
            goq_tracks.append(track)
    return goq_tracks


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


# Evaluate predictions.
def evaluate(label_path, file_path="grounded_question_valid_vgptplus.json"):
    # Load json files.
    results = load_db_json(file_path)
    print(len(results))
    label_dict = load_db_json(label_path)
    print(len(label_dict))

    data = run_iou(results, label_dict)

    hota_evaluator = HOTA()
    eval_sequences(hota_evaluator, data)


def main(input_json, output_json, eval=False, visualize=False, data_path=''):
    split = 'valid'
    label_path = f'{data_path[:-5]}/grounded_question_valid.json'

    if eval:
        evaluate(label_path, file_path=output_json)
        exit()

    # Initialise dataset config.
    cfg = {
        'video_folder_path': data_path,
        'task': 'grounded_question',
        'split': split,
    }

    # Init tracker.
    tracker = Tracker_with_OwlV2()

    label_dict = load_db_json(label_path)
    input_json = json.load(open(input_json, 'r'))

    results = {}

    q_count = 0
    
    try:
        current_results_dict = json.load(open(output_json, 'r'))
    except:
        current_results_dict = {}

    # Initialise dataset.
    gqa_dataset = PerceptionGQADataset(label_path, current_results_dict, **cfg)
    gqa_dataset = DataLoader(gqa_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x)

    for cidx, video_item in enumerate(gqa_dataset):
        video_item = video_item[0]
        video_id = video_item['metadata']['video_id']

        if video_id in current_results_dict:
            results[video_id] = current_results_dict[video_id]
            continue
        
        frames_idx = video_item['frames_idx']

        print(f'\nBeginning to process video: {video_id} -- Count: {cidx}')
        print('----------------------------------------')

        tracking_frames = video_item['tracking_frames']

        video_pred_tracks = {}

        for q_num, q in enumerate(video_item['grounded_question']):
            q_count += 1
            print(f'\n\tProcessing question: {q_num}')
            # print('\t----------------------')

            cur_answers_id = q['answers']
            cur_answers = [label_dict[video_id]['object_tracking'][id]['label'] for id in cur_answers_id]

            match_state = input_json[video_id][str(q_num)]['predicted_objects']
            #match_state=input_json[video_id][q_num]['model_output'].split(',')
            if type(match_state) == str:
                match_state = [match_state]
            # TODO: need better backup plan.
            if len(match_state) == 0:
                match_state = [input_json[video_id][str(q_num)]['model_output'].split(' ')[-1]]
                #match_state = [input_json[video_id][q_num]['model_output'].split(' ')[-1]]
            print(f'\tGrounding: {match_state} -- Answers: {cur_answers}')
            if len(match_state) > 1:
                match_state = [match_state[0]]
            
            # Save for tracking.
            # prompts.append(match_state)
            # q_ids.append(q['id'])
            qid = q['id']
            match_state = match_state[:7]
            c = 0

            try:
                outputs = tracker.detect_and_track_batch(tracking_frames, prompts=match_state)
                #print('outputs:', outputs)
                if visualize:
                    pred_masks = [m1 | m2 for m1, m2 in zip(outputs[0]['pred_masks'], outputs[1]['pred_masks'])]
                    pred_bboxes = outputs[0]['pred_bboxes']
                    tracker.visualize_masks(tracking_frames, pred_masks)
                    tracker.visualize_boxes(tracking_frames, pred_bboxes)
                    exit()
        
                # get into same form as baseline to perform HOTA calculations
                pred_tracks = []
                for id in match_state:
                    pred_track = {}
                    pred_track['id'] = c
                    pred_track['bounding_boxes'] = outputs[c]['pred_bboxes_relative']
                    pred_track['frame_ids'] = frames_idx
                    pred_track['label'] = id
                    pred_tracks.append(pred_track)
                    c += 1
                video_pred_tracks[qid] = [pred_track]
            except:
                print(f'Error in tracking in video {video_id}!')
                pred_tracks = []
                for id in match_state:
                    pred_track = {}
                    pred_track['id'] = c
                    pred_track['bounding_boxes'] = outputs[c]['pred_bboxes_relative']
                    pred_track['frame_ids'] = frames_idx
                    pred_track['label'] = id
                    pred_tracks.append(pred_track)
                    c += 1
                video_pred_tracks[qid] = [pred_track]
            
                

        results[video_id] = {'grounded_question': video_pred_tracks}

        # Incremental saving of results.
        if cidx % 100 == 0 and cidx != 0:
            with open(output_json, 'w') as my_file:
                json.dump(results, my_file)

            print('\nPROGRESS REPORT AT VIDEO ' + str(cidx))
            print('------------------------------------')
            print("total number of questions: " + str(q_count))
            try:
                evaluate(label_path, output_json)
            except:
                print('Error in evaluation...')
            print('------------------------------------\n')

    with open(output_json, 'w') as my_file:
        json.dump(results, my_file)

    print(f'\nFINAL RESULTS for {cidx} videos!')
    print("total number of questions: " + str(q_count))
    evaluate(label_path, output_json)

# Start here.
if __name__ == '__main__':
    input_json = 'valid_vqa_results.json'
    output_json = input_json.replace('vqa', 'vqa_tracking_owl')
    visualize = False # Will exit after visualizing if True.
    bool_evaluate = False

    print(f'PROCESSING TRACKING for {input_json} and saving to {output_json}...')

    # Call main function.
    main(input_json, output_json, eval=bool_evaluate, visualize=visualize, data_path='')
