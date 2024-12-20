import argparse
import numpy as np
import json
import os
import random
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.as_posix())
sys.path.insert(0, os.path.join(Path(__file__).parent.as_posix(), "slowfast_llava"))
from typing import Tuple, List, Dict, Any
import torch
from torch.utils.data import Dataset

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from llama.entity_matching_llama import EntityMatchingModule
from llama.image_tagging import TaggingModule, get_unique_tags

from dataset import load_video
from prompt import get_prompt

def llava_inference(
    video_frames,
    question,
    conv_mode,
    model,
    tokenizer,
    image_processor,
    image_sizes,
    temporal_aggregation,
):
    # Get prompt
    prompt = get_prompt(model, conv_mode, question)

    # Get text inputs
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).cuda()

    # Get image inputs
    image_tensor = process_images(video_frames, image_processor, model.config)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=256,
            use_cache=True,
            temporal_aggregation=temporal_aggregation,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def load_db_json(db_file: str) -> Dict[str, Any]:
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


def inference_wrapper(tokenizer, model, image_processor, context_len, video_frames, sizes, args, question):
    # Run inference on the video
    output = llava_inference(
        video_frames,
        question,
        args.conv_mode,
        model,
        tokenizer,
        image_processor,
        sizes,
        args.temporal_aggregation,
    )
    return output

class PerceptionGQADataset(Dataset):
    """Dataset class to store video items from dataset.

    Attributes:
        video_folder_path: Path to the folder containing the videos.
        task: Task type for annotations.
        split: Dataset split to load.
        task_db: List containing annotations for dataset according to
            split and task availability.
    """

    def __init__(self, db_path: Dict[str, Any], video_folder_path: str,
                task: str, split: str, image_processor, num_frames) -> None:
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
        self.image_processor = image_processor
        self.num_frames = num_frames

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
        return len(self.pt_db_list)

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
        frame_idx = round(data_item['metadata']['num_frames']/2)
        video_file_path = os.path.join(self.video_folder_path,
                                    data_item['metadata']['video_id']) + '.mp4'
        video_frames, sizes = load_video(video_file_path, num_frms=16)

        return {'metadata': metadata,
                'grounded_question': annot,
                'object_tracking': data_item['object_tracking'],
                'video_frames': video_frames,
                'sizes': sizes,
                'frames_idx': frame_idx,
                }


def run_inference(args):
    """
    Run inference

    Args:
        args: Command-line arguments.
    """

    disable_torch_init()

    # Load tokenizer, model and image processor
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name,
        device=torch.cuda.current_device(),
        device_map="cuda",
        rope_scaling_factor=args.rope_scaling_factor,
    )

    # Load tagging entity matching module
    tagging_model = TaggingModule()
    entity_match_module = EntityMatchingModule()

    # Override image aspect ratio if needed
    if args.image_aspect_ratio:
        model.config.image_aspect_ratio = args.image_aspect_ratio

    if args.question != None:
        # If input is a single video, load video
        video_frames, sizes = load_video(args.video_path, num_frms=args.num_frames)
        output = inference_wrapper(tokenizer, model, image_processor, context_len, video_frames, sizes, args, args.question)
        video_id = video_path.replace(".mp4", "")
    else:
        label_path = args.metadata_path
        label_dict = load_db_json(label_path)

        # Assume input is video folder path, use dataloader
        cfg = {
                'video_folder_path': args.video_path,
                'task': 'grounded_question',
                'split': 'valid',
                'image_processor': image_processor,
                'num_frames': args.num_frames,
            }

        gqa_dataset = PerceptionGQADataset(args.metadata_path, **cfg)

        count_check = 50

        try:
            current_results_dict = json.load(open(args.results_path, 'r'))
        except:
            current_results_dict = {}

        results_dict = {}

        for cidx, video_item in enumerate(gqa_dataset):
            video_id = video_item['metadata']['video_id']
            video_file_path = args.video_path + video_id + '.mp4'
            frames_idx = video_item['frames_idx']

            if (cidx % count_check) == 0:
                print(f'{cidx} {video_id}')

            if video_id in current_results_dict:
                results_dict[video_id] = current_results_dict[video_id]
                continue

            video_frames = video_item['video_frames']
            sizes = video_item['sizes']

            # run tagging model
            # convert PIL to tensor
            video_array = np.array(video_frames)

            tags_in_video = tagging_model.run_on_video(video_array)
            entity_list = get_unique_tags(tags_in_video)

            q_count = 0
            results_dict[video_id] = {}

            # processing video
            for q_num, q in enumerate(video_item['grounded_question']):
                q_count += 1

                qs = q['question']
                qs = qs.replace('Track', 'What is/are')
                question = qs + ' Please answer with only the object name.'
                
                #try:
                output = inference_wrapper(tokenizer, model, image_processor, context_len, video_frames, sizes, args, question)
                cur_answers_id = q['answers']
                cur_answers = [label_dict[video_id]['object_tracking'][id]['label'] for id in cur_answers_id]
                    # entity matcher
                match_state, answer_score = entity_match_module(qs, output, entity_list, cur_answers)
                if len(match_state) == 0:
                    match_state = [entity_list[random.randint(0, len(entity_list) - 1)]]
                #except Exception as e:
                #    print(f"Error processing video file '{args.video_path}, {video_id}': {e}")

                results_dict[video_id][q_num] = {'question': q['question'], 'outputs': output, 'predicted_objects': match_state}
            print(results_dict)

            # incrementally save to json
            with open(args.results_path, 'w') as my_file:
                json.dump(results_dict, my_file)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", help="input video path", default='')
    parser.add_argument("--model_path", help="LLaVA model path", type=str, default='')
    parser.add_argument("--metadata_path", help="metadata path", type=str, default='')
    parser.add_argument("--results_path", help="results json file path", type=str, default='valid_vqa_results.json')
    parser.add_argument("--question", help="Input question and prompt", type=str, required=False, default=None)
    parser.add_argument("--conv_mode", type=str, required=False, default="image_seq_v3")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--input_structure", type=str, default="image_seq")
    parser.add_argument("--image_aspect_ratio", type=str, default="resize")
    parser.add_argument("--temporal_aggregation", type=str, default="slowfast-slow_10frms_spatial_1d_max_pool-fast_4x4")
    parser.add_argument("--rope_scaling_factor", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
