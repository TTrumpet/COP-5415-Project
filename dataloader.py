from decord import VideoReader, cpu
# from mmengine import fileio
import io
import numpy as np
import os.path
import tarfile
import torch


def uniform_sample(lst, n):
    assert n <= len(lst)
    m = len(lst)
    step = m // n  # Calculate the step size
    return [lst[i * step] for i in range(n)]


def _get_rawvideo_dec(video_path, image_processor, video_processor, max_frames=8, frame_resolution=224,
                      video_framerate=1, s=None, e=None, min_frames=8, num_video_frames=8, num_context_images=8):
    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    try:
        if os.path.exists(video_path):
            vreader = VideoReader(video_path, num_threads=1)
        else:
            raise Exception("Video path does not exist")
    except Exception as e:
        # with open(video_path, 'rb') as f:
        #     video_bytes = f.read()
        ds_path = 'INSERT-DATASET-PATH'
        remaining_path = video_path.replace(ds_path, '')
        subfolder = remaining_path.split(os.sep)[0]
        dataset = remaining_path.split(os.sep)[1]
        # if 'webvid' in dataset:
        #     dataset = os.path.join('webvid', remaining_path.split(os.sep)[2])
        # Manual dataset name change.
        if dataset == 'Activity_Videos':
            dataset = 'activitynet_videos'
        vid_path = os.path.dirname(remaining_path).replace(subfolder + os.sep, '')
        video = os.path.basename(video_path)

        # print(remaining_path)
        # print(subfolder)
        # print(dataset)
        # print(vid_path)
        # print(video)
        # print(os.path.join(ds_path, subfolder, dataset + '.tgz'))
        # print(os.path.join(vid_path, video))

        with tarfile.open(os.path.join(ds_path, subfolder, dataset + '.tgz'), 'r') as tar:
            video_file = tar.extractfile(os.path.join(vid_path, video))
            video_bytes = video_file.read()
        vreader = VideoReader(io.BytesIO(video_bytes), num_threads=1)

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    # print(num_frames)
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        elif len(all_pos) < min_frames:
            if num_frames < min_frames:
                min_frames = num_frames
            t_stride = max(1, (f_end - f_start) // (min_frames - 1))
            adjusted_f_end = f_start + t_stride * (min_frames - 1)
            sample_pos = list(range(f_start, adjusted_f_end + 1, t_stride))
        else:
            sample_pos = all_pos

        all_images = [f for f in vreader.get_batch(sample_pos).asnumpy()]
        # In case if we can't sample MAX_IMAGE_LENGTH frames
        num_video_frames_sampled = min(num_video_frames, len(all_images))
        num_context_images_sampled = min(num_context_images, len(all_images))

        video_frames = uniform_sample(all_images, num_video_frames_sampled)
        context_images = uniform_sample(all_images, num_context_images_sampled)
        video_frames = video_processor.preprocess(video_frames)['pixel_values']
        context_images = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in context_images]

        if len(context_images) < num_context_images:  # Pad
            while len(context_images) < num_context_images:
                context_images.append(
                    torch.zeros((3, image_processor.crop_size['height'], image_processor.crop_size['width'])))

        slice_len = len(video_frames)

        if slice_len < 1:
            pass
        else:
            while len(video_frames) < num_video_frames:
                video_frames.append(torch.zeros((3, frame_resolution, frame_resolution)))
    else:
        print("video path: {} error.".format(video_path))

    return video_frames, context_images, slice_len


def _get_rawvideo_dec_gqa(video_path, image_processor, video_processor, max_frames=8, frame_resolution=224,
                      video_framerate=1, s=None, e=None, min_frames=8, num_video_frames=8, num_context_images=8):
    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    try:
        vreader = VideoReader(video_path, num_threads=1)
    except Exception as e:
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        vreader = VideoReader(io.BytesIO(video_bytes), num_threads=1)

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    # print(num_frames)
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        elif len(all_pos) < min_frames:
            if num_frames < min_frames:
                min_frames = num_frames
            t_stride = max(1, (f_end - f_start) // (min_frames - 1))
            adjusted_f_end = f_start + t_stride * (min_frames - 1)
            sample_pos = list(range(f_start, adjusted_f_end + 1, t_stride))
        else:
            sample_pos = all_pos

        all_images = [f for f in vreader.get_batch(sample_pos).asnumpy()]

        if video_processor is not None:
            # In case if we can't sample MAX_IMAGE_LENGTH frames
            num_video_frames_sampled = min(num_video_frames, len(all_images))
            num_context_images_sampled = min(num_context_images, len(all_images))

            video_frames = uniform_sample(all_images, num_video_frames_sampled)
            context_images = uniform_sample(all_images, num_context_images_sampled)
            video_frames = video_processor.preprocess(video_frames)['pixel_values']
            context_images = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in context_images]

            if len(context_images) < num_context_images:  # Pad
                while len(context_images) < num_context_images:
                    context_images.append(
                        torch.zeros((3, image_processor.crop_size['height'], image_processor.crop_size['width'])))
                    
            slice_len = len(video_frames)

            if slice_len < 1:
                pass
            else:
                while len(video_frames) < num_video_frames:
                    video_frames.append(torch.zeros((3, frame_resolution, frame_resolution)))
        else:
            video_frames = None
            context_images = None
            slice_len = None

        frames_idx = list(range(f_start, f_end + 1, 1))
        # frames_idx.append(f_end)
        tracking_frames = [f for f in vreader.get_batch(frames_idx).asnumpy()]
    else:
        print("video path: {} error.".format(video_path))

    return video_frames, context_images, slice_len, tracking_frames, frames_idx