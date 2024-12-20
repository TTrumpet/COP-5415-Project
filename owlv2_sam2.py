from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.models.owlv2.modeling_owlv2 import center_to_corners_format

from sam2.build_sam import build_sam2_video_predictor_hf
from sam2.utils.transforms import SAM2Transforms
from sam2.utils.misc import mask_to_box


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Tracker_with_OwlV2:
    def __init__(self):
        self.device = 'cuda'
        self.owl_processor = AutoProcessor.from_pretrained('google/owlv2-large-patch14')
        self.detector = Owlv2ForObjectDetection.from_pretrained('google/owlv2-large-patch14').to(self.device)
        self.sam = build_sam2_video_predictor_hf('facebook/sam2-hiera-large').to(self.device)
        self.sam.init_state = self.init_state
        # self.sam_transform = v2.Compose([
        #     v2.Resize((self.sam.image_size, self.sam.image_size)),
        #     v2.ToDtype(torch.float32, scale=True),
        #     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        self.sam_transform = SAM2Transforms(resolution=self.sam.image_size, mask_threshold=0.0)
        self.sam.fill_hole_area = 0


    @torch.inference_mode()
    def init_state(
        self,
        video,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        
        video_height, video_width = video[0].shape[:2]
        # Modified this line, transforming video here.
        # video = torch.stack([self.sam_transform(torch.tensor(frame).permute(2, 0, 1)) for frame in video], dim=0)
        video = self.sam_transform.forward_batch(video)
        inference_state = {}
        inference_state["images"] = video
        inference_state["num_frames"] = len(video)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self.sam._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state
    
    
    def post_process_object_detection(
        self, outputs, threshold: float = 0.1, target_sizes = None
    ):
        """
        Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`OwlViTObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        logits, boxes = outputs.logits, outputs.pred_boxes
        num_logit_cls = logits.shape[-1]
        # Set to -inf
        for i in range(logits.shape[0]):
            num_cls = len(threshold[i])
            for j in range(num_logit_cls):
                if j >= num_cls:
                    logits[i, :, j] = -float('inf')
        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            img_w, img_h = target_sizes
            # rescale coordinates
            width_ratio = 1
            height_ratio = 1

            if img_w < img_h:
                width_ratio = img_w / img_h
            elif img_h < img_w:
                height_ratio = img_h / img_w
            
            img_w = img_w / width_ratio
            img_h = img_h / height_ratio

            img_w, img_h = torch.tensor([img_w, img_h])
            img_h = torch.Tensor([img_h for _ in range(len(boxes))])
            img_w = torch.Tensor([img_w for _ in range(len(boxes))])

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []

        for i, (s, l, b, t) in enumerate(zip(scores, labels, boxes, threshold)):
            t = len(t)
            # Get top n predictions from s.
            custom_threshold = torch.topk(s, t).values[-1]
            custom_threshold = custom_threshold

            score = s[s >= custom_threshold]
            label = l[s >= custom_threshold]
            box = b[s >= custom_threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results
    
    
    def box_to_relative_box(self, boxes, img_shape):
        img_h, img_w = img_shape
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(boxes.device)
        boxes = boxes / scale_fct
        return boxes


    @torch.inference_mode()
    def detect(self, video, prompts):
        inputs = self.owl_processor(text=prompts, images=video, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.detector(**inputs)
        target_sizes = video[0].shape[:2]
        results = self.post_process_object_detection(
            outputs=outputs, threshold=prompts, target_sizes=target_sizes
        )
        return results


    def track(self, video, boxes, object_ids):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.sam.init_state(video)
            # TODO: can we handle the case where the object is not detected in the first frame?
            frame_idx = 0
            # add new prompts and instantly get the output on the same frame
            for obj_id, box in zip(object_ids, boxes):
                _, _, _ = self.sam.add_new_points_or_box(state, frame_idx, obj_id, box=box)
            # self.visualize_masks(video, out_mask_logits)
            # propagate the prompts to get masklets throughout the video
            video_segments = {}
            for frame_idx, out_obj_ids, out_mask_logits in self.sam.propagate_in_video(state):
                video_segments[frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)}

            self.sam.reset_state(state)
            return video_segments


    def detect_and_track(self, video, prompts):
        detect_vid = np.expand_dims(video[0], axis=0)
        output = self.detect(detect_vid, prompts)[0]
        boxes = output['boxes'].cpu()
        object_ids = output['labels'].cpu().numpy()
        # self.visualize_boxes(video, boxes.unsqueeze(0))
        video_segments = self.track(detect_vid, boxes, object_ids)
        # Dictionary to store results.
        tracking_results = {}
        frame_idx = list(video_segments.keys())
        tracking_results['frames_idx'] = frame_idx
        for obj_id in object_ids:
            pred_masks = [video_segments[frame_idx][obj_id] for frame_idx in video_segments]
            # Convert masks to bounding boxes.
            pred_bboxes = mask_to_box(torch.tensor(np.array(pred_masks)))
            pred_bboxes_relative = self.box_to_relative_box(pred_bboxes, video[0].shape[:2])
            tracking_results[obj_id] = {
                'pred_masks': pred_masks,
                'pred_bboxes': pred_bboxes,
                'pred_bboxes_relative': pred_bboxes_relative.tolist(),
            }
        return tracking_results


    def detect_and_track_batch(self, video, prompts):
        detect_vid = np.repeat(np.expand_dims(video[0], axis=0), len(prompts), axis=0)
        output = self.detect(detect_vid, prompts)
        boxes = [out['boxes'].cpu() for out in output]
        object_ids = [out['labels'].cpu().numpy() for out in output]
        temp_obj_ids = list(range(len(np.concatenate(object_ids))))
        # self.visualize_boxes(detect_vid, boxes)
        video_segments = self.track(video, torch.cat(boxes), temp_obj_ids)
        # Dictionary to store results.
        tracking_results = {}
        for obj_id in temp_obj_ids:
            pred_masks = [video_segments[frame_idx][obj_id] for frame_idx in video_segments]
            # Convert masks to bounding boxes.
            pred_bboxes = mask_to_box(torch.tensor(np.array(pred_masks))).squeeze()
            pred_bboxes_relative = self.box_to_relative_box(pred_bboxes, video[0].shape[:2])
            tracking_results[obj_id] = {
                'pred_masks': pred_masks,
                'pred_bboxes': pred_bboxes,
                'pred_bboxes_relative': pred_bboxes_relative.tolist(),
            }
        return tracking_results
    

    def visualize_detect(self, video, prompts):
        detect_vid = np.expand_dims(video[0], axis=0)
        outputs = self.detect(detect_vid, prompts)
        self.visualize_boxes(detect_vid, [output['boxes'].cpu() for output in outputs])
    
    
    def visualize_boxes(self, video, boxes, num_frames=5):
        plt_count = min(len(video), len(boxes), num_frames if num_frames is not None else len(video))
        fig, ax = plt.subplots(1, plt_count, figsize=(4*plt_count, 4))
        frame_idx = np.linspace(0, len(video)-1, plt_count, dtype=int)
        video_plt = [video[i] for i in frame_idx]
        boxes_plt = [boxes[i] for i in frame_idx]
        for idx, (frame, box) in enumerate(zip(video_plt, boxes_plt)):
            try:
                ax[idx].imshow(frame)
                ax[idx].axis("off")
            except:
                ax.imshow(frame)
                ax.axis("off")
            try:
                b = box[0][1]
            except:
                box = np.expand_dims(box, axis=0)
            for b in box:
                rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=1, edgecolor="r", facecolor="none")
                try:
                    ax[idx].add_patch(rect)
                except:
                    ax.add_patch(rect)

        plt.tight_layout()
        plt.show()

    
    def visualize_masks(self, video, masks, num_frames=5):
        plt_count = min(len(video), len(masks), num_frames if num_frames is not None else len(video))
        fig, ax = plt.subplots(1, plt_count, figsize=(4*plt_count, 4))
        frame_idx = np.linspace(0, len(video)-1, plt_count, dtype=int)
        video_plt = [video[i] for i in frame_idx]
        masks_plt = [masks[i] for i in frame_idx]
        for idx, (frame, mask) in enumerate(zip(video_plt, masks_plt)):
            try:
                ax[idx].imshow(frame)
                ax[idx].axis("off")
                ax[idx].imshow(mask.squeeze(), alpha=0.5, cmap="jet")
            except:
                ax.imshow(frame)
                ax.axis("off")
                ax.imshow(mask.squeeze().cpu().numpy(), alpha=0.5, cmap="jet")
        plt.tight_layout()
        plt.show()
