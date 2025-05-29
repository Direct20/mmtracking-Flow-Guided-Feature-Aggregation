# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import DATASETS
from mmdet.datasets.api_wrappers import COCO

from .coco_video_dataset import CocoVideoDataset
from .parsers import CocoVID
from mmcv.utils import print_log
import random

@DATASETS.register_module()
class ImagenetVIDDataset(CocoVideoDataset):
    """ImageNet VID dataset for video object detection."""

    CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
               'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
               'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
               'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale',
               'zebra')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotations from COCO/COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCO/COCOVID api.
        """
        if self.load_as_video:
            data_infos = self.load_video_anns(ann_file)
        else:
            data_infos = self.load_image_anns(ann_file)
        return data_infos

    def load_image_anns(self, ann_file):
        """Load annotations from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCO api.
        """
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        all_img_ids = self.coco.get_img_ids()
        self.img_ids = []
        data_infos = []
        for img_id in all_img_ids:
            info = self.coco.load_imgs([img_id])[0]
            info['filename'] = info['file_name']
            if info['is_vid_train_frame']:
                self.img_ids.append(img_id)
                data_infos.append(info)
        return data_infos

    def load_video_anns(self, ann_file):
        """Load annotations from COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCOVID api.
        """
        self.coco = CocoVID(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                info['filename'] = info['file_name']
                if self.test_mode:
                    assert not info['is_vid_train_frame'], \
                        'is_vid_train_frame must be False in testing'
                    self.img_ids.append(img_id)
                    data_infos.append(info)
                elif info['is_vid_train_frame']:
                    self.img_ids.append(img_id)
                    data_infos.append(info)
        return data_infos
    
    def ref_img_sampling(self,
                            img_info,
                            frame_range,
                            stride=1,
                            num_ref_imgs=1,
                            filter_key_img=True,
                            method='uniform',
                            return_key_img=True):
            """Sampling reference frames in the same video for key frame.

            Args:
                img_info (dict): The information of key frame.
                frame_range (List(int) | int): The sampling range of reference
                    frames in the same video for key frame.
                stride (int): The sampling frame stride when sampling reference
                    images. Default: 1.
                num_ref_imgs (int): The number of sampled reference images.
                    Default: 1.
                filter_key_img (bool): If False, the key image will be in the
                    sampling reference candidates, otherwise, it is exclude.
                    Default: True.
                method (str): The sampling method. Options are 'uniform',
                    'bilateral_uniform', 'test_with_adaptive_stride',
                    'test_with_fix_stride'. 'uniform' denotes reference images are
                    randomly sampled from the nearby frames of key frame.
                    'bilateral_uniform' denotes reference images are randomly
                    sampled from the two sides of the nearby frames of key frame.
                    'test_with_adaptive_stride' is only used in testing, and
                    denotes the sampling frame stride is equal to (video length /
                    the number of reference images). test_with_fix_stride is only
                    used in testing with sampling frame stride equalling to
                    `stride`. Default: 'uniform'.
                return_key_img (bool): If True, the information of key frame is
                    returned, otherwise, not returned. Default: True.

            Returns:
                list(dict): `img_info` and the reference images information or
                only the reference images information.
            """
            assert isinstance(img_info, dict)
            if isinstance(frame_range, int):
                assert frame_range >= 0, 'frame_range can not be a negative value.'
                frame_range = [-frame_range, frame_range]
            elif isinstance(frame_range, list):
                assert len(frame_range) == 2, 'The length must be 2.'
                assert frame_range[0] <= 0 and frame_range[1] >= 0
                for i in frame_range:
                    assert isinstance(i, int), 'Each element must be int.'
            else:
                raise TypeError('The type of frame_range must be int or list.')

            if 'test' in method and \
                    (frame_range[1] - frame_range[0]) != num_ref_imgs:
                print_log(
                    'Warning:'
                    "frame_range[1] - frame_range[0] isn't equal to num_ref_imgs."
                    'Set num_ref_imgs to frame_range[1] - frame_range[0].',
                    logger=self.logger)
                self.ref_img_sampler[
                    'num_ref_imgs'] = frame_range[1] - frame_range[0]

            if (not self.load_as_video) or img_info.get('frame_id', -1) < 0 \
                    or (frame_range[0] == 0 and frame_range[1] == 0):
                ref_img_infos = []
                for i in range(num_ref_imgs):
                    ref_img_infos.append(img_info.copy())
            else:
                vid_id, img_id, frame_id = img_info['video_id'], img_info[
                    'id'], img_info['frame_id']
                img_ids = self.coco.get_img_ids_from_vid(
                    vid_id)  # valid img ids in this video
                left = max(0, frame_id + frame_range[0])  # left bound
                # right bound
                right = min(frame_id + frame_range[1], len(img_ids) - 1)

                ref_img_ids = []
                if method == 'uniform':
                    valid_ids = img_ids[left:right + 1]
                    if filter_key_img and img_id in valid_ids:
                        valid_ids.remove(img_id)
                    num_samples = min(num_ref_imgs, len(valid_ids))
                    ref_img_ids.extend(random.sample(valid_ids, num_samples))
                elif method == 'bilateral_uniform':
                    assert num_ref_imgs % 2 == 0, \
                        'only support load even number of ref_imgs.'
                    for mode in ['left', 'right']:
                        if mode == 'left':
                            valid_ids = img_ids[left:frame_id + 1]
                        else:
                            valid_ids = img_ids[frame_id:right + 1]
                        if filter_key_img and img_id in valid_ids:
                            valid_ids.remove(img_id)
                        if len(valid_ids) < num_ref_imgs//2:
                            valid_ids = [img_id]*(num_ref_imgs//2)
                        num_samples = min(num_ref_imgs // 2, len(valid_ids))
                        sampled_inds = random.sample(valid_ids, num_samples)
                        ref_img_ids.extend(sampled_inds)
                    # if len(valid_ids) < num_ref_imgs//2:
                    #         valid_ids = [frame_id]*(num_ref_imgs//2)
                elif method == 'test_with_adaptive_stride':
                    if frame_id == 0:
                        stride = float(len(img_ids) - 1) / (num_ref_imgs - 1)
                        for i in range(num_ref_imgs):
                            ref_id = round(i * stride)
                            ref_img_ids.append(img_ids[ref_id])
                elif method == 'test_with_fix_stride':
                    if frame_id == 0:  # new video
                        # pad using img itself, and push self in
                        for i in range(frame_range[0], 1):
                            ref_img_ids.append(img_ids[0])
                        for i in range(1, frame_range[1] + 1):
                            # using stride to sample ref img
                            ref_id = min(round(i * stride), len(img_ids) - 1)
                            ref_img_ids.append(img_ids[ref_id])
                    elif frame_id % stride == 0:  # not new video, push new ref img in and pop old ref img out
                        ref_id = min(
                            round(frame_id + frame_range[1] * stride),
                            len(img_ids) - 1)
                        ref_img_ids.append(img_ids[ref_id])
                    img_info['num_left_ref_imgs'] = abs(frame_range[0]) \
                        if isinstance(frame_range, list) else frame_range
                    img_info['frame_stride'] = stride
                else:
                    raise NotImplementedError

                ref_img_infos = []
                for ref_img_id in ref_img_ids:
                    ref_img_info = self.coco.load_imgs([ref_img_id])[0]
                    ref_img_info['filename'] = ref_img_info['file_name']
                    ref_img_infos.append(ref_img_info)
                ref_img_infos = sorted(ref_img_infos, key=lambda i: i['frame_id'])

            if return_key_img:
                return [img_info, *ref_img_infos]
            else:
                return ref_img_infos
