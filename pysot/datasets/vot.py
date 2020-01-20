import os
import cv2
import json
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from PIL import Image

from .dataset import Dataset
from .video import Video

class VOTVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        camera_motion: camera motion tag
        illum_change: illum change tag
        motion_change: motion change tag
        size_change: size change
        occlusion: occlusion
    """
    def __init__(self, name, root, video_dir, init_rect, img_names, gt_rect,
            camera_motion, illum_change, motion_change, size_change, occlusion, load_img=False):
        super(VOTVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, None, load_img)
        self.tags= {'all': [1] * len(gt_rect)}
        self.tags['camera_motion'] = camera_motion
        self.tags['illum_change'] = illum_change
        self.tags['motion_change'] = motion_change
        self.tags['size_change'] = size_change
        self.tags['occlusion'] = occlusion

        # TODO
        # if len(self.gt_traj[0]) == 4:
        #     self.gt_traj = [[x[0], x[1], x[0], x[1]+x[3]-1,
        #                     x[0]+x[2]-1, x[1]+x[3]-1, x[0]+x[2]-1, x[1]]
        #                         for x in self.gt_traj]

        # empty tag
        all_tag = [v for k, v in self.tags.items() if len(v) > 0 ]
        self.tags['empty'] = np.all(1 - np.array(all_tag), axis=1).astype(np.int32).tolist()
        # self.tags['empty'] = np.all(1 - np.array(list(self.tags.values())),
        #         axis=1).astype(np.int32).tolist()

        self.tag_names = list(self.tags.keys())
        if not load_img:
            img_name = os.path.join(root, self.img_names[0])
            img = np.array(Image.open(img_name), np.uint8)
            self.width = img.shape[1]
            self.height = img.shape[0]

    def select_tag(self, tag, start=0, end=0):
        if tag == 'empty':
            return self.tags[tag]
        return self.tags[tag][start:end]

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_files = glob(os.path.join(path, name, 'baseline', self.name, '*0*.txt'))
            if len(traj_files) == 15:
                traj_files = traj_files
            else:
                traj_files = traj_files[0:1]
            pred_traj = []
            for traj_file in traj_files:
                with open(traj_file, 'r') as f:
                    traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                    pred_traj.append(traj)
            if store:
                self.pred_trajs[name] = pred_traj
            else:
                return pred_traj

class VOTDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'VOT2018', 'VOT2016'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(VOTDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
            meta_data = json.load(f)

        dataset_root = os.path.join(dataset_root, name)

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = VOTVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['camera_motion'],
                                          meta_data[video]['illum_change'],
                                          meta_data[video]['motion_change'],
                                          meta_data[video]['size_change'],
                                          meta_data[video]['occlusion'],
                                          load_img=load_img)

        self.tags = ['all', 'camera_motion', 'illum_change', 'motion_change',
                     'size_change', 'occlusion', 'empty']


class VOTLTVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, load_img=False):
        super(VOTLTVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, None, load_img)
        self.gt_traj = [[0] if np.isnan(bbox[0]) else bbox
                for bbox in self.gt_traj]
        if not load_img:
            img_name = os.path.join(root, self.img_names[0])
            img = np.array(Image.open(img_name), np.uint8)
            self.width = img.shape[1]
            self.height = img.shape[0]
        self.confidence = {}

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            seq_name = os.path.basename(self.name)
            traj_file = os.path.join(path, name, 'unsupervised',
                    seq_name, seq_name+'_001.txt')
            with open(traj_file, 'r') as f:
                traj = [list(map(float, x.strip().split(',')))
                        for x in f.readlines()]
            if store:
                self.pred_trajs[name] = traj
            confidence_file = os.path.join(path, name, 'unsupervised',
                    seq_name, seq_name+'_001_confidence.value')
            with open(confidence_file, 'r') as f:
                score = [float(x.strip()) for x in f.readlines()[1:]]
                score.insert(0, float('nan'))
            if store:
                self.confidence[name] = score
        return traj, score

class VOTLTDataset(Dataset):
    """
    Args:
        name: dataset name, 'VOT2018-LT'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False, list_file=None):
        super(VOTLTDataset, self).__init__(name, dataset_root)

        #with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
        #    meta_data = json.load(f)
        if list_file is None:
                list_file = os.path.join(dataset_root, 'list.txt')
        else:
                list_file = os.path.join(dataset_root, list_file+'.txt')

        with open(list_file, 'r') as f:
            self.seq_names = f.read().strip().split('\n')
        self.seq_dirs = [os.path.join(dataset_root, s) for s in self.seq_names]
        self.anno_files = [os.path.join(s, 'groundtruth.txt')
                        for s in self.seq_dirs]

        # load videos
        pbar = tqdm(self.seq_dirs, desc='loading '+name+'\n', ncols=100)
        self.videos = {}
        for index, video in enumerate(pbar):
            pbar.set_postfix_str(video)

            # make compliant with framework
            # init_rect
            with open(self.anno_files[index], 'r') as f:
                init_rect = f.readline()

            # gt_rects
            gt_rects = pd.read_csv(self.anno_files[index], sep=',').to_numpy()

            # image files
            img_files = sorted(glob(
                os.path.join(self.seq_dirs[index], 'color/*.jpg')))

            self.videos[video] = VOTLTVideo(video,
                                          dataset_root,
                                          self.seq_dirs[index],
                                          init_rect,
                                          img_files,
                                          gt_rects)


class VOTLTRGBDVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        depth_names: depth image names
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, depth_names, load_img=False):
        super(VOTLTRGBDVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, None, load_img)
        self.depth_names = [os.path.join(root, x) for x in depth_names]
        self.gt_traj = [[0] if np.isnan(bbox[0]) else bbox
                for bbox in self.gt_traj]
        if not load_img:
            img_name = os.path.join(root, self.img_names[0])
            img = np.array(Image.open(img_name), np.uint8)
            self.width = img.shape[1]
            self.height = img.shape[0]
        self.confidence = {}
        self.dataset_name = 'VOTRGBD2019'

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            seq = os.path.basename(self.name)
            traj_file = os.path.join(path, self.dataset_name, name, 
                    'unsupervised', seq, seq+'_001.txt')
            with open(traj_file, 'r') as f:
                traj = [list(map(float, x.strip().split(',')))
                        for x in f.readlines()]
            if store:
                self.pred_trajs[name] = traj

            
            confidence_file = os.path.join(path, self.dataset_name, name, 
                    'unsupervised', seq, seq+'_001_confidence.value')
            try:
                with open(confidence_file, 'r') as f:
                    score = [float(x.strip()) for x in f.readlines()[1:]]
                    score.insert(0, float('nan'))
            except:
                #print('There is no confidence file in the result folder. \
                #    Check %s. All 1s will be used as a replacement.' \
                #    % (confidence_file))
                score = np.ones(len(traj))
            if store:
                self.confidence[name] = score
        return traj, score

class VOTLTRGBDDataset(Dataset):
    """
    Args:
        name: dataset name, 'VOT2019-RGBD'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False, list_file=None):
        super(VOTLTRGBDDataset, self).__init__(name, dataset_root)

        if list_file is None:
            list_file = os.path.join(dataset_root, 'list.txt')
        else:
            list_file = os.path.join(dataset_root, list_file+'.txt')

        with open(list_file, 'r') as f:
            self.seq_names = f.read().strip().split('\n')
        self.seq_dirs = [os.path.join(dataset_root, s) for s in self.seq_names]
        self.anno_files = [os.path.join(s, 'groundtruth.txt')
                           for s in self.seq_dirs]


        # load videos
        pbar = tqdm(self.seq_dirs, desc='loading '+name+'\n', ncols=100)
        self.videos = {}
        for index, video in enumerate(pbar):
            pbar.set_postfix_str(video)

            # make compliant with framework
            # init_rect
            with open(self.anno_files[index], 'r') as f:
                init_rect = f.readline()

            # gt_rects
            gt_rects = pd.read_csv(self.anno_files[index], sep=',').to_numpy()

            # image files
            img_files = sorted(glob(
                os.path.join(self.seq_dirs[index], 'color/*.jpg')))
            depth_files = sorted(glob(
                os.path.join(self.seq_dirs[index], 'depth/*.png')))
            assert len(img_files) == len(depth_files)


            self.videos[video] = VOTLTRGBDVideo(video,
                                          dataset_root,
                                          self.seq_dirs[index],
                                          init_rect,
                                          img_files,
                                          gt_rects,
                                          depth_files)
