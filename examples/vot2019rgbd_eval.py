import sys
sys.path.append("./")

from pysot_toolkit.bin.eval import pysot_eval


class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val


def load_eval_config():
    args = DotDict()
    args.dataset_dir = '/home/maxi/datasets/VOT2019RGBD'
    args.dataset = 'VOT2019-RGBD'
    args.tracker_result_dir = '/home/maxi/masterthesis/5_Evaluation/pysot_toolkit/results'
    args.trackers = ['IdentityTrackerRGBD']
    args.vis = True
    args.show_video_level = True
    args.num = 1
    args.list_file = None # None, train_list, val_list or test_list
    return args


if __name__ == '__main__':
    
    # Evaluate tracker if results are already stored in results directory
    pysot_eval(load_eval_config())