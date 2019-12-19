from pysot.datasets import VOTLTRGBDDataset
from got_toolkit.got10k.experiments import ExperimentVOT

# Import trackers
from got_toolkit.got10k.trackers import IdentityTrackerRGBD


dataset_dir = '/home/maxi/datasets/VOT2019RGBD'


if __name__ == "__main__":
    # setup tracker
    tracker = IdentityTrackerRGBD()

    # run experiment
    experiment = ExperimentVOT(
        root_dir=dataset_dir,
        version='RGBD2019',
        read_image=True,
        list_file=None, # None, train_list, test_list, val_list
        experiments='unsupervised', 
        result_dir='results',
        report_dir='reports'
        )
    
    experiment.run(tracker, visualize=False)