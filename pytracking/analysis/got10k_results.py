import os
from got10k.experiments import ExperimentUAV123

if __name__ == '__main__':
    # linux
    # data_path = '/media/hexdjx/907856427856276E/'  # linux
    # result_path = '/home/hexd6/code/Tracking/VisTrack/pytracking/results/'
    # report_path = '/home/hexd6/code/Tracking/VisTrack/pytracking/results/reports'

    # windows
    data_path = 'D:/Tracking/Datasets/'  # windows
    result_path = 'D:/Tracking/VisTrack/pytracking/results'
    report_path = 'D:/Tracking/VisTrack/pytracking/results/reports'
    # run experiments on UAV
    experiment_uav = ExperimentUAV123(root_dir=os.path.join(data_path, 'UAV123'),
                                      result_dir=result_path,
                                      report_dir=report_path)

    # report performance
    # experiment_uav.report(['SuperDiMP', 'SuperDiMP_verifier', 'EnDiMP', 'EnDiMP_verifier'])
    # experiment_uav.report(['FuDiMP_awff_att', 'FuDiMP_awff', 'FuDiMP_ff'])
    # experiment_uav.report(['FuDiMP_psme', 'FuDiMP_apce', 'FuDiMP_psr'])
    experiment_uav.report(['CTP'])
