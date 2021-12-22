from pytracking.evaluation import Tracker, get_dataset, trackerlist


#######################################################################
# @ author Xuedong He
# my add oupt
# Learning Object-Uncertainty Policy for Visual Tracking
def oupt_otb():
    # Run OUPT on OTB dataset
    # target memory size parameter test (10~20)

    # trackers = trackerlist('oupt', 'oupt18_10', range(1)) + \
    #            trackerlist('oupt', 'oupt18_12', range(1)) + \
    #            trackerlist('oupt', 'oupt18_13', range(1)) + \
    #            trackerlist('oupt', 'oupt18_14', range(1)) + \
    #            trackerlist('oupt', 'oupt18_15', range(1)) + \
    #            trackerlist('oupt', 'oupt18_16', range(1)) + \
    #            trackerlist('oupt', 'oupt18_17', range(1)) + \
    #            trackerlist('oupt', 'oupt18_18', range(1)) + \
    #            trackerlist('oupt', 'oupt18_19', range(1)) + \
    #            trackerlist('oupt', 'oupt18_20', range(1))

    # whether consider the initial state
    # trackers = trackerlist('oupt', 'oupt18_0', range(1)) + \
    #            trackerlist('oupt', 'oupt18_1', range(1))

    # final test based on DiMP and PrDiMP
    trackers = trackerlist('oupt', 'oupt18', range(1)) + \
               trackerlist('oupt', 'oupt50', range(1)) + \
               trackerlist('oupt', 'proupt18', range(1)) + \
               trackerlist('oupt', 'proupt50', range(1))

    dataset = get_dataset('otb')
    return trackers, dataset


def oupt_nfs_uav():
    # Run OUPT on NFS and UAV datasets
    trackers = trackerlist('oupt', 'proupt50', range(1))

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset


def oupt_lasot():
    # Run OUPT on LaSOT dataset
    trackers = trackerlist('oupt', 'proupt50', range(1))

    dataset = get_dataset('lasot')
    return trackers, dataset


def oupt_trackingnet():
    # Run OUPT on TrackingNet dataset
    trackers = trackerlist('oupt', 'proupt50', range(1))

    dataset = get_dataset('trackingnet')
    return trackers, dataset


#######################################################################
# Variable Scale Learning for Visual Object Tracking
# ATOMS
def atomS_otb():
    # scale factor choice test 1~10 [1.005, 1.05,step=0.005]
    # trackers = trackerlist('vslt', 'ms_ratio_1') + \
    #            trackerlist('vslt', 'ms_ratio_2') + \
    #            trackerlist('vslt', 'ms_ratio_3') + \
    #            trackerlist('vslt', 'ms_ratio_4') + \
    #            trackerlist('vslt', 'ms_ratio_5') + \
    #            trackerlist('vslt', 'ms_ratio_6') + \
    #            trackerlist('vslt', 'ms_ratio_7') + \
    #            trackerlist('vslt', 'ms_ratio_8') + \
    #            trackerlist('vslt', 'ms_ratio_9') + \
    #            trackerlist('vslt', 'ms_ratio_10')

    # scale iter choice test 1~8
    # trackers = trackerlist('vslt', 'ms_var_1') + \
    #            trackerlist('vslt', 'ms_var_2') + \
    #            trackerlist('vslt', 'ms_var_3') + \
    #            trackerlist('vslt', 'ms_var_4') + \
    #            trackerlist('vslt', 'ms_var_5') + \
    #            trackerlist('vslt', 'ms_var_6') + \
    #            trackerlist('vslt', 'ms_var_7') + \
    #            trackerlist('vslt', 'ms_var_8') + \
    #            trackerlist('vslt', 'ms_var_9') + \
    #            trackerlist('vslt', 'ms_var_10')

    trackers = trackerlist('atom', 'default', range(1)) + \
               trackerlist('atom', 'multiscale', range(1)) + \
               trackerlist('atom', 'no_scale', range(1)) + \
               trackerlist('vslt', 'atomS_ratio', range(1)) + \
               trackerlist('vslt', 'atomS_var', range(1)) + \
               trackerlist('vslt', 'atomS_var_ratio', range(1))

    dataset = get_dataset('otb')
    return trackers, dataset


def atomS_tpl_uav():
    # Run ATOMS on Temple Color and UAV datasets
    trackers = trackerlist('atom', 'default', range(1)) + \
               trackerlist('atom', 'multiscale', range(1)) + \
               trackerlist('atom', 'no_scale', range(1)) + \
               trackerlist('vslt', 'atomS_ratio', range(1)) + \
               trackerlist('vslt', 'atomS_var', range(1)) + \
               trackerlist('vslt', 'atomS_var_ratio', range(1))
    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset


def atomS_lasot():
    # Run ATOMS on Temple Color datasets
    trackers = trackerlist('atom', 'default', range(1)) + \
               trackerlist('atom', 'multiscale', range(1)) + \
               trackerlist('atom', 'no_scale', range(1)) + \
               trackerlist('vslt', 'ms_ratio', range(1)) + \
               trackerlist('vslt', 'ms_var', range(1)) + \
               trackerlist('vslt', 'ms_var_ratio', range(1))
    dataset = get_dataset('lasot')
    return trackers, dataset


def ecoS_test():
    trackers = trackerlist('eco', 'default', range(1)) + \
               trackerlist('vslt', 'ecoS_ratio', range(1)) + \
               trackerlist('vslt', 'ecoS_var', range(1)) + \
               trackerlist('vslt', 'ecoS_var_ratio', range(1))
    dataset = get_dataset('otb', 'uav', 'tpl', 'lasot')
    return trackers, dataset

#######################################################################
