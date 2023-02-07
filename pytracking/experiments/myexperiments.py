from pytracking.evaluation import get_dataset, trackerlist


# @author Xuedong He

# -- EnDiMP-- ######################################################################
# Enhancing Discriminative Appearance Model

def endimp_test():
    # feature enhancement module
    trackers = trackerlist('dimp', 'super_dimp') + \
               trackerlist('endimp', 'endimp')

    # dimp series with verifier
    # trackers = trackerlist('endimp', 'dimp_verifier') + \
    #            trackerlist('endimp', 'prdimp_verifier') + \
    #            trackerlist('endimp', 'super_dimp_verifier') + \
    #            trackerlist('endimp', 'endimp_verifier')

    dataset = get_dataset('otb', 'uav', 'nfs', 'lasot')

    # dataset = get_dataset('trackingnet')
    return trackers, dataset


# --RVT-- ######################################################################
# Reliable Verifier

def dimp_test():
    # origin super dimp exclude advanced localization
    trackers = trackerlist('dimp', 'super_dimp') + \
               trackerlist('dimp', 'super_dimp_no_al')

    dataset = get_dataset('otb', 'uav', 'nfs')

    # dataset = get_dataset('lasot')
    # dataset = get_dataset('trackingnet')

    return trackers, dataset


def rvt_test():
    # adaptive threshold
    # trackers = trackerlist('rvt', 'rvt_0') + \
    #            trackerlist('rvt', 'rvt_1') + \
    #            trackerlist('rvt', 'rvt_2') + \
    #            trackerlist('rvt', 'rvt_3') + \
    #            trackerlist('rvt', 'rvt_4') + \
    #            trackerlist('rvt', 'rvt_5') + \
    #            trackerlist('rvt', 'rvt_6') + \
    #            trackerlist('rvt', 'rvt_7') + \
    #            trackerlist('rvt', 'rvt_8')

    # gauss sampling
    # trackers = trackerlist('rvt', 'rvt_gauss')

    # Portability test
    # trackers = trackerlist('rvt', 'dimp50_rv') + \
    #            trackerlist('rvt', 'prdimp50_rv') + \
    #            trackerlist('dimp', 'dimp50_no_al') + \
    #            trackerlist('dimp', 'prdimp50_no_al')

    trackers = trackerlist('rvt', 'rvt')

    dataset = get_dataset('otb', 'uav', 'nfs')

    # dataset = get_dataset('lasot')
    # dataset = get_dataset('trackingnet')
    return trackers, dataset


# --OUPT-- #######################################################################
# Learning Object-Uncertainty Policy for Visual Tracking
def oupt_otb():
    # Run OUPT on OTB dataset
    # target memory size parameter test (10~20)

    # trackers = trackerlist('oupt', 'oupt18_10') + \
    #            trackerlist('oupt', 'oupt18_12') + \
    #            trackerlist('oupt', 'oupt18_13') + \
    #            trackerlist('oupt', 'oupt18_14') + \
    #            trackerlist('oupt', 'oupt18_15') + \
    #            trackerlist('oupt', 'oupt18_16') + \
    #            trackerlist('oupt', 'oupt18_17') + \
    #            trackerlist('oupt', 'oupt18_18') + \
    #            trackerlist('oupt', 'oupt18_19') + \
    #            trackerlist('oupt', 'oupt18_20')

    # whether consider the initial state
    # trackers = trackerlist('oupt', 'oupt18_0') + \
    #            trackerlist('oupt', 'oupt18_1')

    # final test based on DiMP and PrDiMP
    trackers = trackerlist('oupt', 'oupt18') + \
               trackerlist('oupt', 'oupt50') + \
               trackerlist('oupt', 'proupt18') + \
               trackerlist('oupt', 'proupt50')

    dataset = get_dataset('otb')
    return trackers, dataset


def oupt_nfs_uav():
    # Run OUPT on NFS and UAV datasets
    trackers = trackerlist('oupt', 'proupt50')

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset


def oupt_lasot():
    # Run OUPT on LaSOT dataset
    trackers = trackerlist('oupt', 'oupt50') + \
               trackerlist('oupt', 'proupt50')

    dataset = get_dataset('lasot')
    return trackers, dataset


def oupt_trackingnet():
    # Run OUPT on TrackingNet dataset
    trackers = trackerlist('oupt', 'proupt50')

    dataset = get_dataset('trackingnet')
    return trackers, dataset


# --VSLT-- ######################################################################
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

    trackers = trackerlist('atom', 'default') + \
               trackerlist('atom', 'multiscale') + \
               trackerlist('atom', 'no_scale') + \
               trackerlist('vslt', 'atomS_ratio') + \
               trackerlist('vslt', 'atomS_var') + \
               trackerlist('vslt', 'atomS_var_ratio')

    dataset = get_dataset('otb')
    return trackers, dataset


def atomS_tpl_uav_lasot():
    # Run ATOMS on Temple Color and UAV datasets
    trackers = trackerlist('atom', 'default') + \
               trackerlist('atom', 'multiscale') + \
               trackerlist('atom', 'no_scale') + \
               trackerlist('vslt', 'atomS_ratio') + \
               trackerlist('vslt', 'atomS_var') + \
               trackerlist('vslt', 'atomS_var_ratio')
    dataset = get_dataset('tpl', 'uav', 'lasot')
    return trackers, dataset


def ecoS_test():
    trackers = trackerlist('eco', 'default') + \
               trackerlist('vslt', 'ecoS_ratio') + \
               trackerlist('vslt', 'ecoS_var') + \
               trackerlist('vslt', 'ecoS_var_ratio')
    dataset = get_dataset('otb', 'uav', 'tpl', 'lasot')
    return trackers, dataset
################################################################################
