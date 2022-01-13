from pytracking.evaluation import Tracker, get_dataset, trackerlist


#######################################################################
# Reliable Verifier

def dimp_test():
    # origin super dimp exclude advanced localization
    trackers = trackerlist('dimp', 'super_dimp', range(1)) + \
               trackerlist('dimp', 'super_dimp_no_al', range(1))

    # dataset = get_dataset('otb', 'uav', 'nfs', 'tpl')

    dataset = get_dataset('lasot')
    dataset = get_dataset('trackingnet')

    return trackers, dataset


def rvt_test():
    # adaptive threshold
    # trackers = trackerlist('rvt', 'rvt_0', range(1)) + \
    #            trackerlist('rvt', 'rvt_1', range(1)) + \
    #            trackerlist('rvt', 'rvt_2', range(1)) + \
    #            trackerlist('rvt', 'rvt_3', range(1)) + \
    #            trackerlist('rvt', 'rvt_4', range(1)) + \
    #            trackerlist('rvt', 'rvt_5', range(1)) + \
    #            trackerlist('rvt', 'rvt_6', range(1)) + \
    #            trackerlist('rvt', 'rvt_7', range(1)) + \
    #            trackerlist('rvt', 'rvt_8', range(1))

    trackers = trackerlist('rvt', 'rvt', range(1))

    dataset = get_dataset('otb', 'uav', 'nfs', 'lasot')  # , 'lasot'

    # dataset = get_dataset('trackingnet')
    return trackers, dataset


#######################################################################
# Enhancing Discriminative Appearance Model for Visual Tracking

def endimp_test():
    # feature enhancement module
    # trackers = trackerlist('dimp', 'super_dimp', range(1)) + \
    #            trackerlist('endimp', 'endimp', range(1))

    # dimp series with verifier
    trackers = trackerlist('endimp', 'dimp_verifier', range(1)) + \
               trackerlist('endimp', 'prdimp_verifier', range(1)) + \
               trackerlist('endimp', 'super_dimp_verifier', range(1)) + \
               trackerlist('endimp', 'endimp_verifier', range(1))

    dataset = get_dataset('otb', 'uav', 'nfs', 'lasot')

    # dataset = get_dataset('trackingnet')
    return trackers, dataset
#######################################################################
