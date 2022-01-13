# Generating Results on Datasets

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [7, 4]  # 14，8

from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist

trackers = []
#######################################################
# @author Xuedong He
# my add

# origin paper results
# otb:100, uav:123, tpl:129, nfs:100, lasot:280
# trackers.extend(trackerlist('CCOT', 'default', None, 'CCOT'))  # otb uav nfs tpl
# trackers.extend(trackerlist('ECO', 'default_hc', None, 'ECO-HC'))  # otb uav tpl
# trackers.extend(trackerlist('UPDT', 'default', range(0, 10), 'UPDT'))  # otb uav nfs tpl
# trackers.extend(trackerlist('MDNet', 'default', None, 'MDNet'))  # otb lasot(all data 1400) nfs
# trackers.extend(trackerlist('ECO', 'default_deep', None, 'ECO'))  # otb uav nfs tpl
# trackers.extend(trackerlist('DaSiamRPN', 'default', None, 'DaSiamRPN'))  # otb uav
# trackers.extend(trackerlist('SiamRPN++', 'default', None, 'SiamRPN++'))  # otb uav lasot
# trackers.extend(trackerlist('KYS', 'default', range(0, 5), 'KYS'))  # otb nfs

# trackers.extend(trackerlist('ATOM', 'default', range(0, 5), 'ATOM'))  # otb uav nfs lasot
# trackers.extend(trackerlist('DiMP', 'dimp18', range(0, 5), 'DiMP18'))  # otb uav nfs lasot
# trackers.extend(trackerlist('DiMP', 'dimp50', range(0, 5), 'DiMP50'))  # otb uav nfs lasot
# trackers.extend(trackerlist('DiMP', 'prdimp18', range(0, 5), 'PrDiMP18'))  # otb uav nfs lasot
# trackers.extend(trackerlist('DiMP', 'prdimp50', range(0, 5), 'PrDiMP50'))  # otb uav nfs lasot
# trackers.extend(trackerlist('DiMP', 'super_dimp', range(0, 5), 'SuperDiMP'))  # otb uav nfs lasot
# trackers.extend(trackerlist('DiMP', 'super_dimp_simple', range(0, 5), 'SuperDiMPSimple'))  # otb uav nfs lasot lasotextensionsubset
# trackers.extend(trackerlist('KeepTrack', 'default', range(0, 5), 'KeepTrack'))  # otb uav nfs lasot lasotextensionsubset

# --my add--#####################################################################
# Reliable Verifier

# trackers.extend(trackerlist('dimp', 'super_dimp', range(0, 1), 'SuperDimp'))  # otb, uav, nfs, lasot, tpl # 1
# trackers.extend(trackerlist('dimp', 'super_dimp_no_al', range(0, 1), 'SuperDimp_no_al'))  # otb, uav, nfs, lasot, tpl # 1

# target embedding network
# adaptive threshold
# trackers.extend(trackerlist('rvt', 'rvt_0', range(0, 1), 'rvt_0'))  # otb, nfs, uav # 1
# trackers.extend(trackerlist('rvt', 'rvt_1', range(0, 1), 'rvt_1'))  # otb, nfs, uav # 1 better
# trackers.extend(trackerlist('rvt', 'rvt_2', range(0, 1), 'rvt_2'))  # otb, nfs, uav # 1
# trackers.extend(trackerlist('rvt', 'rvt_3', range(0, 1), 'rvt_3'))  # otb, nfs, uav # 1
# trackers.extend(trackerlist('rvt', 'rvt_4', range(0, 1), 'rvt_4'))  # otb, nfs, uav # 1
# trackers.extend(trackerlist('rvt', 'rvt_5', range(0, 1), 'rvt_5'))  # otb, nfs, uav # 1
# trackers.extend(trackerlist('rvt', 'rvt_6', range(0, 1), 'rvt_6'))  # otb, nfs, uav # 1
# trackers.extend(trackerlist('rvt', 'rvt_7', range(0, 1), 'rvt_7'))  # otb, nfs, uav # 1
# trackers.extend(trackerlist('rvt', 'rvt_8', range(0, 1), 'rvt_8'))  # otb, nfs, uav # 1

# trackers.extend(trackerlist('rvt', 'rvt', range(0, 1), 'RVT'))  # otb, nfs, uav

# --my add--#####################################################################
# Enhanced DiMP

# DiMP and PrDiMP results from origin paper
# trackers.extend(trackerlist('DiMP', 'dimp50', range(0, 1), 'DiMP'))  # otb uav nfs lasot # 5
# trackers.extend(trackerlist('DiMP', 'prdimp50', range(0, 1), 'PrDiMP'))  # otb uav nfs lasot # 5

# SuperDiMP as a baseline
# trackers.extend(trackerlist('dimp', 'super_dimp_no_al', range(0, 1), 'SuperDiMP_no_HML'))  # otb, uav, nfs, lasot, tpl # 1
# trackers.extend(trackerlist('dimp', 'super_dimp', range(0, 1), 'SuperDiMP'))  # otb, uav, nfs, lasot, tpl # 1

# feature enhancement module
# trackers.extend(trackerlist('endimp', 'endimp', range(0, 1), 'EnDiMP'))  # otb, nfs, uav, lasot # 1

# verifier module
# trackers.extend(trackerlist('endimp', 'endimp_verifier', range(0, 1), 'EnDiMP_verifier'))  # otb, nfs, uav, lasot # 1

# trackers.extend(trackerlist('endimp', 'dimp_verifier', range(0, 1), 'DiMP_verifier'))  # otb, nfs, uav, lasot # 1
# trackers.extend(trackerlist('endimp', 'prdimp_verifier', range(0, 1), 'PrDiMP_verifier'))  # otb, nfs, uav, lasot # 1
# trackers.extend(trackerlist('endimp', 'superdimp_verifier', range(0, 1), 'SuperDiMP_verifier'))  # otb, nfs, uav, lasot # 1
###################################################################################################
trackers.extend(trackerlist('DiMP', 'super_dimp', range(0, 5), 'SuperDiMP'))  # otb uav nfs lasot
trackers.extend(trackerlist('fudimp', 'dimp_awff', range(0, 1), 'SuperDiMP_awff'))  # otb, nfs, uav
trackers.extend(trackerlist('fudimp', 'default1', range(0, 1), 'FuDiMP1'))  # otb, nfs, uav

# --plot results--##############################################################################
dataset = get_dataset('otb')
plot_results(trackers, dataset, 'OTB', merge_results=True, plot_types=('success', 'prec'),
             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

# dataset = get_dataset('uav')
# plot_results(trackers, dataset, 'UAV', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

dataset = get_dataset('nfs')
plot_results(trackers, dataset, 'NFS', merge_results=True, plot_types=('success', 'prec'),
             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

# dataset = get_dataset('lasot')
# plot_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

# dataset = get_dataset('lasot_extension_subset')
# plot_results(trackers, dataset, 'LaSOTExtSub', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

# dataset = get_dataset('tpl')  # tpl tpl_nootb
# plot_results(trackers, dataset, 'TPL', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

##################################################################################

# --print tables--##############################################################################
# dataset = get_dataset('otb')
# print_results(trackers, dataset, 'OTB', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))

# dataset = get_dataset('nfs')
# print_results(trackers, dataset, 'NFS', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))

# dataset = get_dataset('uav')
# print_results(trackers, dataset, 'UAV', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))

# dataset = get_dataset('otb', 'nfs', 'uav')
# print_results(trackers, dataset, 'OTB+NFS+UAV', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))

# dataset = get_dataset('lasot')
# print_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
##################################################################################################

# --Filtered per-sequence results--##############################################################
# Print per sequence results for sequences where all trackers fail, i.e. all trackers have average overlap in percentage of less than 10.0
# filter_criteria = {'mode': 'ao_max', 'threshold': 10.0}
# dataset = get_dataset('otb', 'nfs', 'uav')
# print_per_sequence_results(trackers, dataset, 'OTB+NFS+UAV', merge_results=True, filter_criteria=filter_criteria,
#                            force_evaluation=False)

# Print per sequence results for sequences where at least one tracker fails, i.e. a tracker has average overlap in percentage of less than 10.0
# filter_criteria = {'mode': 'ao_min', 'threshold': 10.0}
# dataset = get_dataset('otb', 'nfs', 'uav')
# print_per_sequence_results(trackers, dataset, 'OTB+NFS+UAV', merge_results=True, filter_criteria=filter_criteria,
#                            force_evaluation=False)

# Print per sequence results for sequences where the trackers have differing behavior.
# i.e. average overlap in percentage for different trackers on a sequence differ by at least 40.0
# filter_criteria = {'mode': 'delta_ao', 'threshold': 40.0}
# dataset = get_dataset('otb', 'nfs', 'uav')
# print_per_sequence_results(trackers, dataset, 'OTB+NFS+UAV', merge_results=True, filter_criteria=filter_criteria,
#                            force_evaluation=False)

# Print per sequence results for all sequences
# filter_criteria = None
# dataset = get_dataset('otb', 'nfs', 'uav')
# print_per_sequence_results(trackers, dataset, 'OTB+NFS+UAV', merge_results=True, filter_criteria=filter_criteria,
#                            force_evaluation=False)
######################################################################################
