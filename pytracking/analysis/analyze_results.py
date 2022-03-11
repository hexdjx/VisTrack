# Generating Results on Datasets

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [7, 4]  # 14，8

from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist

trackers = []
#################################################################################################
# @author Xuedong He

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
trackers.extend(trackerlist('DiMP', 'super_dimp', range(0, 5), 'SuperDiMP'))  # otb uav nfs lasot
# trackers.extend(trackerlist('DiMP', 'super_dimp_simple', range(0, 5), 'SuperDiMPSimple'))  # otb uav nfs lasot lasotextensionsubset
# trackers.extend(trackerlist('KeepTrack', 'default', range(0, 5), 'KeepTrack'))  # otb uav nfs lasot lasotextensionsubset
#############################################################################################################

# --RVT-- #####################################################################
# Reliable Verifier

# target embedding network
# adaptive threshold
# trackers.extend(trackerlist('rvt', 'rvt_0', range(0, 1), 'rvt_0'))  # otb, nfs, uav
# trackers.extend(trackerlist('rvt', 'rvt_1', range(0, 1), 'rvt_1'))  # otb, nfs, uav # Optional
# trackers.extend(trackerlist('rvt', 'rvt_2', range(0, 1), 'rvt_2'))  # otb, nfs, uav
# trackers.extend(trackerlist('rvt', 'rvt_3', range(0, 1), 'rvt_3'))  # otb, nfs, uav
# trackers.extend(trackerlist('rvt', 'rvt_4', range(0, 1), 'rvt_4'))  # otb, nfs, uav
# trackers.extend(trackerlist('rvt', 'rvt_5', range(0, 1), 'rvt_5'))  # otb, nfs, uav
# trackers.extend(trackerlist('rvt', 'rvt_6', range(0, 1), 'rvt_6'))  # otb, nfs, uav
# trackers.extend(trackerlist('rvt', 'rvt_7', range(0, 1), 'rvt_7'))  # otb, nfs, uav
# trackers.extend(trackerlist('rvt', 'rvt_8', range(0, 1), 'rvt_8'))  # otb, nfs, uav # Optional

# gauss sampling
# trackers.extend(trackerlist('rvt', 'rvt', range(0, 1), 'RVT'))  # otb, nfs, uav
# trackers.extend(trackerlist('rvt', 'rvt_gauss', range(0, 1), 'RVT_gauss'))  # otb, nfs, uav

#  Portability test
# trackers.extend(trackerlist('DiMP', 'dimp50', range(0, 1), 'DiMP'))  # otb uav nfs
# trackers.extend(trackerlist('DiMP', 'prdimp50', range(0, 1), 'PrDiMP'))  # otb uav nfs
# trackers.extend(trackerlist('dimp', 'dimp50_no_al', range(0, 1), 'DiMP_noAL'))  # otb uav nfs
# trackers.extend(trackerlist('dimp', 'prdimp50_no_al', range(0, 1), 'PrDiMP_noAL'))  # otb uav nfs
# trackers.extend(trackerlist('rvt', 'dimp50_rv', range(0, 1), 'DiMP_RV'))  # otb, nfs, uav
# trackers.extend(trackerlist('rvt', 'prdimp50_rv', range(0, 1), 'PrDiMP_RV'))  # otb, nfs, uav

# trackers.extend(trackerlist('dimp', 'super_dimp', range(0, 1), 'SuperDimp'))  # otb, uav, nfs, lasot
# trackers.extend(trackerlist('dimp', 'super_dimp_no_al', range(0, 1), 'SuperDimp_no_al'))  # otb, uav, nfs, lasot
# trackers.extend(trackerlist('rvt', 'rvt', range(0, 1), 'RVT'))  # otb, nfs, uav lasot


# --OUPT-- #####################################################################
# Object Uncertainty Policy

# OTB datasets results
# oupt target set num test
# trackers.extend(trackerlist('oupt', 'oupt18_10', range(0, 1), 'DiMP18_oup_10'))  # otb
# trackers.extend(trackerlist('oupt', 'oupt18_11', range(0, 1), 'DiMP18_oup_11'))  # otb
# trackers.extend(trackerlist('oupt', 'oupt18_12', range(0, 1), 'DiMP18_oup_12'))  # otb
# trackers.extend(trackerlist('oupt', 'oupt18_13', range(0, 1), 'DiMP18_oup_13'))  # otb
# trackers.extend(trackerlist('oupt', 'oupt18_14', range(0, 1), 'DiMP18_oup_14'))  # otb
# trackers.extend(trackerlist('oupt', 'oupt18_15', range(0, 1), 'DiMP18_oup_15'))  # otb
# trackers.extend(trackerlist('oupt', 'oupt18_16', range(0, 1), 'DiMP18_oup_16'))  # otb
# trackers.extend(trackerlist('oupt', 'oupt18_17', range(0, 1), 'DiMP18_oup_17'))  # otb this is better!
# trackers.extend(trackerlist('oupt', 'oupt18_18', range(0, 1), 'DiMP18_oup_18'))  # otb
# trackers.extend(trackerlist('oupt', 'oupt18_19', range(0, 1), 'DiMP18_oup_19'))  # otb
# trackers.extend(trackerlist('oupt', 'oupt18_20', range(0, 1), 'DiMP18_oup_20'))  # otb

# whether consider the initial state
# trackers.extend(trackerlist('oupt', 'oupt18_0', range(0, 1), 'ours(DiMP18_oup0)'))  # otb
# trackers.extend(trackerlist('oupt', 'oupt18_1', range(0, 1), 'ours(DiMP18_oup1)'))  # otb

# DiMP and PrDiMP based OTB results
# trackers.extend(trackerlist('oupt', 'oupt18', range(0, 1), 'ours(DiMP18_oup)'))
# trackers.extend(trackerlist('oupt', 'oupt50', range(0, 1), 'ours(DiMP50_oup)'))
# trackers.extend(trackerlist('oupt', 'proupt18', range(0, 1), 'ours(PrDiMP18_oup)'))
# trackers.extend(trackerlist('oupt', 'proupt50', range(0, 1), 'ours(PrDiMP50_oup)'))

# OTB, UAV, NFS, and LaSOT results
# trackers.extend(trackerlist('oupt', 'proupt50', range(0, 1), 'OUPT'))


# --VSLT-- #####################################################################
# Variable Scale Learning

# scale factor choice test 1~10 [1.005, 1.05,step=0.005]
# trackers.extend(trackerlist('vslt', 'ratio_1', None, 'ATOMS_ratio_1'))
# trackers.extend(trackerlist('vslt', 'ratio_2', None, 'ATOMS_ratio_2'))
# trackers.extend(trackerlist('vslt', 'ratio_3', None, 'ATOMS_ratio_3'))
# trackers.extend(trackerlist('vslt', 'ratio_4', None, 'ATOMS_ratio_4'))  # choose 4
# trackers.extend(trackerlist('vslt', 'ratio_5', None, 'ATOMS_ratio_5'))
# trackers.extend(trackerlist('vslt', 'ratio_6', None, 'ATOMS_ratio_6'))
# trackers.extend(trackerlist('vslt', 'ratio_7', None, 'ATOMS_ratio_7'))
# trackers.extend(trackerlist('vslt', 'ratio_8', None, 'ATOMS_ratio_8'))
# trackers.extend(trackerlist('vslt', 'ratio_9', None, 'ATOMS_ratio_9'))
# trackers.extend(trackerlist('vslt', 'ratio_10', None, 'ATOMS_ratio_10'))

# scale iter choice test 1~10
# trackers.extend(trackerlist('vslt', 'var_1', None, 'ATOMS_var_1'))
# trackers.extend(trackerlist('vslt', 'var_2', None, 'ATOMS_var_2'))  # choose 2
# trackers.extend(trackerlist('vslt', 'var_3', None, 'ATOMS_var_3'))
# trackers.extend(trackerlist('vslt', 'var_4', None, 'ATOMS_var_4'))
# trackers.extend(trackerlist('vslt', 'var_5', None, 'ATOMS_var_5'))
# trackers.extend(trackerlist('vslt', 'var_6', None, 'ATOMS_var_6'))
# trackers.extend(trackerlist('vslt', 'var_7', None, 'ATOMS_var_7'))
# trackers.extend(trackerlist('vslt', 'var_8', None, 'ATOMS_var_8'))
# trackers.extend(trackerlist('vslt', 'var_9', None, 'ATOMS_var_9'))
# trackers.extend(trackerlist('vslt', 'var_10', None, 'ATOMS_var_10'))

# ECO
# trackers.extend(trackerlist('eco', 'default', range(0, 1), 'ECO'))

# ATOM
# trackers.extend(trackerlist('atom', 'default', range(0, 1), 'ATOM'))
# trackers.extend(trackerlist('atom', 'multiscale', range(0, 1), 'ATOM_multiscale'))
# trackers.extend(trackerlist('atom', 'no_scale', range(0, 1), 'ATOM_no_scale'))

# VSLT
# trackers.extend(trackerlist('vslt', 'ecoS_ratio', range(0, 1), 'ECOS_ratio'))
# trackers.extend(trackerlist('vslt', 'ecoS_var', range(0, 1), 'ECOS_var'))
# trackers.extend(trackerlist('vslt', 'ecoS_var_ratio', range(0, 1), 'ECOS_var_ratio'))
# trackers.extend(trackerlist('vslt', 'atomS_ratio', range(0, 1), 'ATOMS_ratio'))
# trackers.extend(trackerlist('vslt', 'atomS_var', range(0, 1), 'Ours(ATOMS_var)'))
# trackers.extend(trackerlist('vslt', 'atomS_var_ratio', range(0, 1), 'ATOMS_var_ratio'))

###################################################################################################

# --plot results-- ##############################################################################
# dataset = get_dataset('otb')
# plot_results(trackers, dataset, 'OTB', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
#
# dataset = get_dataset('nfs')
# plot_results(trackers, dataset, 'NFS', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
#
# dataset = get_dataset('uav')
# plot_results(trackers, dataset, 'UAV', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

dataset = get_dataset('lasot')
plot_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'prec'),
             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

# dataset = get_dataset('lasot_extension_subset')
# plot_results(trackers, dataset, 'LaSOTExtSub', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

# dataset = get_dataset('tpl')  # tpl tpl_nootb
# plot_results(trackers, dataset, 'TPL', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

##################################################################################

# --print tables-- ##############################################################################
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

# --Filtered per-sequence results-- ##############################################################
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
