# Generating Results on Datasets

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [7, 4]

from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist


# trackers = []

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

# trackers.extend(trackerlist('TrDiMP', 'TrDiMP', range(0, 3), 'TrDiMP'))  # lasot
# trackers.extend(trackerlist('TrSiam', 'TrSiam', range(0, 3), 'TrSiam'))  # lasot
# trackers.extend(trackerlist('TransT', 'N2', None, 'TransT_N2'))  # otb, uav, nfs, lasot
# trackers.extend(trackerlist('TransT', 'N4', None, 'TransT_N4'))  # otb, uav, nfs, lasot
# trackers.extend(trackerlist('STARK', 'stark_s', None, 'STARK_s'))  # otb, uav, nfs
# trackers.extend(trackerlist('STARK', 'stark_st', None, 'STARK_st'))  # otb, uav, nfs
# trackers.extend(trackerlist('STARK', 'stark_st_R101', None, 'STARK'))  # otb, uav, nfs

# trackers.extend(trackerlist('KYS', 'default', range(0, 5), 'KYS'))  # otb nfs
# trackers.extend(trackerlist('ATOM', 'default', range(0, 5), 'ATOM'))  # otb uav nfs lasot
# trackers.extend(trackerlist('DiMP', 'dimp18', range(0, 5), 'DiMP18'))  # otb uav nfs lasot
# trackers.extend(trackerlist('DiMP', 'dimp50', range(0, 5), 'DiMP50'))  # otb uav nfs lasot
# trackers.extend(trackerlist('DiMP', 'prdimp18', range(0, 5), 'PrDiMP18'))  # otb uav nfs lasot
# trackers.extend(trackerlist('DiMP', 'prdimp50', range(0, 5), 'PrDiMP50'))  # otb uav nfs lasot
# trackers.extend(trackerlist('DiMP', 'super_dimp', range(0, 5), 'SuperDiMP'))  # otb uav nfs lasot
# trackers.extend(trackerlist('DiMP', 'super_dimp_simple', range(0, 5), 'SuperDiMPSimple'))  # otb uav nfs lasot lasotextensionsubset
# trackers.extend(trackerlist('KeepTrack', 'default', range(0, 5), 'KeepTrack'))  # otb uav nfs lasot lasotextensionsubset
# trackers.extend(trackerlist('ToMP', 'tomp50', None, 'ToMP50'))  # otb uav nfs lasot lasotextensionsubset
# trackers.extend(trackerlist('ToMP', 'tomp101', None, 'ToMP101'))  # otb uav nfs lasot lasotextensionsubset
#############################################################################################################

# --VSLT-- #####################################################################
# Variable Scale Learning
def analysis_VSLT():
    trackers = []
    # scale factor choice test 1~10 [1.005, 1.05,step=0.005]
    # trackers.extend(trackerlist('vslt', 'ratio_1', None, 'ATOM_ratio_1'))
    # trackers.extend(trackerlist('vslt', 'ratio_2', None, 'ATOM_ratio_2'))
    # trackers.extend(trackerlist('vslt', 'ratio_3', None, 'ATOM_ratio_3'))
    # trackers.extend(trackerlist('vslt', 'ratio_4', None, 'ATOM_ratio_4'))  # choose 4
    # trackers.extend(trackerlist('vslt', 'ratio_5', None, 'ATOM_ratio_5'))
    # trackers.extend(trackerlist('vslt', 'ratio_6', None, 'ATOM_ratio_6'))
    # trackers.extend(trackerlist('vslt', 'ratio_7', None, 'ATOM_ratio_7'))
    # trackers.extend(trackerlist('vslt', 'ratio_8', None, 'ATOM_ratio_8'))
    # trackers.extend(trackerlist('vslt', 'ratio_9', None, 'ATOM_ratio_9'))
    # trackers.extend(trackerlist('vslt', 'ratio_10', None, 'ATOM_ratio_10'))

    # scale iter choice test 1~10
    # trackers.extend(trackerlist('vslt', 'var_1', None, 'ATOM_var_1'))
    # trackers.extend(trackerlist('vslt', 'var_2', None, 'ATOM_var_2'))  # choose 2
    # trackers.extend(trackerlist('vslt', 'var_3', None, 'ATOM_var_3'))
    # trackers.extend(trackerlist('vslt', 'var_4', None, 'ATOM_var_4'))
    # trackers.extend(trackerlist('vslt', 'var_5', None, 'ATOM_var_5'))
    # trackers.extend(trackerlist('vslt', 'var_6', None, 'ATOM_var_6'))
    # trackers.extend(trackerlist('vslt', 'var_7', None, 'ATOM_var_7'))
    # trackers.extend(trackerlist('vslt', 'var_8', None, 'ATOM_var_8'))
    # trackers.extend(trackerlist('vslt', 'var_9', None, 'ATOM_var_9'))
    # trackers.extend(trackerlist('vslt', 'var_10', None, 'ATOM_var_10'))

    # ECO
    # trackers.extend(trackerlist('eco', 'default', None, 'ECO'))

    # ATOM
    # trackers.extend(trackerlist('atom', 'default', None, 'ATOM'))
    # trackers.extend(trackerlist('atom', 'multiscale', None, 'ATOM_multiscale'))
    # trackers.extend(trackerlist('atom', 'no_scale', None, 'ATOM_no_scale'))

    # trackers.extend(trackerlist('CCOT', 'default', None, 'CCOT'))  # otb uav nfs tpl
    # trackers.extend(trackerlist('eco', 'default_hc', None, 'ECO-HC'))  # otb uav tpl
    # trackers.extend(trackerlist('UPDT', 'default', range(0, 10), 'UPDT'))  # otb uav nfs tpl
    # trackers.extend(trackerlist('MDNet', 'default', None, 'MDNet'))  # otb lasot(all data 1400) nfs
    # trackers.extend(trackerlist('eco', 'default', None, 'ECO'))  # otb uav nfs tpl
    # trackers.extend(trackerlist('DaSiamRPN', 'default', None, 'DaSiamRPN'))  # otb uav
    # trackers.extend(trackerlist('SiamRPN++', 'default', None, 'SiamRPN++'))  # otb uav lasot

    # trackers.extend(trackerlist('KYS', 'default', range(0, 5), 'KYS'))  # otb nfs
    # trackers.extend(trackerlist('ATOM', 'default', range(0, 5), 'ATOM'))  # otb uav nfs lasot
    # trackers.extend(trackerlist('dimp', 'dimp18', range(0, 5), 'DiMP18'))  # otb uav nfs lasot
    # trackers.extend(trackerlist('DiMP', 'dimp50', range(0, 5), 'DiMP50'))  # otb uav nfs lasot
    # trackers.extend(trackerlist('dimp', 'prdimp18', range(0, 5), 'PrDiMP18'))  # otb uav nfs lasot
    # trackers.extend(trackerlist('DiMP', 'prdimp50', range(0, 5), 'PrDiMP50'))  # otb uav nfs lasot

    # VSLT
    # trackers.extend(trackerlist('vslt', 'ecoS_ratio', None, 'ECO_ratio'))
    trackers.extend(trackerlist('vslt', 'ecoS_var', None, 'Ours(ECO_var)'))
    # trackers.extend(trackerlist('vslt', 'ecoS_var_ratio', None, 'ECO_var_ratio'))
    # trackers.extend(trackerlist('vslt', 'atomS_ratio', None, 'Ours(ATOM_ratio)'))
    trackers.extend(trackerlist('vslt', 'atomS_var', None, 'Ours(ATOM_var)'))
    # trackers.extend(trackerlist('vslt', 'atomS_var_ratio', None, 'Ours(ATOM_var_ratio)'))
    return trackers


# --OUPT-- #####################################################################
# Object Uncertainty Policy
def analysis_OUPT():
    trackers = []
    # OTB datasets results
    # oupt target set num test
    # trackers.extend(trackerlist('oupt', 'oupt18_10', None, 'DiMP18_oup_10'))  # otb
    # trackers.extend(trackerlist('oupt', 'oupt18_11', None, 'DiMP18_oup_11'))  # otb
    # trackers.extend(trackerlist('oupt', 'oupt18_12', None, 'DiMP18_oup_12'))  # otb
    # trackers.extend(trackerlist('oupt', 'oupt18_13', None, 'DiMP18_oup_13'))  # otb
    # trackers.extend(trackerlist('oupt', 'oupt18_14', None, 'DiMP18_oup_14'))  # otb
    # trackers.extend(trackerlist('oupt', 'oupt18_15', None, 'DiMP18_oup_15'))  # otb
    # trackers.extend(trackerlist('oupt', 'oupt18_16', None, 'DiMP18_oup_16'))  # otb
    # trackers.extend(trackerlist('oupt', 'oupt18_17', None, 'DiMP18_oup_17'))  # otb this is better!
    # trackers.extend(trackerlist('oupt', 'oupt18_18', None, 'DiMP18_oup_18'))  # otb
    # trackers.extend(trackerlist('oupt', 'oupt18_19', None, 'DiMP18_oup_19'))  # otb
    # trackers.extend(trackerlist('oupt', 'oupt18_20', None, 'DiMP18_oup_20'))  # otb

    # whether consider the initial state
    # trackers.extend(trackerlist('oupt', 'oupt18_0', None, 'ours(DiMP18_oup0)'))  # otb
    # trackers.extend(trackerlist('oupt', 'oupt18_1', None, 'ours(DiMP18_oup1)'))  # otb

    # DiMP and PrDiMP based OTB results
    # trackers.extend(trackerlist('oupt', 'oupt18', None, 'ours(DiMP18_oup)'))
    # trackers.extend(trackerlist('oupt', 'oupt50', None, 'ours(DiMP50_oup)'))
    # trackers.extend(trackerlist('oupt', 'proupt18', None, 'ours(PrDiMP18_oup)'))
    # trackers.extend(trackerlist('oupt', 'proupt50', None, 'ours(PrDiMP50_oup)'))

    # OTB, UAV, NFS, and LaSOT results
    trackers.extend(trackerlist('oupt', 'proupt50', None, 'OUPT'))
    return trackers


# --RVT-- #####################################################################
# Reliable Verifier
def analysis_RVT():
    trackers = []
    # target embedding network
    # adaptive threshold
    # trackers.extend(trackerlist('rvt', 'rvt_0', None, 'rvt_0'))  # otb, nfs, uav
    # trackers.extend(trackerlist('rvt', 'rvt_1', None, 'rvt_1'))  # otb, nfs, uav # Optional
    # trackers.extend(trackerlist('rvt', 'rvt_2', None, 'rvt_2'))  # otb, nfs, uav
    # trackers.extend(trackerlist('rvt', 'rvt_3', None, 'rvt_3'))  # otb, nfs, uav
    # trackers.extend(trackerlist('rvt', 'rvt_4', None, 'rvt_4'))  # otb, nfs, uav
    # trackers.extend(trackerlist('rvt', 'rvt_5', None, 'rvt_5'))  # otb, nfs, uav
    # trackers.extend(trackerlist('rvt', 'rvt_6', None, 'rvt_6'))  # otb, nfs, uav
    # trackers.extend(trackerlist('rvt', 'rvt_7', None, 'rvt_7'))  # otb, nfs, uav
    # trackers.extend(trackerlist('rvt', 'rvt_8', None, 'rvt_8'))  # otb, nfs, uav # Optional

    # gauss sampling
    # trackers.extend(trackerlist('rvt', 'rvt', None, 'RVT'))  # otb, nfs, uav
    # trackers.extend(trackerlist('rvt', 'rvt_gauss', None, 'RVT_gauss'))  # otb, nfs, uav

    #  Portability test
    # trackers.extend(trackerlist('DiMP', 'dimp50', None, 'DiMP'))  # otb uav nfs
    # trackers.extend(trackerlist('DiMP', 'prdimp50', None, 'PrDiMP'))  # otb uav nfs
    # trackers.extend(trackerlist('dimp', 'dimp50_no_al', None, 'DiMP_noAL'))  # otb uav nfs
    # trackers.extend(trackerlist('dimp', 'prdimp50_no_al', None, 'PrDiMP_noAL'))  # otb uav nfs
    # trackers.extend(trackerlist('rvt', 'dimp50_rv', None, 'DiMP_RV'))  # otb, nfs, uav
    # trackers.extend(trackerlist('rvt', 'prdimp50_rv', None, 'PrDiMP_RV'))  # otb, nfs, uav

    trackers.extend(trackerlist('dimp', 'super_dimp', None, 'SuperDimp'))  # otb, uav, nfs, lasot
    # trackers.extend(trackerlist('dimp', 'super_dimp_no_al', None, 'SuperDimp_no_al'))  # otb, uav, nfs, lasot
    trackers.extend(trackerlist('rvt', 'rvt', None, 'Ours(RVT)'))  # otb, nfs, uav lasot

    trackers.extend(trackerlist('CCOT', 'default', None, 'CCOT'))  # otb uav nfs tpl
    trackers.extend(trackerlist('UPDT', 'default', range(0, 10), 'UPDT'))  # otb uav nfs tpl
    trackers.extend(trackerlist('MDNet', 'default', None, 'MDNet'))  # otb lasot(all data 1400) nfs
    trackers.extend(trackerlist('eco', 'default_deep', None, 'ECO'))  # otb uav nfs tpl
    # trackers.extend(trackerlist('DaSiamRPN', 'default', None, 'DaSiamRPN'))  # otb uav
    # trackers.extend(trackerlist('SiamRPN++', 'default', None, 'SiamRPN++'))  # otb uav lasot

    trackers.extend(trackerlist('KYS', 'default', range(0, 5), 'KYS'))  # otb nfs
    trackers.extend(trackerlist('atom', 'default', range(0, 5), 'ATOM'))  # otb uav nfs lasot
    # trackers.extend(trackerlist('DiMP', 'dimp18', range(0, 5), 'DiMP18'))  # otb uav nfs lasot
    trackers.extend(trackerlist('dimp', 'dimp50', range(0, 1), 'DiMP'))  # otb uav nfs lasot
    # trackers.extend(trackerlist('DiMP', 'prdimp18', range(0, 5), 'PrDiMP18'))  # otb uav nfs lasot
    trackers.extend(trackerlist('dimp', 'prdimp50', range(0, 1), 'PrDiMP'))  # otb uav nfs lasot
    # trackers.extend(trackerlist('dimp', 'super_dimp', range(0, 5), 'SuperDiMP'))  # otb uav nfs lasot
    return trackers


# --EnDiMP-- #####################################################################
# Enhancing Discriminative Model Prediction
def analysis_EnDiMP():
    trackers = []
    # DiMP and PrDiMP results from origin paper
    # trackers.extend(trackerlist('DiMP', 'dimp50', None, 'DiMP'))  # otb uav nfs lasot # 5
    # trackers.extend(trackerlist('DiMP', 'prdimp50', None, 'PrDiMP'))  # otb uav nfs lasot # 5

    # SuperDiMP as a baseline, experiment on my device
    trackers.extend(trackerlist('dimp', 'super_dimp_no_al', None, 'SuperDiMP_no_HML'))  # otb, uav, nfs, lasot, tpl
    trackers.extend(trackerlist('dimp', 'super_dimp', None, 'SuperDiMP'))  # otb, uav, nfs, lasot, tpl

    # feature enhancement module
    # trackers.extend(trackerlist('endimp', 'endimp', None, 'EnDiMP'))  # otb, nfs, uav, lasot

    # verifier module
    # trackers.extend(trackerlist('endimp', 'dimp_verifier', None, 'DiMP_verifier'))  # otb, nfs, uav, lasot
    # trackers.extend(trackerlist('endimp', 'prdimp_verifier', None, 'PrDiMP_verifier'))  # otb, nfs, uav, lasot
    # trackers.extend(trackerlist('endimp', 'superdimp_verifier', None, 'SuperDiMP_verifier'))  # otb, nfs, uav, lasot
    # trackers.extend(trackerlist('endimp', 'endimp_verifier', None, 'EnDiMP_verifier'))  # otb, nfs, uav, lasot

    return trackers


# --FuDiMP-- #########################################################################
# adaptive feature fusion with attention and target-uncertain detection
def analysis_FuDiMP():
    trackers = []
    # trackers.extend(trackerlist('dimp', 'super_dimp', range(0, 5), 'SuperDiMP'))  # otb uav nfs lasot
    # trackers.extend(trackerlist('dimp', 'dimp50', range(0, 5), 'DiMP50'))  # otb uav nfs lasot

    # trackers.extend(trackerlist('fudimp', 'fudimp_ff', None, 'FuDiMP_ff'))  # otb, nfs, uav, lasot
    # trackers.extend(trackerlist('fudimp', 'fudimp_awff', None, 'FuDiMP_awff'))  # otb, nfs, uav, lasot
    # trackers.extend(trackerlist('fudimp', 'fudimp_awff_att', None, 'FuDiMP_awff_att'))  # otb, nfs, uav, lasot

    # additional score index
    # trackers.extend(trackerlist('fudimp', 'fudimp_psme_05', None, 'FuDiMPpsme05'))  # lasot
    # trackers.extend(trackerlist('fudimp', 'fudimp_apce_05', None, 'FuDiMPapce05'))  # lasot
    # trackers.extend(trackerlist('fudimp', 'fudimp_psme', None, 'FuDiMPpsme06'))  # otb, nfs, uav, lasot
    # trackers.extend(trackerlist('fudimp', 'fudimp_apce', None, 'FuDiMPapce06'))  # otb, nfs, uav, lasot
    # trackers.extend(trackerlist('fudimp', 'fudimp_psme_07', None, 'FuDiMPpsme07'))  # lasot
    # trackers.extend(trackerlist('fudimp', 'fudimp_apce_07', None, 'FuDiMPapce07'))  # lasot

    # ------------------------------------------------------------------------------
    # trackers.extend(trackerlist('fudimp', 'dimp50_psme05', None, 'DiMP_psme05'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'dimp50_apce05', None, 'DiMP_apce05'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'dimp50_psr05', None, 'DiMP_psr05'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'dimp50_psme06', None, 'DiMP_psme06'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'dimp50_apce06', None, 'DiMP_apce06'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'dimp50_psr06', None, 'DiMP_psr06'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'dimp50_psme07', None, 'DiMP_psme07'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'dimp50_apce07', None, 'DiMP_apce07'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'dimp50_psr07', None, 'DiMP_psr07'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'dimp50_psme08', None, 'DiMP_psme08'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'dimp50_apce08', None, 'DiMP_apce08'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'dimp50_psr08', None, 'DiMP_psr08'))  # otb, nfs, uav

    # trackers.extend(trackerlist('fudimp', 'super_dimp_psme_05', None, 'SuperDiMP_psme_05'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'super_dimp_apce_05', None, 'SuperDiMP_apce_05'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'super_dimp_psr_05', None, 'SuperDiMP_psr_05'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'super_dimp_psme_06', None, 'SuperDiMP_psme_06'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'super_dimp_apce_06', None, 'SuperDiMP_apce_06'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'super_dimp_psr_06', None, 'SuperDiMP_psr_06'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'super_dimp_psme_07', None, 'SuperDiMP_psme_07'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'super_dimp_apce_07', None, 'SuperDiMP_apce_07'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'super_dimp_psr_07', None, 'SuperDiMP_psr_07'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'super_dimp_psme_08', None, 'SuperDiMP_psme_08'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'super_dimp_apce_08', None, 'SuperDiMP_apce_08'))  # otb, nfs, uav
    # trackers.extend(trackerlist('fudimp', 'super_dimp_psr_08', None, 'SuperDiMP_psr_08'))  # otb, nfs, uav
    # ------------------------------------------------------------------------------

    # trackers.extend(trackerlist('fudimp', 'fudimp_psme', None, 'FuDiMP_psme'))  # otb, nfs, uav, lasot
    # trackers.extend(trackerlist('fudimp', 'fudimp_apce', None, 'FuDiMP_apce'))  # otb, nfs, uav, lasot
    # trackers.extend(trackerlist('fudimp', 'fudimp_psr', None, 'FuDiMP_psr'))  # otb, nfs, uav, lasot

    trackers.extend(trackerlist('fudimp', 'fudimp_awff_att', None, 'FuDiMP'))  # otb, nfs, uav, lasot
    trackers.extend(trackerlist('fudimp_mu', 'default', None, 'FuDiMP_MU'))  # otb, nfs, uav, lasot

    # tomp test
    trackers.extend(trackerlist('tomp', 'fu_tomp50', None, 'FuToMP'))  # otb, nfs, uav, lasot

    return trackers


# --CAT--- #########################################################################
# Color Attention Tracking with Score Matching
def analysis_CAT():
    trackers = []
    # trackers.extend(trackerlist('cat', 'default', None, 'default'))
    # trackers.extend(trackerlist('cat', 'default_64', None, 'default_64'))

    # trackers.extend(trackerlist('cat', 'default_mean', None, 'default_mean'))
    # trackers.extend(trackerlist('cat', 'default_no_mlp', None, 'default_no_mlp'))

    # trackers.extend(trackerlist('cat', 'default_init_prob', None, 'default'))
    # trackers.extend(trackerlist('cat', 'default_replace_prob', None, 'default_replace_prob'))
    # trackers.extend(trackerlist('cat', 'default_update_prob', None, 'default_update_prob'))

    # trackers.extend(trackerlist('cat', 'ctp', None, 'ctp'))  # CTP+SM
    # trackers.extend(trackerlist('cat', 'ctp_update', None, 'ctp_update'))  # CTPu+SM #default_prob_match

    # trackers.extend(trackerlist('DiMP', 'super_dimp', range(0, 5), 'SuperDiMP'))  # nfs  #(1, 3)
    # trackers.extend(trackerlist('cat', 'default', None, 'SuperDiMP+CTP'))
    # trackers.extend(trackerlist('cat', 'default_update_prob', None, 'SuperDiMP+CTPu'))
    trackers.extend(trackerlist('cat', 'ctp_update', None, 'CAT'))
    # trackers.extend(trackerlist('cat', 'ctp', None, 'SuperDiMP+CTP+SM'))

    # trackers.extend(trackerlist('ToMP', 'tomp50', None, 'ToMP50'))
    # trackers.extend(trackerlist('cat', 'ca_tomp_match', None, 'CAT++'))

    return trackers


# --Prompt Tracking--###################################################################
def analysis_PrompT():
    trackers = []

    # SuperDiMP as baseline
    # trackers.extend(trackerlist('prompt', 'prompt_prob_default', None, 'prompt_prob_default'))  # otb nfs lasot
    # trackers.extend(trackerlist('prompt', 'prompt_prob_init', None, 'prompt_prob_init'))  # otb nfs lasot
    # trackers.extend(trackerlist('prompt', 'prompt_prob_update', None, 'prompt_prob_update'))  # otb nfs lasot
    # trackers.extend(trackerlist('prompt', 'prompt_prob_replace', None, 'prompt_prob_replace'))  # otb nfs lasot

    # trackers.extend(trackerlist('prompt', 'prompt_prob_nohan', None, 'prompt_prob_nohan'))  # otb nfs lasot
    # trackers.extend(trackerlist('prompt', 'prompt_prob_nohan_update', None, 'prompt_prob_nohan_update'))  # otb nfs lasot

    # distance
    # trackers.extend(trackerlist('prompt', 'prompt_prob_fixed_dis', None, 'prompt_prob_fixed_dis'))  # otb nfs lasot
    # trackers.extend(trackerlist('prompt', 'prompt_prob_dynamic_dis', None, 'prompt_prob_dynamic_dis'))  # otb nfs lasot
    # trackers.extend(trackerlist('prompt', 'prompt_prob_update_dis', None, 'prompt_prob_dis'))  # otb nfs lasot  #initial

    # trackers.extend(trackerlist('prompt', 'prompt_prob_dis3_noaverage', None, 'prompt_prob_dis3_noaverage'))  # otb  nfs lasot
    # trackers.extend(trackerlist('prompt', 'prompt_prob_dis3', None, 'prompt_prob_dis3'))  # otb  nfs lasot

    # trackers.extend(trackerlist('prompt', 'prompt_prob_dis5', None, 'prompt_prob_dis5'))  # otb  nfs lasot

    # trackers.extend(trackerlist('prompt', 'prompt_prob_update', None, 'SuperDiMP+CPE'))  # otb nfs lasot
    # trackers.extend(trackerlist('prompt', 'prompt_prob_update_dis', None, 'SuperDiMP+CPE+PCM'))  # otb nfs lasot

    # down sampling bilinear interp
    # trackers.extend(trackerlist('prompt', 'prompt_prob_update', None, 'CPE_bi_mlp'))

    # patch embedding
    # trackers.extend(trackerlist('prompt', 'prompt_prob_add', None, 'CPE_pe_add'))
    # trackers.extend(trackerlist('prompt', 'prompt_prob_cat', None, 'CPE_pe_cat'))
    # trackers.extend(trackerlist('prompt', 'prompt_prob_att', None, 'CPE_pe_att'))
    # trackers.extend(trackerlist('prompt', 'prompt_prob_mlp', None, 'CPE_pe_mlp'))

    # tomp50 as baseline
    # trackers.extend(trackerlist('prompt', 'pro_tomp50_init', None, 'pro_tomp50_init'))
    # trackers.extend(trackerlist('prompt', 'pro_tomp50_default', None, 'pro_tomp50_default'))
    # trackers.extend(trackerlist('prompt', 'pro_tomp50_update', None, 'pro_tomp50_update'))
    # trackers.extend(trackerlist('prompt', 'pro_tomp50_replace', None, 'pro_tomp50_replace'))

    # trackers.extend(trackerlist('prompt', 'pro_tomp50_dis', None, 'pro_tomp50_dis')) #update
    # trackers.extend(trackerlist('prompt', 'pro_tomp50_dis3', None, 'pro_tomp50_dis3')) #update
    # trackers.extend(trackerlist('prompt', 'pro_tomp50_dis5', None, 'pro_tomp50_dis5')) #update

    # trackers.extend(trackerlist('prompt', 'pro_tomp50_update', None, 'ToMP50+CPE'))
    # trackers.extend(trackerlist('prompt', 'pro_tomp50_dis', None, 'ToMP50+CPE+PCM'))

    return trackers

# -----------------------------------------------------------------------------------------------------------

# --published trackers------------------------
# trackers = analysis_VSLT()...
# trackers = analysis_OUPT()
# trackers = analysis_RVT()
# trackers = analysis_EnDiMP()
# trackers = analysis_FuDiMP()
# trackers = analysis_CAT()
trackers = analysis_PrompT()

# --plot results-- ##############################################################################
# dataset = get_dataset('nfs')
# plot_results(trackers, dataset, 'NFS', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
#
# dataset = get_dataset('otb')
# plot_results(trackers, dataset, 'OTB', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
#
# dataset = get_dataset('tnl2k')
# plot_results(trackers, dataset, 'TNL2K', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
#
# dataset = get_dataset('lasot')
# plot_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'norm_prec', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

# dataset = get_dataset('lasot_extension_subset')
# plot_results(trackers, dataset, 'LaSOTExtSub', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

# dataset = get_dataset('uav')
# plot_results(trackers, dataset, 'UAV', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

# dataset = get_dataset('tpl')  # tpl tpl_nootb
# plot_results(trackers, dataset, 'TPL', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

##################################################################################

# --print tables-- ##############################################################################
dataset = get_dataset('lasot')
print_results(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))

dataset = get_dataset('nfs')
# print_results(trackers, dataset, 'NFS', merge_results=True, plot_types=('success', 'prec'))
#
dataset = get_dataset('otb')
print_results(trackers, dataset, 'OTB', merge_results=True, plot_types=('success', 'prec'))

dataset = get_dataset('tnl2k')
print_results(trackers, dataset, 'TNL2K', merge_results=True, plot_types=('success', 'prec'))

# # dataset = get_dataset('uav')
# # print_results(trackers, dataset, 'UAV', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
#
dataset = get_dataset('lasot_extension_subset')
print_results(trackers, dataset, 'LaSOTExtSub', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
#################################################################################################

# --print per-sequence results-- ##############################################################
# Print per sequence results for all sequences
# filter_criteria = None
# dataset = get_dataset('lasot')
# print_per_sequence_results(trackers, dataset, 'LaSOT', merge_results=True, filter_criteria=filter_criteria,
#                            force_evaluation=False)
######################################################################################
