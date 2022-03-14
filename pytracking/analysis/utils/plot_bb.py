import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

from pytracking.evaluation import get_dataset


# enhancing DiMP
def plot_endimp(base_path):
    _tracker_disp_colors = {1: (0, 255, 0), 2: (0, 255, 255), 3: (255, 0, 0),
                            4: (0, 0, 255)}

    SuperDiMP = base_path + 'dimp/super_dimp_000'
    EnDiMP = base_path + 'endimp/endimp_000'
    EnDiMP_v = base_path + 'endimp/endimp_verifier_000'

    dataset = get_dataset('otb')

    sequence = ['Skating2_1', 'DragonBaby', 'Diving']
    dataset = [dataset[s] for s in sequence]

    for seq in dataset:
        gt = seq.ground_truth_rect
        SuperDiMP_file = '{}/{}.txt'.format(SuperDiMP, seq.name)
        EnDiMP_file = '{}/{}.txt'.format(EnDiMP, seq.name)
        EnDiMP_v_file = '{}/{}.txt'.format(EnDiMP_v, seq.name)

        SuperDiMP_bb = np.loadtxt(SuperDiMP_file)
        EnDiMP_bb = np.loadtxt(EnDiMP_file)
        EnDiMP_v_bb = np.loadtxt(EnDiMP_v_file)

        output_path = './results/endimp/img/{}/'.format(seq.name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for frame_num, frame_path in enumerate(seq.frames):
            im = cv.imread(seq.frames[frame_num])
            # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            # plt.imshow(im)

            pred_bb = [gt[frame_num], SuperDiMP_bb[frame_num], EnDiMP_bb[frame_num], EnDiMP_v_bb[frame_num]]
            # pred_bb =[gt[frame_num], SuperDiMP[frame_num], EnDiMP[frame_num]]
            for i, s in enumerate(pred_bb, start=1):
                pred_s = s
                tl = tuple(map(int, [pred_s[0], pred_s[1]]))
                br = tuple(map(int, [pred_s[0] + pred_s[2], pred_s[1] + pred_s[3]]))
                col = _tracker_disp_colors[i]
                cv.rectangle(im, tl, br, col, 2)
                # plt.imshow(im)
                # plt.show()

            cv.imwrite('{}/{}.jpg'.format(output_path, frame_num+1), im)


# reliable verifier
def plot_rvt(base_path):
    _tracker_disp_colors = {1: (0, 255, 0), 2: (0, 255, 255), 3: (255, 0, 0), 4: (0, 0, 255)}

    Baseline = base_path + 'dimp/super_dimp_000'
    SuperDiMP = base_path + 'dimp/super_dimp_no_al_000'
    Ours = base_path + 'rvt/rvt_000'

    dataset = get_dataset('otb')

    sequence = ['Basketball', 'Bird2', 'Liquor', 'Soccer']
    dataset = [dataset[s] for s in sequence]

    for seq in dataset:
        Baseline_file = '{}/{}.txt'.format(Baseline, seq.name)
        SuperDiMP_file = '{}/{}.txt'.format(SuperDiMP, seq.name)
        Ours_file = '{}/{}.txt'.format(Ours, seq.name)

        gt = seq.ground_truth_rect
        Baseline_bb = np.loadtxt(Baseline_file)
        SuperDiMP_bb = np.loadtxt(SuperDiMP_file)
        Ours_bb = np.loadtxt(Ours_file)

        output_path = './results/rvt/img/{}/'.format(seq.name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for frame_num, frame_path in enumerate(seq.frames):
            im = cv.imread(seq.frames[frame_num])
            # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            # plt.imshow(im)

            pred_bb = [gt[frame_num], Baseline_bb[frame_num], SuperDiMP_bb[frame_num], Ours_bb[frame_num]]

            for i, s in enumerate(pred_bb, start=1):
                pred_s = s
                tl = tuple(map(int, [pred_s[0], pred_s[1]]))
                br = tuple(map(int, [pred_s[0] + pred_s[2], pred_s[1] + pred_s[3]]))
                col = _tracker_disp_colors[i]
                cv.rectangle(im, tl, br, col, 2)
                # plt.imshow(im)
                # plt.show()

            cv.imwrite('{}/{}.jpg'.format(output_path, frame_num + 1), im)


# object-uncertainty polcy
def plot_oupt(base_path):
    _tracker_disp_colors = {1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255),
                            4: (123, 0, 123), 5: (255, 255, 255), 6: (0, 0, 0),
                            7: (123, 123, 123), 8: (199, 18, 237), 9: (255, 255, 0)}

    OUPT = base_path + 'oupt/proupt50_000'
    ATOM = base_path + 'ATOM_raw/default_000'
    DaSiamRPN = base_path + 'DaSiamRPN/default'
    DiMP = base_path + 'DiMP_raw/dimp50_000'
    ECO = base_path + 'ECO/default_deep'
    KYS = base_path + 'KYS/default_000'
    PrDiMP = base_path + 'DiMP_raw/prdimp50_000'
    SiamRPN = base_path + 'SiamRPN++/default'

    dataset = get_dataset('otb')

    sequence = ['Basketball', 'Soccer', 'Matrix', 'MotorRolling']

    dataset = [dataset[s] for s in sequence]

    for seq in dataset:
        ATOM_file = '{}/{}.txt'.format(ATOM, seq.name)
        DaSiamRPN_file = '{}/{}.txt'.format(DaSiamRPN, seq.name)
        DiMP_file = '{}/{}.txt'.format(DiMP, seq.name)
        ECO_file = '{}/{}.txt'.format(ECO, seq.name)
        KYS_file = '{}/{}.txt'.format(KYS, seq.name)
        PrDiMP_file = '{}/{}.txt'.format(PrDiMP, seq.name)
        SiamRPN_file = '{}/{}.txt'.format(SiamRPN, seq.name)
        OUPT_file = '{}/{}.txt'.format(OUPT, seq.name)

        gt = seq.ground_truth_rect
        ATOM_bb = np.loadtxt(ATOM_file)
        DaSiamRPN_bb = np.loadtxt(DaSiamRPN_file, delimiter=',')
        DiMP_bb = np.loadtxt(DiMP_file)
        ECO_bb = np.loadtxt(ECO_file)
        KYS_bb = np.loadtxt(KYS_file)
        PrDiMP_bb = np.loadtxt(PrDiMP_file)
        SiamRPN_bb = np.loadtxt(SiamRPN_file)
        OUPT_bb = np.loadtxt(OUPT_file)

        output_path = './results/oupt/img/{}/'.format(seq.name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for frame_num, frame_path in enumerate(seq.frames):
            im = cv.imread(seq.frames[frame_num])
            # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            # plt.imshow(im)

            pred_bb = [gt[frame_num], OUPT_bb[frame_num], ATOM_bb[frame_num], DaSiamRPN_bb[frame_num],
                       DiMP_bb[frame_num], ECO_bb[frame_num],
                       KYS_bb[frame_num], PrDiMP_bb[frame_num], SiamRPN_bb[frame_num]]

            for i, s in enumerate(pred_bb, start=1):
                pred_s = s
                tl = tuple(map(int, [pred_s[0], pred_s[1]]))
                br = tuple(map(int, [pred_s[0] + pred_s[2], pred_s[1] + pred_s[3]]))
                col = _tracker_disp_colors[i]
                cv.rectangle(im, tl, br, col, 2)
                # plt.imshow(im)
                # plt.show()

            cv.imwrite('{}/{}.jpg'.format(output_path, frame_num + 1), im)


# variable scale learning
def plot_vslt(base_path):
    _tracker_disp_colors = {1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255),
                            4: (255, 255, 255), 5: (0, 0, 0), 6: (255, 128, 0)
                            }

    default = base_path + 'atom/default_000'
    multiscale = base_path + 'atom/multiscale_000'
    var = base_path + 'vslt/atomS_var_000'
    ratio = base_path + 'vslt/atomS_ratio_000'
    var_ratio = base_path + 'vslt/atomS_var_ratio_000'

    dataset = get_dataset('otb')

    sequence = ['Basketball', 'David', 'Soccer']

    dataset = [dataset[s] for s in sequence]

    for seq in dataset:
        d_file = '{}/{}.txt'.format(default, seq.name)
        ms_file = '{}/{}.txt'.format(multiscale, seq.name)
        v_file = '{}/{}.txt'.format(var, seq.name)
        r_file = '{}/{}.txt'.format(ratio, seq.name)
        vr_file = '{}/{}.txt'.format(var_ratio, seq.name)

        gt = seq.ground_truth_rect
        v_bb = np.loadtxt(v_file)
        r_bb = np.loadtxt(r_file)
        vr_bb = np.loadtxt(vr_file)
        d_bb = np.loadtxt(d_file)
        ms_bb = np.loadtxt(ms_file)

        output_path = './results/vslt/img/{}'.format(seq.name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for frame_num, frame_path in enumerate(seq.frames):
            im = cv.imread(seq.frames[frame_num])
            # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            # plt.imshow(im)

            pred_bb = [gt[frame_num], v_bb[frame_num], r_bb[frame_num], vr_bb[frame_num], d_bb[frame_num],
                       ms_bb[frame_num]]

            for i, s in enumerate(pred_bb, start=1):
                pred_s = s
                tl = tuple(map(int, [pred_s[0], pred_s[1]]))
                br = tuple(map(int, [pred_s[0] + pred_s[2], pred_s[1] + pred_s[3]]))
                col = _tracker_disp_colors[i]
                cv.rectangle(im, tl, br, col, 2)
                # plt.imshow(im)

            cv.imwrite('{}/{}.jpg'.format(output_path, frame_num + 1), im)


if __name__ == "__main__":
    base_path = "D:/Tracking/VisTrack/pytracking/results/tracking_results/"
    # plot_vslt(base_path)
    # plot_oupt(base_path)
    # plot_rvt(base_path)
    plot_endimp(base_path)
    print('done!')
