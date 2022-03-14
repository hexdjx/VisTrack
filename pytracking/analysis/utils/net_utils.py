import os
import cv2 as cv
import numpy as np
import torch
from pytracking.features.net_wrappers import NetWithBackbone
from pytracking.evaluation import get_dataset
from pytracking.features.preprocessing import numpy_to_torch


def read_image(image_file: str):
    im = cv.imread(image_file)
    return cv.cvtColor(im, cv.COLOR_BGR2RGB)

def cnn_paras_count(net):
    """cnn参数量统计"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print("Number of total parameter: %.2fM" % (total_params / 1e6))
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of trainable parameter: %.2fM" % (total_trainable_params / 1e6))

def crop_target(base_path):
    Ours = base_path + 'rvt/rvt_000'

    dataset = get_dataset('otb')

    sequence = ['Basketball', 'Liquor', 'Lemming', 'Soccer']
    dataset = [dataset[s] for s in sequence]
    for seq in dataset:
        Ours_file = '{}/{}.txt'.format(Ours, seq.name)
        Ours_bb = np.loadtxt(Ours_file)

        output_path = './results/rvt/crop_img/{}/'.format(seq.name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for frame_num, frame_path in enumerate(seq.frames):
            im = cv.imread(seq.frames[frame_num])

            pred_bb = Ours_bb[frame_num]

            tl = tuple(map(int, [pred_bb[0], pred_bb[1]]))
            br = tuple(map(int, [pred_bb[0] + pred_bb[2], pred_bb[1] + pred_bb[3]]))

            target_patch = im[tl[1]:br[1], tl[0]:br[0]]

            cv.imwrite('{}/{}.jpg'.format(output_path, frame_num + 1), target_patch)


if __name__ == "__main__":

    # base_path = "D:/Tracking/VisTrack/pytracking/results/tracking_results/"
    # crop_target(base_path)

    verify_net = NetWithBackbone(net_path='Verify_Net.pth.tar', use_gpu=True)
    verify_net.initialize()

    # cnn_paras_count(verify_net.net)

    img_path = 'D:/Tracking/VisTrack/pytracking/analysis/utils/results/rvt/crop_img/Liquor/'  # Basketball, Liquor
    img1 = read_image(img_path + '1.jpg')  #
    img_resized1 = cv.resize(img1, (128, 128))
    image_tensor1 = numpy_to_torch(img_resized1)

    with torch.no_grad():
        ref_target_embedding = verify_net.extract_backbone(image_tensor1)

    for i in range(5):  # 725, 1336
        i = i + 1
        img2 = read_image(img_path + str(i) + '.jpg')
        img_resized2 = cv.resize(img2, (128, 128))
        image_tensor2 = numpy_to_torch(img_resized2)
        with torch.no_grad():
            test_target_embedding = verify_net.extract_backbone(image_tensor2)
        sim = torch.cosine_similarity(ref_target_embedding, test_target_embedding)
        print('the cos similarity of the {} frame is {}'.format(i, sim))

    print('done!')
