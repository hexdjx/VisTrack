import os
import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

from PIL import Image
from pytracking.features.net_wrappers import NetWithBackbone
from pytracking.evaluation import get_dataset
from pytracking.features.preprocessing import numpy_to_torch, torch_to_numpy, sample_patch
from pytracking.utils.target_prob import get_target_probability


def read_image(image_file: str):
    im = cv.imread(image_file)
    return cv.cvtColor(im, cv.COLOR_BGR2RGB)


def im_save(im, save_path):
    im_patch = torch_to_numpy(im)
    img = Image.fromarray(np.uint8(im_patch))
    img.save(save_path)


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


def rvt_net_analysis():
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


def endimp_net_analysis():
    # super_dimp.pth.tar EnDiMPnet.pth.tar
    net = NetWithBackbone(net_path='EnDiMPnet.pth.tar', use_gpu=True)
    net.initialize()
    # special net branch params
    # net_params = net.net.classifier.feature_extractor.parameters()
    # total_params = sum(p.numel() for p in net_params)
    # print("Number of total parameter: %.2fM" % (total_params / 1e6))

    # Visualization of feature map
    img = '/media/hexd6/aede3fa6-c741-4516-afe7-4954b8572ac9/907856427856276E/LaSOT/Test/dog-15/img/00000660.jpg'  # LaSOT/Test/ zebra-17/img/00000001.jpg dog-15/img/00000660.jpg
    state = [180, 97, 98, 178]  # zebra-17: 402,320,261,304 dog-15: 180,97,98,178, Basketball: 198,214,34,81
    pos = torch.Tensor([state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
    target_sz = torch.Tensor([state[3], state[2]])

    sz = torch.Tensor([352, 352])

    search_area = torch.prod(target_sz * 6).item()
    target_scale = math.sqrt(search_area) / sz.prod().sqrt()
    im = read_image(img)
    im = numpy_to_torch(im)

    img_patch, _ = sample_patch(im, pos.round(), target_scale * sz, sz)
    # im_save(img_patch.cpu(), 'dog-15')
    # im_save(img_patch.cpu(), 'zebra-17')
    # im_save(img_patch.cpu(), 'basketball')

    # plt.imshow(torch_to_numpy(img_patch.int()))
    # plt.show()
    with torch.no_grad():
        backbone_feat = net.extract_backbone(img_patch, layers=['layer3'])['layer3']
        clf_feat = net.net.classifier.extract_classification_feat(backbone_feat)
        backbone_feat1 = torch.sum(backbone_feat, dim=1, keepdim=True)
        backbone_feat2 = F.interpolate(backbone_feat1, [352, 352], mode='bilinear', align_corners=False)

        backbone_feat3 = torch_to_numpy(backbone_feat2.cpu())
        # cv.imwrite('/home/hexd6/code/Tracking/VisTrack/pytracking/analysis/utils/results/feat.png', cv.cvtColor(backbone_feat3, cv.COLOR_BGR2RGB))

        plt.imshow(backbone_feat3)
        plt.axis('off')
        plt.axis('equal')
        plt.show()
        # plt.savefig('/home/hexd6/code/Tracking/VisTrack/pytracking/analysis/utils/results/fudimp/feat.png', format='png', dpi=300)


def fudimp_net_analysis():
    # super_dimp.pth.tar FuDiMPnet_ff.pth.tar FuDiMPnet_awff.pth.tar FuDiMPnet_awff_att.pth.tar
    net = NetWithBackbone(net_path='super_dimp.pth.tar', use_gpu=True)
    net.initialize()
    # special net branch params
    # net_params = net.net.classifier.feature_extractor.parameters()
    # total_params = sum(p.numel() for p in net_params)
    # print("Number of total parameter: %.2fM" % (total_params / 1e6))

    # Visualization of feature map
    img = '/media/hexd6/aede3fa6-c741-4516-afe7-4954b8572ac9/907856427856276E/LaSOT/Test/zebra-17/img/00000001.jpg'  # zebra-17/img/00000001.jpg dog-15/img/00000660.jpg
    state = [402, 320, 261, 304]  # zebra-17: 402,320,261,304 dog-15: 180,97,98,178
    pos = torch.Tensor([state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
    target_sz = torch.Tensor([state[3], state[2]])

    sz = torch.Tensor([352, 352])

    search_area = torch.prod(target_sz * 6).item()
    target_scale = math.sqrt(search_area) / sz.prod().sqrt()
    im = read_image(img)
    im = numpy_to_torch(im)

    img_patch, _ = sample_patch(im, pos.round(), target_scale * sz, sz)
    # im_save(img_patch.cpu(), 'dog-15')
    # im_save(img_patch.cpu(), 'zebra-17')

    # plt.imshow(torch_to_numpy(img_patch.int()))
    # plt.show()
    with torch.no_grad():
        backbone_feat = net.extract_backbone(img_patch, layers=['layer3'])['layer3']
        # clf_feat = net.net.classifier.extract_classification_feat(backbone_feat)
        backbone_feat1 = torch.sum(backbone_feat, dim=1, keepdim=True)
        backbone_feat2 = F.interpolate(backbone_feat1, [352, 352], mode='bilinear', align_corners=False)

        backbone_feat3 = torch_to_numpy(backbone_feat2.cpu())
        # cv.imwrite('/home/hexd6/code/Tracking/VisTrack/pytracking/analysis/utils/results/feat.png', cv.cvtColor(backbone_feat3, cv.COLOR_BGR2RGB))

        plt.imshow(backbone_feat3)
        plt.axis('off')
        plt.axis('equal')
        plt.show()
        # plt.savefig('/home/hexd6/code/Tracking/VisTrack/pytracking/analysis/utils/results/fudimp/feat.png', format='png', dpi=300)


def ctp_net_analysis():
    # color target probability

    # Visualization of feature map
    # dataset = get_dataset('otb')
    # dataset = [dataset[s] for s in ['Basketball', 'Bolt']]

    # dataset = get_dataset('nfs')
    # dataset = [dataset[s] for s in ['nfs_dog_1', 'nfs_Skiing_red']]

    dataset = get_dataset('lasot')
    dataset = [dataset[s] for s in ['airplane-1', 'shark-6']]
    for s in dataset:
        sequence = s.name

        net_name = 'super_dimp'  # ProbDiMP super_dimp
        feat_type = 'cls'  # bk cls

        net = NetWithBackbone(net_path=net_name + '.pth.tar', use_gpu=True)
        net.initialize()

        img = s.frames[0]

        gt = s.ground_truth_rect
        state = gt[0]

        pos = torch.Tensor([state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
        target_sz = torch.Tensor([state[3], state[2]])

        sz = torch.Tensor([352, 352])  # image_sample_size=22*16

        search_area = torch.prod(target_sz * 6).item()  # search_area_scale = 6
        target_scale = math.sqrt(search_area) / sz.prod().sqrt()

        img = read_image(img)

        img_patch, _ = sample_patch(numpy_to_torch(img), pos.round(), target_scale * sz, sz)

        base_path = os.path.join(os.path.dirname(__file__), 'results/ctp/img/')
        if not os.path.isdir(base_path):
            os.makedirs(base_path)

        save_path = os.path.join(base_path, '{}.png'.format(sequence))
        if not os.path.isfile(base_path):
            im_save(img_patch, save_path)

        # plt.imshow(torch_to_numpy(img_patch.int()))
        # plt.show()

        with torch.no_grad():
            backbone_feat = net.extract_backbone(img_patch, layers=['layer3'])['layer3']

            if net_name == 'ProbDiMP':
                target_prob = get_target_probability(img_patch.squeeze(), target_sz / target_scale)
                target_prob = F.interpolate(target_prob.reshape(-1, *target_prob.shape[-3:]), size=(22, 22),
                                            mode='bilinear', align_corners=False)

                clf_feat = net.net.classifier.extract_classification_feat(backbone_feat, target_prob.cuda())
            else:
                clf_feat = net.net.classifier.extract_classification_feat(backbone_feat)

            if feat_type == 'bk':
                feat = torch.mean(backbone_feat, dim=1, keepdim=True)
            else:
                feat = torch.mean(clf_feat, dim=1, keepdim=True)

            heatmap = (feat - feat.min()) / (feat.max() - feat.min())

            heatmap = F.interpolate(heatmap, [352, 352], mode='bilinear', align_corners=False)
            heatmap = torch_to_numpy(heatmap.cpu())

            # opencv
            # heatmap[heatmap < 0.5] = 0

            heatmap = np.uint8(255 * heatmap)
            heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_RAINBOW)  # COLORMAP_JET COLORMAP_RAINBOW COLORMAP_HSV

            im = torch_to_numpy(img_patch)

            heat_img = heatmap * 0.5 + im
            heat_img = np.clip(heat_img, 0, 255)

            # plt.imshow(heat_img.astype(int))
            # plt.axis('off')
            # plt.axis('equal')
            # plt.show()

            cv.imwrite(os.path.join(base_path, '{}_{}_{}_feat.png'.format(net_name, sequence, feat_type)),
                       cv.cvtColor(heat_img.astype(np.float32), cv.COLOR_BGR2RGB))


if __name__ == "__main__":
    # rvt_net_analysis()
    # endimp_net_analysis()
    # fudimp_net_analysis()
    ctp_net_analysis()
    print('done!')
