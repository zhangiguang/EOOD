# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
import mmrotate  # noqa: F401


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='/home/iguang/Desktop/img', default='/media/iguang/5AB64D2CB64D0A49/DOTA/val/VOC2007/JPEGImages')
    # parser.add_argument('--config', help='Config file', default='/home/iguang/detection/base/mmrotate/configs/eood/eood_r50_fpn_1x_dota_le90_2heads.py')
    # parser.add_argument('--checkpoint', help='Checkpoint file',default='/media/iguang/5AB64D2CB64D0A49/DOTA/EE_noaug/eood/3heads/6988.pth')
    parser.add_argument('--config', help='Config file', default='/home/iguang/detection/base/mmrotate/configs/eood/eood_onehead_r50_fpn_1x_dota_le90.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',default='/media/iguang/5AB64D2CB64D0A49/DOTA/EE_noaug/eood/1head_nfl/epoch_24.pth')
    # parser.add_argument('--config', help='Config file', default='/home/iguang/detection/base/mmrotate/configs/eood/eood_r50_fpn_1x_dota_le90_1head.py')
    # parser.add_argument('--checkpoint', help='Checkpoint file',default='/media/iguang/5AB64D2CB64D0A49/DOTA/EE_noaug/eood/1head_retinanet/epoch_24.pth')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.01, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    dir = os.listdir(args.img)
    length = len(dir)
    for i in range(0, length, 10):
        img_path = os.path.join(args.img, dir[i])
        result = inference_detector(model, img_path)
    # with open('/home/iguang/Desktop/img/p27581111.txt',"w") as f:
    #     for i in result[6]:
    #         f.write(str(i[-1]) + "\n")
    # show the results
    # show_result_pyplot(
    #     model,
    #     args.img,
    #     result,
    #     palette=args.palette,
    #     score_thr=args.score_thr,
    #     out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
