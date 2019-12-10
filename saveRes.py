from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import argparse
import os

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def cv2_demo(net, transform):
    def predict(frame,roi,x1,y1):
        height, width = roi.shape[:2]
        x = torch.from_numpy(transform(roi)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0)).cuda()
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0])+x1, int(pt[1])+y1),
                              (int(pt[2])+x1, int(pt[3])+y1),
                              COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0])+x1, int(pt[1])+y1),
                            FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    # start video stream thread, allow buffer to fill
    img_path = "/data/shw/zhixing/cut"
    save_res_path = "/data/shw/zhixing/cut_res"
    test_img = os.listdir(img_path)

    left =1311
    top =736
    right =3613
    bottom =2080

    for m in test_img:
        frame = cv2.imread(os.path.join(img_path, m))
        print(m)
        cut = frame[top:bottom, left:right].copy()
        frame = cv2.GaussianBlur(frame, (41, 41), 0)
        frame[top:bottom, left:right] = cut
        roi = frame[top:bottom,left:right]
        res = predict(frame,roi,left,top)
        cv2.imwrite(os.path.join(save_res_path,m),res)



if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    parser = argparse.ArgumentParser(description="Weather Classification Project.")
    parser.add_argument('--config', default='./configs/sample.yaml')
    args = parser.parse_args()

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    if opt.DEVICE:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.DEVICE_ID
        num_gpus = len(opt.DEVICE_ID.split(','))

    if torch.cuda.is_available():
        if opt.DEVICE:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not opt.DEVICE:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                  "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    net = build_ssd('test', 300, 5)    # initialize SSD
    net.load_state_dict(torch.load("/home/shw/code/ZhiXing/checkpoint/Exp-1/ZX.pth"))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
    cv2_demo(net.eval(), transform)
    # stop the timer and display FPS information

    # cleanup
    cv2.destroyAllWindows()

