import argparse
import os
import sys
import logging
import yaml
import random
import numpy as np
import SimpleITK as sitk

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append('./src/')

from seq2seq.dataloader.lungmr import Dataset_lungmr
from seq2seq.models.seq2seq import Generator
from tsf.models.tsf_seq2seq import TSF_seq2seq


def test(args, net, seq2seq, device, dir_results):
    test_data = Dataset_lungmr(args, mode='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    c_in = args['seq2seq']['c_in']
    c_s = args['seq2seq']['c_s']
    valid_size = args['train']['valid_size']

    with open(os.path.join(dir_results, 'result_metrics.csv'), 'w') as f:
        f.write('name,n_src,tgt,psnr,ssim,lpips\n')
        
    with torch.no_grad():
        net.eval()
        seq2seq.eval()
        with torch.no_grad():
            for batch in test_loader:
                img_vibe = batch['vibe']
                img_rmax = batch['rmax']
                img_pbf = batch['pbf']
                img_postvibe = batch['postvibe']
                img_prm = batch['prm']
                flags = [i[0] for i in batch['flag']]
                path = batch['path'][0][0]
                name = os.path.basename(path)
                if len(flags)==0:
                    raise Exception('No available sequence in {}!'.format(path))

                d = img_vibe.shape[3]

                _, nw, nh = valid_size
                nd = 128
                b, d, t, c, w, h = img_vibe.shape
                rd = (d - nd) // 2 if d > nd else 0  # random.randint(0, d-nd-1) if d>nd else 0
                rw = (w - nw) // 2 if w > nw else 0  # random.randint(0, w-nw-1) if w>nw else 0
                rh = (h - nh) // 2 if h > nh else 0  # random.randint(0, h-nh-1) if h>nh else 0

                inputs = [
                    img_vibe[:, rd:rd + nd, :, rw:rw + nw, rh:rh + nh].to(device=device, dtype=torch.float32),
                    img_rmax[:, rd:rd + nd, :, rw:rw + nw, rh:rh + nh].to(device=device, dtype=torch.float32),
                    img_pbf[:, rd:rd + nd, :, rw:rw + nw, rh:rh + nh].to(device=device, dtype=torch.float32),
                    img_postvibe[:, rd:rd + nd, :, rw:rw + nw, rh:rh + nh].to(device=device, dtype=torch.float32),
                ]
                output = img_prm[:, rd:rd + nd, :, rw:rw + nw, rh:rh + nh].to(device=device, dtype=torch.float32)

                tgt_flags = [i for i in range(5) if i not in flags]
                n_src = len(flags)
                mask = (img_vibe >-1).to(device=device, dtype=torch.float32)

                for tgt in tgt_flags:
                    indices =[1,2]
                    flags_3 = [flags[i] for i in indices]
                    source_imgs = [inputs[src] for src in flags_3]
                    print(len(source_imgs))

                    target_img = output
                    target_code = torch.from_numpy(np.array([1 if i==tgt else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)
                    output_target = net(seq2seq, source_imgs, flags_3, target_code, n_outseq=3,skip_attn=True)
                    class_indices = torch.argmax(output_target, dim=1).unsqueeze(dim=1).to(device=device,
                                                                                           dtype=torch.float32)
                    class_indices = torch.mul(class_indices, mask) + mask
                    target_img = target_img + mask
                    tgtimg = target_img[0,0,0,:]
                    preimg = class_indices[0,0,0,:]
                    dir_pred = os.path.join(dir_results, 'predictunet')
                    os.makedirs(dir_pred, exist_ok=True)

                    sitk.WriteImage(sitk.GetImageFromArray(tgtimg.cpu()), os.path.join(dir_pred, '{}_tgt_{}.nii.gz'.format(name, tgt)))
                    sitk.WriteImage(sitk.GetImageFromArray(preimg.cpu()), os.path.join(dir_pred, '{}_pred_{}.nii.gz'.format(name, tgt)))

def get_args():
    parser = argparse.ArgumentParser(description='Test seq2seq model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', dest='config', type=str, default='./config/config.yaml',
                        help='config file')
    parser.add_argument('-l', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-m', '--seq2seq', dest='seq2seq', type=str, default=None,
                        help='Load seq2seq model from a .pth file')
    parser.add_argument('-d', '--device', dest='device', type=str, default='cpu',
                        help='cuda or cpu')
    parser.add_argument('-o', '--output', dest='output', type=str, default=None,
                        help='output')
    
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dir_output = ''
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    seq2seq = Generator(config)
    seq2seq.to(device=device)
    pretrained_weight = config['seq2seq']['pretrain']
    load_dict = torch.load(pretrained_weight, map_location=device)
    seq2seq.load_state_dict(load_dict)

    net = TSF_seq2seq(config)
    net.to(device=device)

    if args.load:
        load_dict = torch.load(args.load, map_location=device)
        net.load_state_dict(load_dict)
        print('[*] Load model from', args.load)
    
    if args.seq2seq:
        load_dict = torch.load(args.seq2seq, map_location=device)
        seq2seq.load_state_dict(load_dict)
        print('[*] Load seq2seq model from', args.seq2seq)
    try:
        test(
            config,
            net=net,
            seq2seq=seq2seq,
            device=device,
            dir_results=dir_output,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)