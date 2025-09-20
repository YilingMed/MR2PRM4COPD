import argparse
import os
import logging
import  yaml
import random
import tqdm
from tqdm import *
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append('./src/')

from seq2seq.utils import poly_lr, Recorder, Plotter, save_grid_images
from seq2seq.losses import PerceptualLoss
from seq2seq.dataloader.lungmr import Dataset_lungmr
from seq2seq.models.seq2seq import Generator
from tsf.models.tsf_seq2seq import TSF_seq2seq
from tsf.train.losses import prm_losses
import matplotlib.pyplot as plt
from tsf.models.Gaussianblur import gaussian_pooling
output_path = ''

plt.switch_backend('agg')
import pdb
def train(args, net, seq2seq, device):
    train_data = Dataset_lungmr(args, mode='train')
    valid_data = Dataset_lungmr(args, mode='valid')

    n_train = len(train_data)
    n_valid = len(valid_data)

    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    c_in = args['seq2seq']['c_in']
    c_s = args['seq2seq']['c_s']
    epochs = args['train']['epochs']
    lr = np.float32(args['train']['lr'])
    dir_visualize = args['train']['vis']
    dir_checkpoint = args['train']['ckpt']
    rep_step = args['train']['rep_step']
    crop_size = args['train']['crop_size']
    valid_size = args['train']['valid_size']
    lambda_rec = args['train']['lambda_rec']
    lambda_per = args['train']['lambda_per']
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {1}
        Learning rate:   {lr}
        Training size:   {n_train}
        Valid size:      {n_valid}
        Device:          {device.type}
    ''')
    seq2seq_param = list(seq2seq.decoder.parameters())+list(seq2seq.style_fc.parameters())#+list(seq2seq.dec_convlstm.parameters())
    optimizer = torch.optim.Adam(list(net.parameters())+seq2seq_param, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: poly_lr(epoch, epochs, lr, min_lr=1e-6)/lr)
    #perceptual = PerceptualLoss().to(device=device)
        
    recorder = Recorder(['train_loss', 'psnr'])
    plotter = Plotter(dir_visualize, keys1=['train_loss'], keys2=['psnr'])
    
    with open(os.path.join(dir_checkpoint, 'log.csv'), 'w') as f:
        f.write('epoch,train_loss,psnr\n')

    total_step = 0
    best_valid_acc = -2
    for epoch in range(epochs):
        if epoch!=0:
            scheduler.step()
        net.train()
        seq2seq.train()
        train_losses = []
        #pdb.set_trace()
        with tqdm(total=n_train * rep_step, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                img_vibe = batch['vibe']
                img_rmax = batch['rmax']
                img_pbf = batch['pbf']
                img_postvibe = batch['postvibe']
                img_prm = batch['prm']
                #print('img_vibe shape',img_vibe.shape)
                flags = [i[0] for i in batch['flag']]
                #print('load from batch flags', flags)
                path = batch['path'][0][0]
                if len(flags)==0:
                    raise Exception('No available sequence in {}!'.format(path))


                d = img_vibe.shape[3]   #[b,1,d,w,h]

                for _ in range(rep_step):
                    nd, nw, nh = crop_size

                    b,d,t,c,w,h = img_vibe.shape #b:batch
                    rd = random.randint(0, d-nd-1) if d>nd else 0
                    rw = random.randint(0, w-nw-1) if w>nw else 0
                    rh = random.randint(0, h-nh-1) if h>nh else 0
                    inputs = [
                        img_vibe[:,rd:rd+nd,:,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                        img_rmax[:,rd:rd+nd,:,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                        img_pbf[:,rd:rd+nd,:,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                        img_postvibe[:,rd:rd+nd,:,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32),
                    ]
                    output = img_prm[:,rd:rd+nd,:,rw:rw+nw,rh:rh+nh].to(device=device, dtype=torch.float32)

                    mask = (img_vibe>-1).to(device=device, dtype=torch.float32)
                    source_seqs = flags

                    skip_attn = random.randint(0, 1)>0.5

                    source_imgs = inputs
                    target_img = output
                    if torch.abs(torch.mean(source_imgs[0]) + 1) < 1e-8:
                        continue
                    one_hot_target = nn.functional.one_hot(target_img.long(), num_classes=3).squeeze(dim=1).permute(0,5,1,2,3,4).to(device=device)
                    source_code = torch.from_numpy(np.array([1 if i == source_seqs[0] else 0 for i in range(c_s)])).reshape((1, c_s)).to(device=device, dtype=torch.float32)  #cs sequence code length (?)
                    target_code = torch.from_numpy(np.array([1 if i==4 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)



                    output_target = net(seq2seq, source_imgs, source_seqs, target_code, n_outseq=3, task_attn=True, skip_attn=True) #target_code? target_seq one_hot_target.shape[1]
                    print('output_target before argmax', output_target.shape)
                    class_indices = torch.argmax(output_target, dim=1).unsqueeze(dim=1).to(device=device,
                                                                                           dtype=torch.float32)

                    class_indices = torch.mul(class_indices, mask) + mask
                    target_img = target_img + mask

                    pooled_one_hot_target =gaussian_pooling(one_hot_target.squeeze(dim=2).float())
                    loss_rec = 0.5*nn.L1Loss()(torch.mul(output_target,mask), one_hot_target)+  0.5*nn.L1Loss()(torch.mul(output_target.squeeze(dim=2),mask), torch.mul(pooled_one_hot_target,mask))
                    if (total_step  % 1) == 0:
                        numpy_output = class_indices.cpu()[0,0,0,30,:].detach().numpy()
                        plt.imsave('outputint.png',np.round(numpy_output).astype(int),vmin=0, vmax=3)
                        plt.imsave('output.png',numpy_output,vmin=0, vmax=3)
                        plt.imsave('gt.png',target_img.cpu()[0,0,0,30, :].detach().numpy(),vmin=0, vmax=3)
                        np.save(output_path + 'trainout', class_indices.cpu()[0,0,0:].detach().numpy())
                        np.save(output_path + 'traingt', target_img.cpu()[0, 0, 0:].detach().numpy())
                        np.save(output_path + 'trainmask', mask.cpu()[0, 0, 0, :].detach().numpy())
                    prm_loss = prm_losses(class_indices, target_img)
                    loss = lambda_rec*loss_rec + (1-lambda_rec)*prm_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss.item())
                    pbar.set_postfix(**{'rec': loss_rec.item()})
                    pbar.update(1)
                    print(total_step)
                    if ((total_step+1) % args['train']['vis_steps']) == 0:
                        with torch.no_grad():
                            output1 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==0 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1)
                            output2 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==1 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1)
                            output3 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==2 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1)
                            output4 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==3 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1)
                            
                            view_list = [
                                [source_imgs[0][:,0:1,(c_in-1)//2], output_source[:,0:1,(c_in-1)//2], target_img[:,0:1,(c_in-1)//2], output_target[:,0:1,(c_in-1)//2]],
                                [inputs[0][:,0:1,(c_in-1)//2], inputs[1][:,0:1,(c_in-1)//2], inputs[2][:,0:1,(c_in-1)//2], inputs[3][:,0:1,(c_in-1)//2]],
                                [output1[:,0:1,(c_in-1)//2], output2[:,0:1,(c_in-1)//2], output3[:,0:1,(c_in-1)//2], output4[:,0:1,(c_in-1)//2]],
                            ]

                            output1 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==0 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1, skip_attn=True)
                            output2 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==1 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1, skip_attn=True)
                            output3 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==2 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1, skip_attn=True)
                            output4 = net(seq2seq, source_imgs, source_seqs, torch.from_numpy(np.array([1 if i==3 else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32), n_outseq=1, skip_attn=True)

                            view_list.append(
                                [output1[:,0:1,(c_in-1)//2], output2[:,0:1,(c_in-1)//2], output3[:,0:1,(c_in-1)//2], output4[:,0:1,(c_in-1)//2]],
                            )
                        
                        
                        save_grid_images(view_list, os.path.join(dir_visualize, '{:03d}.jpg'.format(epoch+1)), clip_range=(-1,1), normalize=True)
                        torch.cuda.empty_cache()
                    
                    if (total_step % args['train']['ckpt_steps']) == 0:
                        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_tmp.pth'))
                        torch.save(seq2seq.state_dict(), os.path.join(dir_checkpoint, 'ckpt_seq2seq_tmp.pth'))

                    total_step += 1
                    if total_step > args['train']['total_steps']:
                        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_latest.pth'))
                        return
                #break
        
        net.eval()
        seq2seq.eval()
        valid_acc = []
        torch.cuda.empty_cache()
        with torch.no_grad():
            valii = 0
            for batch in valid_loader:
                valii = valii +1
                print('*****************vali batch**********',valii)
                img_vibe = batch['vibe']
                img_rmax = batch['rmax']
                img_pbf = batch['pbf']
                img_postvibe = batch['postvibe']
                img_prm = batch['prm']
                flags = [i[0] for i in batch['flag']]
                path = batch['path'][0][0]
                if len(flags)==0:
                    raise Exception('No available sequence in {}!'.format(path))

                nd, nw, nh = valid_size

                b,d,t,c,w,h = img_vibe.shape
                rd = (d-nd)//2 if d>nd else 0
                rw = (w-nw)//2 if w>nw else 0
                rh = (h-nh)//2 if h>nh else 0

                inputs = [
                    img_vibe[:,rd:rd + nd, :, rw:rw + nw, rh:rh + nh].to(device=device, dtype=torch.float32),
                    img_rmax[:,rd:rd + nd, :, rw:rw + nw, rh:rh + nh].to(device=device, dtype=torch.float32),
                    img_pbf[:,rd:rd + nd, :, rw:rw + nw, rh:rh + nh].to(device=device, dtype=torch.float32),
                    img_postvibe[:,rd:rd + nd, :, rw:rw + nw, rh:rh + nh].to(device=device, dtype=torch.float32),
                ]

                mask = (img_vibe >-1).to(device=device, dtype=torch.float32)

                tgt_flags = [tgt for tgt in range(5) if tgt not in flags] if args['data']['nomiss'] else flags #vali
                for tgt in tgt_flags:
                    source_imgs = [inputs[src] for src in flags]
                    target_img =img_prm[rd:rd + nd, :, rw:rw + nw, rh:rh + nh].to(device=device, dtype=torch.float32)# inputs[tgt]

                    target_code = torch.from_numpy(np.array([1 if i==tgt else 0 for i in range(c_s)])).reshape((1,c_s)).to(device=device, dtype=torch.float32)

                    one_hot_target = nn.functional.one_hot(target_img.long(), num_classes=3).squeeze(dim=1).permute(0,5,1,2,3,4).to(device=device)
                    output_target = net(seq2seq, source_imgs, source_seqs, target_code, n_outseq=3,skip_attn=True)
                    pooled_one_hot_target = gaussian_pooling(one_hot_target.squeeze(dim=2).float())
                    loss_rec = 0.5*nn.L1Loss()(torch.mul(output_target,mask), one_hot_target)+ 0.5*nn.L1Loss()(torch.mul(output_target.squeeze(dim=2),mask), torch.mul(pooled_one_hot_target,mask))

                    class_indices = torch.argmax(output_target, dim=1).unsqueeze(dim=1).to(device=device,dtype=torch.float32)

                    valid_evaluation = -(1-lambda_rec)*prm_losses(class_indices, target_img).item()-lambda_rec*loss_rec.item()
                    valid_acc.append(valid_evaluation)
                #break

        mean_valid_acc = np.mean(valid_acc)
        print('valid_acc', mean_valid_acc, valid_acc)
        train_losses = np.mean(train_losses)
        recorder.update({'train_loss': train_losses, 'psnr': mean_valid_acc})
        plotter.send(recorder.call())
        if best_valid_acc< (mean_valid_acc-train_losses)/2:
            best_valid_acc =  (mean_valid_acc-train_losses)/2
            print('          bset valid acc',best_valid_acc)
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_best.pth'))
            torch.save(seq2seq.state_dict(), os.path.join(dir_checkpoint, 'ckpt_seq2seq_best.pth'))
        with open(os.path.join(dir_checkpoint, 'log.csv'), 'a+') as f:
            f.write('{},{},{}\n'.format(epoch+1, train_losses, mean_valid_acc))
        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_latest.pth'))
        torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser(description='Train TSF-seq2seq model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', dest='config', type=str, default='./config/config.yaml',
                        help='config file')
    parser.add_argument('-l', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-m', '--seq2seq', dest='seq2seq', type=str, default=None,
                        help='Load seq2seq model from a .pth file')
    parser.add_argument('-d', '--device', dest='device', type=str, default='cpu',
                        help='cuda or cpu')
    
    return parser.parse_args()


if __name__ == '__main__':
    torch.cuda.empty_cache()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dir_checkpoint = config['train']['ckpt']
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    
    dir_visualize = config['train']['vis']
    if not os.path.exists(dir_visualize):
        os.makedirs(dir_visualize)

    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    seq2seq = Generator(config)
    seq2seq.to(device=device)

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
        train(
            config,
            net=net,
            seq2seq=seq2seq,
            device=device,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)