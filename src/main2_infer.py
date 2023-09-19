#
# Copyright 2023. Triad National Security, LLC. All rights reserved. 
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
# Department of Energy/National Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting on its behalf a
# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
# others to do so.
#
# Author:
#   Kai Gao <kaigao@lanl.gov>
# 

import os
import warnings
import sys
import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torchmetrics
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

warnings.filterwarnings("ignore")

from utility import *
from model2 import *

#==============================================================================
parser = argparse.ArgumentParser(description='MTL-Net options')
parser.add_argument('--ntrain', type=int, default=1000, help='training size')
parser.add_argument('--nvalid', type=int, default=100, help='validation size')
parser.add_argument('--batch_train', type=int, default=1, help='training batch size')
parser.add_argument('--batch_valid', type=int, default=1, help='validation batch size')
parser.add_argument('--epochs', type=int, default=100, help='max number of epochs')
parser.add_argument('--lr', type=float, default=0.5e-4, help='learning rate')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader')
parser.add_argument('--dir_output', type=str, default='./result', help='directory')
parser.add_argument('--dir_data_train', type=str, default='./dataset/data_train', help='directory')
parser.add_argument('--dir_target_train', type=str, default='./dataset/target_train', help='directory')
parser.add_argument('--dir_data_valid', type=str, default='./dataset/data_valid', help='directory')
parser.add_argument('--dir_target_valid', type=str, default='./dataset/target_valid', help='directory')
parser.add_argument('--resume', type=str, default=None, help='restart training from resume checkopoint')
parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--gpus_per_node', type=int, default=4, help='number of gpus per node')
parser.add_argument('--seed', type=int, default=12345, help='random seed for initialization')
parser.add_argument('--check', type=str, default=None, help='test model using test checkpoint')
parser.add_argument('--n1', '-n1', type=int, default=256, help='number of sampling points in x1')
parser.add_argument('--n2', '-n2', type=int, default=256, help='number of sampling points in x2')
parser.add_argument('--input', '-in', type=str, default=None, help='test model using test checkpoint')
parser.add_argument('--model', '-model', type=str, default=None, help='test model using test checkpoint')
parser.add_argument('--output', '-out', type=str, default=None, help='test model using test checkpoint')
parser.add_argument('--rgt', '-rgt', type=str2bool, default='y', help='turn on/off RGT module')
parser.add_argument('--dhr', '-dhr', type=str2bool, default='y', help='turn on/off DHR module')
parser.add_argument('--fault', '-fault', type=str2bool, default='y', help='turn on/off fault module')
opts = parser.parse_args()

assert opts.n1 >= 1
assert opts.n2 >= 1

if torch.cuda.is_available() and opts.gpus_per_node >= 1:
    device = torch.device('cuda')
    print(date_time(), ' >> Using GPU')
else:
    device = torch.device('cpu')
    print(date_time(), ' >> Using CPU')

torch.set_float32_matmul_precision('high')


#==============================================================================
class BasicDataset(Dataset):
    def __init__(self, dir_data, dir_target, data_ids, dim=(opts.n1, opts.n2)):
        self.dir_data = dir_data
        self.dir_target = dir_target
        self.ids = data_ids
        self.n1 = dim[0]
        self.n2 = dim[1]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):

        idx = str(self.ids[i])

        # data
        data = read_array(self.dir_data + '/' + idx + '.bin', (1, self.n1, self.n2))

        # target
        target = {}

        if opts.rgt:
            target['rgt'] = read_array(self.dir_target + '/' + idx + '_rgt.bin', (1, self.n1, self.n2))

        if opts.dhr:
            target['dhr'] = read_array(self.dir_target + '/' + idx + '_dhr.bin', (1, self.n1, self.n2))

        if opts.fault:
            target['fsem'] = read_array(self.dir_target + '/' + idx + '_fsem.bin', (1, self.n1, self.n2))
            target['fdip'] = read_array(self.dir_target + '/' + idx + '_fdip.bin', (1, self.n1, self.n2))

        return data, target


def custom_loss(y_pred, y_true):

    # rgt
    if opts.rgt:
        rp = y_pred['rgt']
        rt = y_true['rgt']
        loss_rgt = 1 - ssim(rp, rt) + F.l1_loss(rp, rt)
        loss_rgt = loss_rgt * 5
    else:
        loss_rgt = 0

    # dhr
    if opts.dhr:
        sp = y_pred['dhr']
        st = y_true['dhr']
        loss_dhr = F.mse_loss(sp, st) * 10
    else:
        loss_dhr = 0

    # fault
    if opts.fault:
        # fault semantic
        mp = y_pred['fsem']
        mt = y_true['fsem']
        loss_fault_semantic = 1.0 - (2.0 * torch.sum(mp * mt) + 1.0) / (torch.sum(mp + mt) + 1.0)

        # fault dip
        dp = y_pred['fdip']
        dt = y_true['fdip']
        loss_fault_dip = F.l1_loss(dp, dt) * 10
    else:
        loss_fault_semantic = 0
        loss_fault_dip = 0

    # sum
    loss = loss_rgt + loss_dhr + loss_fault_semantic + loss_fault_dip

    return loss, loss_rgt, loss_dhr, loss_fault_semantic, loss_fault_dip


def custom_accuracy(y_pred, y_true):

    return None


#==============================================================================
class mtlnet(pl.LightningModule):
    def __init__(self, lr: float = 1.0e-4):

        super(mtlnet, self).__init__()

        self.lr = lr
        self.in_ch = 1

        # encoder
        self.l1 = 16
        self.l2 = 32
        self.l3 = 64
        self.encoder1 = resu1(self.in_ch, self.l1)
        self.encoder2 = resu2(self.l1, self.l2)
        self.encoder3 = resu3(self.l2, self.l3)

        # decoders
        if opts.rgt:
            self.decoder_rgt = mtl_decoder(self.l1, self.l2, self.l3, out_ch=1, out_activation='sigmoid')

        if opts.dhr:
            self.decoder_dhr = mtl_decoder(self.l1, self.l2, self.l3, out_ch=1, out_activation=None)

        if opts.fault:
            self.decoder_fault = mtl_decoder(self.l1, self.l2, self.l3, out_ch=self.l1, last_kernel_size=3)
            self.subdecoder_fault_semantic = mtl_subdecoder(in_ch=self.l1,
                                                            out_ch=1,
                                                            bn=False,
                                                            mid_activation='relu',
                                                            activation='sigmoid')
            self.subdecoder_fault_dip = mtl_subdecoder(in_ch=self.l1,
                                                       out_ch=1,
                                                       bn=True,
                                                       mid_activation='relu',
                                                       activation='sigmoid')

    def forward(self, x):

        out_encoder1 = self.encoder1(x)
        out_encoder2 = self.encoder2(maxpool(out_encoder1, 2))
        out_encoder3 = self.encoder3(maxpool(out_encoder2, 2))

        # decoders
        out = {}
        if opts.rgt:
            out['rgt'] = self.decoder_rgt(x, out_encoder1, out_encoder2, out_encoder3)

        if opts.dhr:
            out['dhr'] = self.decoder_dhr(x, out_encoder1, out_encoder2, out_encoder3)

        if opts.fault:
            out_fault = self.decoder_fault(x, out_encoder1, out_encoder2, out_encoder3)
            out_fault_semantic = self.subdecoder_fault_semantic(out_fault)
            out_fault_dip = self.subdecoder_fault_dip(out_fault) * out_fault_semantic
            out['fsem'] = out_fault_semantic
            out['fdip'] = out_fault_dip

        return out

    def training_step(self, batch, batch_idx):

        image, y_true = batch
        y_pred = self.forward(image)

        loss, loss_rgt, loss_dhr, loss_fault_semantic, loss_fault_dip = custom_loss(y_pred, y_true)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if opts.rgt:
            self.log("train_loss_rgt", loss_rgt, on_step=False, on_epoch=True, prog_bar=False)
        if opts.dhr:
            self.log("train_loss_dhr", loss_dhr, on_step=False, on_epoch=True, prog_bar=False)
        if opts.fault:
            self.log("train_loss_fsem", loss_fault_semantic, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_loss_fdip", loss_fault_dip, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):

        image, y_true = batch
        y_pred = self.forward(image)

        loss, loss_rgt, loss_dhr, loss_fault_semantic, loss_fault_dip = custom_loss(y_pred, y_true)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if opts.rgt:
            self.log("valid_loss_rgt", loss_rgt, on_step=False, on_epoch=True, prog_bar=False)
        if opts.dhr:
            self.log("valid_loss_dhr", loss_dhr, on_step=False, on_epoch=True, prog_bar=False)
        if opts.fault:
            self.log("valid_loss_fsem", loss_fault_semantic, on_step=False, on_epoch=True, prog_bar=False)
            self.log("valid_loss_fdip", loss_fault_dip, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            'name': 'lr',
            'monitor': 'val_loss'
            }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


#==============================================================================
if __name__ == '__main__':

    if opts.input is None and opts.check is None:
        ## Training phase

        logger = TensorBoardLogger(opts.dir_output)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath=opts.dir_output,
                                              filename='{epoch:d}',
                                              mode='min',
                                              save_top_k=opts.epochs,
                                              save_last=True)

        params = {
            'max_epochs': opts.epochs,
            'default_root_dir': opts.dir_output,
            'logger': logger,
            'callbacks': [checkpoint_callback]
        }
        if torch.cuda.is_available() and opts.gpus_per_node >= 1:
            params['devices'] = opts.gpus_per_node
            params['num_nodes'] = opts.nodes
            params['accelerator'] = 'gpu'
            params['strategy'] = 'ddp'

        trainer = pl.Trainer(**params)

        t = BasicDataset(opts.dir_data_train,
                         opts.dir_target_train,
                         data_ids=np.arange(0, opts.ntrain),
                         dim=(opts.n1, opts.n2))
        train_loader = DataLoader(t, batch_size=opts.batch_train, num_workers=opts.threads, shuffle=True)

        v = BasicDataset(opts.dir_data_valid,
                         opts.dir_target_valid,
                         data_ids=np.arange(0, opts.nvalid),
                         dim=(opts.n1, opts.n2))
        valid_loader = DataLoader(v, batch_size=opts.batch_valid, num_workers=opts.threads)

        set_random_seed(opts.seed)
        params = {'lr': opts.lr}
        net = mtlnet(**params)

        if opts.resume:
            trainer.fit(net, train_loader, valid_loader, ckpt_path=opts.resume)
        else:
            trainer.fit(net, train_loader, valid_loader)

        print(date_time(), ' >> Training finished')

    if opts.input is None and opts.check is not None:
        ## Validation phase
        
        v = BasicDataset(opts.dir_data_valid,
                         opts.dir_target_valid,
                         data_ids=np.arange(0, opts.nvalid),
                         dim=(opts.n1, opts.n2))
        valid_loader = DataLoader(v, batch_size=opts.batch_valid, num_workers=opts.threads)

        net = mtlnet()
        net.load_state_dict(torch.load(opts.check, map_location=device)['state_dict'])
        net.to(device)
        l = 1

        with tqdm(total=len(v), desc='', unit='image') as pbar:

            for (input, target) in valid_loader:

                with torch.no_grad():
                    predict = net(input.to(device))

                for i in range(0, input.shape[0]):

                    ir = (l - 1) * opts.batch_valid + i

                    fig, ax = plt.subplots(2, 5)
                    im = input[i, 0, :, :].squeeze().data.cpu().numpy()

                    if opts.rgt:
                        rgt = target['rgt'][i].squeeze().data.cpu().numpy()
                    else:
                        rgt = 0 * im
                    if opts.dhr:
                        dhr = target['dhr'][i].squeeze().data.cpu().numpy()
                    else:
                        dhr = 0 * im
                    if opts.fault:
                        fsem = target['fsem'][i].squeeze().data.cpu().numpy()
                        fdip = target['fdip'][i].squeeze().data.cpu().numpy()
                    else:
                        fsem = 0 * im
                        fdip = 0 * im

                    ax[0][0].imshow(im, cmap='gray'), ax[0][0].set_title('noisy')
                    ax[0][1].imshow(rgt, cmap='jet'), ax[0][1].set_title('rgt')
                    ax[0][2].imshow(dhr, cmap='gray'), ax[0][2].set_title('dhr')
                    ax[0][3].imshow(fsem, cmap='viridis'), ax[0][3].set_title('fsem')
                    ax[0][4].imshow(fdip, cmap='jet'), ax[0][4].set_title('fdip')

                    if opts.rgt:
                        rgt = predict['rgt'][i].squeeze().data.cpu().numpy()
                    else:
                        rgt = 0 * im
                    if opts.dhr:
                        dhr = predict['dhr'][i].squeeze().data.cpu().numpy()
                    else:
                        dhr = 0 * im
                    if opts.fault:
                        fsem = predict['fsem'][i].squeeze().data.cpu().numpy()
                        fdip = predict['fdip'][i].squeeze().data.cpu().numpy()
                    else:
                        fsem = 0 * im
                        fdip = 0 * im

                    ax[1][0].imshow(im, cmap='gray'), ax[0][0].set_title('noisy')
                    ax[1][1].imshow(rgt, cmap='jet'), ax[1][1].set_title('rgt')
                    ax[1][2].imshow(dhr, cmap='gray'), ax[1][2].set_title('dhr')
                    ax[1][3].imshow(fsem, cmap='viridis'), ax[1][3].set_title('fsem')
                    ax[1][4].imshow(fdip, cmap='jet'), ax[1][4].set_title('fdip')

                    plt.show()

                pbar.update(input.shape[0])
                l = l + 1

        print(date_time(), ' >> Validation finished')

    if opts.input is not None:
        ## Inference phase

        # Read image
        n1 = opts.n1
        n2 = opts.n2

        img = read_array(opts.input, (1, 1, n1, n2))

        # Load trained model
        net = mtlnet()

        if not opts.rgt or not opts.dhr or not opts.fault:
            # Only select parts of the model needed
            pretrained_dict = torch.load(opts.model, map_location=device)['state_dict']
            model_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
        else:
            net.load_state_dict(torch.load(opts.model, map_location=device)['state_dict'])

        net.to(device)

        print(date_time(), " >> Pretrained model loaded")

        with torch.no_grad():
            predict = net(img.to(device))

        if opts.rgt:
            rgt = get_numpy(predict['rgt'])
            rgt = rgt - np.min(rgt)
            rgt = rgt / np.max(rgt)
            write_array(rgt, opts.output + '.rgt')
        if opts.dhr:
            dhr = get_numpy(predict['dhr'])
            write_array(dhr, opts.output + '.dhr')
        if opts.fault:
            fsem = get_numpy(predict['fsem'])
            fdip = get_numpy(predict['fdip'])
            write_array(fsem, opts.output + '.fsem')
            write_array(fdip, opts.output + '.fdip')

        print(date_time(), ' >> Inference finished')
