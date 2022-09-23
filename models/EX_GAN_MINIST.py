import numpy as np
import math
import functools
import random
import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.utils import data

from .layers import SNConv2d, ccbn, identity, SNLinear, SNEmbedding

from .losses_original import loss_dis_real, loss_dis_fake
from .pyod_utils import AUC_and_Gmean, get_measure

from .Imbalanced_MINIST import MNIST
from torchvision import transforms

class Generator(nn.Module):
    def __init__(self, init='ortho', SN_used=True):
        super(Generator, self).__init__()
        self.init = init
        if SN_used:
            self.which_conv = functools.partial(SNConv2d,
                                num_svs=1, num_itrs=1)
        else:
            self.which_conv = nn.Conv2d
        
        self.conv1 = self.which_conv(1, 32, 3, 1, 1)
        self.conv2 = self.which_conv(32, 64, 3, 1, 1)
        self.conv3 = self.which_conv(64, 1, 3, 1, 1)
        
        self.model = nn.Sequential(self.conv1,
                                    nn.ReLU(),
                                    self.conv2,
                                    nn.ReLU(),
                                    self.conv3)
                                    #nn.Sigmoid())

        self.init_weights()
        
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) 
                or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    def forward(self, x):
        #print(x.shape)
        h = self.model(x)
        #print(h.shape)
        return h

class Discriminator(nn.Module):
    def __init__(self, num_class=2, init='ortho', SN_used=True):
        super(Discriminator, self).__init__()
        self.n_classes = num_class
        self.init = init

        if SN_used:
            self.which_conv = functools.partial(SNConv2d,
                                num_svs=1, num_itrs=1)
            self.which_linear = functools.partial(SNLinear,
                                num_svs=1, num_itrs=1)
        else:
            self.which_conv = nn.Conv2d
            self.which_linear = nn.Linear

        self.which_embedding = nn.Embedding


        self.conv1 = self.which_conv(1, 32, 3, 1)
        self.conv2 = self.which_conv(32, 128, 3, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        #self.fc1 = self.which_linear(9216, 128)

        self.model = nn.Sequential(self.conv1,
                                    nn.ReLU(),
                                    self.conv2,
                                    nn.ReLU(),
                                    self.avg_pool)

        self.output_fc = self.which_linear(128, 1)
        self.output_category = nn.Sequential(self.which_linear(128, 1),
                                            nn.Sigmoid())
        # Embedding for projection discrimination
        self.embed = self.which_embedding(self.n_classes, 128)
        
        self.init_weights()
    
    #Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding) or isinstance(module, nn.Conv2d)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        #print('Param count for D''s initialized parameters: %d' % self.param_count)
    
    def forward(self, x, y=None, mode=0):
        #mode 0: train the whole discriminator network
        if mode==0:
            h = self.model(x)
            #h = F.max_pool2d(h, 2)
            #h = F.adaptive_avg_pool2d(h)
            #print(h.shape)
            #print(h.shape)
            #h = torch.flatten(h, 1)
            h = h.view(h.shape[0], -1)
            #h = self.fc1(h)
            #print(h.shape)
            out = self.output_fc(h)
            # Get projection of final featureset onto class vectors and add to evidence
            out_real_fake = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
            out_category = self.output_category(h)
            return out_real_fake, out_category
        #mode 1: train self.output_fc, only classify whether an input is fake or real
        elif mode==1:
            h = self.model(x)
            h = h.view(h.shape[0], -1)
            out = self.output_fc(h)
            return out
        #mode 2: train self.output_category, used in fine_tunning stage
        else:
            h = self.model(x)
            h = h.view(h.shape[0], -1)
            out = self.output_category(h)
            return out

class EX_GAN(nn.Module):
    def __init__(self, args, lamd=1.0):
        super(EX_GAN, self).__init__()
        self.args = args
        
        self.Cycle_Loss = nn.L1Loss()
        self.l1_loss = nn.L1Loss()
        self.lamd = lamd

        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
    
        #1: prepare Generator
        self.netG_A_to_B = Generator(init=args.init_type, SN_used=args.SN_used)
        self.netG_B_to_A = Generator(init=args.init_type, SN_used=args.SN_used)

        self.optimizerG = optim.Adam(list(self.netG_A_to_B.parameters())+list(self.netG_B_to_A.parameters()), lr=args.lr_g, betas=(0.00, 0.99))
        
        #2: create ensemble of discriminator
        self.NetD_Ensemble = []
        self.opti_Ensemble = []
        lr_ds = np.random.rand(args.ensemble_num)*(args.lr_d*5-args.lr_d)+args.lr_d  #learning rate
        for index in range(args.ensemble_num):
            netD = Discriminator(num_class=2, init=args.init_type, SN_used=args.SN_used)
            
            optimizerD = optim.Adam(netD.parameters(), lr=lr_ds[index], betas=(0.00, 0.99))
            self.NetD_Ensemble += [netD]
            self.opti_Ensemble += [optimizerD]

    def fit(self):
        log_dir = os.path.join(self.args.log_path, self.args.data_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Start iteration
        Best_Measure_Recorded = -1
        Best_AUC_test = -1
        Best_F_test = -1
        Best_AUC_train = -1
        Best_F_train = -1
        self.train_history = defaultdict(list)
        transform=transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,))
                ])

        train_dataset = MNIST('./data', train=True, download=True,
                            transform=transform)
        test_dataset = MNIST('./data', train=False, download=True,
                            transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size)
        
        for epoch in range(self.args.max_epochs):
            train_AUC, train_score, train_Gmean, test_auc, test_score, test_gmean = self.train_one_epoch(epoch, train_loader, test_loader)
            if train_score*train_AUC > Best_Measure_Recorded:
                Best_Measure_Recorded = train_score*train_AUC
                Best_AUC_test = test_auc
                Best_F_test = test_score
                Best_AUC_train = train_AUC
                Best_F_train = train_score

                states = {
                    'epoch':epoch,
                    'gen_A_to_B':self.netG_A_to_B.state_dict(),
                    'gen_B_to_A':self.netG_B_to_A.state_dict(),
                    'max_auc':train_AUC
                }
                for i in range(self.args.ensemble_num):
                    netD = self.NetD_Ensemble[i]
                    optimi_D = self.opti_Ensemble[i]
                    states['dis_dict'+str(i)] = netD.state_dict()
                
                torch.save(states, os.path.join(log_dir, 'checkpoint_best.pth'))
            #print(train_AUC, test_AUC, epoch)
            if self.args.print:
                print('Epoch %d: Train_AUC=%.4f train_fscore=%.4f train_Gmean=%.4f Test_AUC=%.4f test_fscore=%.4f Test_Gmean=%.4f' % (epoch + 1, train_AUC, train_score, train_Gmean, test_auc, test_score, test_gmean))
        
       
        #step 1: load the best models
        self.Best_Ensemble = []
        states = torch.load(os.path.join(log_dir, 'checkpoint_best.pth'))
        self.netG_A_to_B.load_state_dict(states['gen_A_to_B'])
        self.netG_B_to_A.load_state_dict(states['gen_B_to_A'])
        for i in range(self.args.ensemble_num):
            netD = self.NetD_Ensemble[i]
            netD.load_state_dict(states['dis_dict'+str(i)])
            self.Best_Ensemble += [netD]
        
        return Best_AUC_train, Best_F_train, Best_AUC_test, Best_F_test
    
    def predict(self, data_loader, dis_Ensemble=None, need_explain=False):
        y_pred = []
        y_true = []
        data_ex = []
        data = []
        for i, (digits, labels) in enumerate(data_loader):
            for i in range(self.args.ensemble_num):
                pt = self.Best_Ensemble[i](digits, mode=2) if dis_Ensemble is None else dis_Ensemble[i](digits, mode=2)
                if i==0:
                    final_pt = pt.detach()
                else:
                    final_pt += pt
            final_pt /= self.args.ensemble_num
            final_pt = final_pt.view(final_pt.shape[0],)
            y_pred += [final_pt]
            y_true += [labels.view(labels.shape[0],)]

            digits_ex = torch.zeros_like(digits)
            data += [digits.detach()]

            if need_explain:
                index0 = (final_pt<0.5)
                index1 = (final_pt>=0.5)
                data0 = digits[index0]
                data1 = digits[index1]
                if data0.shape[0]>0:
                    data0_ex = self.netG_A_to_B(data0)
                    digits_ex[index0] = data0_ex
                if data1.shape[0]>0:
                    data1_ex = self.netG_B_to_A(data1)
                    digits_ex[index1] = data1_ex
                
                data_ex += [digits_ex]
        y_pred = torch.cat(y_pred, 0)
        y_true = torch.cat(y_true, 0)
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        if need_explain:
            data_ex = torch.cat(data_ex, dim=0)
            
        return y_pred, y_true, data_ex


    def train_one_epoch(self, epoch=1, train_loader=None, test_loader=None):
        #train discriminator & generator for one specific spoch
        data0 = []
        data1 = []
        targets0 = []
        targets1 = []
        count0 = 0
        count1 = 0
        for i, (digits, labels) in enumerate(train_loader):
            index0 = (labels==0)
            index1 = (labels==1)
            if torch.sum(index0)>0:
                data0 += [digits[index0]]
                targets0 += [labels[index0]]
            if torch.sum(index1)>0:
                data1 += [digits[index1]]
                targets1 += [labels[index1]]
            count0 += torch.sum(index0)
            count1 += torch.sum(index1)
            if count0>=self.args.batch_size and count1>=self.args.batch_size:
                break
        data0 = torch.cat(data0, dim=0)
        data1 = torch.cat(data1, dim=0)
        data0 = data0[0:self.args.batch_size]
        data1 = data1[0:self.args.batch_size]
        #print(torch.max(data0), "    ", torch.min(data1))
        #_, _, _ = self.predict(test_loader, self.NetD_Ensemble)
        
        for index, (digits, labels) in enumerate(train_loader):
            # step 1: train the ensemble of discriminator
            # Get training data
            index0 = (labels==0)
            index1 = (labels==1)
            real_x0 = digits[index0]
            real_x1 = digits[index1]

            if real_x0.shape[0] > 0:
                fake_B = self.netG_A_to_B(real_x0)
            else:
                real_x0 = data0.clone().detach()
                fake_B = self.netG_A_to_B(real_x0)
            reconstrcuted_A = self.netG_B_to_A(fake_B)
            
            if real_x1.shape[0]>0:
                fake_A = self.netG_B_to_A(real_x1)
            else:
                real_x1 = data1.clone().detach()
                fake_A = self.netG_B_to_A(real_x1)
            reconstrcuted_B = self.netG_A_to_B(fake_A)

            fake_x = torch.cat([fake_A, fake_B], 0)
            fake_y = torch.cat([torch.zeros(fake_A.shape[0],), torch.ones(fake_B.shape[0],)], 0).long()
            real_x = torch.cat([real_x0, real_x1], 0)
            real_y = torch.cat([torch.zeros(real_x0.shape[0],), torch.ones(real_x1.shape[0],)], 0).long()
            
            #train D with real data and fake data
            dis_loss = 0
            adv_weight_real, cat_weight_real, adv_weight_fake, cat_weight_fake = None, None, None, None
            for i in range(self.args.ensemble_num):
                optimizer = self.opti_Ensemble[i]
                netD = self.NetD_Ensemble[i]

                #train the GAN with real data
                out_adv_real, out_cat_real = netD(real_x, real_y)
            
                loss_adv_real, loss_cat_real, adv_weight_real, cat_weight_real = loss_dis_real(out_adv_real, out_cat_real, real_y, adv_weight_real, cat_weight_real)
                real_loss = loss_adv_real+loss_cat_real
                
                #train the GAN with fake data
                out_adv_fake, out_cat_fake = netD(fake_x.detach(), fake_y.detach())
                loss_adv_fake, loss_cat_fake, adv_weight_fake, cat_weight_fake = loss_dis_fake(out_adv_fake, out_cat_fake, fake_y.detach(), adv_weight_fake, cat_weight_fake)
                fake_loss = loss_adv_fake+loss_cat_fake
                sum_loss = real_loss+fake_loss
                dis_loss += sum_loss

                self.train_history['discriminator_loss_'+str(i)].append(sum_loss)
                optimizer.zero_grad()
                sum_loss.backward(retain_graph=True)
                optimizer.step()
            self.train_history['discriminator_loss'].append(dis_loss)
            
            #step 2: train the generator
            gen_loss = 0
            adv_weight_gen, cat_weight_gen = None, None
            for i in range(self.args.ensemble_num):
                #optimizer = names['optimizerD_' + str(i)]
                netD = self.NetD_Ensemble[i]
                out_adv, out_cat = netD(fake_x, fake_y)
                loss_adv, loss_cat, adv_weight_gen, cat_weight_gen = loss_dis_real(out_adv, out_cat, fake_y, adv_weight_gen, cat_weight_gen)
                gen_loss += (loss_adv+loss_cat)
            cycle_loss = self.l1_loss(reconstrcuted_A, real_x0) + self.l1_loss(reconstrcuted_B, real_x1)
            consistency_loss = self.l1_loss(real_x0, fake_B) + self.l1_loss(real_x1, fake_A)
            gen_loss += cycle_loss  + 1.0*epoch/self.args.max_epochs * consistency_loss
            self.train_history['generator_loss'].append(gen_loss)

            self.optimizerG.zero_grad()
            gen_loss.backward()
            self.optimizerG.step()
        
        y_pred_train, y_true_train, _ = self.predict(train_loader, self.NetD_Ensemble)
        #y_pred_train = y_pred_train.detach().cpu().numpy()
        #y_true_train = y_true_train.detach().cpu().numpy()
        auc_train, fscore_train, gmean_train = get_measure(y_true_train, y_pred_train)

        self.train_history['train_auc'].append(auc_train)
        self.train_history['train_Gmean'].append(gmean_train)
        
        y_pred_test, y_true_test, _ = self.predict(test_loader, self.NetD_Ensemble)
        #y_pred_test = y_pred_test.detach().cpu().numpy()
        #y_true_test = y_true_test.detach().cpu().numpy()
        auc_test, fscore_test, gmean_test = get_measure(y_true_test, y_pred_test)

        return auc_train, fscore_train, gmean_train, auc_test, fscore_test, gmean_test
