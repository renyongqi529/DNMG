import torch
from torchvision import datasets, transforms
from torch.utils.data import dataloader
from torch.utils import data
import matplotlib.pyplot as plt
from wgan import discrimination, generate
import torch.nn as nn
from decoding import *
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import tqdm
torch.backends.cudnn.enable =True
import pandas as pd
import torch.autograd as autograd
from functions import LatentLoss, DiffLoss
from generators import queue_datagen
from keras.utils.data_utils import GeneratorEnqueuer
import argparse
from loss import vae_loss
#####################

#####################

device = torch.device('cuda')
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Path to input .smi file.")
parser.add_argument("-o", "--output_dir",default='./cdk2/', help="Path to model save folder.")
args = vars(parser.parse_args())

cap_loss = 0.
batch_size =32

savedir = args["output_dir"]
os.makedirs(savedir, exist_ok=True)
smiles = np.load(args["input"])

import multiprocessing
multiproc = multiprocessing.Pool(6)
my_gen = queue_datagen(smiles, batch_size=batch_size, mp_pool=multiproc)
print('my_gen:',my_gen)
mg = GeneratorEnqueuer(my_gen)
mg.start()
mt_gen = mg.get()
print('mt_gen:',mt_gen)
# num_epoch = 100
num_epoch = 30
beta_weight = 0.075
loss_diff = DiffLoss()
loss_latent = LatentLoss()
loss_diff = loss_diff.cuda()
loss_latent = loss_latent.cuda()
class NoamOpt:
    "Optimizer wrapper that implements rate decay (adapted from\
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

        self.state_dict = self.optimizer.state_dict()
        self.state_dict['step'] = 0
        self.state_dict['rate'] = 0

    def step(self):
        "Update parameters and rate"
        self.state_dict['step'] += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.state_dict['rate'] = rate
        self.optimizer.step()
        for k, v in self.optimizer.state_dict().items():
            self.state_dict[k] = v

    def rate(self, step=None):
        "Implement 'lrate' above"
        if step is None:
            step = self.state_dict['step']
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict
def noise(size):
    n = torch.randn(size, 128,8,8,8).cuda()
    return n.to(device)
D_weights =  os.path.join("./testmodel6/discriminator-70000.pkl")
G_weights =  os.path.join("./testmodel6/generator-70000.pkl")
encoder_weights =  os.path.join("./testmodel6/shareencoder-70000.pkl")
decoder_weights =os.path.join("./testmodel6/decoder-70000.pkl")

shareencoder = ShareEncoder()
decoder = Decode()
shareencoder.load_state_dict(torch.load(encoder_weights, map_location='cpu'))
decoder.load_state_dict(torch.load(decoder_weights, map_location='cpu'))

shareencoder.to(device)
decoder.to(device)
caption_params = list(decoder.parameters()) + list(shareencoder.parameters())
# caption_optimizer = torch.optim.Adam(caption_params, lr=0.001)
caption_optimizer = NoamOpt(128, 1, 10000,
                                 torch.optim.Adam(caption_params, lr=0,
                                 betas=(0.9,0.98), eps=1e-9))
shareencoder.train()
decoder.train()
# lr = 5e-5
# lr = 2e-4
lr = 1e-4
discriminator = discrimination().to(device)
discriminator.load_state_dict(torch.load(D_weights, map_location='cpu'))
d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

generator = generate().to(device)
generator.load_state_dict(torch.load(G_weights, map_location='cpu'))
g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=lr)

d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.99)
g_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=0.99)

loss_fn = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

tq_gen = enumerate(mt_gen)
log_file = open(os.path.join(savedir, "log.txt"), "w")
# caption_start = 4000
caption_start = 0
one = torch.tensor(1, dtype=torch.float)  ###torch.FloatTensor([1])
mone = one * -1
def calc_gradient_penalty(netD, real_data, fake_data):
        LAMBDA = .1
        alpha = torch.rand(32, 19,16,16,16)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda() 
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        
        interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = netD(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda() , create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty
CRITIC_ITERS = 5
def train_discriminator(optimizer, loss_fn, real_data, fake_data):
    for param in discriminator.parameters():
        param.requires_grad = True
    for _ in range(CRITIC_ITERS):
        optimizer.zero_grad()
        # real_data1 = Variable(real_data[:, :])
        # print('real_data1:',real_data1.shape)
        discriminator_real_data = discriminator(real_data.to(torch.float32))
        # print('discriminator_real_data:',discriminator_real_data.shape)
        # loss_real = loss_fn(discriminator_real_data, torch.ones(real_data.size(0),1).to(device))
        loss_real = discriminator_real_data.mean(0).view(1)
        loss_real.backward()

        discriminator_fake_data = discriminator(fake_data)
        # loss_fake = loss_fn(discriminator_fake_data, torch.zeros(fake_data.size(0), 1).to(device))
        loss_fake = discriminator_fake_data.mean(0).view(1)
        loss_fake.backward()
        gradient_penalty = calc_gradient_penalty(discriminator, real_data.data, fake_data.data)
        gradient_penalty.backward()
        
        optimizer.step()
    for parm in discriminator.parameters():
        parm.data.clamp_(-0.01, 0.01)
    # print('loss_real + loss_fake:',loss_real + loss_fake)
    # return loss_real + loss_fake, discriminator_real_data, discriminator_fake_data
    return  loss_fake - loss_real+ gradient_penalty, discriminator_real_data, discriminator_fake_data

def train_generator(optimizer, loss_fn, fake_data):
    for param in discriminator.parameters():
        param.requires_grad = False
    optimizer.zero_grad()

    output_discriminator = discriminator(fake_data)
    # loss = loss_fn(output_discriminator, torch.ones(output_discriminator.size(0), 1).to(device))
    loss = torch.mean(output_discriminator).view(1)
    loss.backward()
    optimizer.step()
    return loss

plt.figure()
lists=[]
lists1=[]
lists2=[]

# for epoch in range(num_epoch):
    # i=0 
    # # while i < 30:
    # #     train_data_iter = iter(train_data)
    # #     data_ = train_data_iter.__next__()
    # #     real_data,caption, length = data_
    # cap_loss1 = 0
    # current_step =0
    # train_loss =0
cap_loss = 0. 
caption_start =0    
for i, (real_data,caption, length) in tq_gen:
        cap_loss = 0. 
        print('i:',i)  
        real_data=real_data.cuda()
            # print('123',real_data.shape)
            # real_data = images2vectors(input_real_batch).to(device)
        
        generated_fake_data = generator(noise(real_data.size(0))).detach()
        
        d_loss, discriminated_real, discriminated_fake = train_discriminator(d_optimizer, loss_fn, real_data,
                                                                                generated_fake_data)

        generated_fake_data = generator(noise(real_data.size(0)))
        g_loss = train_generator(g_optimizer, loss_fn, generated_fake_data)

        re_con = generator(noise(real_data.size(0)).detach())
        # print('re_con.shape:',re_con.shape)
        if i >=caption_start:
            caption_= Variable(caption.long()).to(device)
            # print('caption_.shape:',caption_.shape)#torch.Size([32, 120])
            # print('111:',pack_padded_sequence(caption_, length, batch_first=True,enforce_sorted=False))
            #pack_padded_sequence是按照列压缩的，
            target = pack_padded_sequence(caption_, length, batch_first=True,enforce_sorted=False)[0]#按列取出元素
            decoder.zero_grad()
            shareencoder.zero_grad()
            # caption_optimizer.zero_grad()
            
            result  = shareencoder(re_con)
            
            single_mu,single_logvar,features = result
            latloss = loss_latent(single_mu,single_logvar)  
            print('latloss:',latloss)
            cap_loss += latloss 
            # diffloss = beta_weight * loss_diff(single_latent,share_latent)
            # # print('diffloss:',diffloss)
            # cap_loss += diffloss
            outputs = decoder(features, caption_, length)
            # print('outputs:',outputs.shape)
            # print('target:',target.shape)
            # print('outputs:',outputs)
            # print('target:',target)
            # print('target:',target[-1])
            cap_loss=criterion(outputs, target)
            # cap_loss +=criterion_loss
            # cap_loss1 +=(cap_loss.item())
            cap_loss.backward()
            # caption_optimizer = exp_lr_scheduler(optimizer=caption_optimizer, step=current_step)
            caption_optimizer.step()
            # print(epoch,'d_loss: ', d_loss.item(), 'g_loss: ', g_loss.item(),'cap_loss:',cap_loss.item())
        
            lists.append(d_loss.item())
            lists1.append(g_loss.item())
            lists2.append(cap_loss.item())
    # train_loss +=cap_loss1
    
        print('cap_loss1:',cap_loss)  
        # i += 1
        # current_step +=1      
        # break
    # d_scheduler.step()
    # g_scheduler.step()
        # print(len(train_data))
        # if train_idx == len(train_data) - 1:
    # if (epoch + 1) % 100 == 0:
    #     print(epoch, 'd_loss: ', d_loss.item(), 'g_loss: ', g_loss.item(),'cap_loss:',cap_loss.item())
    
        if (i + 1) % 10 == 0:
            print('epoch:',i+1,'d_loss: ', d_loss.item(), 'g_loss: ', g_loss.item(),'cap_loss:',cap_loss)  
      

        if i == 1020:
            # We are Done!
            break    
torch.save(shareencoder.state_dict(),
                   os.path.join(savedir,
                                'shareencoder-%d.pkl' % (545)))
torch.save(decoder.state_dict(),
                   os.path.join(savedir,
                                'decoder-%d.pkl' % (545)))            
torch.save(generator.state_dict(),
                   os.path.join(savedir,
                                'generator-%d.pkl' % (545)))
torch.save(discriminator.state_dict(),
                   os.path.join(savedir,
                                'discriminator-%d.pkl' % (545)))  
df_smiles = pd.DataFrame(lists)
df_smiles.to_csv('./lossD.csv')
df_smiles1 = pd.DataFrame(lists1)
df_smiles1.to_csv('./lossG.csv')
df_smiles2 = pd.DataFrame(lists2)
df_smiles2.to_csv('./losscap.csv')

print('End Training')