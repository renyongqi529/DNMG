import torch
from torchvision import datasets, transforms
from torch.utils.data import dataloader
from torch.utils import data
import matplotlib.pyplot as plt
import torch.nn as nn
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
from functions import LatentLoss
from data_generator import queue_datagen
from keras.utils.data_utils import GeneratorEnqueuer
import argparse
import multiprocessing
from wgan import discrimination, generate,Encoder,Decode
#####################
#####################
device = torch.device('cuda')
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Path to input .smi file.")
parser.add_argument("-o", "--output_dir",default='./savemodel/', help="Path to model save folder.")
args = vars(parser.parse_args())

cap_loss = 0.
batch_size =32

savedir = args["output_dir"]
os.makedirs(savedir, exist_ok=True)
smiles = np.load(args["input"])

multiproc = multiprocessing.Pool(6)
wgen = queue_datagen(smiles, batch_size=batch_size, mp_pool=multiproc)
print('wgen:',wgen)
wg = GeneratorEnqueuer(wgen)
wg.start()
wgen = wg.get()
print('wgen:',wgen)

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
###############################################



loss_latent = LatentLoss()
loss_latent = loss_latent.cuda()

def noise(size):
    n = torch.randn(size, 128,8,8,8).cuda()
    return n

cap_encoder = Encoder()
decoder = Decode()

cap_encoder.to(device)
decoder.to(device)
caption_params = list(decoder.parameters()) + list(cap_encoder.parameters())
caption_optimizer = NoamOpt(128, 1, 10000,
                                 torch.optim.Adam(caption_params, lr=0,
                                 betas=(0.9,0.98), eps=1e-9))
cap_encoder.train()
decoder.train()


lr = 1e-4
discriminator = discrimination().to(device)
discriminator.weight_init()
d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

generator = generate().to(device)
generator.weight_init()
g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=lr)

d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.99)
g_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=0.99)

loss_fn = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
####################

CRITIC_ITERS = 5
def train_discriminator(optimizer, real_data, fake_data):
    for param in discriminator.parameters():
        param.requires_grad = True
    for _ in range(CRITIC_ITERS):
        optimizer.zero_grad()
        # real_data1 = Variable(real_data[:, :])
        # print('real_data1:',real_data.shape)
        real_label = Variable(torch.ones(real_data.size(0))).cuda()
        # print('real_label.shape:',real_label.shape)
        
        discriminator_real_data = discriminator(real_data.to(torch.float32))
        loss_real = discriminator_real_data.mean(0).view(1)
        loss_real.backward()

        discriminator_fake_data = discriminator(fake_data)     
        loss_fake = discriminator_fake_data.mean(0).view(1)
        loss_fake.backward()

        gradient_penalty = calc_gradient_penalty(discriminator, real_data.data, fake_data.data)
        gradient_penalty.backward()
  
        optimizer.step()
    for parm in discriminator.parameters():
        parm.data.clamp_(-0.01, 0.01)
    return  loss_fake - loss_real+ gradient_penalty, discriminator_real_data, discriminator_fake_data

def train_generator(optimizer, fake_data):
    for param in discriminator.parameters():
        param.requires_grad = False
    optimizer.zero_grad()

    output_discriminator = discriminator(fake_data)
    loss = torch.mean(output_discriminator).view(1)
    loss.backward()
    optimizer.step()
    return loss

plt.figure()
lists=[]
lists1=[]
lists2=[]

   
     
cap_loss = 0.
# latloss =0.
# criterion_loss=0.
caption_start = 40       
# for train_idx, (real_data,caption, length) in enumerate(tq_gen):
for i, (real_data,caption, length) in enumerate(wgen):
    
    current_step =0
    cap_loss = 0.    
    real_data=real_data.cuda()
    generated_fake_data = generator(noise(real_data.size(0))).detach()
    d_loss, discriminated_real, discriminated_fake = train_discriminator(d_optimizer, real_data,
                                                                                generated_fake_data)
    generated_fake_data = generator(noise(real_data.size(0)))
    g_loss = train_generator(g_optimizer, generated_fake_data)
    re_con = generator(noise(real_data.size(0))).detach()
    print('re_con.shape:',re_con.shape)    
    if i >= caption_start:
        caption_= Variable(caption.long()).to(device)
        target = pack_padded_sequence(caption_, length, batch_first=True,enforce_sorted=False)[0]#按列取出元素
        decoder.zero_grad()
        cap_encoder.zero_grad()
            # caption_optimizer.zero_grad()
            
        result = cap_encoder(re_con)
        single_mu,single_logvar,features = result
        latloss = loss_latent(single_mu,single_logvar)  
        cap_loss += latloss 
            
        outputs = decoder(features, caption_,length)
        cap_loss=criterion(outputs, target)
        cap_loss.backward()

        caption_optimizer.step()
        print('d_loss: ', d_loss.item(), 'g_loss: ', g_loss.item(),'cap_loss:',cap_loss.item())
        lists.append(d_loss.item())
        lists1.append(g_loss.item())
        lists2.append(cap_loss.item())
    
        # print('cap_loss1:',cap_loss)  
        # i += 1
        current_step +=1      
        # break
    d_scheduler.step()
    g_scheduler.step()
        # print(len(train_data))
    if (i + 1) % 100 == 0:
        print('epoch:',i+1,'d_loss: ', d_loss.item(), 'g_loss: ', g_loss.item(),'cap_loss:',cap_loss)  
    if (i + 1) % 5000 == 0:
        torch.save(decoder.state_dict(),os.path.join(savedir,'decoder-%d.pkl' % (i + 1)))
        torch.save(cap_encoder.state_dict(),
                    os.path.join(savedir,
                                    'cap_encoder-%d.pkl' % (i + 1)))
        torch.save(generator.state_dict(),
                    os.path.join(savedir,
                                    'generator-%d.pkl' % (i + 1)))
        torch.save(discriminator.state_dict(),
                    os.path.join(savedir,
                                    'discriminator-%d.pkl' % (i + 1)))      
    if i == 100000:
        # We are Done!
        break      

df_smiles = pd.DataFrame(lists)
df_smiles.to_csv('./lossD.csv')
df_smiles1 = pd.DataFrame(lists1)
df_smiles1.to_csv('./lossG.csv')
df_smiles2 = pd.DataFrame(lists2)
df_smiles2.to_csv('./losscap.csv')

print('End Training')


wgen.close()
multiproc.close()