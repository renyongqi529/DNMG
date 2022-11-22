import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
voc_set=['pad', 'bos', 'eos', '5', 'Y', ')', 'Z', '[', ']', '-', 
    'S', '1', 'O', 'N', "'", ' ', 'C', '(', 'n', 'c', '#', 's', '6', 
    'X', '4', ',', '2', 'o', 'F', '=', '3', '.', 'I', '/', '+', '\\', '@', 'H', 'P']
vocab_i2c_v1 = {i: x for i, x in enumerate(voc_set)}
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}
def decode_smiles(in_tensor):
    """
    Decodes input tensor to a list of strings.
    :param in_tensor:
    :return:
    """
    gen_smiles = []
    for sample in in_tensor:
        csmile = ""
        for xchar in sample[1:]:
            if xchar == 2:
                break
            csmile += vocab_i2c_v1[xchar]
        gen_smiles.append(csmile)
    return gen_smiles
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        enc = []
        shr_enc = []
        self.relu = nn.ReLU()
        in_channels=19
        out_channels = 32
        for i in range(8):
            enc.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            enc.append(nn.BatchNorm3d(out_channels))
            enc.append(nn.ReLU())
            in_channels=out_channels

            if (i+1) % 2 ==0: 
                out_channels *= 2
                enc.append(nn.MaxPool3d((2, 2, 2)))
        enc.pop()
        self.fc11 = nn.Linear(256,512)
        self.fc12 = nn.Linear(256,512)
        self.single_encoder=nn.Sequential(*enc)

        
    def reparametrize(self, mu, logvar,factor):
        std = logvar.mul(0.5).exp_()# 矩阵点对点相乘之后再把这些元素作为e的指数
        #eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = torch.FloatTensor(std.size()).normal_()# 生成随机数组
        #eps = Variable(eps)
        eps = Variable(eps).cuda()
        return (eps.mul(std)* factor).add_(mu)# 用一个标准正态分布乘标准差，再加上均值，使隐含向量变为正太分布
    
    def forward(self, input_data,factor=1.):
        result=[]   

        x = self.single_encoder(input_data)
        x = x.mean(dim=2).mean(dim=2).mean(dim=2) 
        single_mu,single_logvar = self.fc11(x), self.fc12(x)
        single_latent = self.reparametrize(single_mu,single_logvar,factor=factor)# 重新参数化成正态分布
        result.extend([single_mu,single_logvar, single_latent])
        return result
    
class Decode(nn.Module):
    def __init__(self):
        """Set the hyper-parameters and build the layers."""
        super(Decode, self).__init__()
        embed_size=512
        hidden_size=1024
        vocab_size=39
        num_layers=1
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        self.linear1 =  nn.Linear(embed_size,960)
        self.linear2 =  nn.Linear(embed_size,512-8)
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decode shapes feature vectors and generates SMILES."""
        embedding = self.embedding(captions)
        
        embedding = torch.cat((features.unsqueeze(1), embedding), 1)

        packed = pack_padded_sequence(embedding, lengths, batch_first=True,enforce_sorted=False)

        hiddens, _ = self.lstm(packed)

        outputs = self.linear(hiddens[0])
   
        return outputs
    
    def sample(self, features,states=None):
        """Samples SMILES tockens for given shape features (Greedy search)."""
        sampled_ids = []
  
        inputs = features.unsqueeze(1)

        for i in range(80):
            hiddens, states = self.lstm(inputs,states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]

            sampled_ids.append(predicted)
            inputs = self.embedding(predicted) 
            inputs = inputs.unsqueeze(1)
            
        return sampled_ids
    def sample_prob(self, features, states=None):
        """Samples SMILES tockens for given shape features (probalistic picking)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(80):  # maximum sampling length
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            if i == 0:
                predicted = outputs.max(1)[1]
            else:
                probs = F.softmax(outputs, dim=1)

                # Probabilistic sample tokens
                if probs.is_cuda:
                    probs_np = probs.data.cpu().numpy()
                else:
                    probs_np = probs.data.numpy()

                rand_num = np.random.rand(probs_np.shape[0])
                iter_sum = np.zeros((probs_np.shape[0],))
                tokens = np.zeros(probs_np.shape[0], dtype=np.int)

                for i in range(probs_np.shape[1]):
                    c_element = probs_np[:, i]
                    iter_sum += c_element
                    valid_token = rand_num < iter_sum
                    update_indecies = np.logical_and(valid_token,
                                                     np.logical_not(tokens.astype(np.bool)))#astype:0代表False 非0代表True
                    tokens[update_indecies] = i

                # put back on the GPU.
                if probs.is_cuda:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)).cuda())
                else:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)))#int:0代表0，非0代表1

            sampled_ids.append(predicted)
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)
   
        return sampled_ids
    

class discrimination(nn.Module):
    def __init__(self, nc=19, ngf=128, ndf=128, latent_variable_size=512, use_cuda=False):
        super(discrimination, self).__init__()
        self.use_cuda = use_cuda
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        
        self.e1 = nn.Conv3d(nc, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(32)

        self.e2 = nn.Conv3d(32, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm3d(32)

        self.e3 = nn.Conv3d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm3d(64)

        self.e4 = nn.Conv3d(64, ndf * 4, 3, 2, 1)
        self.bn4 = nn.BatchNorm3d(ndf * 4)

        self.e5 = nn.Conv3d(ndf * 4, ndf * 4, 3, 2, 1)
        self.bn5 = nn.BatchNorm3d(ndf * 4)

        # self.fc1 = nn.Linear(512 * 27, latent_variable_size)
        
        self.fc1 = nn.Linear(512 * 8, latent_variable_size)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
    def forward(self, x):
     
        h1 = self.leakyrelu(self.bn1(self.e1(x)))

        h2 = self.leakyrelu(self.bn2(self.e2(h1)))

        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
     
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
 
        h5 = h5.view(h5.size(0),-1)

        h6 = self.fc1(h5)
        h7 = self.fc2(h6)
        

        return h7
    
    def weight_init(m):
        # weight_initialization: important for wgan
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            m.weight.data.normal_(0, 0.02)
        elif class_name.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)


class generate(nn.Module):
    def __init__(self, nc=19, ngf=128, ndf=128, latent_variable_size=512, use_cuda=False):
        super(generate, self).__init__()
        self.use_cuda = use_cuda
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
     
        # up2 12 -> 24
        self.d5 = nn.ConvTranspose3d(ndf, 32, 3, 2, padding=1, output_padding=1)
        self.bn9 = nn.BatchNorm3d(32, 1.e-3)

        # Output layer
        self.d6 = nn.Conv3d(32, nc, 3, 1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self, x):
       
        h5 = self.leakyrelu(self.bn9(self.d5(x)))
        print(h5.shape)
        return h5
    
    def weight_init(m):
        # weight_initialization: important for wgan
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            m.weight.data.normal_(0, 0.02)
        elif class_name.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)

