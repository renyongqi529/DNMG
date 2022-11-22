from rdkit import Chem
from rdkit.Chem import AllChem
from torch.autograd import Variable
import torch
from torch.utils import data
import pandas as pd
import numpy as np
import torch.nn.functional as F
import pybel
from data_generator import Featurizer,rotation_matrix,rotate,make_grid
from gan1 import  generate, discrimination,Encoder,Decode
device = torch.device('cuda')
featurizer = Featurizer()

def filter_unique_canonical(in_mols):
    """
    :param in_mols - list of SMILES strings
    :return: list of uinique and valid SMILES strings in canonical form.
    """
    xresults = [Chem.MolFromSmiles(x) for x in in_mols]  # Convert to RDKit Molecule
    xresults = [Chem.MolToSmiles(x) for x in xresults if x is not None]  # Filter out invalids
    return  xresults# Check for duplicates and filter out invalids

voc_set=['pad', 'bos', 'eos', '5', 'Y', ')', 'Z', '[', ']', '-', 
    'S', '1', 'O', 'N', "'", ' ', 'C', '(', 'n', 'c', '#', 's', '6', 
    'X', '4', ',', '2', 'o', 'F', '=', '3', '.', 'I', '/', '+', '\\', '@', 'H', 'P']
# voc_set=['pad', 'bos', 'eos', 'Y', '-', 'o', '[', ' ', 'N', 'P', '1', 's', '8', '7', '(', '5', 
#      'C', '3', 'X', 'n', 'Z', 'c', 'H', '+', ',', ')', "'", 'S', 'O', '4', ']', '#', '2', 'I', 'F', '=', '6','9','@']
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

class MolecularGenerator:
    def __init__(self, use_cuda=True):

        self.use_cuda = False

        self.encoder = Encoder()
        self.decoder = Decode()

        self.D = discrimination()
        self.G = generate()

        self.D.eval()
        self.G.eval()
        self.encoder.eval()
        self.decoder.eval()

        if use_cuda:
            assert torch.cuda.is_available()
            self.encoder.cuda()
            self.decoder.cuda()
            self.D.cuda()
            self.G.cuda()
            self.use_cuda = True
   
    def load_weight(self, D_weights, G_weights, encoder_weights, decoder_weights):
        """
        Load the weights of the models.
        :param vae_weights: str - VAE model weights path
        :param encoder_weights: str - captioning model encoder weights path
        :param decoder_weights: str - captioning model decoder model weights path
        :return: None
        """
        self.D.load_state_dict(torch.load(D_weights, map_location='cpu'))
        self.G.load_state_dict(torch.load(G_weights, map_location='cpu'))
        self.encoder.load_state_dict(torch.load(encoder_weights, map_location='cpu'))
        self.decoder.load_state_dict(torch.load(decoder_weights, map_location='cpu'))
        
    def generate_molecules(self, n_attemps=300, lam_fact=1., probab=False,filter_unique_valid=True):
        """
        Generate novel compounds from a seed compound.
        :param smile_str: string - SMILES representation of a molecule
        :param n_attemps: int - number of decoding attempts
        :param lam_fact: float - latent space pertrubation factor
        :param probab: boolean - use probabilistic decoding
        :param filter_unique_canonical: boolean - filter for valid and unique molecules
        :return: list of RDKit molecules.
        """
  
        z = Variable(torch.randn(n_attemps, 128, 12, 12, 12)).cuda()
        recoded_shapes = self.G(z)
        
        union_x = self.encoder(recoded_shapes,lam_fact)
        single_mu,single_logvar,features = union_x
        if probab:
            captions = self.decoder.sample_prob(features)
        else:
            captions = self.decoder.sample(features)

        captions = torch.stack(captions, 1)
        
        captions = captions.cpu().data.numpy()

        lists = []
        lists1 = []
        for i in captions:
            lists.append(i)
        print(lists)
        smiles = decode_smiles(lists)
        for i in smiles:
            i = i.replace("X", "Cl").replace("Y", "[nH]").replace("Z", "Br")
            lists1.append(i)
        if filter_unique_valid:
            return filter_unique_canonical(smiles)
        return [Chem.MolFromSmiles(x) for x in smiles]
        