#!/usr/bin/env python
# coding: utf-8

from molecular_generation import MolecularGenerator
import os
import torch
import pandas as pd
import csv
from rdkit import Chem
from rdkit.Chem import AllChem


torch.backends.cudnn.enabled = False
my_gen = MolecularGenerator(use_cuda=True)  # set use_cuda=False if you do not have a GPU.

# Load the weights of the models
D_weights =  os.path.join("./2qd9_gan/discriminator-100000.pkl")
G_weights =  os.path.join("./2qd9_gan/generator-100000.pkl")
encoder_weights =  os.path.join("./2qd9_gan/encoder-100000.pkl")
decoder_weights =os.path.join("./2qd9_gan/decoder-100000.pkl")
my_gen.load_weight(D_weights, G_weights, encoder_weights, decoder_weights)


gen_mols = my_gen.generate_molecules(
                                        n_attemps=1000,  # How many attemps of generations will be carried out
                                        lam_fact=1.,  # Variability factor
                                        probab=True, # Probabilistic RNN decoding
                                        filter_unique_valid=True)  # Filter out invalids and replicates
# with open('ippigan_p53.csv', 'a') as f:
#     for i in range(len(gen_mols)):
#         smi = gen_mols[i]
#         f.write(smi+"\n")
f=open('./2qd9_gan/3.csv','w')
wr=csv.writer(f)
count=0
rows=[]
for m in gen_mols:
    count+=1
    rows.append(m)

def list_of_groups(init_list, children_list_len):
    list_of_groups = zip(*(iter(init_list),) *children_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % children_list_len
    end_list.append(init_list[-count:]) if count !=0 else end_list
    return end_list

code_list = list_of_groups(rows,1)
print(code_list)
for i in range(count):
    # print('\t',code_list[i])
    # print(code_list[i])
    x=code_list[i]
    print(x)      
    wr.writerows([x])
f.close()
