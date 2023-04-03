"""
Generate the vocabulary of the selfies of the smiles in the dataset
"""
import yaml
import selfies as sf
import tqdm
import pandas as pd
import pickle
import os
import random
from rdkit import Chem

def split_smiles_file(path,proportion):

    with open(path, 'r') as f:
        smiles = [line.strip("\n") for line in f.readlines()]
    for i in proportion:
        if not os.path.exists(os.path.join('../chembl',f'database_smiles_{i}.pkl')):
            with open(os.path.join('../chembl',f'database_smiles_{i}.pkl'),'wb') as file:
                print(f'{i*100}% data had generated...')
                sample_length = int(len(smiles)* i)
                samples = random.sample(smiles, sample_length)
                pickle.dump(samples,file,protocol=4)


if __name__ == "__main__":

    dataset_path = "../chembl/database.cleaned.smi"
    output_vocab = "chembl_selfies_vocab.yaml"
    proportion = [0.5,0.25]

    split_smiles_file(dataset_path,proportion)

    i = 0
    with open(dataset_path) as f:
        for line in f.readlines():
            i += 1

    smiles = [0]*(i+len(drug_smiles))

    index = 0
    with open(dataset_path, 'r') as f:
        for i in f.readlines():
            smile = i.strip("\n")
            # mol = Chem.MolFromSmiles(smile)
            # if mol is not None:
            smiles[index] = smile
            index += 1
    for drug in drug_smiles:
        smiles[index] = drug
        index += 1
    print(f'{len(smiles)} valid molecules in the dataset')
    # smiles = [line.strip("\n") for line in f.readlines() if Chem.MolToSmiles(Chem.MolFromSmiles(line.strip("\n"))) is not None]




    selfies_tokens = []
    for x in tqdm.trange(len(smiles)):
        try:
            x = sf.encoder(smiles[x])
            if x is not None:
                selfies_tokens.append(x)
        except BaseException:
                continue

    print('getting alphabet from selfies...')
    vocab = sf.get_alphabet_from_selfies(selfies_tokens)

    i = 0
    vocab_dict = {}
    for i, token in enumerate(vocab):
        vocab_dict[token] = i

    i += 1
    vocab_dict['<eos>'] = i
    i += 1
    vocab_dict['<sos>'] = i
    i += 1
    vocab_dict['<pad>'] = i

    with open(output_vocab, 'w') as f:
        yaml.dump(vocab_dict, f)











