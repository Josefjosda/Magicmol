import pickle
import torch
import re
import yaml
import selfies as sf
import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def dataloader_gen(dataset_dir, percentage, which_vocab, vocab_path,
                   batch_size, shuffle, drop_last=True, mission_type = None):
    """
    Genrate the dataloader for training
    """
    if which_vocab == "selfies":
        vocab = SELFIEVocab(vocab_path)
    elif which_vocab == "regex":
        vocab = RegExVocab(vocab_path)
    elif which_vocab == "char":
        vocab = CharVocab(vocab_path)
    else:
        raise ValueError("Wrong vacab name for configuration which_vocab!")

    dataset = SMILESDataset(dataset_dir, percentage, vocab , mission_type= mission_type)

    PADDING_IDX = len(vocab.vocab)-1

    def pad_collate(batch):
        """
        Put the sequences of different lengths in a minibatch by paddding.
        """
        lengths = [len(x) for x in batch]

        # embedding layer takes long tensors
        batch = [torch.tensor(x, dtype=torch.long) for x in batch]

        x_padded = pad_sequence(
            batch, 
            batch_first=True,
            padding_value=PADDING_IDX
        )

        return x_padded, lengths

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=drop_last, 
        collate_fn=pad_collate
    )

    return dataloader, len(dataset)


class SMILESDataset(Dataset):
    def __init__(self, smiles_file, percentage, vocab , mission_type = None):
        """
        smiles_file: path to the .smi file containing SMILES.
        percantage: percentage of the dataset to use.
        """
        super(SMILESDataset, self).__init__()
        assert(0 < percentage <= 1)

        self.percentage = percentage
        self.vocab = vocab

        # load eaqual portion of data from each tranche
        self.data = self.read_smiles_file(smiles_file)
        print("total number of SMILES loaded: ", len(self.data))

        # convert the smiles to selfies
        temp = []
        if self.vocab.name == "selfies":
            if mission_type != 'fine-tune':
                if not os.path.exists(os.path.join('vocab','database_selfies_encode.pkl')):
                    for i in tqdm.trange(len(self.data)):
                        try:
                            if sf.encoder(self.data[i]) is not None:
                                try:
                                    mol = sf.encoder(self.data[i])
                                    if self.vocab.tokenize_smiles(mol):
                                        temp.append(sf.encoder(self.data[i]))
                                except KeyError as K:
                                    continue
                        except sf.exceptions.EncoderError as E:
                            continue

                    # self.data = [sf.encoder(self.data[x])
                    #              for x in tqdm.trange(len(self.data)) if sf.encoder(self.data[x]) is not None]
                    with open(os.path.join('vocab','database_selfies_encode.pkl'),'wb') as f:
                        pickle.dump(temp,f,protocol=4)
                else:
                    with open(os.path.join('vocab','database_selfies_encode.pkl'),'rb') as f:
                        self.data = pickle.load(f)
            else:
                data_ = []
                for i in tqdm.trange(len(self.data)):
                    try:
                        if sf.encoder(self.data[i]) is not None:
                            if self.vocab.tokenize_smiles(sf.encoder(self.data[i])) != None:
                                data_.append(sf.encoder(self.data[i]))
                    except BaseException:
                        continue
                self.data = data_
            print("total number of valid SELFIES: ", len(self.data))

    def read_smiles_file(self, path):
        # need to exclude first line which is not SMILES
        with open(path, "rb") as f:
            smiles = pickle.load(f)
        #     smiles = [line.strip("\n") for line in f.readlines()]

        num_data = len(smiles)

        return smiles[0:int(num_data * self.percentage)]

    def get_selfie_chars(self,selfie):
        '''Obtain a list of all selfie characters in string selfie

        Parameters:
        selfie (string) : A selfie string - representing a molecule

        Example:
        # >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
        ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']

        Returns:`
        chars_selfie: list of selfie characters present in molecule selfie
        '''
        chars_selfie = []  # A list of all SELFIE sybols from string selfie
        while selfie != '':
            chars_selfie.append(selfie[selfie.find('['): selfie.find(']') + 1])
            selfie = selfie[selfie.find(']') + 1:]
        return chars_selfie

    def __getitem__(self, index):
        mol = self.data[index]

        # convert the data into integer tokens
        mol = self.vocab.tokenize_smiles(mol)

        return mol

    def __len__(self):
        return len(self.data)


class CharVocab:
    def __init__(self, vocab_path):
        self.name = "char"

        # load the pre-computed vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = yaml.full_load(f)

        # a dictionary to map integer back to SMILES
        # tokens for sampling
        self.int2tocken = {}
        for token, num in self.vocab.items():
            self.int2tocken[num] = token

        # a hashset of tokens for O(1) lookup
        self.tokens = self.vocab.keys()

    def tokenize_smiles(self, smiles):
        """
        Takes a SMILES string and returns a list of tokens.
        Atoms with 2 characters are treated as one token. The 
        logic references this code piece:
        https://github.com/topazape/LSTM_Chem/blob/master/lstm_chem/utils/smiles_tokenizer2.py
        """
        n = len(smiles)
        tokenized = ['<sos>']
        i = 0

        # process all characters except the last one
        while (i < n - 1):
            # procoss tokens with length 2 first
            c2 = smiles[i:i + 2]
            if c2 in self.tokens:
                tokenized.append(c2)
                i += 2
                continue

            # tokens with length 2
            c1 = smiles[i]
            if c1 in self.tokens:
                tokenized.append(c1)
                i += 1
                continue

            raise ValueError(
                "Unrecognized charater in SMILES: {}, {}".format(c1, c2))

        # process last character if there is any
        if i == n:
            pass
        elif i == n - 1 and smiles[i] in self.tokens:
            tokenized.append(smiles[i])
        else:
            raise ValueError(
                "Unrecognized charater in SMILES: {}".format(smiles[i]))

        tokenized.append('<eos>')

        tokenized = [self.vocab[token] for token in tokenized]
        return tokenized

    def combine_list(self, smiles):
        return "".join(smiles)


class RegExVocab:
    def __init__(self, vocab_path):
        self.name = "regex"

        # load the pre-computed vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = yaml.full_load(f)

        # a dictionary to map integer back to SMILES
        # tokens for sampling
        self.int2tocken = {}
        for token, num in self.vocab.items():
            if token == "R":
                self.int2tocken[num] = "Br"
            elif token == "L":
                self.int2tocken[num] = "Cl"
            else:
                self.int2tocken[num] = token

    def tokenize_smiles(self, smiles):
        """Takes a SMILES string and returns a list of tokens.
        This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
        '[xx]' as one token."""
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = self.replace_halogen(smiles)
        char_list = re.split(regex, smiles)

        tokenized = ['<sos>']

        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append('<eos>')

        # convert tokens to integer tokens
        tokenized = [self.vocab[token] for token in tokenized]

        return tokenized

    def replace_halogen(self, string):
        """Regex to replace Br and Cl with single letters"""
        br = re.compile('Br')
        cl = re.compile('Cl')
        string = br.sub('R', string)
        string = cl.sub('L', string)

        return string

    def combine_list(self, smiles):
        return "".join(smiles)


class SELFIEVocab:
    def __init__(self, vocab_path):
        self.name = "selfies"

        # load the pre-computed vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = yaml.full_load(f)

        self.int2tocken = {value: key for key, value in self.vocab.items()}

    def tokenize_smiles(self, mol):
        """convert the smiles to selfies, then return 
        integer tokens."""
        ints = [self.vocab['<sos>']]

        #encoded_selfies = sf.encoder(smiles)
        selfies_list = list(sf.split_selfies(mol))

        for token in selfies_list:
               ints.append(self.vocab[token])

        ints.append(self.vocab['<eos>'])

        return ints

    def combine_list(self, selfies):
        return "".join(selfies)
