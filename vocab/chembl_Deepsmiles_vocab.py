"""generate the vocabulary accorrding to the regular expressions of
DeepSMILES of molecules."""
import yaml
import re
from tqdm import tqdm,trange
import deepsmiles


def read_deepsmiles_file(path, percentage):
    smiles = []
    with open(path, 'r') as f:
        temp = [line.strip("\n") for line in f.readlines()]
    for i in trange(len(temp)):
        encoded = converter.encode(temp[i])
        smiles.append(encoded)
    num_data = len(smiles)
    return smiles[0:int(num_data * percentage)]


def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string


def tokenize(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[*]' as one token."""
    regex = '(\[[^\[\]]{1,6}\])'
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokenized = []
    for char in char_list:
        if char.startswith('['):
            tokenized.append(char)
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    return tokenized


if __name__ == "__main__":

    print("DeepSMILES version: %s" % deepsmiles.__version__)
    converter = deepsmiles.Converter(rings=True, branches=True)
    print(converter)  # record the options used

    dataset_dir = "../chembl/database.cleaned.smi"
    output_vocab = "./chembl_deepsmiles_vocab.yaml"

    # read smiles as strings
    deepsmiles = read_deepsmiles_file(dataset_dir, 1)

    print("computing token set...")
    tokens = []
    [tokens.extend(tokenize(x)) for x in tqdm(deepsmiles)]
    tokens = set(tokens)
    print("finish.")

    vocab_dict = {}
    for i, token in enumerate(tokens):
        vocab_dict[token] = i

    i += 1
    vocab_dict['<eos>'] = i
    i += 1
    vocab_dict['<sos>'] = i
    i += 1
    vocab_dict['<pad>'] = i

    with open(output_vocab, 'w') as f:
        yaml.dump(vocab_dict, f)
