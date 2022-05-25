import os
from dataloader import SELFIEVocab, RegExVocab, CharVocab
from model import RNN
import argparse
import torch
import yaml
import selfies as sf
from tqdm import tqdm
from rdkit import Chem

def get_args():
    parser = argparse.ArgumentParser("python")
    parser.add_argument("-result_dir",
                        required=False,
                        default= './generate_result',
                        help="directory of result files including configuration, \
                         loss, trained model, and sampled molecules"
                        )
    parser.add_argument("-batch_size",
                        required=False,
                        default=1024,
                        help="number of samples to generate per mini-batch"
                        )
    parser.add_argument("-num_batches",
                        required=False,
                        default=100,
                        help="number of batches to generate"
                        )
    parser.add_argument("-num_epochs",
                        required=False,
                        default=10,
                        help="number of batches to generate"
                        )

    return parser.parse_args()


def sample(model,num_batches,num_samples,vocab,device,with_file=False,return_value=False,out_file_path=None):

    model.eval()
    # sample, filter out invalid molecules, and save the valid molecules
    total_mols = []
    num_valid, num_invalid = 0, 0
    for i in tqdm(range(num_batches)):
        # sample molecules as integers
        sampled_ints = model.sample(
            batch_size= num_samples,
            vocab=vocab,
            device=device
        )
        selfies_mol = []
        # convert integers back to SMILES
        sampled_ints = sampled_ints.tolist()
        for ints in sampled_ints:
            molecule = []
            for x in ints:
                if vocab.int2tocken[x] == '<eos>':
                    break
                else:
                    molecule.append(vocab.int2tocken[x])
            selfies_mol.append("".join(molecule))


        # convert SELFIES back to SMILES
        molecules = [sf.decoder(x) for x in selfies_mol]
        total_mols.extend(molecules)

        if with_file:
            out_file = out_file_path
            with open(out_file,'w') as f:
            # save the valid sampled SMILES to output file,
                for smiles in molecules:
                    if smiles is None:
                        num_invalid += 1
                        continue
                    mol = Chem.MolFromSmiles(smiles)
                    if not mol:
                        num_invalid += 1
                    else:
                        num_valid += 1
                        out_file.write(smiles + '\n')
        else:
            for smiles in molecules:
                if smiles is None:
                    num_invalid += 1
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    num_invalid += 1
                else:
                    num_valid += 1
            # and compute the valid rate

    print("sampled {} valid SMILES out of {}, success rate: {}".format(
        num_valid, num_valid + num_invalid, num_valid / (num_valid + num_invalid))
    )
    if return_value:
        return total_mols





if __name__ == "__main__":

    args = get_args()
    result_dir = args.result_dir
    batch_size = int(args.batch_size)
    num_batches = int(args.num_batches)
    num_epochs = int(args.num_epochs)

    # load the configuartion file in output
    config_dir = os.path.join( result_dir , "config.yaml")
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)

    # detect cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    # load vocab
    which_vocab, vocab_path = config["which_vocab"], config["vocab_path"]

    if which_vocab == "selfies":
        vocab = SELFIEVocab(vocab_path)
    elif which_vocab == "regex":
        vocab = RegExVocab(vocab_path)
    elif which_vocab == "char":
        vocab = CharVocab(vocab_path)
    else:
        raise ValueError("Wrong vacab name for configuration which_vocab!")

    # load model
    rnn_config = config['rnn_config']
    model = RNN(rnn_config).to(device)

    # model.load_state_dict(torch.load(
    #     os.path.join('./model_parameters',f'epoch{i}.pth'),
    #     # os.path.join('./model_parameters','reinforced_model.pth'),
    #     map_location=torch.device(device)))

    path = './model_parameters/reinforced_model.pth'
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    out_file = os.path.join(result_dir,f"reinforced_molecules.smi")


    #For testing
    # molecules = sample(model,out_file_path=out_file,num_batches=100,
    #                    num_samples=1024,vocab=vocab,device=device,with_file=False)



