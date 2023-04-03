import os
from dataloader import SELFIEVocab, RegExVocab, DSVocab
from model import RNN
import argparse
import torch
import yaml
import selfies as sf
from tqdm import tqdm
from rdkit import Chem
import pickle
import deepsmiles


def sample(model,num_batches,num_samples,vocab,device,with_file=False,return_mols=False,out_file_path=None
           ,which_vocab=None,return_valid_mols=False):

    model.eval()
    # sample, filter out invalid molecules, and save the valid molecules
    valid_mols = []
    total_mols = []
    num_valid, num_invalid = 0, 0
    error_count = 0
    for i in tqdm(range(num_batches)):
        # sample molecules as integers
        sampled_ints = model.sample(
            batch_size= num_samples,
            vocab=vocab,
            device=device
        )
        translated_mols = []

        # convert integers back to SMILES
        sampled_ints = sampled_ints.tolist()
        for ints in sampled_ints:
            molecule = []
            for x in ints:
                if vocab.int2tocken[x] == '<eos>':
                    break
                else:
                    molecule.append(vocab.int2tocken[x])
            translated_mols.append("".join(molecule))

        # convert SELFIES back to SMILES
        if which_vocab == 'selfies':
            molecules = [sf.decoder(x) for x in translated_mols]
            total_mols.extend(molecules)
        elif which_vocab == 'DeepSMILES':
            _molecules = []
            converter = deepsmiles.Converter(rings=True, branches=True)
            for mol in translated_mols:
                try:
                    mol = converter.decode(mol)
                    _molecules.append(mol)
                except deepsmiles.DecodeError as e:
                    print("DecodeError! Error message was '%s'" % e.message)
                    error_count += 1
                    continue
                except IndexError:
                    error_count += 1
                    continue
            molecules = _molecules
            total_mols.extend(molecules)
        else:
            molecules = translated_mols

        for smiles in molecules:
            if smiles is None:
                num_invalid += 1
                continue
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                num_invalid += 1
            else:
                num_valid += 1
                valid_mols.append(smiles)
        # and compute the valid rate

    if which_vocab == 'DeepSMILES':
        print(f"{error_count} molecules can't be decoded.")
        print("Remained {} valid SMILES out of {}, success rate: {}".format(
            num_valid, num_valid + num_invalid, num_valid / (num_valid + num_invalid))
        )
    else:
        print("sampled {} valid SMILES out of {}, success rate: {}".format(
            num_valid, num_valid + num_invalid, num_valid / (num_valid + num_invalid))
        )

    if return_mols:
        return total_mols

    if return_valid_mols:
        return valid_mols

    if with_file:
        smi_out_file = os.path.join(out_file_path,f'{which_vocab}.smi')
        with open(smi_out_file, 'w') as f:
            # save the valid sampled SMILES to output file,
            for mol in total_mols:
                if mol is not None:
                    f.write(mol + '\n')
        pkl_out_file = os.path.join(out_file_path,f'{which_vocab}.pkl')
        with open(pkl_out_file,'wb') as fi:
            pickle.dump(total_mols,fi,protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Magicmol")
    parser.add_argument("-result_dir",
                        required=False,
                        default='./generate_result',
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
    parser.add_argument("--result_dir",
                        default='./generate_result'
                        )
    parser.add_argument("--which_vocab",
                        required=False,
                        default='DeepSMILES'
                        )
    parser.add_argument("--vocab_path",
                        required=False,
                        default='./vocab/chembl_deepsmiles_vocab.yaml'
                        )
    parser.add_argument("--model_weight_path",
                        required=False,
                        default='model_parameters/7_trained_model_DeepSMILES.pth'
                        )
    parser.add_argument("--num_samples",
                        required=False,
                        default=1024,
                        help="sampled molecules for a single step"
                        )
    parser.add_argument("--num_batches",
                        required=False,
                        default=100,
                        )
    parser.add_argument('--rnn_type', type=str, default='GRU')
    #SELFIES - 148 , regex - 101, DeepSMILES - 100
    parser.add_argument('--num_embeddings', type=int, default=100)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_epoch', type=int, default=10)
    config = parser.parse_args()

    # detect cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    # load vocab
    which_vocab, vocab_path = config.which_vocab, config.vocab_path

    if which_vocab == "selfies":
        vocab = SELFIEVocab(vocab_path)
    elif which_vocab == "regex":
        vocab = RegExVocab(vocab_path)
    elif which_vocab == "DeepSMILES":
        vocab = DSVocab(vocab_path)
    else:
        raise ValueError("Wrong vocab name for configuration which_vocab!")

    # load model
    rnn_config = {'num_embeddings': config.num_embeddings, 'embedding_dim': config.embedding_dim,
                  'rnn_type': config.rnn_type, 'input_size': config.input_size,
                  'hidden_size': config.hidden_size, 'num_layers': config.num_layers, 'dropout': config.dropout}
    model = RNN(rnn_config).to(device)
    path = config.model_weight_path
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    # model.load_state_dict(checkpoint)
    # model_name = checkpoint['model_name']
    out_file = config.result_dir


    #For testing
    molecules = sample(model,out_file_path=out_file,num_batches=config.num_batches,
                       num_samples=config.num_samples,vocab=vocab,device=device,with_file=True,which_vocab=config.which_vocab
                       ,return_valid_mols=False)



