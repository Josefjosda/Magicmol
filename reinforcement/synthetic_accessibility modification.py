import yaml
import os
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
import selfies as sf
import pickle
import numpy as np
from torch.utils.data import Dataset,DataLoader
import glob
import sys
sys.path.insert(0,'../')
from model import RNN
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from tqdm import trange,tqdm
from torch.nn.utils.rnn import pad_sequence
from dataloader import dataloader_gen
from  syba.syba import SybaClassifier
from sample import sample



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

        # encoded_selfies = sf.encoder(smiles)
        selfies_list = list(sf.split_selfies(mol))

        for token in selfies_list:
            ints.append(self.vocab[token])

        ints.append(self.vocab['<eos>'])

        return ints


def combine_list(self, selfies):
    return "".join(selfies)

def mol2image(x, n=2048):
    try:
        m = Chem.MolFromSmiles(x)
        fp = Chem.RDKFingerprint(m, maxPath=4, fpSize=n)
        res = np.zeros(len(fp))
        Chem.DataStructs.ConvertToNumpyArray(fp, res)
        return res
    except:
        return [np.nan]

def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

def get_fp(smiles):

    fp = []
    processed_indices = []
    invalid_indices = []
    for i in range(len(smiles)):
        mol = smiles[i]
        tmp = np.array(mol2image(mol, n=2048))
        if np.isnan(tmp[0]):
            invalid_indices.append(i)
        else:
            fp.append(tmp)
            processed_indices.append(i)
    return np.array(fp), processed_indices, invalid_indices

def predict(smiles, average=True, get_features=None,predictor=None,use_syba=True):

    x, processed_indices, invalid_indices = get_features(smiles)
    processed_objects = processed_indices
    invalid_objects = invalid_indices
    prediction = []

    if not get_features:
        x = smiles
    if len(x) == 0:
        processed_objects = []
        prediction = []
        invalid_objects = smiles

    if use_syba:
        processed_objects = processed_indices
        invalid_objects = invalid_indices
        for i in processed_objects:
            try:
                prediction.append(predictor.predict(smi=smiles[i]))
            except BaseException:
                return
    else:
        for i in range(len(predictor)):
            # if normalization:
            #     x, _ = normalize_desc(x, self.desc_mean[i])
            prediction.append(predictor[i].predict(x))
        prediction = np.array(prediction)
        if average:
            prediction = prediction.mean(axis=0)
    return  processed_objects,prediction,invalid_objects

def get_score(smiles, get_features=get_fp, predictor=None):

    mol, syba_score, nan_smiles = predict(smiles, get_features=get_features,predictor=predictor)
    x = np.exp(  np.multiply(1.0/150 , syba_score) )
    rl_score = x + np.exp(1)
    return syba_score, rl_score , nan_smiles


def fit_model(valid_model,feature,score,ensemble_size=None):
    eval_metrics = []
    X = feature
    y = score.values
    kf = KFold(n_splits=ensemble_size, shuffle=True)

    i = 0
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        valid_model[i].fit(X_train, y_train)
        predicted = valid_model[i].predict(X_test)

        r2 = r2_score(y_test, predicted)
        print(r2)
        eval_metrics.append(r2)
        # metrics_type = 'R^2 score'

        i += 1

    # return eval_metrics, metrics_type

import seaborn as sns

def plot_hist(prediction,label):

    prediction = np.array(prediction)
    percentage_in_threshold = np.sum((prediction >= 0.0) &
                                     (prediction <= 300))/len(prediction)
    print("Percentage of predictions within easy to systhesis region:", percentage_in_threshold)
    ax = sns.distplot(prediction,kde_kws={"label":f"{label}"})
    plt.axvline(x=0.0)
    ax.set(xlabel='Predicted systhetic score',
           title='Distribution of predicted systhetic score for generated molecules')
    plt.legend()
    # plt.show()

class Reinforcement(object):
    def __init__(self, generator, predictor, get_loss, vocab, device, optimizer):
        super(Reinforcement, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.get_loss = get_loss
        self.vocab = vocab
        self.device = device
        self.optimizer = optimizer
        self.reinforce_threshold = 20

    def to_tensor(self, token_list):
        # if self.device == 'cuda':
        return torch.tensor(token_list).cuda()
        # else:
        #     return torch.tensor(token_list)


    def policy_gradient(self,gamma=0.97,**kwarg):

        self.generator.train()
        rl_loss = 0
        rl_score_sum = 0
        #Define the sample batch
        while True:
            sample_size = 10
            sampled_result = self.generator.sample(batch_size=sample_size,vocab=self.vocab,device=self.device,max_length=140)
            sampled_ints = sampled_result.tolist()
            if sample_size == 1 :
                sampled_ints = [sampled_ints]
            molecules,int_tokens = [],sampled_ints
            for ints in sampled_ints:
                molecule = []
                for x in ints:
                    if self.vocab.int2tocken[x] == '<eos>':
                        break
                    else:
                        molecule.append(self.vocab.int2tocken[x])
                molecules.append(molecule)

            smiles = [sf.decoder(''.join(x)) for x in molecules]
            print(f'Decode {len(smiles)} smiles.')

            invalid_index = []
            for i in range(len(smiles)):
                try:
                    _ = Chem.MolFromSmiles(smiles[i])
                except BaseException:
                    invalid_index.append(i)
                    print('Found an invalid molecule.')
                    continue

            for i in invalid_index:
                del sampled_ints[i]
                print(f"Del index {i} from sampled_ints.")
                del smiles[i]
                print(f"Del index {i} from smiles.")

            #Get systhetic score of generated molecules using predictor.
            syba_score, rl_score , nan_smiles = self.get_loss(smiles= smiles, predictor=self.predictor)
            if not rl_score.all():
                continue

            #because of the unbalanced distribution of the molecules, we take two positive-score molecules and
            # a single negative_score molecules for learning during one certain epoch
            # negamols_index_logi = syba_score < 0
            # index = np.where(negamols_index_logi == True)[0]
            # min_val = np.sort(syba_score[index])
            # # if len(min_val) >= 2:
            # min_val = min_val[0]
            # nega_index = np.argwhere(syba_score == min_val).flatten()[0]
            # #
            # n_batch = 5
            # min_score_list = np.sort(syba_score)
            # min_score = min_score_list[0]
            # index =  [np.argwhere(syba_score == min_score).flatten()[0] ]
            # index = [index]
            # index = np.where(posi_index_logi == True)[0]

            valid_mols_count = len(rl_score)
            waive_optim_threshold = 150
            syba_score = np.array(syba_score)
            posi_index_logi = syba_score > 0
            waive_index_logi = syba_score > waive_optim_threshold
            for i in range(len(rl_score)):
                if waive_index_logi[i]:
                    rl_score[i] = 0
                    continue
                elif posi_index_logi[i]:
                    rl_score[i] /= 2


            # min_val = min_val[0]
            # posi_index = np.argwhere(syba_score == min_val).flatten()[0]

            # index = [posi_index , nega_index]

            if valid_mols_count == 0:
                continue
            else:
                break

        torch.cuda.empty_cache()

        for i in range(valid_mols_count):

            trajectory_input = self.to_tensor(sampled_ints[i])
            discounted_reward = rl_score[i]
            if discounted_reward == 0:
                continue
            rl_score_sum += discounted_reward
            hidden = None

            for item in range(len(trajectory_input)):

                if item == 0:
                    last_output,x,hidden = self.generator.sample(batch_size=sample_size,
                                                          provided_prefix=trajectory_input[item],device=device,rein=True
                                                          ,vocab=vocab,first_time=True)
                else:
                    last_output,x,hidden = self.generator.sample(batch_size=sample_size,
                                                          provided_prefix= x,device=device,rein=True,
                                                          vocab=vocab,first_time=False,hidden=hidden)

                softmax_probs = F.log_softmax(torch.squeeze(last_output,dim=1), dim=1)
                top_i = trajectory_input[item]
                if kwarg['task'] == 'optim_n':
                    rl_loss -= (softmax_probs[i, top_i] * discounted_reward)
                elif kwarg['task'] == 'optim_p':
                    rl_loss += (softmax_probs[i, top_i] * discounted_reward)
                discounted_reward = discounted_reward * gamma

        if rl_loss == 0:
            return rl_score_sum,rl_loss
            # Doing backward pass and parameters update
        # rl_loss = rl_loss / n_batch
        rl_loss = rl_loss / valid_mols_count
        self.optimizer.zero_grad()
        rl_loss.backward()
        self.optimizer.step()


            # print(f"rl_loss batch:{epoch} - {rl_loss}" )
            # total_reward = total_reward / n_batch
            # if rl_loss >= torch.tensor(0):
            # if grad_clipping is not None:
            #     torch.nn.utils.clip_grad_norm_(self.generator.parameters(),
            #                                    grad_clipping)

        return rl_score_sum , rl_loss.detach().item()

if __name__ == "__main__":


    torch.backends.cudnn.enabled = True
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print('device: ', device)

    config_dir = "reinforcement.yaml"
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)


    vocab_path = config['vocab_path']
    # create dataloader
    batch_size = config['batch_size']
    shuffle = config['shuffle']

    rnn_config = config['rnn_config']
    model = RNN(rnn_config).to(device)
    model.train()


    # for param in model.parameters():
    #     param.requires_grad = True

    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    # if config['which_optimizer'] == "adam":
    #     optimizer = torch.optim.Adam(
    #         model.parameters(), lr=learning_rate,
    #         weight_decay=weight_decay, amsgrad=True
    #     )

    # path = '../model_parameters/15_epoch.pth'
    # checkpoint = torch.load(path, map_location=device)
    # model.load_state_dict(checkpoint['net'])
    # optimizer_state_dict = checkpoint['optimizer']
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
    #                                      weight_decay=weight_decay)
    # optimizer.load_state_dict(optimizer_state_dict)
    # print(f"Load model parameters from {path}.")

    if os.path.exists('../model_parameters/trained_model.pth'):
        path = '../model_parameters/trained_model.pth'
        checkpoint = torch.load(path,map_location=device)
        model.load_state_dict(checkpoint['net'])
        optimizer_state_dict = checkpoint['optimizer']
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
            weight_decay=weight_decay)
        optimizer.load_state_dict(optimizer_state_dict)
        # start_epoch = checkpoint['epoch']
        print(f"Load model parameters from {path}.")
    else:
        optimizer = torch.optim.Adadelta(
            model.parameters(), lr=learning_rate,
            weight_decay=weight_decay)

    use_syba = True
    ensemble_size = 5


    #It should be better to use syba, if syba is not available for you,
    # use a random forest model as the substitution.
    if use_syba :
        validate_model = SybaClassifier()
        validate_model.fitDefaultScore()
        sys_valid_model = validate_model
        print('Predictor: syba.')
    else:
        sys_valid_model_instance = RFR
        sys_valid_model_params = {'n_estimators': 250, 'n_jobs': 10}

        sys_valid_model = []
        for i in range(ensemble_size):
           sys_valid_model.append(sys_valid_model_instance(n_estimators=sys_valid_model_params['n_estimators'],
                                                           n_jobs=sys_valid_model_params['n_jobs']))
        print('Predictor: machine learning model.')
        data = pd.read_csv('../chembl/with_score.csv')
        smiles_name = data['cmpdname']
        smiles = data['isosmiles']
        smiles_score =  data['estimated_sys_score']

        feature,_,_ = get_fp(smiles)
        #Use machine learning model to fit the sys_score.
        fit_model(valid_model=sys_valid_model,feature=feature,score=smiles_score,ensemble_size=ensemble_size)

    # generated_smiles_path = glob.glob(pathname=config['datapath']+'/sampled_molecules_epoch*.smi')
    # generated_smiles = []
    # for file_path in generated_smiles_path:
    #     with open(file_path) as f:
    #         for line in f.readlines():
    #             generated_smiles.append(line.strip())


    # # Use selfies vocab to encode smiles.
    vocab_path = config['vocab_path']
    vocab = SELFIEVocab(vocab_path)
    num_batches = 10
    num_samples = 1024
    sampled_mols = sample(model, num_batches=num_batches, num_samples=num_samples,
           vocab=vocab, device=device, with_file=False, return_value=True)

    sys_score = []

    if use_syba:
        for i in range(len(sampled_mols)):
            try:
                mol = Chem.MolFromSmiles(sampled_mols[i])
                sys_score.append(sys_valid_model.predict(mol=mol))
            except:
                continue
        print(f'Average systhesis score = {np.mean(sys_score)}')

    plot_hist(sys_score,label='former synthetic prediction')
    # plt.show()


    n_policy = 10
    nega_train_iterations =  15
    posi_iterations =  10
    RL_max = Reinforcement(model, sys_valid_model, get_score, vocab=vocab, device=device, optimizer=optimizer)

    reward_sum = []
    rl_losses = []

    # optimize negative molecules
    # for i in range(nega_train_iterations):
    #     for j in trange(n_policy, desc='Policy gradient...'):
    #         torch.cuda.empty_cache()
    #         cur_reward, cur_loss = RL_max.policy_gradient(get_features=get_fp,task_type='optim_n')
    #         reward_sum.append(cur_reward)
    #         rl_losses.append(cur_loss)
    #
    #     # plt.plot(reward_sum)
    #     # plt.xlabel('Training iteration')
    #     # plt.ylabel('score_sum')
    #     # plt.savefig(f'../figure/{i}_{j+1}_score_sum.png')
    #     # plt.show()
    #
    #     plt.plot(rl_losses)
    #     plt.xlabel('Training iteration')
    #     plt.ylabel('Loss')
    #     plt.savefig(f'../figure/{i}_{j+1}_Loss.png')
    #     plt.show()
    #
    #     sampled_mols = sample(model, num_batches=num_batches, num_samples=num_samples,
    #                           vocab=vocab, device=device, with_file=False, return_value=True)
    #     if use_syba:
    #         for i in range(len(sampled_mols)):
    #             try:
    #                 mol = Chem.MolFromSmiles(sampled_mols[i])
    #                 sys_score.append(sys_valid_model.predict(mol=mol))
    #             except:
    #                 continue
    #         print(f'Average systhesis score = {np.mean(sys_score)}')
    #
    #     # plot_hist(sys_score, label='systhetic prediction')
    #     # plt.show()
    #
    #
    #
    #     checkpoint = {
    #         "net": model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #     }
    #     torch.save(checkpoint, os.path.join(config['out_dir'], f'{nega_train_iterations}_epoch_negative.pth'))


    # optimize positive molecules
    # for i in range(posi_iterations):
    #     for j in trange(n_policy, desc='Policy gradient...'):
    #         cur_reward, cur_loss = RL_max.policy_gradient(get_features=get_fp,task='optim_p')
    #         # RL_max.policy_gradient(get_features=get_fp)
    #         reward_sum.append(cur_reward)
    #         rl_losses.append(cur_loss)
    #
    #     # plt.plot(reward_sum)
    #     # plt.xlabel('Training iteration')
    #     # plt.ylabel('score_sum')
    #     # plt.savefig(f'../figure/{i}_{j + 1}_score_sum.png')
    #     # plt.show()
    #
    #     plt.plot(rl_losses)
    #     plt.xlabel('Training iteration')
    #     plt.ylabel('Loss')
    #     plt.savefig(f'../figure/{i}_{j + 1}_positive_Loss.png')
    #     plt.show()
    #
    #     checkpoint = {
    #         "net": model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #     }
    #     torch.save(checkpoint, os.path.join(config['out_dir'], f'{posi_iterations}_epoch_positive.pth'))




    # plot HS molecules' SA-score distribution.
    path = f'../model_parameters/{nega_train_iterations}_epoch_negative.pth'
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    print(f"Load model parameters from {path}.")

    sampled_mols = sample(model, num_batches=num_batches, num_samples=num_samples,
                          vocab=vocab, device=device, with_file=False, return_value=True)

    with open('./negative.pkl', 'wb') as f:
        pickle.dump(sampled_mols, f, protocol=4)

    sys_score = []
    if use_syba:
        for i in range(len(sampled_mols)):
            try:
                mol = Chem.MolFromSmiles(sampled_mols[i])
                sys_score.append(sys_valid_model.predict(mol=mol))
            except:
                continue
        print(f'Average synthesis score = {np.mean(sys_score)}')

    plot_hist(sys_score, label='Negative optimized molecules')



    # plot ES molecules' SA-score distribution.
    path = f'../model_parameters/{posi_iterations}_epoch_positive.pth'
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    print(f"Load model parameters from {path}.")

    sampled_mols = sample(model, num_batches=num_batches, num_samples=num_samples,
                          vocab=vocab, device=device, with_file=False, return_value=True)

    with open('./positive.pkl', 'wb') as f:
        pickle.dump(sampled_mols, f, protocol=4)

    sys_score = []

    if use_syba:
        for i in range(len(sampled_mols)):
            try:
                mol = Chem.MolFromSmiles(sampled_mols[i])
                sys_score.append(sys_valid_model.predict(mol=mol))
            except:
                continue
        print(f'Average synthesis score = {np.mean(sys_score)}')

    plot_hist(sys_score, label='Positive optimized molecules')
    plt.show()
















