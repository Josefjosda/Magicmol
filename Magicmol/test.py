import pandas as pd
import tqdm
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.Chem.Draw import IPythonConsole
RDLogger.DisableLog('rdApp.*')
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
import os
import glob
import pickle
import yaml
from syba.syba import SybaClassifier

#Next we calculate the metrics for these molecules.
generated_files = glob.glob('./generate_result/6lu7-generated_molecule_*-0.pdb')
config_dir = "./generate_result/config.yaml"

with open(config_dir, 'r') as f:
    config = yaml.full_load(f)

with open(config['dataset_dir'] , 'rb') as f:
    original_smile = pickle.load(f)


#
valid_original_smiles = [0]*len(original_smile)
# if not os.path.exists('./plot_pickle.pkl'):

for i in tqdm.trange(len(original_smile)):
    # if Chem.MolFromSmiles(original_smile[i]) != None:
    valid_original_smiles[i] = original_smile[i]

#     with open('./plot_pickle.pkl','wb') as f:
#         pickle.dump(valid_original_smiles, f, pickle.HIGHEST_PROTOCOL)
#
# else:
#     with open('./plot_pickle.pkl', 'rb') as f:
#         valid_original_smiles = pickle.load(f)
#         print("Loading original_smiles.")
# #
#
# with open('./fine-tune/fine_tune_mols_cleaned.pkl','rb') as f:
#     fine_tune_smiles = pickle.load(f)
#
# fine_tune_smiles = [Chem.MolFromSmiles(i) for i in fine_tune_smiles]

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

def plot_scatter(prediction,label):

    plt.figure(figsize=(8,6))
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



# x = SybaClassifier()
# x.fitDefaultScore()

# score =[]
#
# for i in tqdm.trange(len(valid_original_smiles)):
#     try:
#
#         score.append(x.predict(smi=valid_original_smiles[i]))
#     except BaseException:
#         continue
# #
# with open('./score.pkl','rb') as f:
#     score = pickle.load(f)
#     # score = score[10]
#
# # plt.figure(figsize=(8,6))
# # sns.countplot(x="deck", data=score , palette="Greens_d")
#
# class_ = [0] * len(score)
#
# for i in tqdm.trange(len(score)):
#     if score[i] > 0 :
#         class_[i] = 'Easy to synthesize'
#     else:
#         class_[i] = 'Hard to synthesize'
#
#
#
# dic = {'class':class_ , 'score' :score}
#
# df = pd.DataFrame(dic)
# # sns.boxplot(x=class_, y=score , orient= 'h') # hue分类依据 data=titanic)
# plt.figure(figsize=(8,6))
#
# # plot_hist(score,label='')
# # iris = sns.load_dataset("iris")
# # ax = sns.boxplot(x=class_,y=score,orient ='h', palette="Set2")
#
# sns.set(style="darkgrid")
# # sns.set_title(title='Training molecules classification by predicted systhetic score')
# ax = sns.countplot(x = df['class'] , data = df )
# ax.set(title = 'Training molecules classification by predicted systhetic score' , ylabel ='count' ,xlabel='')
# # # ax = sns.jointplot(x = class_, y = score , kind = 'kde' )
# plt.legend()
# #
# #
# plt.show()

# valid_mols = []
# for i in range(len(generated_files)):
#     line_count = 0
#     with open(generated_files[i],'rb') as f:
#         temp = pickle.load(f)
#         for smi in temp:
#             line_count += 1
#             mol = Chem.MolFromSmiles(smi.strip())
#             if mol is not None:
#                 valid_mols.append(mol)

    # print("Load file from {}".format(generated_files[i-1]))
    #
    # print(f'Validity of generated_smiles - epoch{i}: ', f'{len(valid_mols) / line_count:.2%}')
    #
    # print(f'Uniqueness of generated_smiles - epoch{i}:' , f'{len(set(valid_mols)) /  line_count:.2%}')
    #
    # print(f'Novelty of generated_smiles - epoch{i}:' , f'{ 1 - (len(set(valid_original_smiles) &  set(valid_mols)) /  line_count ):.2%}')
    # valid_mols.clear()
    #

# re = '    CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@](C#N)([C@H](O)[C@@H]1O)C1=CC=C2N1N=CN=C2N)OC1=CC=CC=C1'
#
import rdkit
# from rdkit import Chem
#
# with open('./reinforcement/positive.pkl','rb') as f:
#     smiles_files = pickle.load(f)
#
# mol = []
# for i in smiles_files :
#     temp =  Chem.MolFromSmiles(i)
#
#     mol.append(temp)
#
# # mol = [Chem.MolFromSmiles(i) for i in mol]
#
# # m = [Chem.MolToSmiles(i) for i in mol]
#
# # print(m)
#
#
#
# x = Draw.MolsToGridImage(mol[:30] ,molsPerRow=5, subImgSize=(400,300), returnPNG=False)
#
# x.save('./mo.png')
#
#
#
# file = pd.read_csv('molecule selection/mol_candidates.csv')
#
# logP_path = file['log_p']
# QED_path = file['QED']
#
# fig, ax = plt.subplots()
# cm = plt.cm.get_cmap('viridis')
# sc = ax.scatter(logP_path, QED_path, cmap=cm, s=13)
# clb = plt.colorbar(sc)
# fig.tight_layout()
#
# # plt.savefig('./logP_v_QED_scatter.png', dpi=1000)
#
# plt.show()
#
#
# x = ['COC1=CC=C(Cl)C=C1NC(=O)C2=CC=C(F)C=C2O','COC1=CC=C(Br)C=C1C(=O)NC2=CC=C(F)C=C2']
#
# x = [Chem.MolFromSmiles(i)for i in x]
#
# y = Draw.MolsToImage(x)
# y.show()

#
x = SybaClassifier()
x.fitDefaultScore()


with open('./reinforcement/positive.pkl','rb') as f:
    psmiles = pickle.load(f)

with open('./reinforcement/negative.pkl','rb') as f:
    nsmiles = pickle.load(f)

pscore = []
nscore = []

for i in psmiles:
    try:
        pscore.append(x.predict(mol = Chem.MolFromSmiles(i)))
    except BaseException:
        continue

pscore = np.array(pscore)

pscore_index = np.argsort(pscore)[::-1] [:5]


for i in nsmiles:
    try:
        nscore.append(x.predict(mol = Chem.MolFromSmiles(i)))
    except BaseException:
        continue


nscore = np.array(nscore)

nscore_index = np.argsort(nscore) [10:15]


a = [Chem.MolFromSmiles(psmiles[i]) for i in pscore_index]
b = [Chem.MolFromSmiles(nsmiles[i]) for i in nscore_index]

print(pscore[pscore_index])
print(nscore[nscore_index])

x = Draw.MolsToGridImage(a,molsPerRow=5, subImgSize=(400,300), returnPNG=False)

x.save('./p.png')


x = Draw.MolsToGridImage(b,molsPerRow=5, subImgSize=(400,300), returnPNG=False)

x.save('./n.png')























