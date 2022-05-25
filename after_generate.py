import tqdm
from rdkit import RDLogger, Chem, DataStructs
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

#Next we calculate the metrics for these molecules.
generated_files = glob.glob('./reinforcement/*.pkl')
config_dir = "./generate_result/config.yaml"

with open(config_dir, 'r') as f:
    config = yaml.full_load(f)

with open(config['dataset_dir'] , 'rb') as f:
    original_smile = pickle.load(f)


valid_original_smiles = [0]*len(original_smile)
# if not os.path.exists('./plot_pickle.pkl'):

for i in tqdm.trange(len(original_smile)):
    # if Chem.MolFromSmiles(original_smile[i]) != None:
    valid_original_smiles[i] = original_smile[i]



valid_mols = []
for i in range(len(generated_files)):
    line_count = 0
    with open(generated_files[i],'rb') as f:
        temp = pickle.load(f)
        for smi in temp:
            line_count += 1
            try:
                mol = Chem.MolFromSmiles(smi.strip())
            except BaseException:
                continue
            if mol is not None:
                valid_mols.append(mol)

    print("Load file from {}".format(generated_files[i]))

    print(f'Validity of generated_smiles - epoch{i}: ', f'{len(valid_mols) / line_count:.2%}')

    print(f'Uniqueness of generated_smiles - epoch{i}:' , f'{len(set(valid_mols)) /  line_count:.2%}')

    print(f'Novelty of generated_smiles - epoch{i}:' , f'{ 1 - (len(set(valid_original_smiles) &  set(valid_mols)) /  line_count ):.2%}')
    valid_mols.clear()


    #fine_tune
    # ffps = []
    # for mol in range(len(fine_tune_smiles)):
    #     bv = AllChem.GetMACCSKeysFingerprint(fine_tune_smiles[mol])
    #     fp = np.zeros(len(bv))
    #     DataStructs.ConvertToNumpyArray(bv, fp)
    #     ffps.append(fp)
    #
    # #generated
    # Vfps = []
    # for mol in range(0,len(valid_mols),100):
    #     bv = AllChem.GetMACCSKeysFingerprint(valid_mols[mol])
    #     fp = np.zeros(len(bv))
    #     DataStructs.ConvertToNumpyArray(bv, fp)
    #     Vfps.append(fp)
    #
    # #trained
    # Ofps = []
    # for mol in range(0,len(valid_original_smiles), 100):
    #     bv = AllChem.GetMACCSKeysFingerprint(valid_original_smiles[mol])
    #     fp = np.zeros(len(bv))
    #     DataStructs.ConvertToNumpyArray(bv, fp)
    #     Ofps.append(fp)
    #
    #
    # print(f"Ploting figure epoch{i}")
    # from sklearn.decomposition import PCA
    # Vlen = len(Vfps)
    # v_f_len = len(Vfps) + len(ffps)
    # x = Vfps + ffps + Ofps
    # pca = PCA(n_components=2, random_state=71)
    # X = pca.fit_transform(x)
    #
    #
    # plt.figure(figsize=(12, 9))
    # # plt.scatter(X[v_f_len:, 0], X[v_f_len:, 1], marker='o',color='#00FF7F',edgecolors='k', label='Original SMILES')
    # plt.scatter(X[:Vlen, 0], X[:Vlen, 1], marker='o',color='#9370DB',edgecolors='k', label='Generated SMILES')
    # plt.scatter(X[Vlen:v_f_len, 0], X[Vlen:v_f_len, 1], marker='o', color='#A52A2A', edgecolors='k', label='Fine-tune SMILES')
    # #00BFFF 蓝色参数
    # #00FF7F 绿色参数
    #
    # plt.xlabel('Principal component 1')
    # plt.ylabel('Principal component 2')
    # plt.title("Chemical space comparison")
    # plt.legend(['Original molecules','Generated molecules','Fine-tune SMILES'])
    # plt.show()
    # plt.savefig(f'chemical space.png')
    # valid_mols.clear()
    #



#original chems
# import os
# import tqdm
# with open('./dataset/chembl28-cleaned.smi') as f:
#     org_smiles = [l.rstrip() for l in f.readlines()]
#
# temp = []
# if not os.path.exists(os.path.join('vocab','total_valid_chems.pkl')):
#     with open(os.path.join('vocab','total_valid_chems.pkl'),'wb') as f:
#         for i in tqdm.trange(len(org_smiles)):
#             if Chem.MolFromSmiles(org_smiles[i]) is not None:
#                 temp.append(Chem.MolFromSmiles(org_smiles[i].strip()))
#         pickle.dump(temp,f,pickle.HIGHEST_PROTOCOL)
#     org_mols = temp
# else:
#     with open(os.path.join('vocab', 'total_valid_chems.pkl'), 'rb') as f:
#         org_mols = pickle.load(f)
#
#
#








































