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
from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors
import pickle
import yaml
from  syba.syba import SybaClassifier
from numpy import mean



#Next we calculate the metrics for these molecules.
generated_files = glob.glob('./generate_result/*.pkl')
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



total = []
# plt.figure(figsize=(300, 100), dpi=10)
labels = ['SELFIES', 'SMILES']
x = np.arange(len(generated_files))
validate_model = SybaClassifier()
validate_model.fitDefaultScore()

for i in range(len(generated_files)):
    valid_mols = []
    QED_ = []
    match_rules = []
    sa_ = []
    ES = 0
    with open(generated_files[i],'rb') as f:
        temp = pickle.load(f)
        for j in tqdm.trange(len(temp)):
            try:
                mol = Chem.MolFromSmiles(temp[j].strip())
            except BaseException:
                continue
            if mol is not None:
                SMILES = Chem.MolToSmiles(mol,isomericSmiles=False, canonical=True)
                valid_mols.append(SMILES)
                QED_.append(QED.qed(mol))
                sa = validate_model.predict(mol=mol)
                sa_.append(sa)
                if sa > 0:
                    ES += 1
                #Rule of Five
                mw = np.round(Descriptors.MolWt(mol), 1)
                if mw <= 500:
                    logp = np.round(Descriptors.MolLogP(mol), 2)
                    if  -0.4 < logp < 5.6:
                        hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
                        if hba < 10:
                            rob = rdMolDescriptors.CalcNumRotatableBonds(mol)
                            if rob < 10:
                                hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
                                if hbd < 5:
                                    match_rules.append(SMILES)



    total.append(valid_mols)

    Sm = valid_mols
    Vm = set(valid_mols)


    # a = set(valid_original_smiles)
    # b = Vm & set(valid_original_smiles)
    print("Load file from {}".format(generated_files[i]))

    print(f'Validity of generated_smiles - epoch{i}: ', f'{ len(Sm) / len(temp):.2%}')

    print(f'Uniqueness of generated_smiles - epoch{i}:' , f'{ len(Vm) / len(Sm) :.2%}')

    print(f'Novelty of generated_smiles - epoch{i}:' , f'{ 1 - ( len( Vm & set(valid_original_smiles) ) / len(Vm)  ):.2%}')

    print(f'Average QED:{mean(QED_)}' )

    print(f'Average SA:{mean(sa_)}' )

    print(f"{ES/len(valid_mols):.2%} percentage molecules belong to ES.")

    print(f"{len(match_rules)/len(valid_mols):.2%} molecules match Lipinski‘s Rule of Five.")

    # valid_mols.clear()X


















