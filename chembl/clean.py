"""
Pre-process the Chembl dataset:
    1. Normalize the molecule (this step also removes stereochemical info).
    2. Convert the SMILES to canonical form.
"""
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import MolStandardize
import pickle
RDLogger.DisableLog('rdApp.*')


class MolCleaner:
    def __init__(self):
        self.normarizer = MolStandardize.normalize.Normalizer()
        self.choose_frag = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()
        self.te = rdMolStandardize.TautomerEnumerator()

    def process(self, mol):
        mol = Chem.MolFromSmiles(mol)
        if mol is not None:
            mol = self.normarizer.normalize(mol)
            mol = self.choose_frag.choose(mol)
            mol = self.uc.uncharge(mol)
            mol = self.te.Canonicalize(mol)
            # Adding this may evidently prolong the time of data processing.
            mol = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return mol
        else:
            return None


if __name__ == "__main__":

    in_path = "database.smi"
    out_path = "mols_cleaned.smi"
    pkl_out_path = "mols_cleaned.pkl"

    # with open(in_path, 'rb') as f:
    #     smiles = pickle.load(f)
    with open(in_path) as f:
        smiles = f.readlines()
    print("number of SMILES before cleaning:", len(smiles))

    # clean the molecules
    cleaner = MolCleaner()
    processed = []
    for mol in tqdm(smiles):
        mol = cleaner.process(mol)
        if mol is not None and 40 < len(mol) < 120:
            processed.append(mol)

    processed = set(processed)

    print("number of SMILES after cleaning:", len(processed))

    with open(out_path, "w") as f:
        for mol in processed:
            f.write(mol + "\n")
    f.close()

    with open(pkl_out_path,'wb') as f:
        pickle.dump(processed,f)
