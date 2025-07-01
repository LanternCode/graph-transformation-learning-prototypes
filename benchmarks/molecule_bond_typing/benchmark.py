import random
import numpy as np
from rdkit import Chem
from sklearn.metrics import classification_report

# Electronegativity dictionary
ELECTRONEGATIVITY = {
    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66,
}


def atom_rich_features(atom):
    atomic_num = atom.GetAtomicNum()
    return [
        atomic_num,
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        ELECTRONEGATIVITY.get(atomic_num, 0.0),
        atom.GetTotalValence(),
        atom.GetExplicitValence()
    ]


# Bond feature extractor
def get_edge_examples(mol):
    feats = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        f1 = atom_rich_features(a1)
        f2 = atom_rich_features(a2)
        edge_feat = f1 + f2
        feats.append((edge_feat, bond.GetBondTypeAsDouble()))
    return feats


# Benchmark data generator
def generate_benchmark_dataset(n=300):
    fragments = [
        'C=C', 'C#C', 'C1=CC=CC=C1', 'CC(=O)O', 'CC#N', 'C1CC1',
        'c1ccccc1O', 'CC=O', 'C=N', 'C=CC=C', 'CC#CC', 'C1=CC=CN=C1',
        'C1CCCCC1', 'c1ccncc1', 'CC(N)=O', 'CC(C)=O', 'c1ccco1',
        'C#N', 'O=C=O', 'C1=CN=CN1', 'CCCl', 'CCBr', 'CC=CC',
        'C1CCOC1', 'C1=COC=C1', 'CC(C)O', 'CN(C)C', 'CC(C)N',
        'c1cccs1', 'C(=O)N'
    ]

    smiles_list = []
    while len(smiles_list) < n:
        frag = random.choice(fragments)
        mol = Chem.MolFromSmiles(frag)
        if mol is not None:
            smiles_list.append(Chem.MolToSmiles(mol))

    X, y = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        examples = get_edge_examples(mol)
        for feat, label in examples:
            X.append(feat)
            y.append(label)

    # Remap labels
    bond_type_map = {1.0: 0, 2.0: 1, 3.0: 2, 1.5: 3}
    y = np.array([bond_type_map.get(b, -1) for b in y])
    X = np.array(X)
    X = X[y != -1]
    y = y[y != -1]

    return X, y


def run_bond_classification_benchmark(predict_fn, n=750):
    X, y = generate_benchmark_dataset(n=n)
    y_pred = predict_fn(X)

    print("Unique predictions:", set(y_pred))
    print("Label counts:", {label: list(y_pred).count(label) for label in set(y_pred)})

    print("Benchmark Evaluation:")
    print(classification_report(y, y_pred, target_names=['Single', 'Double', 'Triple', 'Aromatic']))
