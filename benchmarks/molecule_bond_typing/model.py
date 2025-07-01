import random
import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# Feature dictionary
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


def get_edge_examples(mol):
    feats = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        f1 = atom_rich_features(a1)
        f2 = atom_rich_features(a2)
        edge_feat = f1 + f2  # Concatenate
        feats.append((edge_feat, bond.GetBondTypeAsDouble()))
    return feats


def generate_dataset(n=300):
    # Common fragments with varied bond types
    fragments = [
        'C#N', 'c1ccncc1', 'O=C=O', 'C1=CN=CN1', 'CCCl', 'CCBr', 'CC=CC',
        'C1CCOC1', 'C1=COC=C1', 'CC(C)O', 'CN(C)C', 'CC(C)N', 'c1cccs1', 'C(=O)N'
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

    return np.array(X), np.array(y)


# Generate dataset
X, y = generate_dataset(n=1000)
bond_type_map = {1.0: 0, 2.0: 1, 3.0: 2, 1.5: 3}
y = np.array([bond_type_map.get(b, -1) for b in y])
X = X[y != -1]
y = y[y != -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Train model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred, digits=4))

# Save to a file
joblib.dump(clf, 'rf_bond_classifier.pth')
