from rdkit import Chem


def trim_molecule(mol, ref_mol):
    tmp_rw_mol = Chem.RWMol(mol)
    match_set = set(tmp_rw_mol.GetSubstructMatch(ref_mol))
    all_set = set(list(range(0, tmp_rw_mol.GetNumAtoms())))
    for idx in sorted(list(all_set.difference(match_set)), reverse=True):
        tmp_rw_mol.RemoveAtom(idx)
    Chem.SanitizeMol(tmp_rw_mol)
    return Chem.Mol(tmp_rw_mol)


def fix_oaa(mol):
    oaa_smiles = 'CC(=O)N[C@H]1[C@@H](O)O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O[C@@H]3O[C@H](C=O)[C@@H](O[C@@H]4O[C@H](CO[C@H]5O[C@H](CO)[C@@H](O[C@@H]6OC(C=O)=C[C@H](O)[C@H]6O)[C@H](O)[C@H]5NC(C)=O)[C@@H](O)[C@H](O)[C@H]4NC(C)=O)[C@H](O)[C@H]3O)[C@H](O)[C@H]2NC(C)=O)[C@@H](O)[C@@H]1O'
    oaa_tmplt = Chem.MolFromSmiles(oaa_smiles)
    return trim_molecule(mol, oaa_tmplt)


def fix_mq7(mol):
    rw_mol = Chem.RWMol(mol)
    ref_smiles = "CC(C)=CCC\C(C)=C\CC\C(C)=C\CC1=C(C)C(=O)c2ccccc2C1=O"
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    trim_mol = trim_molecule(rw_mol, ref_mol)
    return trim_mol


def fix_f3s(pdb_mol):
    new_mol = Chem.RWMol(pdb_mol)
    for bnd in pdb_mol.GetBonds():
        bgn_atm = bnd.GetBeginAtom()
        end_atm = bnd.GetEndAtom()
        if (bgn_atm.GetAtomicNum() == 26) and (end_atm.GetAtomicNum() == 26):
            bgn_idx = bgn_atm.GetIdx()
            end_idx = end_atm.GetIdx()
            new_mol.RemoveBond(bgn_idx, end_idx)
    Chem.SanitizeMol(new_mol)
    return new_mol
