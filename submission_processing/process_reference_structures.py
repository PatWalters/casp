#!/usr/bin/env python

import sys
from pathlib import Path, PurePath
from process_submission_files import LigandInfo, heavy_atom_mf
from casp_utils import atomgroup_to_rdmol
from biopandas.pdb import PandasPdb
from glob import glob
from collections import Counter
import pandas as pd
from tqdm.auto import tqdm
import prody
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import Chem
from rdkit.Chem import AllChem
import useful_rdkit_utils as uru

home = str(Path.home())
LIGAND_DIR = f"{home}/DATA/CASP/FINAL/LIGAND"
SUBMISSION_DIR = f"{home}/DATA/CASP/FINAL/SUBMISSIONS"
SOLUTIONS_DIR = f"{home}/DATA/CASP/FINAL/SOLUTIONS"


def find_close_residues(prot, chid, resnum, resname, cutoff):
    query_str = f"(protein or nucleic) within {cutoff} of (chid {chid} resnum {resnum} resname {resname})"
    prot.select(query_str)
    sel = prot.select(query_str)
    sel_atomnum_set = []
    if sel:
        sel_atomnum_set = list(set(sel.getResnums()))
    return [int(x) for x in sel_atomnum_set]


def get_ligand_rdmol(atmgrp, chid, resname, resnum):
    query_str = f"(chid is {chid}) and (resname is {resname}) and (resnum is {resnum})"
    sel_ag = atmgrp.select(query_str)
    sel_mol = atomgroup_to_rdmol(sel_ag)
    return sel_mol


def process_ref_protein(filename):
    uru.rd_shut_the_hell_up()
    prot_ag = prody.parsePDB(filename)
    basename = PurePath(filename).parts[-1]
    ligand_filename = basename.replace(".pdb", ".txt")
    lig_info = LigandInfo(f"{LIGAND_DIR}/{ligand_filename}")
    lig_df = lig_info.get_df()
    lig_df = lig_df.query('Relevant == "Yes"')
    lig_name_set = set(lig_df.Name.values)
    ppdb = PandasPdb()
    prot = ppdb.read_pdb(filename)
    res = []
    df_list = []
    for chain_id, v in prot.df['HETATM'].groupby("chain_id"):
        for row in v[['residue_name', 'residue_number']].drop_duplicates().values:
            res_name, res_num = row
            if res_name in lig_name_set:
                ref_smiles, ref_name, res_relevant, ref_hvy_mf = lig_info.lookup_ligand_info_by_name(res_name)
                ligand_rd_mol = get_ligand_rdmol(prot_ag, chain_id, res_name, res_num)
                ligand_mf = CalcMolFormula(ligand_rd_mol)
                ref_mol = Chem.MolFromSmiles(ref_smiles)
                if ref_mol.GetNumBonds() > 0:
                    try:
                        AllChem.AssignBondOrdersFromTemplate(ref_mol, ligand_rd_mol)
                    except ValueError as e:
                        print(e, ref_smiles)
                ligand_mol_block = Chem.MolToMolBlock(ligand_rd_mol)
                pdb_hvy_mf = heavy_atom_mf(ligand_mf)
                close_3 = find_close_residues(prot_ag, chain_id, res_num, res_name, 3.0)
                close_5 = find_close_residues(prot_ag, chain_id, res_num, res_name, 5.0)
                res.append([chain_id, res_name, res_num, ref_smiles, ref_hvy_mf, pdb_hvy_mf, ligand_mol_block, close_3,
                            close_5])

    ref_df = pd.DataFrame(res, columns=["chain", "res_name", "res_num", "ref_smiles", "ref_hvy_mf", "pdb_hvy_mf",
                                        "ligand_mol_block", "close_3", "close_5"])
    ref_df['mf_ok'] = ref_df.ref_hvy_mf == ref_df.pdb_hvy_mf
    ref_df['target'] = basename
    ref_counter = sorted(Counter(ref_df.res_name.values).items())
    lig_counter = sorted(Counter(lig_df.Name.values).items())
    match = str(ref_counter) == str(lig_counter)
    return ref_df, [basename, len(ref_df), len(lig_df), str(ref_counter), str(lig_counter), match]


def main():
    df_list = []
    check_list = []
    for filename in tqdm(glob(f"{SOLUTIONS_DIR}/*.pdb")):
        ref_df, ref_res = process_ref_protein(filename)
        check_list.append(ref_res)
        df_list.append(ref_df)
    combo_df = pd.concat(df_list)
    combo_df.to_csv("proteins_ok.csv", index=False)
    check_df = pd.DataFrame(check_list, columns=["Target", "Num_PDB_Ligands", "Num_Provided_ligands", "PDB_Ligands",
                                                 "Provided_Ligands", "OK"])
    check_df.to_csv("check_proteins.csv", index=False)


if __name__ == "__main__":
    main()
