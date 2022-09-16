#!/usr/bin/env  python

import sys

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem import AllChem
import pandas as pd
import os
from glob import glob
import re
from casp_utils import pdb_str_to_atomgroup, molfile_str_to_atomgroup, get_close_residues, read_submission_file
import numpy as np
from prody import confProDy
from pathlib import Path
from tqdm.auto import tqdm
import json
from read_rotation_matrices import build_rotation_dataaframe

home = str(Path.home())
LIGAND_DIR = f"/{home}/DATA/CASP/FINAL/LIGAND"
SUBMISSION_DIR = f"/{home}/DATA/CASP/FINAL/SUBMISSIONS"


def heavy_atom_mf(mf_in):
    element_re = re.compile("([A-G,I-Z,a-z]+(\d+)?)")
    element_list = [x[0] for x in element_re.findall(mf_in)]
    return "".join([x for x in element_list if not x.startswith("H")])


def count_model_lines(filename):
    """ Count the number of lines in a submission that start with MODEL
    :param filename: input file name
    :return: number of lines that start with MODEL
    """
    model_lines = [x for x in open(filename) if x.startswith("MODEL")]
    return len(model_lines)


def fix_molfile(mf_in):
    """ Fix broken V3000 molfiles
    :param mf_in: input molfile as a text string
    :return: fixed V3000 molfile as a text string
    """
    hdr = ['', '  Mrv2004 09022213143D          ', '', '  0  0  0     0  0            999 V3000']
    lines = mf_in.split("\n")
    start = 0
    for idx, line in enumerate(lines):
        if line.find("M  V30 BEGIN CTAB") >= 0:
            start = idx
            break
    return "\n".join(hdr + lines[start:])


def read_ligand_info(filename):
    """ Read the ligand information from the CASP ligand files
    :param filename: input filename, files were copied from the website
    :return: dataframe with ligand info
    """
    lines = open(filename).readlines()
    lines = [x.split() for x in lines]
    ligand_df = pd.DataFrame(lines[1:], columns=lines[0])
    ligand_df['mol'] = ligand_df.SMILES.apply(Chem.MolFromSmiles)
    ligand_df['MF'] = ligand_df.mol.apply(CalcMolFormula)
    # added
    ligand_df['Name'] = [x.upper() for x in ligand_df.Name]
    ligand_df['Number'] = ligand_df.ID.astype(int)
    ligand_df['Relevant'] = [x.capitalize() for x in ligand_df.Relevant]
    ligand_df['Hvy_MF'] = ligand_df.MF.apply(heavy_atom_mf)
    ligand_df = ligand_df.set_index('Number')
    return ligand_df


class LigandInfo:
    """ A class to store ligand info for CASP submissions
    """

    def __init__(self, filename):
        self.ligand_df = read_ligand_info(filename)

    def lookup_ligand_info_by_number(self, num):
        row = self.ligand_df.loc[num]
        smiles, name, relevant, hvy_mf = row[["SMILES", "Name", "Relevant", "Hvy_MF"]].values
        name = name.upper()
        return smiles, name, relevant, hvy_mf

    def lookup_ligand_info_by_name(self, name):
        res = self.ligand_df.query("Name == @name")
        ret_val = (None, None, None, None)
        if len(res):
            row = res.iloc[0]
            smiles, name, relevant, hvy_mf = row[["SMILES", "Name", "Relevant", "Hvy_MF"]].values
            name = name.upper()
            ret_val = (smiles, name, relevant, hvy_mf)
        return ret_val

    def get_df(self):
        return self.ligand_df


def check_ligand_bonds(mol, smiles):
    tmplt_mol = Chem.MolFromSmiles(smiles)
    ok = True
    if mol.GetNumBonds() > 0:
        try:
            AllChem.AssignBondOrdersFromTemplate(tmplt_mol, mol)
        except ValueError as e:
            ok = False
    return ok


def check_protein(pdb_block):
    atmgrp = pdb_str_to_atomgroup(pdb_block)
    return atmgrp is None


def archive_set(set_in):
    """Save a set as a string
    :param set_in: set to save
    :return: string representing the set
    """
    res = None
    if set_in:
        arr_in = np.array(list(set_in))
        res = np.array2string(arr_in)
    return res


def process_ligand_molfile(model_ligand):
    """ Attempt to read and fix a molfile
    :param model_ligand:  molfile as a text string
    :return: RDKit molecule, error code
    """
    mol_status = "BAD"
    mol = Chem.MolFromMolBlock(model_ligand)
    if mol:
        mol_status = "OK"
    # fix for broken V3000 molfiles
    if mol is None:
        fixed_molfile = fix_molfile(model_ligand)
        mol = Chem.MolFromMolBlock(fixed_molfile)
        if mol:
            mol_status = "FIXED_MOLFILE"
    # try parsing the molfile as a pdb
    if mol is None:
        mol = Chem.MolFromPDBBlock(model_ligand)
        if mol:
            mol_status = "PARSED_PDB"
    return mol, mol_status


def find_close_residues(mol, protein_ag):
    ligand_atmgrp_ok = False
    close_str_3, close_str_5 = None, None
    if protein_ag:
        mb = Chem.MolToMolBlock(mol)
        max_res = max(protein_ag.getResnums())
        mol_ag = molfile_str_to_atomgroup(mb, max_res + 1)
        if mol_ag:
            ligand_atmgrp_ok = True
            close_res_3 = get_close_residues(protein_ag, mol_ag, 3.0)
            close_res_5 = get_close_residues(protein_ag, mol_ag, 5.0)
            close_str_3 = json.dumps([int(x) for x in close_res_3])
            close_str_5 = json.dumps([int(x) for x in close_res_5])
    return ligand_atmgrp_ok, close_str_3, close_str_5


def process_submission_file(filename, ligand_info, find_interactions=False):
    num_model_lines = count_model_lines(filename)
    confProDy(verbosity='none')
    # file_path, submission = os.path.split(filename)
    submission_results = read_submission_file(filename)

    bad_protein, len_protein, protein_ag, protein_atmgrp_ok = process_protein(find_interactions,
                                                                              submission_results)
    ligand_res = []
    for pose_id, model_ligands_pose in submission_results['ligands'].items():
        pose_number = int(pose_id.split()[1])
        for model_ligand_id, model_ligand in model_ligands_pose.items():
            model_ligand_toks = model_ligand_id.split()
            model_ligand_name = model_ligand_toks[-1]
            try:
                model_ligand_number = int(model_ligand_toks[1])
            except ValueError as e:
                model_ligand_number = 1

            mol, mol_status = process_ligand_molfile(model_ligand)
            bad_mol = mol is None

            smiles, name_out, relevant, mf = None, None, None, None
            bonds_ok, mol_smiles, ligand_atmgrp_ok = None, None, None
            close_3, close_5 = None, None
            mf, hvy_mf, ref_hvy_mf, hvy_mf_ok = None, None, None, None
            mol_block = None

            if mol:
                mf = CalcMolFormula(mol)
                hvy_mf = heavy_atom_mf(mf)
                mol_smiles = Chem.MolToSmiles(mol)
                smiles, name_out, relevant, ref_hvy_mf = ligand_info.lookup_ligand_info_by_number(model_ligand_number)
                hvy_mf_ok = hvy_mf == ref_hvy_mf
                if not hvy_mf_ok:
                    smiles, name_out, relevant, ref_hvy_mf = ligand_info.lookup_ligand_info_by_name(
                        model_ligand_name)
                    hvy_mf_ok = hvy_mf == ref_hvy_mf
                # bonds_ok = check_ligand_bonds(mol,smiles)
                mol_block = Chem.MolToMolBlock(mol)
                if find_interactions:
                    ligand_atmgrp_ok, close_3, close_5 = find_close_residues(mol, protein_ag)
            ligand_res.append(
                [num_model_lines, pose_id, pose_number,
                 model_ligand_id, model_ligand_name, model_ligand_number,
                 name_out, relevant,
                 mf, hvy_mf, ref_hvy_mf, hvy_mf_ok,
                 mol_smiles, smiles,
                 bad_mol, bad_protein, mol_status, bonds_ok, ligand_atmgrp_ok, len_protein, protein_atmgrp_ok,
                 close_3, close_5, mol_block])
    return ligand_res


def process_protein(find_interactions, submission_results):
    protein_str = submission_results['protein']
    bad_protein = len(protein_str) == 0
    len_protein = len(submission_results['protein'])
    protein_ag = None
    protein_atmgrp_ok = False
    if find_interactions:
        protein_ag = pdb_str_to_atomgroup(protein_str)
        protein_atmgrp_ok = protein_ag is not None
    return bad_protein, len_protein, protein_ag, protein_atmgrp_ok


def get_rotation_matrix(submission, rot_df):
    rot_id = submission.replace("LG", "TS")
    sel_df = rot_df.query("name == @rot_id")
    if len(sel_df) > 0:
        rot_mat = sel_df.tm.values[0]
        res = json.dumps(rot_mat.tolist())
    else:
        res = None
    return res


def process_ligands():
    rotation_df = build_rotation_dataaframe()
    rotation_df.set_index("name",inplace=True)
    df_list = []

    cols = ["num_model_lines", "pose_id", "pose_num",
            "ligand_id", "ligand_name", "ligand_number", "corrected_name",
            "relevant",
            "mol_formula", "hvy_mol_formula", "ref_hvy_mol_formula", "mf_ok",
            "mol_zmiles", "zmiles",
            "bad_ligand", "bad_protein", "mol_status", "bonds_ok", "ligand_atmgrp_ok", "len_protein",
            "protein_atmgrp_ok",
            "close_3", "close_5", "mol_block"]
    for dirpath in glob(f"{SUBMISSION_DIR}/T1187*"):
        base_name, target_name = os.path.split(dirpath)
        ligand_file = base_name.replace("SUBMISSIONS", "LIGAND") + f"/{target_name}_lig.txt"
        lig_info = LigandInfo(ligand_file)
        sub_file_list = sorted(glob(dirpath + f"/{target_name}*"))
        for sub_filepath in tqdm(sub_file_list, desc=dirpath):
            data_path, sub_filename = os.path.split(sub_filepath)
            row_df = pd.DataFrame(process_submission_file(sub_filepath, lig_info, find_interactions=False),
                                  columns=cols)
            row_df['target'] = target_name
            row_df['submission'] = sub_filename
            rmat = get_rotation_matrix(sub_filename, rotation_df)
            row_df['rotation_matrix'] = rmat
            df_list.append(row_df)
            combo_df = pd.concat(df_list)
            group_re = re.compile("LG([0-9]+)_")
            combo_df['group'] = [group_re.findall(x)[0] for x in combo_df.submission.values]
            combo_df.to_csv("casp_ligands_with_mols.csv", index=False)
            # return combo_df


def debug():
    cols = ["num_model_lines", "pose_id", "pose_num",
            "ligand_id", "ligand_name", "ligand_number", "corrected_name",
            "relevant",
            "mol_formula", "hvy_mol_formula", "ref_hvy_mol_formula", "mf_ok",
            "mol_zmiles", "zmiles",
            "bad_ligand", "bad_protein", "mol_status", "bonds_ok", "ligand_atmgrp_ok", "len_protein",
            "protein_atmgrp_ok",
            "close_str_3", "close_str_5"]
    entry = "H1114LG132_1"
    stub = entry[0:5]
    lig_info = LigandInfo(f"{LIGAND_DIR}/{stub}_lig.txt")
    res = process_submission_file(f"{SUBMISSION_DIR}/{stub}/{entry}", lig_info)
    print(pd.DataFrame(res, columns=cols))


if __name__ == "__main__":
    # debug()
    process_ligands()
