#!/usr/bin/env python

from io import StringIO
from pathlib import PurePath

import pandas as pd
from prody import parsePDBStream, confProDy, writePDBStream
from rdkit import Chem
from tqdm.auto import tqdm
from glob import glob
from pathlib import Path


def read_submission_file(submission_file):
    read_protein = True
    read_ligand = False
    protein_pdb = ''
    ligands_poses = {}

    with open(submission_file) as f:
        lines = f.readlines()

        for line in lines:
            if read_ligand:
                try:
                    if not (line.startswith('END') or line.startswith('POSE') or line.startswith('LIGAND')):
                        ligands_poses[current_pose][current_ligand] += line
                except:
                    ligands_poses.setdefault(current_pose, {})
                    ligands_poses[current_pose].setdefault(current_ligand, line)

            if line.startswith('PFRMAT LG'):
                read_protein = False
            elif line.startswith('LIGAND'):
                current_ligand = line.strip()
            elif line.startswith('POSE'):
                current_pose = line.strip()
                read_ligand = True

            if read_protein:
                if not (line.startswith('PFRMAT') or line.startswith('TARGET') or line.startswith(
                        'PARENT') or line.startswith('END')):
                    protein_pdb += line

    results = {'protein': protein_pdb, 'ligands': ligands_poses}

    return results


# ================ Prody functions =======================

def atomgroup_to_rdmol(ag):
    buff = StringIO()
    writePDBStream(buff, ag)
    mol = Chem.MolFromPDBBlock(buff.getvalue())
    return mol


def pdb_str_to_atomgroup(pdb_str):
    """ Read a pdb file as a string and covert it to a Prody AtomGroup
    :param pdb_str: pdb file as a string
    :return: a Prody AtomGroup with protein molecule represented by the PDB string
    """
    atmgrp = None
    # Make sure the PDB string isn't empty
    if len(pdb_str):
        pdb_io = StringIO(pdb_str)
        atmgrp = parsePDBStream(pdb_io)
    return atmgrp


def molfile_str_to_atomgroup(mol_str, res_num):
    """ Read a molfile as a string and convert it to a Prody AtomGroup
    :param mol_str: input molfile as a string
    :param res_num: residue number for the Prody AtomGroup
    :return: a Prody AtomGroup with the molecule represented by the molfile string, the residue names wll be LIG
    """
    rd_mol = Chem.MolFromMolBlock(mol_str)
    atmgrp = None
    if rd_mol:
        pdb_block = Chem.MolToPDBBlock(rd_mol)
        atmgrp = parsePDBStream(StringIO(pdb_block))
        atmgrp.setResnames([f'LIG'] * len(atmgrp.getResnames()))
        atmgrp.setResnums([res_num] * len(atmgrp.getResnums()))
    return atmgrp


def process_ligands(entry_dict, lig_res_num):
    """Process the dictionary of ligands returned by read_submission file
    :param entry_dict: dictionary returned by read_submission_file
    :param lig_res_num: the residue number to be assigned to the ligand
    :return: a dataframe with ligand information
    """
    res = []
    for k1 in entry_dict['ligands']:
        for k2 in entry_dict['ligands'][k1]:
            molfile_buff = entry_dict['ligands'][k1][k2]
            molfile_atmgrp = molfile_str_to_atomgroup(molfile_buff, lig_res_num)
            res.append(k1.split() + k2.split() + [molfile_atmgrp])
    res_df = pd.DataFrame(res, columns=["TAG_1", "POSE_NUM", "TAG_2", "LIGNUM", "LIGNAME", "ATMGRP"])
    return res_df


def process_entry(entry_dict):
    """ read the dictionary generated by read_submission_file and generate
    Prody AtomGroup with the protein and a dataframe with the ligands
    :param entry_dict: dictionary returned by read_submission_file
    :return: protein as a Prody AtomGroup, Pandas dataframe with ligands
    """
    prot_atmgrp = pdb_str_to_atomgroup(entry_dict['protein'])
    lig_df = None
    if prot_atmgrp:
        max_res = max(prot_atmgrp.getResnums())
        lig_df = process_ligands(entry_dict, max_res + 1)
    return prot_atmgrp, lig_df


def generate_entry_df(entry_name):
    """Process ligand information generated by read_submission_file
    :param entry_name: dictionary produced by read_submission_file
    :return: dataframe with ligand information
    """
    entry_dict = read_submission_file(entry_name)
    prot_atmgrp, lig_df = process_entry(entry_dict)
    if prot_atmgrp:
        close_list = []
        for idx, row in lig_df.iterrows():
            close_list.append(get_close_residues(prot_atmgrp, row.ATMGRP))
        lig_df['CLOSE'] = close_list
        lig_df['ENTRY'] = PurePath(entry_name).parts[-1]
        lig_df['NUM_ATOMS'] = [len(x) for x in lig_df.ATMGRP]
    return lig_df


def get_close_residues(prot, lig, dist_cutoff=3):
    """Find protein residues within a cutoff distance of a ligand
    :param prot: protein as a Prody AtomGroup
    :param lig: ligand as a Prody AtomGroup
    :param dist_cutoff: cutoff distance
    :return: a set with residue numbers
    """
    combo_atmgrp = prot + lig
    close_atmgrp = combo_atmgrp.select(f"protein and within {dist_cutoff} of resname LIG")
    close_res = set()
    if close_atmgrp:
        close_res = set(close_atmgrp.getResnums())
    return close_res


def write_entry(entry, prefix):
    """Write a protein and ligands from the dictionary produced by read_submission_file
    for debugging.  Protein is written to prefix_prot.pdb, ligands are written to
    prefix_lig.sdf
    :param entry: dictionary generated by read_submission_file
    :param prefix: prefix for output filenames
    :return: None
    """
    ofs = open(f"{prefix}_prot.pdb", "w")
    pdb_buff = entry['protein']
    print(pdb_buff, file=ofs)
    ofs.close()
    writer = Chem.SDWriter(f"{prefix}_lig.sdf")
    for k1 in entry['ligands']:
        for k2 in entry['ligands'][k1]:
            molfile_buff = entry['ligands'][k1][k2]
            mol = Chem.MolFromMolBlock(molfile_buff)
            writer.write(mol)
    writer.close()


def get_reference_interactions(ref_pdb, ref_sdf, dist_cutoff=3):
    """Get residues from ref_pdb that are within dist_cutoff for each ligand in ref_sdf.
    The ligand names from ref_sdf will be used to label the interactions
    :param ref_pdb: reference pdb file
    :param ref_sdf: reference sdf
    :param dist_cutoff: distance cutoff
    :return: a dictionary with ligand names and close residues
    """
    close_list = []
    ref_pdb_buff = open(ref_pdb).read()
    ref_pdb_io = StringIO(ref_pdb_buff)
    ref_pdb = parsePDBStream(ref_pdb_io)
    max_num = max(ref_pdb.getResnums())
    suppl = Chem.SDMolSupplier(ref_sdf)
    for mol in suppl:
        mol_block = Chem.MolToMolBlock(mol)
        mol_name = mol.GetProp("_Name")
        ref_lig = molfile_str_to_atomgroup(mol_block, max_num + 1)
        complex_mol = ref_pdb + ref_lig
        close_mol = complex_mol.select(f"protein and within {dist_cutoff} of resname LIG")
        close_set = None
        if close_mol:
            close_set = set(close_mol.getResnums())
        close_list.append([mol_name, close_set])
    close_df = pd.DataFrame(close_list, columns=["LIGNAME", "CLOSE"])
    return close_df


def tanimoto(set_1, set_2):
    """Calcuate the tanimoto coefficient between two sets of residue numbers
    :param set_1: the first set of residue numbers
    :param set_2: the second set of residue numbers
    :return: tanimoto coefficient
    """
    numerator = len(set_1.intersection(set_2))
    denominator = len(set_1.union(set_2))
    return numerator / denominator


def compare_interactions(ref_df, test_df):
    """Compare interactions between a reference and set of predictions
    :param ref_df: dataframe produced by get_reference_interactions
    :param test_df: dataframe with collected predictions
    :return: list of tanimoto coefficients
    """
    score_list = []
    for idx, row in test_df.iterrows():
        ligname = row.LIGNAME
        close = row.CLOSE
        ref_close_list = ref_df.query("LIGNAME == @ligname ").CLOSE.values
        score = 0.0
        if close:
            score = max([tanimoto(close, x) for x in ref_close_list])
        score_list.append(score)
    return score_list


def main():
    confProDy(verbosity='none')
    ref_df = get_reference_interactions("T1124_ref_protein.pdb", "T1124_ref_ligands.sdf")
    df_list = []
    for filename in tqdm(glob("/Users/pwalters/DATA/CASP/T1124/*")):
        basename = PurePath(filename).parts[-1]
        df_list.append([basename, generate_entry_df(filename)])
    bad_list = [x for x in df_list if x[1] is None]
    print("There is a problem with the following entries")
    print("\n".join([x[0] for x in bad_list]))
    all_df = pd.concat([x[1] for x in df_list if x[1] is not None])
    name_dict = {26: 'SAH', 13: 'TYR'}
    all_df.LIGNAME = [name_dict[x] for x in all_df.NUM_ATOMS]
    score_list = compare_interactions(ref_df, all_df)
    all_df['SCORE'] = score_list
    all_df.sort_values('SCORE', inplace=True, ascending=False)
    out_columns = [x for x in all_df.columns if x != "ATMGRP"]
    all_df[out_columns].to_csv("T1124_report.csv", index=False)


def debug_entry(name):
    home = str(Path.home())
    SUBMISSION_DIR = f"{home}/DATA/CASP/FINAL/SUBMISSIONS"
    base_dir = name.split("LG")[0]
    entry = read_submission_file(f"{SUBMISSION_DIR}/{base_dir}/{name}")
    write_entry(entry,name)


if __name__ == "__main__":
    debug_entry("H1172LG035_3")
