#!/usr/bin/env python

import sys
from casp_utils import read_submission_file, atomgroup_to_rdmol
from pathlib import Path, PurePath
from rdkit import Chem
from rdkit.Geometry import Point3D
import numpy as np
from itertools import zip_longest
import pandas as pd
import prody
from rdkit.Chem import AllChem
import useful_rdkit_utils as uru
from featuremap_score import FeatureMapScore
from glob import glob

home = str(Path.home())
LIGAND_DIR = f"{home}/DATA/CASP/FINAL/LIGAND"
SUBMISSION_DIR = f"{home}/DATA/CASP/FINAL/SUBMISSIONS"
ROTATION_DIR = f"{home}/DATA/CASP/FINAL/ROTATION"
SOLUTIONS_DIR = f"{home}/DATA/CASP/FINAL/SOLUTIONS"


def parse_lga_rotation_matrix(lines):
    name = lines[0].strip()
    lst = []
    for line in lines[1:4]:
        toks = line.split()
        lst.append([float(toks[x]) for x in [2, 6, 10, 14]])
    return name, np.array(lst, dtype=np.float64)


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def read_lga_matrix_file(filename):
    mat_list = []
    with open(filename) as f:
        for lines in grouper(f, 5, ''):
            name, tm = parse_lga_rotation_matrix(lines)
            mat_list.append([name, tm])
    mat_df = pd.DataFrame(mat_list, columns=["name", "tm"])
    return mat_df


def parse_mmalign_rotation_matrix(lines):
    lst = []
    for line in lines:
        toks = line.split()
        lst.append([float(toks[x]) for x in [2, 3, 4, 1]])
    return np.array(lst, dtype=np.float64)


def read_mmalign_matrix_files(dirname):
    mat_list = []
    dir_base = PurePath(dirname).parts[-1]
    for filename in glob(f"{dirname}/{dir_base}*.ROT"):
        basename = PurePath(filename).parts[-1]
        name = basename.replace(".ROT", "")
        with open(filename) as ifs:
            lines = ifs.readlines()
            tm = parse_mmalign_rotation_matrix(lines[2:5])
            mat_list.append([name, tm])
    mat_df = pd.DataFrame(mat_list, columns=["name","tm"])
    print(mat_df.shape)


def test_read(infile_name):
    lig_df = pd.read_csv("casp_ligands_with_mols.csv")
    m = Chem.MolFromMolBlock(lig_df.mol_block.values[0])
    print(Chem.MolToSmiles(m))


def transform_molecule(mol, tm):
    conf = mol.GetConformer(0)
    for i in range(0, mol.GetNumAtoms()):
        x, y, z = conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z
        x_new = tm[0][0] * x + tm[0][1] * y + tm[0][2] * z + tm[0][3]
        y_new = tm[1][0] * x + tm[1][1] * y + tm[1][2] * z + tm[1][3]
        z_new = tm[2][0] * x + tm[2][1] * y + tm[2][2] * z + tm[2][3]
        conf.SetAtomPosition(i, Point3D(x_new, y_new, z_new))


def read_reference(filename, solutions_dir):
    uru.rd_shut_the_hell_up()
    df = pd.read_csv(filename)
    tgt_dict = {}
    mol_list = []
    for idx, row in df.iterrows():
        protein_pdb = row.target
        prot_ag = tgt_dict.get(protein_pdb)
        if prot_ag is None:
            prot_ag = prody.parsePDB(f"{solutions_dir}/{protein_pdb}")
            tgt_dict[protein_pdb] = prot_ag
        query_str = f"chid {row.chain} and resnum {row.res_num} and resname {row.res_name}"
        sel_ag = prot_ag.select(query_str)
        sel_mol = atomgroup_to_rdmol(sel_ag)
        tmplt_mol = Chem.MolFromSmiles(row.ref_smiles)

        if tmplt_mol.GetNumBonds() > 0:
            try:
                AllChem.AssignBondOrdersFromTemplate(tmplt_mol, sel_mol)
            except ValueError as e:
                pass
                # print(row.target, row.chain, row.res_name, row.res_num)
                # print(Chem.MolToSmiles(sel_mol))
        mol_list.append(sel_mol)
    df['mol'] = mol_list
    return df


def eval_ligands(ligand_csv):
    print("called")
    df_ligand = pd.read_csv(ligand_csv)
    df_ref = read_reference("proteins_ok_close.csv", SOLUTIONS_DIR)
    for target in df_ligand.target.unique():
        df_target = df_ligand.query("target == @target").copy()
        if target.startswith("T"):
            try:
                df_rot = read_lga_matrix_file(f"{ROTATION_DIR}/{target}.rot")
                evaluate_overlap(df_target, df_ref, df_rot)
            except FileNotFoundError as e:
                print(e)
            break


def evaluate_overlap(df_target, df_ref, df_rot):
    missing_set = set()
    for sub in df_target.submission:
        rot_name = sub.replace("LG", "TS")
        df_rot_sel = df_rot.query("name == @rot_name")
        df_sub = get_submission_ligands(sub)
        print("names = ", df_sub.model_ligand_name.unique())
        if len(df_rot_sel) == 1:
            pass
        else:
            if rot_name not in missing_set:
                missing_set.add(rot_name)


def rotate_and_compare(target, sub_df, ref_df):
    sub_ligand_name_list = sub_df.corrected_name.unique()
    contact_dict = {}
    for ligand_name in sub_ligand_name_list:
        print(ligand_name)


def get_submission_ligands(submission):
    target_name = submission.split("LG")[0]
    sub = read_submission_file(f"{SUBMISSION_DIR}/{target_name}/{submission}")
    mol_list = []
    for pose_id, model_ligands_pose in sub['ligands'].items():
        pose_number = int(pose_id.split()[1])
        for model_ligand_id, model_ligand in model_ligands_pose.items():
            model_ligand_toks = model_ligand_id.split()
            model_ligand_name = model_ligand_toks[-1]
            try:
                model_ligand_number = int(model_ligand_toks[1])
            except ValueError as e:
                model_ligand_number = 1

            mol_list.append([pose_id,
                             model_ligands_pose,
                             model_ligand_id,
                             model_ligand,
                             model_ligand_name,
                             model_ligand_number])
    return pd.DataFrame(mol_list, columns=["pose_id", "model_ligands_pose", "model_ligand_id", "model_ligand",
                                           "model_ligand_name", "model_ligand_number"])


def main():
    submission = "T1146LG205_2"
    target_name = submission.split("LG")[0]
    target_pdb = f"{target_name}_lig.pdb"

    df_tm = read_lga_matrix_file(f"{ROTATION_DIR}/{target_name}.rot")
    df_ref = read_reference("proteins_ok_close.csv", SOLUTIONS_DIR)

    sub = read_submission_file(f"{SUBMISSION_DIR}/{target_name}/{submission}")
    rot_name = submission.replace("LG", "TS")
    name, tm = df_tm.query("name == @rot_name").values[0]

    df_submission_ligands = get_submission_ligands()

    print(df_submission_ligands.model_ligand_name.unique())

    # mol_list = []
    # for pose_id, model_ligands_pose in sub['ligands'].items():
    #     pose_number = int(pose_id.split()[1])
    #     for model_ligand_id, model_ligand in model_ligands_pose.items():
    #         mol_list.append(Chem.MolFromMolBlock(model_ligand))
    sys.exit(0)

    ref_mol_list = df_ref.query("target == @target_pdb").mol.values

    writer = Chem.SDWriter("transformed.sdf")
    for mol in mol_list:
        transform_molecule(mol, tm)
        for ref_mol in ref_mol_list:
            fm_score = featuremap_score.score(ref_mol, mol)
            print(fm_score)
        writer.write(mol)
    writer.close()
    print("done")


# eval_ligands("casp_ligands.csv")
# test_read("casp_ligands_with_mols.csv")

read_mmalign_matrix_files(f"{ROTATION_DIR}/ROT/H1114")
