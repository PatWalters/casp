#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import prody
import json
from tqdm.auto import tqdm
from rdkit import Chem
from casp_ligand_comparison import transform_molecule
from casp_utils import read_submission_file, pdb_str_to_atomgroup
import prody
import numpy as np
import sys


def transform_atomgroup(ag, tm):
    crds = ag.getCoords()
    new_crds = []
    for x, y, z in crds:
        x_new = tm[0][0] * x + tm[0][1] * y + tm[0][2] * z + tm[0][3]
        y_new = tm[1][0] * x + tm[1][1] * y + tm[1][2] * z + tm[1][3]
        z_new = tm[2][0] * x + tm[2][1] * y + tm[2][2] * z + tm[2][3]
        new_crds.append([x_new, y_new, z_new])
    ag.setCoords(np.array(new_crds))


home = str(Path.home())
LIGAND_DIR = f"{home}/DATA/CASP/FINAL/LIGAND"
SUBMISSION_DIR = f"{home}/DATA/CASP/FINAL/SUBMISSIONS"
SOLUTIONS_DIR = f"{home}/DATA/CASP/FINAL/XR/SOLUTIONS"

#submission = "T1152LG347_3"
#submission = "T1152LG472_1"
submission = sys.argv[1]
pose_num = int(sys.argv[2])
ligand_df = pd.read_csv("2022_10_02_casp_ligand_eval.csv",low_memory=False)
sub_df = ligand_df.query("submission == @submission and pose_num == @pose_num")
target = sub_df.target.values[0]
ref_pdb_file = f"{SOLUTIONS_DIR}/{target}_lig.pdb"
submission_file = f"{SUBMISSION_DIR}/{target}/{submission}"
sub = read_submission_file(submission_file)
sub_protein_ag = pdb_str_to_atomgroup(sub['protein'])
rmat = sub_df.rotation_matrix.values[0]
if isinstance(rmat,str):
    tm = json.loads(sub_df.rotation_matrix.values[0])
    transform_atomgroup(sub_protein_ag, tm)
prody.writePDB("sub_prot.pdb", sub_protein_ag)

try:
    ref_ag = prody.parsePDB(ref_pdb_file)
except OSError as e:
    print("fallback")
    ref_pdb_file = f"{SOLUTIONS_DIR}/{target}v2_lig.pdb"
    ref_ag = prody.parsePDB(ref_pdb_file)
print(ref_pdb_file)
lig_ag = ref_ag.select("not protein and not water")
prody.writePDB("ref.pdb", ref_ag)
prody.writePDB("ref_lig.pdb",lig_ag)
writer = Chem.SDWriter("sub_lig.sdf")
for idx, row in sub_df.iterrows():
    mol = Chem.MolFromMolBlock(row.mol_block)
    rot_mat = row.rotation_matrix
    if isinstance(rot_mat,str):
        tm = json.loads(rot_mat)
        transform_molecule(mol, tm)
    writer.write(mol)
writer.close()
print("pymol ref.pdb sub_prot.pdb ref_lig.pdb sub_lig.sdf")
