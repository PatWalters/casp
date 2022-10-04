#!/usr/bin/env python

import sys
import pandas as pd
import json
from rdkit import Chem
from featuremap_score import FeatureMapScore
from rdkit.Geometry import Point3D
from tqdm.auto import tqdm
from rdkit.Chem.AllChem import CalcRMS
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np
import useful_rdkit_utils as uru
from spyrmsd import rmsd, molecule
from spyrmsd.optional import rdkit as rd
from spyrmsd.exceptions import NonIsomorphicGraphs
from fix_submission_molecules import fix_mq7, fix_oaa

MAX_RMSD = 10000.0

def spyrmsd(rd_mol, rd_ref):
    mol = rd.to_molecule(rd_mol)
    coords = mol.coordinates
    anum = mol.atomicnums
    adj = mol.adjacency_matrix

    ref = rd.to_molecule(rd_ref)
    coords_ref = ref.coordinates
    anum_ref = ref.atomicnums
    adj_ref = ref.adjacency_matrix

    rmsd_val = rmsd.symmrmsd(
        coords_ref,
        coords,
        anum_ref,
        anum,
        adj_ref,
        adj,
        minimize=False,
    )
    return rmsd_val


def tanimoto(set_a, set_b):
    num_intersection = len(set_a.intersection(set_b))
    num_union = len(set_a.union(set_b))
    res = 0.0
    if num_union > 0:
        res = num_intersection / num_union
    # print(f"{len(a)} {len(b)} {num_intersection} {num_union} {a} {b} tc = {res}")
    return res


def best_tanimoto(sub, ref):
    best_tan = -1.0
    if True:
        for r in ref:
            tc = tanimoto(r, sub)
            best_tan = max(best_tan, tc)
    return best_tan


def evaluate_site_overlap(protein_df, ligand_df, contact_type):
    prot_lig_dict = {}
    for prot_lig_name, prot_lig_recs in protein_df.groupby("res_name"):
        prot_lig_dict[prot_lig_name] = prot_lig_recs[contact_type].values

    for prot_lig_name, close_str_list in prot_lig_dict.items():
        prot_lig_dict[prot_lig_name] = [set(json.loads(x)) for x in close_str_list]
    tanimoto_list = []
    for idx, ligand_rec in ligand_df.iterrows():
        ligand_name = ligand_rec.corrected_name
        try:
            ref_contact_list = prot_lig_dict[ligand_name]
            ligand_contact_str = ligand_rec[contact_type]
            ligand_contact_set = set(json.loads(ligand_contact_str))
            best_val = best_tanimoto(ligand_contact_set, ref_contact_list)
        except (KeyError, TypeError) as e:
            best_val = -1.0
        tanimoto_list.append(best_val)
    return tanimoto_list


def transform_molecule(mol, tm):
    conf = mol.GetConformer(0)
    for i in range(0, mol.GetNumAtoms()):
        x, y, z = conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z
        x_new = tm[0][0] * x + tm[0][1] * y + tm[0][2] * z + tm[0][3]
        y_new = tm[1][0] * x + tm[1][1] * y + tm[1][2] * z + tm[1][3]
        z_new = tm[2][0] * x + tm[2][1] * y + tm[2][2] * z + tm[2][3]
        conf.SetAtomPosition(i, Point3D(x_new, y_new, z_new))


def best_featuremap_score(mol, ref_mol_list, featuremap_score):
    best_fms = -1
    for ref_mol in ref_mol_list:
        best_fms = max(best_fms, featuremap_score.score(mol, ref_mol))
    return best_fms


def single_atom_rmsd(m1, m2):
    conf_1 = m1.GetConformer(0)
    conf_2 = m2.GetConformer(0)
    x1, y1, z1 = conf_1.GetAtomPosition(0).x, conf_1.GetAtomPosition(0).y, conf_1.GetAtomPosition(0).z
    x2, y2, z2 = conf_2.GetAtomPosition(0).x, conf_2.GetAtomPosition(0).y, conf_2.GetAtomPosition(0).z
    crds_1 = np.array([x1, y1, z1])
    crds_2 = np.array([x2, y2, z2])
    rmsd = np.sqrt((((crds_1 - crds_2) ** 2) * 3).mean())
    return rmsd


def best_shape_tanimoto(mol, ref_mol_list):
    best_score = 1.1
    writer = Chem.SDWriter("test.sdf")
    writer.write(mol)
    writer.write(ref_mol_list[0])
    writer.close()
    for ref_mol in ref_mol_list:
        shape_tan = AllChem.ShapeTanimotoDist(mol, ref_mol)
        best_score = min(best_score, shape_tan)
    return best_score


def best_rms(mol, ref_mol_list):
    best_rms_val = MAX_RMSD
    for ref_mol in ref_mol_list:
        rms_val = spyrmsd(mol, ref_mol)
        best_rms_val = min(best_rms_val, rms_val)
    return best_rms_val


def xbest_rms(mol, ref_mol_list):
    uncharger = rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)
    best_rms_val = MAX_RMSD
    for ref_mol in ref_mol_list:
        if ref_mol.GetNumAtoms() != mol.GetNumAtoms():
            mol = Chem.RWMol(mol)
            match_set = set(mol.GetSubstructMatch(ref_mol))
            all_set = set(list(range(0, mol.GetNumAtoms())))
            for idx in sorted(list(all_set.difference(match_set)), reverse=True):
                mol.RemoveAtom(idx)
            Chem.SanitizeMol(mol)
        try:
            mol = AllChem.AssignBondOrdersFromTemplate(ref_mol, mol)
            if mol.GetNumAtoms() == 1:
                best_rms_val = min(best_rms_val, single_atom_rmsd(mol, ref_mol))
            else:
                best_rms_val = min(best_rms_val, CalcRMS(mol, ref_mol))
        except (RuntimeError, ValueError) as e:
            pass
            # print("ref = ", Chem.MolToSmiles(ref_mol), ref_mol.GetNumAtoms())
            # print("mol = ", Chem.MolToSmiles(mol), mol.GetNumAtoms())
    return best_rms_val


def evaluate_ligand_overlap(protein_df, ligand_df):
    uru.rd_shut_the_hell_up()
    uncharger = rdMolStandardize.Uncharger()
    featuremap_score = FeatureMapScore()
    prot_lig_dict = {}
    for prot_lig_name, prot_lig_recs in protein_df.groupby("res_name"):
        prot_lig_dict[prot_lig_name] = prot_lig_recs.ligand_mol_block.values
    for prot_lig_name, prot_mol_block_list in prot_lig_dict.items():
        prot_lig_dict[prot_lig_name] = [uncharger.uncharge(Chem.MolFromMolBlock(x)) for x in prot_mol_block_list]
    fms_list = []
    rms_list = []
    shape_tanimoto_list = []
    ref_tmplt_mol = Chem.Mol()
    ligand_mol = Chem.Mol()
    for idx, ligand_rec in tqdm(ligand_df.iterrows(), total=len(ligand_df)):
        ligand_name = ligand_rec.corrected_name
        try:
            ref_mol_list = prot_lig_dict[ligand_name]
            ligand_mol_block = ligand_rec.mol_block
            ref_tmplt_mol = Chem.MolFromSmiles(ligand_rec.zmiles)
            ligand_mol = Chem.MolFromMolBlock(ligand_mol_block)
            #ligand_mol = AllChem.AssignBondOrdersFromTemplate(ref_tmplt_mol, ligand_mol)
            if ligand_name == "MQ7":
                ligand_mol = fix_mq7(ligand_mol)
            if ligand_name == "OAA":
                ligand_mol = fix_oaa(ligand_mol)
            rmat = json.loads(ligand_rec.rotation_matrix)
            transform_molecule(ligand_mol, rmat)
            best_fms_val = best_featuremap_score(ligand_mol, ref_mol_list, featuremap_score)
            best_rms_val = best_rms(ligand_mol, ref_mol_list)
            best_shape_score = best_shape_tanimoto(ligand_mol, ref_mol_list)
        except (TypeError, KeyError, AssertionError, NonIsomorphicGraphs, ValueError) as e:
            print(e,Chem.MolToSmiles(ref_tmplt_mol),Chem.MolToSmiles(ligand_mol))
            best_fms_val = -1.0
            best_rms_val = MAX_RMSD
            best_shape_score = -1.0
        shape_tanimoto_list.append(best_shape_score)
        fms_list.append(best_fms_val)
        rms_list.append(best_rms_val)
    return shape_tanimoto_list, fms_list, rms_list


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} ref_proteins.csv ligands.csv")
        sys.exit(0)
    protein_ref_df = pd.read_csv(sys.argv[1])
    ligand_df = pd.read_csv(sys.argv[2])
    protein_target_list = protein_ref_df.target.unique()

    df_list = []
    for target in ligand_df.target.unique():
        df_ligand = ligand_df.query("target == @target").copy()
        #df_ligand = ligand_df.query("submission == @submission_id and pose_num == 1").copy()
        target_protein_list = [x for x in protein_target_list if x.startswith(target)]
        for tp in target_protein_list:
            df_protein = protein_ref_df.query("target == @tp")
            print(tp, len(df_ligand), len(df_protein))
            close_3_list = evaluate_site_overlap(df_protein, df_ligand, "close_3")
            close_5_list = evaluate_site_overlap(df_protein, df_ligand, "close_5")
            # print(target, len(df_ligand), len(close_3_list), len(close_5_list))
            # rms_list, fms_list = evaluate_ligand_overlap(df_protein, df_ligand)
            shape_tanimoto_list, fms_list, rms_list = evaluate_ligand_overlap(df_protein, df_ligand)
            df_ligand['ref_protein'] = tp
            df_ligand['close_3'] = close_3_list
            df_ligand['close_5'] = close_5_list
            df_ligand['shape_tanimoto'] = shape_tanimoto_list
            df_ligand['rmsd'] = rms_list
            df_ligand['fms'] = fms_list
        df_list.append(df_ligand)
    combo_df = pd.concat(df_list)
    combo_df.to_csv("2022_10_02_casp_ligand_eval.csv", index=False)


def debug():
    #submission_id = 'H1114LG119_1'
    submission_id = 'H1114LG086_2'
    target_name = submission_id.split('LG')[0]
    protein_ref_df = pd.read_csv("proteins_ok.csv")
    ligand_df = pd.read_csv("2022_09_19_casp_ligands_with_mols.csv")
    protein_target_list = protein_ref_df.target.unique()

    df_list = []
    for target in [target_name]:
        df_ligand = ligand_df.query("target == @target").copy()
        #df_ligand = ligand_df.query("submission == @submission_id and pose_num == 1").copy()
        target_protein_list = [x for x in protein_target_list if x.startswith(target)]
        for tp in target_protein_list:
            df_protein = protein_ref_df.query("target == @tp")
            print(tp, len(df_ligand), len(df_protein))
            close_3_list = evaluate_site_overlap(df_protein, df_ligand, "close_3")
            close_5_list = evaluate_site_overlap(df_protein, df_ligand, "close_5")
            # print(target, len(df_ligand), len(close_3_list), len(close_5_list))
            # rms_list, fms_list = evaluate_ligand_overlap(df_protein, df_ligand)
            shape_tanimoto_list, fms_list, rms_list = evaluate_ligand_overlap(df_protein, df_ligand)
            df_ligand['ref_protein'] = tp
            df_ligand['close_3'] = close_3_list
            df_ligand['close_5'] = close_5_list
            df_ligand['shape_tanimoto'] = shape_tanimoto_list
            df_ligand['rmsd'] = rms_list
            df_ligand['fms'] = fms_list
        df_list.append(df_ligand)
    combo_df = pd.concat(df_list)
    combo_df.to_csv("debug_eval.csv", index=False)


if __name__ == "__main__":
    main()
    #debug()
