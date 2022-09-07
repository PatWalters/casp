#!/usr/bin/env python

import sys
from pathlib import Path, PurePath
from process_submission_file import LigandInfo
from biopandas.pdb import PandasPdb
from glob import glob
from collections import Counter
import pandas as pd
from tqdm.auto import tqdm

home = str(Path.home())
LIGAND_DIR = f"{home}/DATA/CASP/FINAL/LIGAND"
SUBMISSION_DIR = f"{home}/DATA/CASP/FINAL/SUBMISSIONS"
SOLUTIONS_DIR = f"{home}/DATA/CASP/FINAL/SOLUTIONS"


def process_ref_protein(filename):
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
    for k, v in prot.df['HETATM'].groupby("chain_id"):
        for row in v[['residue_name', 'residue_number']].drop_duplicates().values:
            res_name, res_num = row
            if res_name in lig_name_set:
                res.append([k, res_name, res_num])
    ref_df = pd.DataFrame(res, columns=["chain", "res_name", "res_num"])
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
