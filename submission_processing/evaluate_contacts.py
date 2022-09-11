#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import prody
import json
from tqdm.auto import tqdm

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


protein_dict = {}

df = pd.read_csv("proteins_ok.csv")
df['close_3'] = ['[]'] * len(df)
df['close_5'] = ['[]'] * len(df)
for idx, row in tqdm(df.iterrows()):
    tgt = row.target
    prody_prot = protein_dict.get(tgt)
    if prody_prot is None:
        protein_dict[tgt] = prody.parsePDB(f"{SOLUTIONS_DIR}/{tgt}")
        prody_prot = protein_dict[tgt]
    close_3 = find_close_residues(prody_prot, row.chain, row.res_num, row.res_name, 3.0)
    close_5 = find_close_residues(prody_prot, row.chain, row.res_num, row.res_name, 5.0)
    df['close_3'].at[idx] = json.dumps(close_3)
    df['close_5'].at[idx] = json.dumps(close_5)
df.to_csv("proteins_ok_close.csv",index=False)

