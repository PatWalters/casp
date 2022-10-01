#!/usr/bin/env python

from glob import glob
from itertools import zip_longest
from pathlib import Path, PurePath

import numpy as np
import pandas as pd

home = str(Path.home())
LIGAND_DIR = f"{home}/DATA/CASP/FINAL/LIGAND"
SUBMISSION_DIR = f"{home}/DATA/CASP/FINAL/SUBMISSIONS"
ROTATION_DIR = f"{home}/DATA/CASP/FINAL/ROTATION"
SOLUTIONS_DIR = f"{home}/DATA/CASP/FINAL/SOLUTIONS"


def parse_lga_rotation_matrix(lines):
    name = lines[0].strip()
    name = name.split(".")[0]
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
    for filename in glob(f"{dirname}/*.ROT"):
        basename = PurePath(filename).parts[-1]
        basename = basename.replace("o.ROT",".ROT")
        name = basename.replace(".ROT", "")
        with open(filename) as ifs:
            lines = ifs.readlines()
            tm = parse_mmalign_rotation_matrix(lines[2:5])
            mat_list.append([name, tm])
    mat_df = pd.DataFrame(mat_list, columns=["name", "tm"])
    return mat_df


def build_rotation_dataframe():
    df_list = []
    monomer_list = sorted(glob(f"{ROTATION_DIR}/MONOMER_1/*.rot")) + sorted(glob(f"{ROTATION_DIR}/MONOMER_2/*.rot"))
    for filename in monomer_list:
        basename = PurePath(filename).parts[-1]
        target = basename.replace(".rot", "")
        df = read_lga_matrix_file(filename)
        df["target"] = target
        df_list.append(df)
    multimer_list = glob(f"{ROTATION_DIR}/MULTIMER_1/H*") + glob(f"{ROTATION_DIR}/MULTIMER_2/H*")
    for dirname in multimer_list:
        target = PurePath(dirname).parts[-1]
        df = read_mmalign_matrix_files(dirname)
        df["target"] = target
        df_list.append(df)
    rot_df = pd.concat(df_list)
    return rot_df


def test_rotation_dataframe():
    rot_df = build_rotation_dataaframe()
    for filename in sorted(glob(f"{SOLUTIONS_DIR}/*_lig.pdb")):
        basename = PurePath(filename).parts[-1]
        target = basename.replace("_lig.pdb", "")
        df_target = rot_df.query("target == @target")
        print(target, df_target.shape)


if __name__ == "__main__":
    test_rotation_dataframe()

