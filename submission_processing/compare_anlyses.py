#!/usr/bin/env python

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.backends.backend_pdf import PdfPages


def compare_analyses(xr_file, pw_file, pdf_pages):
    try:
        pw_df = pd.read_csv(pw_file, low_memory=False)
        xr_df = pd.read_csv(xr_file, low_memory=False)
        if "target_idsubmission_file" not in xr_df.columns:
            xr_df["target_idsubmission_file"] = xr_df.target_id + xr_df.submission_file
        pw_df["target_idsubmission_file"] = pw_df.ref_protein + pw_df.submission
        pw_df = pw_df.query("(bad_protein == False) and (bad_ligand == False) and (relevant == 'Yes')")
        pw_df = pw_df.rename({"rmsd": "PW_RMSD"}, axis="columns")
        merge_df = pw_df.merge(xr_df, left_on=["target_idsubmission_file", "pose_num", "ligand_number"],
                               right_on=["target_idsubmission_file", "pose_num", "ref_lig_num"])
        target = xr_file.split(".")[0]
        num_pw = len(pw_df.query("target == @target"))

        sns.set(rc={'figure.figsize': (6, 6)})
        sns.set_style('whitegrid')
        sns.set_context('talk')

        plt.figure()

        ax = sns.scatterplot(x="PW_RMSD", y="lddt_pli", data=merge_df)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 10)
        ax.set_xlabel("PW RMSD")
        ax.set_ylabel("lDDT PLI")
        ax.set_title(f"{target} - PW {num_pw} - XR {len(xr_df)} - Merge {len(merge_df)}")

        plt.tight_layout()
        pdf_pages.savefig(ax.figure)
        #plt.savefig(f"{target}.png")
        return merge_df
    except pd.errors.ParserError:
        print(f"Error with {xr_file}")
        return None


def main():
    with PdfPages('plots.pdf') as pdf_pages:
        for filename in sorted(glob("*.csv")):
            print(filename)
            res = compare_analyses(filename,
                                   "/Users/pwalters/software/casp/submission_processing/tmp.csv",
                                   pdf_pages)
            if res is not None:
                name = filename.split(".")[0]
                print(name)
                res.to_csv(f"./MERGE/{name}_merged.csv")

main()
