import sys
import pandas as pd
import json
import numpy as np
import seaborn as sns


def tanimoto(a, b):
    set_a = set(a)
    set_b = set(b)
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


def get_reference_contacts(target_pdb, res_name, ref_df, contact_type):
    res_df = ref_df.query("target == @target_pdb and res_name == @res_name")
    contact_list = []
    for r in res_df[contact_type].values:
        contact_list.append(json.loads(r))
    return contact_list


def compare_submissions(target_pdb, target, sub_df, ref_df, contact_type):
    sub_ligand_name_list = sub_df.corrected_name.unique()
    contact_dict = {}
    for ligand_name in sub_ligand_name_list:
        contact_dict[ligand_name] = get_reference_contacts(target_pdb, ligand_name, ref_df, contact_type)
    tanimoto_list = []
    for idx, row in sub_df.iterrows():
        lig_contacts = row[contact_type]
        lig_name = row.corrected_name
        ref_contacts = contact_dict[lig_name]
        try:
            lig_contacts = json.loads(lig_contacts)
            tanimoto_list.append(best_tanimoto(lig_contacts, ref_contacts))
        except TypeError as e:
            tanimoto_list.append(-1)
    return tanimoto_list


def xcompare_submissions(target_pdb, target, sub_df, ref_df, contact_type):
    tanimoto_list = []
    for res_name in sub_df.corrected_name.unique():
        query_str = "target == @target_pdb and res_name == @res_name"
        raw_contacts = ref_df.query(query_str)[contact_type].values
        ref_contacts = []
        for r in raw_contacts:
            ref_contacts.append(json.loads(r))
        sub_sel_df = sub_df.query("target == @target and corrected_name == @res_name")
        for idx, row in sub_sel_df.iterrows():
            lig_contacts = row[contact_type]
            try:
                lig_contacts = json.loads(lig_contacts)
                tanimoto_list.append(best_tanimoto(lig_contacts, ref_contacts))
            except TypeError as e:
                tanimoto_list.append(-1)

    return tanimoto_list


def debug():
    df_ref = pd.read_csv("proteins_ok_close.csv")
    df_lig = pd.read_csv("casp_ligands.csv")
    df_z = df_lig.query("submission == 'H1114LG248_1'")
    tgt = 'H1114_lig.pdb'
    target_name, num_lig, lig_tanimoto_list_3 = compare_submissions(tgt, df_z, df_ref, "close_3")
    print(lig_tanimoto_list_3)


def get_target_name(target_pdb, target_list):
    target = target_pdb.replace("_lig.pdb", "")
    if target not in target_list:
        target = target.split("v")[0]
    return target


def test():
    df_ref = pd.read_csv("proteins_ok_close.csv")
    get_reference_contacts("T1124_lig.pdb", "SAH", df_ref, "close_3")


def main():
    df_ref = pd.read_csv("proteins_ok_close.csv")
    df_lig = pd.read_csv("casp_ligands.csv")
    df_z = df_lig.query("submission == 'H1114LG248_1'")
    target_list = df_lig.target.unique()

    tgt_df_list = []
    for tgt in df_ref.target.unique():
        # for tgt in ["H1114_lig.pdb"]:
        tgt_name = get_target_name(tgt, target_list)
        tgt_df = df_lig.query("target == @tgt_name and relevant == 'Yes'").dropna(axis=0,
                                                                                  subset=["corrected_name"]).copy()
        lig_tanimoto_list_3 = compare_submissions(tgt, tgt_name, tgt_df, df_ref, "close_3")
        lig_tanimoto_list_5 = compare_submissions(tgt, tgt_name, tgt_df, df_ref, "close_5")
        tgt_df["tanimoto_3"] = lig_tanimoto_list_3
        tgt_df["tanimoto_5"] = lig_tanimoto_list_5
        tgt_df_list.append(tgt_df)
        num_target_ligs = len(tgt_df)
        print(tgt, min(lig_tanimoto_list_3), max(lig_tanimoto_list_3))
    combo_df = pd.concat(tgt_df_list)
    combo_df.to_csv("tanimoto_summary.csv", index=False)


main()
