#!/usr/bin/env python

from io import StringIO
from rdkit import Chem
from openeye import oechem, oeshape

def rdmol_to_oemol(rdmol):
    buff = Chem.MolToMolBlock(rdmol)
    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SDF)
    ims.openstring(buff)
    oemol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ims,oemol)
    return oemol

def best_rocs_scores(refmol, fitmol_list):
    best_shape_score = -1
    best_color_score = -1
    for fitmol in fitmol_list:
        shape_score, color_score = rocs_score_rdmols(refmol, fitmol)
        if shape_score > best_shape_score:
            best_shape_score = shape_score
            best_color_score = color_score
    return best_shape_score, best_color_score

def rocs_score_rdmols(refmol, fitmol):
    ref_oemol = rdmol_to_oemol(refmol)
    fit_oemol = rdmol_to_oemol(fitmol)
    return simple_rocs_score(ref_oemol, fit_oemol)

def simple_rocs_score(refmol, fitmol):
    shapeFunc = oeshape.OEAnalyticShapeFunc()
    shapeFunc.SetupRef(refmol)
    shape_res = oeshape.OEOverlapResults()
    shapeFunc.Overlap(fitmol,shape_res)
    
    prep = oeshape.OEOverlapPrep()
    prep.Prep(refmol)
    prep.Prep(fitmol)
    colorFunc = oeshape.OEAnalyticColorFunc()
    colorFunc.SetupRef(refmol)
    color_res = oeshape.OEOverlapResults()
    colorFunc.Overlap(fitmol, color_res)
    return shape_res.GetShapeTanimoto(),color_res.GetColorTanimoto()
    
    
def main():
    ref_mol = Chem.MolFromMolFile("ref_lig.sdf")
    suppl = Chem.SDMolSupplier("sub_lig.sdf")
    fit_mol_list = [x for x in suppl]
    shape_tan, color_tan = best_rocs_scores(ref_mol, fit_mol_list)
    print(f"{shape_tan:.2f} {color_tan:.2f}")

if __name__ == "__main__":
    main()


    
