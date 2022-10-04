import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig


### A minor refactoring of the code from https://github.com/susanhleung/SuCOS
### Also related to https://chemrxiv.org/engage/chemrxiv/article-details/60c741a99abda23230f8bed5
### and http://rdkit.blogspot.com/2017/11/using-feature-maps.html

class FeatureMapScore:
    def __init__(self):
        self.fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        self.fmParams = {}
        for k in self.fdef.GetFeatureFamilies():
            fparams = FeatMaps.FeatMapParams()
            self.fmParams[k] = fparams

        self.keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder',
                     'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')
        
    def score(self, small_m, large_m, score_mode=FeatMaps.FeatMapScoreMode.All):
        featLists = []
        for m in [small_m, large_m]:
            rawFeats = self.fdef.GetFeaturesForMol(m)
            # filter that list down to only include the ones we're intereted in
            featLists.append([f for f in rawFeats if f.GetFamily() in self.keep])
        fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=self.fmParams) for x in featLists]
        fms[0].scoreMode = score_mode
        denominator = min(fms[0].GetNumFeatures(), len(featLists[1]))
        fm_score = 0.0
        if denominator > 0:
            fm_score = fms[0].ScoreFeats(featLists[1]) / denominator
        return fm_score
