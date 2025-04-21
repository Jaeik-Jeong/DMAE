from MissingImpute import LIImpute, HAImpute, RecommendedImpute
from AnomalyDetect import Zscore

LIImpute("ID1")

HAImpute("ID1", resol=5)

RecommendedImpute("ID1", resol=5)

Zscore("ID1_li_impute", window=12, threshold=3)

Zscore("ID1_ha_impute", window=12, threshold=3)

Zscore("ID1_recommended_impute", window=12, threshold=3)
