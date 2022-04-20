BIRADS_DICT = {
    "1": 0,
    "2": 1,
    "3": 2,
    "4a": 3,
    "4": 4,
    "4b": 4,
    "4c": 5,
    "5": 6,
    "6": 7,
}

BCDR_BIRADS_DICT = {
    "benign": 3,
    "malignant": 6,
}

DENSITY_DICT = {
    1: 0.0,
    2: 0.33,
    3: 0.67,
    4: 1.0,
}

BCDR_SUBDIRECTORIES = {
    "d01": "BCDR-D01_dataset",
    "d02": "BCDR-D02_dataset",
    "f01": "BCDR-F01_dataset",
    "f02": "BCDR-F02_dataset",
    "f03": "BCDR-F03",
}

BCDR_HEALTHY_SUBDIRECTORIES = {
    "dn01": "BCDR-DN01_dataset",
}

BCDR_VIEW_DICT = {
    1: {"laterality": "R", "view": "CC"},
    2: {"laterality": "L", "view": "CC"},
    3: {"laterality": "R", "view": "MLO"},
    4: {"laterality": "L", "view": "MLO"},
}

ROI_TYPES = ["calcification", "mass", "other", "healthy"]
LATERALITIES = ["R", "L"]
VIEWS = ["CC", "MLO"]
BIOPSY_STATUS = ["benign", "malignant", "unknown", "benign_without_callback"]

CBIS_DDSM_CSV_DICT = {
    "train_mass": "mass_case_description_train_set.csv",
    "train_calc": "calc_case_description_train_set.csv",
    "test_mass": "mass_case_description_test_set.csv",
    "test_calc": "calc_case_description_test_set.csv",
}

# Precomputed mean and std of the MMG radiomics values for normalization
# so that they all have the same distribution.

RADIOMICS_NORMALIZATION_PARAMS = {
    "firstorder_10Percentile": (142.43366762, 44.11266766),
    "firstorder_90Percentile": (209.10859599, 48.464337),
    "firstorder_Energy": (9.2277712e08, 2.59525722e09),
    "firstorder_Entropy": (1.84706376, 0.53080851),
    "firstorder_InterquartileRange": (32.96382521, 22.05303094),
    "firstorder_Kurtosis": (4.23572049, 3.11921052),
    "firstorder_Maximum": (236.75931232, 34.40875073),
    "firstorder_MeanAbsoluteDeviation": (21.63719368, 9.45757861),
    "firstorder_Mean": (178.99395081, 48.14307399),
    "firstorder_Median": (182.30229226, 53.78933513),
    "firstorder_Minimum": (95.12320917, 37.93489756),
    "firstorder_Range": (141.63610315, 40.65896269),
    "firstorder_RobustMeanAbsoluteDeviation": (15.08395872, 9.0401025),
    "firstorder_RootMeanSquared": (181.44247138, 47.97221452),
    "firstorder_Skewness": (-0.3589202, 1.21181496),
    "firstorder_TotalEnergy": (9.2277712e08, 2.59525722e09),
    "firstorder_Uniformity": (0.36920271, 0.16007276),
    "firstorder_Variance": (866.11378599, 694.50121623),
    "glcm_Autocorrelation": (23.7478749, 16.83773733),
    "glcm_ClusterProminence": (151.09406971, 237.16392805),
    "glcm_ClusterShade": (-4.90521626, 21.58987166),
    "glcm_ClusterTendency": (5.12075403, 4.2831002),
    "glcm_Contrast": (0.59812566, 0.39496862),
    "glcm_Correlation": (0.73520569, 0.14865705),
    "glcm_DifferenceAverage": (0.39802944, 0.16869139),
    "glcm_DifferenceEntropy": (1.06552942, 0.28572911),
    "glcm_DifferenceVariance": (0.40811543, 0.26256818),
    "glcm_Id": (0.82707221, 0.06492896),
    "glcm_Idm": (0.82000425, 0.06980721),
    "glcm_Idmn": (0.98765033, 0.00751467),
    "glcm_Idn": (0.9510605, 0.01927901),
    "glcm_Imc1": (-0.33813392, 0.12017381),
    "glcm_Imc2": (0.79439217, 0.14319389),
    "glcm_InverseVariance": (0.2846229, 0.10762846),
    "glcm_JointAverage": (4.47974746, 1.59647117),
    "glcm_JointEnergy": (0.2579466, 0.18312335),
    "glcm_JointEntropy": (2.94297081, 0.88318512),
    "glcm_MCC": (0.77152843, 0.13635375),
    "glcm_MaximumProbability": (0.40344095, 0.20557345),
    "glcm_SumAverage": (8.95949491, 3.19294234),
    "glcm_SumEntropy": (2.47752737, 0.7102723),
    "glcm_SumSquares": (1.42971992, 1.12655987),
    "glrlm_GrayLevelNonUniformity": (2669.23655655, 6853.51038902),
    "glrlm_GrayLevelNonUniformityNormalized": (0.25379554, 0.07927515),
    "glrlm_GrayLevelVariance": (1.77590353, 1.18685675),
    "glrlm_HighGrayLevelRunEmphasis": (20.18603371, 11.53007498),
    "glrlm_LongRunEmphasis": (40.42298578, 71.11345727),
    "glrlm_LongRunHighGrayLevelEmphasis": (1109.07673212, 3317.9802143),
    "glrlm_LongRunLowGrayLevelEmphasis": (5.63402361, 25.35486709),
    "glrlm_LowGrayLevelRunEmphasis": (0.11914425, 0.07528684),
    "glrlm_RunEntropy": (4.39445274, 0.594008),
    "glrlm_RunLengthNonUniformity": (2761.65533251, 7723.09385359),
    "glrlm_RunLengthNonUniformityNormalized": (0.32370296, 0.10878795),
    "glrlm_RunPercentage": (0.35167213, 0.12416027),
    "glrlm_RunVariance": (26.00749446, 49.09485122),
    "glrlm_ShortRunEmphasis": (0.5700265, 0.10539516),
    "glrlm_ShortRunHighGrayLevelEmphasis": (11.08726289, 7.07468246),
    "glrlm_ShortRunLowGrayLevelEmphasis": (0.07158835, 0.04401274),
    "gldm_DependenceEntropy": (4.26998248, 0.95559285),
    "gldm_DependenceNonUniformity": (6016.98715802, 18379.66886007),
    "gldm_DependenceNonUniformityNormalized": (0.20131688, 0.11284484),
    "gldm_DependenceVariance": (5.25149548, 1.62693271),
    "gldm_GrayLevelNonUniformity": (9851.36370988, 26154.10469654),
    "gldm_GrayLevelVariance": (1.55295919, 1.18810652),
    "gldm_HighGrayLevelEmphasis": (23.46400646, 16.38190989),
    "gldm_LargeDependenceEmphasis": (44.5124087, 12.05403165),
    "gldm_LargeDependenceHighGrayLevelEmphasis": (1132.8075111, 1035.35717176),
    "gldm_LargeDependenceLowGrayLevelEmphasis": (4.42103531, 4.55700028),
    "gldm_LowGrayLevelEmphasis": (0.10521841, 0.07616953),
    "gldm_SmallDependenceEmphasis": (0.08768416, 0.04288177),
    "gldm_SmallDependenceHighGrayLevelEmphasis": (1.80101472, 1.52441773),
    "gldm_SmallDependenceLowGrayLevelEmphasis": (0.01126947, 0.00867757),
    "glszm_GrayLevelNonUniformity": (304.18558885, 801.11355641),
    "glszm_GrayLevelNonUniformityNormalized": (0.23822761, 0.08128095),
    "glszm_GrayLevelVariance": (1.89735366, 0.893046),
    "glszm_HighGrayLevelZoneEmphasis": (18.70694244, 10.00307326),
    "glszm_LargeAreaEmphasis": (192568.24047119, 608028.46675035),
    "glszm_LargeAreaHighGrayLevelEmphasis": (3706619.35307794, 13028292.25888132),
    "glszm_LargeAreaLowGrayLevelEmphasis": (19258.77346448, 70176.82141362),
    "glszm_LowGrayLevelZoneEmphasis": (0.14861526, 0.10364029),
    "glszm_SizeZoneNonUniformity": (499.04293533, 1399.66502046),
    "glszm_SizeZoneNonUniformityNormalized": (0.32429765, 0.09127774),
    "glszm_SmallAreaEmphasis": (0.57454793, 0.0989412),
    "glszm_SmallAreaHighGrayLevelEmphasis": (10.53597032, 6.3853404),
    "glszm_SmallAreaLowGrayLevelEmphasis": (0.08771519, 0.0609889),
    "glszm_ZoneEntropy": (4.39880743, 0.65041586),
    "glszm_ZonePercentage": (0.08132152, 0.05733178),
    "glszm_ZoneVariance": (191360.99625282, 606125.04928976),
    "ngtdm_Busyness": (73.74896467, 156.81818329),
    "ngtdm_Coarseness": (0.01310976, 0.02125925),
    "ngtdm_Complexity": (7.56734463, 4.76530197),
    "ngtdm_Contrast": (0.0269077, 0.02063085),
}
