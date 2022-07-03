class GDataSubDir:
    PROMOTERS = "promoters"
    TEST = "test"
    HELSEP = "helical_separation"
    ML_MODEL = "ml_model"
    MOTIF = "motif"
    BOUNDARIES = "boundaries"
    NUCLEOSOMES = "nucleosomes"
    LINKERS = "linkers"
    DOMAINS = "domains"


class FigSubDir:
    PROMOTERS = "promoters"
    TEST = "test"
    LINKERS = "linkers"
    CROSSREGIONS = "crossregions"
    BOUNDARIES = "boundaries"
    NDRS = "ndrs"
    CHROMOSOME = "chromosome"
    MOTIFS = "motifs"
    LOOP_ANCHORS = "loop_anchors"
    NUCLEOSOMES = "nucleosomes"
    LOGOS = "logos"
    MODELS = "models"
    GENES = "genes"


YeastChrNumList = (
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
)

ChrIdList = YeastChrNumList + ("VL",)

CNL = "cnl"
RL = "rl"
TL = "tl"
CHRVL = "chrvl"
LIBL = "libl"

CNL_LEN = 19907
RL_LEN = 12472
TL_LEN = 82368
CHRVL_LEN = 82404
LIBL_LEN = 92918

SEQ_LEN = 50
ONE_INDEX_START = 1

CHRV_TOTAL_BP = 576871
CHRV_TOTAL_BP_ORIGINAL = 576874
MAX_TOTAL_BP = 2e6
