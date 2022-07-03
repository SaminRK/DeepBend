from typing import Literal, Union, NewType

PosOneIdx = int

DNASeq = str
DiNc = str
KMerSeq = str
C0 = float

PositiveInt = int
NonNegativeInt = int
ZeroToOne = float

LIBRARY_NAMES = Literal["cnl", "rl", "tl", "chrvl", "libl"]

YeastChrNum = Literal[
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
]

ChrId = Union[YeastChrNum, Literal["VL"]]
