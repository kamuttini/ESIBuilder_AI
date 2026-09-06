"""Legacy ESIBuilder constants and .fss house defaults.

Values come from OldSoftwareEsiBuilder/general_define.h and from the field survey of
templates/DB_setup/*.fss. See docs/specifiche_app_esibuilder_ai_2026-08-26.md.
"""

from __future__ import annotations

# --- #06 VIDEO_INPUT -------------------------------------------------------
VIDEO_NOT_DEFINED = -1
VIDEO_HDMI = 0
VIDEO_VGA = 1
VIDEO_INPUT_LABELS = {VIDEO_HDMI: "HDMI", VIDEO_VGA: "VGA"}

# Acceptable frame sizes (general_define.h:266-269).
VIDEO_INPUT_MIN = (640, 480)
VIDEO_INPUT_MAX = (2000, 2000)

# --- #04 PROBE_TYPE --------------------------------------------------------
PROBE_TYPE_NOT_DEFINED = 0
PROBE_TYPE_LINEAR = 1
PROBE_TYPE_CONVEX = 2
PROBE_TYPE_TRANS_L = 3
PROBE_TYPE_TRANS_T = 4
PROBE_TYPE_TRANS_SINGLE = 5
PROBE_TYPE_LABELS = {
    PROBE_TYPE_LINEAR: "Lineare",
    PROBE_TYPE_CONVEX: "Convex",
    PROBE_TYPE_TRANS_L: "Transrettale L",
    PROBE_TYPE_TRANS_T: "Transrettale T",
    PROBE_TYPE_TRANS_SINGLE: "Transrettale singola",
}
BIPLANE_PROBE_TYPES = (PROBE_TYPE_TRANS_L, PROBE_TYPE_TRANS_T)

# --- #12 GROUP_ORIENTATION -------------------------------------------------
ID_CHECK_NOT_DEFINE = 0
ID_CHECK_ECHO = 1
ID_CHECK_PROBE = 2
ID_CHECK_PROIBITED = 3
ID_CHECK_ORIENTATION = 4  # "symbol": the orientation is read from a marker
ID_CHECK_DEPTH = 5  # the orientation is inferred from the depth
GROUP_ORIENTATION_LABELS = {
    ID_CHECK_ORIENTATION: "Simbolo (marker di orientamento)",
    ID_CHECK_DEPTH: "Depth",
}

# --- biplane recognition (#26) ---------------------------------------------
BIPLANA_NO_SIGN_NO_DEPTH = 0
BIPLANA_TEST_DEPTH = 1
BIPLANA_TEST_SIGN = 2
BIPLANA_MODE_LABELS = {
    BIPLANA_NO_SIGN_NO_DEPTH: "Nessun segno ne depth diverse",
    BIPLANA_TEST_DEPTH: "Nessun segno ma depth diverse tra L e T",
    BIPLANA_TEST_SIGN: "Segno speciale su schermo",
}

# --- orientation groups ----------------------------------------------------
# Order of the four blocks inside #15, #16, #17 and #24 (general_define.h:331).
FLIP_NO = 0
FLIP_LR = 1
FLIP_UD = 2
FLIP_LR_UD = 3
ORIENTATION_ORDER = (FLIP_NO, FLIP_LR, FLIP_UD, FLIP_LR_UD)
ORIENTATION_KEYS = ("NF", "LR", "UD", "LRUD")

# --- match defaults (spec sezione 8) --------------------------------------
# House convention measured on templates/DB_setup: P1=20 / P2=120 in 621 of ~640
# template blocks, CH=7 dominant. MM: euclidean distance after threshold.
DEFAULT_CHANNEL = 7
DEFAULT_MATCH_METHOD = 6  # FE_TH_METHOD; set to 0 for plain CV_TM_SQDIFF
DEFAULT_P1 = 20.0
DEFAULT_P2 = 120.0

# --- ESI display limit ----------------------------------------------------
# The echo rectangle cannot exceed the ESI screen (brainstorming.md).
RECT_MAX_WIDTH = 1450
RECT_MAX_HEIGHT = 820

# --- file layout ----------------------------------------------------------
FSS_LINES_SINGLE_PROBE = 23
FSS_LINES_BIPLANE = 26

FSS_LINE_NAMES = {
    1: "VERSION",
    2: "ID_ECHO",
    3: "ID_PROBE",
    4: "PROBE_TYPE",
    5: "KIT_NEEDLE_GUIDE",
    6: "VIDEO_INPUT",
    7: "VIDEO_INPUT_SIZE_X",
    8: "VIDEO_INPUT_SIZE_Y",
    9: "VIDEO_X_SIZE",
    10: "VIDEO_Y_SIZE",
    11: "RECT_ECHO",
    12: "GROUP_ORIENTATION",
    13: "RECT_NAME_ECHO",
    14: "RECT_NAME_PROBE",
    15: "PROIBITED_SCREEN",
    16: "RECT_ORIENTATION",
    17: "RECT_DEPTH",
    18: "VECT_DEPTH",
    19: "PIXEL_RATIO_X",
    20: "PIXEL_RATIO_Y",
    21: "SCALE_LINE",
    22: "CENTRE_DISTANCE",
    23: "ANGLE",
    24: "RECT_TRANS",
    25: "ID_NEXT_PROBE",
    26: "BIPLANA_RECOGNITION_MODE",
}
