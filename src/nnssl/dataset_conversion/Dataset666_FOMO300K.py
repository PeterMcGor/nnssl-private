import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import save_json
from tqdm import tqdm

from nnssl.data.raw_dataset import Dataset, Image, Session, Subject, AssociatedMasks, Collection
from nnssl.paths import nnssl_raw

sep = os.path.sep


# fmt: off
_GROUP_MAPPING = {
    # ── Control / Healthy ──────────────────────────────────────────────────
    'Control':                   'Control',
    'CONTROL':                   'Control',
    'control':                   'Control',
    'CN':                        'Control',
    'HC':                        'Control',
    'normal':                    'Control',
    'Normal':                    'Control',
    'NeuroTypical':              'Control',
    'Typically Developing':      'Control',
    'Typically developing':      'Control',
    'healthy control':           'Control',
    'Health control':            'Control',
    'Never-Depressed Control':   'Control',
    'nondemented':               'Control',
    'Nondemented':               'Control',
    'nh':                        'Control',
    'normalweight':              'Control',
    'LNCG Non-Substance User (used Marijuana less than 4x/past year)':  'Control',
    'LNCG Substance User (used Marijuana once per month or more)':       'Control',

    # ── Dementia ───────────────────────────────────────────────────────────
    'Dementia':          'Dementia',
    'AD':                'Dementia',
    'FTD':               'Dementia',
    'Demented':          'Dementia',
    'Converted':         'Dementia',
    'very_mild_dementia':'Dementia',
    'mild_dementia':     'Dementia',
    'moderate_dementia': 'Dementia',
    'Memory complaints': 'Dementia',

    # ── Parkinson ──────────────────────────────────────────────────────────
    'Parkinson':                                     'Parkinson',
    'PD':                                            'Parkinson',
    "Parkinson's disease":                           'Parkinson',
    "Parkinson's disease - mild cognitive impairment":'Parkinson',
    "Parkinson's disease - normal cognition":        'Parkinson',
    "Parkinson's - no/mild hyposmia":                'Parkinson',
    "Parkinson's - severe hyposmia":                 'Parkinson',

    # ── Movement disorder ──────────────────────────────────────────────────
    'Upper limb dystonia':        'Movement disorder',
    'Cervical dystonia':          'Movement disorder',
    'Spinocerebellar Ataxia Type 2': 'Movement disorder',

    # ── ADHD ───────────────────────────────────────────────────────────────
    'ADHD':                        'ADHD',
    'ADHD-MPH':                    'ADHD',
    'ADHD-LPX':                    'ADHD',
    'ADHD-GFC':                    'ADHD',
    'ADHD-NAIVE':                  'ADHD',
    'ADHD-Inattentive':            'ADHD',
    'ADHD-Combined':               'ADHD',
    'ADHD-Hyperactive/Impulsive':  'ADHD',
    'Attention deficit hyperactivity disorder': 'ADHD',
    'ADHD Non-Substance User (used Marijuana less than 4x/past year)':  'ADHD',
    'ADHD Substance User (used Marijuana once per month or more)':       'ADHD',
    'Disruptive mood dysregulation disorder or attention-deficit hyperactivity disorder': 'ADHD',
    'Anxiety disorders, disruptive mood dysregulation disorder, or Attention deficit hyperactivity disorder': 'ADHD',

    # ── Brain tumor ────────────────────────────────────────────────────────
    'Brain Tumor':                  'Brain tumor',
    'Brain tumor':                  'Brain tumor',
    'Tumor':                        'Brain tumor',
    'GMB':                          'Brain tumor',  # GBM typo
    'astrocytoma':                  'Brain tumor',
    'Anaplastic astrocytoma':       'Brain tumor',
    'Anaplastic astrocytoma II-III':'Brain tumor',
    'Anaplastic astrocytoma III':   'Brain tumor',
    'pylocytic astrocytom':         'Brain tumor',  # pilocytic astrocytoma typo
    'oligodendroglioma':            'Brain tumor',
    'Oligodendroglioma':            'Brain tumor',
    'Oligodendroglioma II':         'Brain tumor',
    'Anaplastic oligodendroglioma': 'Brain tumor',
    'Oligoastrocytoma':             'Brain tumor',
    'Oligo-astrocytoma II':         'Brain tumor',
    'Mixed oligoastrocytoma':       'Brain tumor',
    'Anaplastic mixed oligoastrocytoma': 'Brain tumor',
    'Anaplastic oligoastrocytoma':  'Brain tumor',
    'glioblastoma':                 'Brain tumor',
    'Glioblastoma':                 'Brain tumor',
    'Anaplastic glioma':            'Brain tumor',
    'Low-grade diffuse glioma':     'Brain tumor',
    'Glioma II':                    'Brain tumor',
    'Glioma WHO II':                'Brain tumor',
    'Glioma WHO III':               'Brain tumor',
    'Glioma WHO IV':                'Brain tumor',
    'Meningioma':                   'Brain tumor',
    'Meningioma I':                 'Brain tumor',
    'Meningioma II':                'Brain tumor',
    'Meningioma WHO I':             'Brain tumor',
    'Ependymoma':                   'Brain tumor',
    'Ependymoma II':                'Brain tumor',
    'Ependymom':                    'Brain tumor',  # typo
    'Low-grade glioneuronal tumor': 'Brain tumor',
    'Ganglioglioma':                'Brain tumor',
    'DNET':                         'Brain tumor',
    'neuroectodermal tumor':        'Brain tumor',
    'medulloblastoma':              'Brain tumor',
    'Pons Gliom':                   'Brain tumor',  # Pons Glioma typo
    'Metastatic lung carcinoma':    'Brain tumor',
    'Diffuse large B-cell lymphoma':'Brain tumor',
    'Pituitary adenomas':           'Brain tumor',

    # ── Epilepsy ───────────────────────────────────────────────────────────
    'Epilepsy':                      'Epilepsy',
    'Temporal or parietal lobe epilepsy': 'Epilepsy',
    'Temporal lobe epilepsy':        'Epilepsy',
    'Epilepsy surgery':              'Epilepsy',
    'focal cortical dysplasia':      'Epilepsy',
    'Focal cortical dysplasia':      'Epilepsy',
    'Intractable epilepsy':          'Epilepsy',
    'Drug-resistant epilepsy':       'Epilepsy',
    'Medication-resistant epilepsy': 'Epilepsy',
    'Medically refractory epilepsy': 'Epilepsy',

    # ── Depression ─────────────────────────────────────────────────────────
    'Depression - no treatment':                    'Depression',
    'Depression - Cognitive behavioral therapy':    'Depression',
    'Depression - rt-fmri neurofeedback treatment': 'Depression',
    'Mild or moderate depression':                  'Depression',
    'Major depressive disorder':                    'Depression',
    'Major Depressive Disorder':                    'Depression',
    'Persistent depressive disorder':               'Depression',
    'Mild depression':                              'Depression',

    # ── Schizophrenia / Psychosis ──────────────────────────────────────────
    'SCHZ':                                                    'Schizophrenia',
    'Schizophrenia/Schizoaffective':                           'Schizophrenia',
    'First-episode schizophrenia':                             'Schizophrenia',
    'First-episode affective psychosis':                       'Schizophrenia',
    'Schizophrenia without current auditory hallucinations':   'Schizophrenia',
    'Schizophrenia with current auditory hallucinations':      'Schizophrenia',
    'Clinical high risk for psychosis':                        'Schizophrenia',

    # ── Bipolar ────────────────────────────────────────────────────────────
    'BIPOLAR': 'Bipolar',
    'Bipolar':  'Bipolar',
    'BP':       'Bipolar',

    # ── Multiple sclerosis ─────────────────────────────────────────────────
    'MS':                'Multiple sclerosis',
    'Multiple sclerosis':'Multiple sclerosis',

    # ── Autism ─────────────────────────────────────────────────────────────
    'Autism Spectrum Disorder': 'Autism',

    # ── TBI ────────────────────────────────────────────────────────────────
    'TBI':                  'TBI',
    'Traumatic brain injury':'TBI',

    # ── Stroke / Vascular ──────────────────────────────────────────────────
    'Stroke':                           'Stroke',
    'Infarct':                          'Stroke',
    'infarct':                          'Stroke',
    'Premature Infarct':                'Stroke',
    'premature infarct':                'Stroke',
    'Cortical cerebral infarction':     'Stroke',
    'Perinatal stroke':                 'Stroke',
    'Intracerebral hemorrhage':         'Stroke',
    'intraventricular hemorrhage':      'Stroke',
    'premature intraventricular hemorrhage': 'Stroke',
    'subdurale hemorrhagy':             'Stroke',   # subdural hemorrhage typo
    'subararachnoidal hemorrhage':      'Stroke',   # subarachnoid hemorrhage typo
    'subdural hygrom':                  'Stroke',   # subdural hygroma typo
    'Intracranial aneurysms':           'Stroke',
    'Brain aneurysm(s)':                'Stroke',

    # ── Neonatal / Perinatal ───────────────────────────────────────────────
    'premature':                     'Neonatal',
    'postoperative':                 'Neonatal',
    'HIE':                           'Neonatal',
    'HIE cerebral oedeme':           'Neonatal',
    'PVL':                           'Neonatal',
    'Premature PVL':                 'Neonatal',
    'Premature PVL hydrocephalus':   'Neonatal',
    'Premature PVL Hydrocephalus':   'Neonatal',
    'premature HIE':                 'Neonatal',
    'premature hydrocephalus':       'Neonatal',
    'premature Polymicrogyria':      'Neonatal',
    'Congenital CMV infection':      'Neonatal',
    'Congenital CMV':                'Neonatal',
    'encephalopathy (biliribun)':    'Neonatal',   # bilirubin encephalopathy typo

    # ── Structural anomaly / Malformation ──────────────────────────────────
    'Hydrocephalus':              'Structural anomaly',
    'VP shunt Hydrocephalus':     'Structural anomaly',
    'cerebral atrophy hydrocephalus': 'Structural anomaly',
    'heterotopia':                'Structural anomaly',
    'Gray matter heterotopia':    'Structural anomaly',
    'polymicroglia':              'Structural anomaly',  # polymicrogyria typo
    'encephalocele':              'Structural anomaly',
    'Meningocele':                'Structural anomaly',
    'Corpus callosum agenesis':   'Structural anomaly',
    'Arnold Chiari Malformation': 'Structural anomaly',
    'Dandy-Walker continuum':     'Structural anomaly',
    'macrocephaly':               'Structural anomaly',
    'Macrocephaly':               'Structural anomaly',
    'Brachycephaly':              'Structural anomaly',
    'Plagiocephaly':              'Structural anomaly',
    'craniosynostosis':           'Structural anomaly',
    'Crouzon syndrome':           'Structural anomaly',
    'Septooptic dysplasy':        'Structural anomaly',  # septo-optic dysplasia typo
    'Sturge-Weber-Syndrom':       'Structural anomaly',
    'NF1':                        'Structural anomaly',
    'NF 1':                       'Structural anomaly',  # typo
    'Fragile X syndrome':         'Structural anomaly',
    'Arachnoidal cyst':           'Structural anomaly',

    # ── Other neurological ─────────────────────────────────────────────────
    'Neurological condition(s)':              'Other neurological',
    'Motor neuron disease':                   'Other neurological',
    'Polyneuropathy':                         'Other neurological',
    'Leigh Syndrome':                         'Other neurological',
    'Mithocondriopathy':                      'Other neurological',  # mitochondriopathy typo
    'Meningitis':                             'Other neurological',
    'Brain abscess':                          'Other neurological',
    'cerebral atrophy':                       'Other neurological',
    'cerebellar atrophy':                     'Other neurological',
    'cerebral/cerebellar atrophy':            'Other neurological',
    'Delayed myelinisation cerebral atrophy': 'Other neurological',
    'Gliotic foci':                           'Other neurological',
    'melanin deposition':                     'Other neurological',
    'brain oedem':                            'Other neurological',
    'cerebellar calcification':               'Other neurological',
    'encephalomalasy':                        'Other neurological',  # encephalomalacia typo
    'searing injuries':                       'Other neurological',  # shearing injuries typo
    'cranial fasciitis':                      'Other neurological',
    'No-Apparent-olfactory-bulb':             'Other neurological',
    'Synaesthesia':                           'Other neurological',
    'Hearing loss':                           'Other neurological',

    # ── Other psychiatric ──────────────────────────────────────────────────
    'Borderline Personality Disorder':            'Other psychiatric',
    'Subclinical signs of emotional dysregulation':'Other psychiatric',
    'Obsessive compulsive disorder':              'Other psychiatric',
    'Psychiatric symptoms':                       'Other psychiatric',
    'Psychiatric illness':                        'Other psychiatric',

    # ── Substance use ──────────────────────────────────────────────────────
    'Cocaine use disorder':           'Substance use',
    'Cocaine use disorder - Sham':    'Substance use',
    'Cocaine use disorder - Treatment':'Substance use',
    'Heavy cannabis user':            'Substance use',

    # ── Learning / Developmental ────────────────────────────────────────────
    'Dyslexia':                    'Learning disability',
    'dyslexia':                    'Learning disability',
    'Spelling Disorder':           'Learning disability',
    'Developmental prosopagnosic': 'Learning disability',
    'Developmental prosopagnosics':'Learning disability',  # plural typo

    # ── Other ──────────────────────────────────────────────────────────────
    'Fibromyalgia':    'Other',
    'Osteoarthritis':  'Other',
    'Unilateral Glaucoma': 'Other',
    'overweight':      'Other',
    't1d':             'Other',   # Type 1 Diabetes

    # ── Unknown (scan artifacts, uninterpretable tokens) ───────────────────
    'Motion artefact': 'Unknown',
}
# fmt: on


def _clean_group(value):
    if pd.isna(value):
        return np.nan
    # normalise curly apostrophes so both variants hit the same key
    v = str(value).strip().replace('’', "'").replace('‘', "'")
    return _GROUP_MAPPING.get(v, 'Unknown')


def _parse_age_sex(value):
    """
    Returns (age_assumed, sex_assumed) for a raw 'age' field value.
    Handles: numerics, ranges (mean), comma decimals, 'y' suffix,
             '+' suffix, sex mislabels, and known invalid tokens.
    """
    if pd.isna(value):
        return np.nan, np.nan

    v = str(value).strip()

    # Sex mislabeled in age column
    if v.upper() in ('M', 'F'):
        return np.nan, v.upper()

    # Known non-parseable tokens
    if v in ('#', 'NoPhenotypicData', 'A', ''):
        return np.nan, np.nan

    # Qualitative — no reliable numeric mapping
    if v in ('Young', 'Old'):
        return np.nan, np.nan

    # Strip trailing 'y' (e.g. '38y' → '38')
    if v.endswith('y'):
        v = v[:-1]

    # Strip trailing '+' (e.g. '89+' → '89', '36+' → '36')
    if v.endswith('+'):
        v = v[:-1]

    # Comma as decimal separator (e.g. '39,5' → '39.5')
    if ',' in v:
        v = v.replace(',', '.')

    # Range (e.g. '18-30') — '-' only between digits, not a negative sign
    range_match = re.match(r'^(\d+\.?\d*)-(\d+\.?\d*)$', v)
    if range_match:
        low, high = float(range_match.group(1)), float(range_match.group(2))
        return (low + high) / 2, np.nan

    # Direct numeric conversion
    try:
        age = float(v)
        return age if age > 0 else np.nan, np.nan
    except ValueError:
        return np.nan, np.nan


# --------------------------------------------------------------------------
# MRI modality parsing
# --------------------------------------------------------------------------

# fmt: off
# (regex, modality) — matched against lowercased filename; first match wins.
# More-specific BIDS suffixes must come before shorter overlapping ones.
_FILENAME_MODALITY_RULES: list[tuple[str, str]] = [
    (r'_T2starw',   'T2star'),   # before _T2w
    (r'_T1map',     'T1map'),    # before _T1w
    (r'_T2map',     'T2map'),    # before _T2w
    (r'_T1w',       'T1'),
    (r'_T2w',       'T2'),
    (r'_FLAIR',     'FLAIR'),
    (r'_SWI',       'SWI'),
    (r'_PDw',       'PD'),
    (r'_dwi',       'DWI'),
    (r'_adc',       'ADC'),
    (r'_bold',      'fMRI'),
    (r'_angio',     'MRA'),
    (r'/dwi/',      'DWI'),      # directory fallback
    (r'/func/',     'fMRI'),
    (r'/perf/',     'ASL'),
    (r'/fmap/',     'fieldmap'),
]

# (regex, modality) — matched against lowercased, stripped SeriesDescription.
# Order matters: most specific / easily confused entries come first.
_SERIES_MODALITY_RULES: list[tuple[str, str]] = [
    # Non-MRI / calibration
    (r'\bpet\b',                                                          'PET'),
    (r'localizer|localiser|\baahscout\b|\bscout\b',                       'localizer'),
    (r'\bb1[\s_-]?map\b',                                                 'fieldmap'),
    # Perfusion
    (r'\basl\b|pcasl|\bcasl\b|\bcbf\b|cerebral[\s_-]blood[\s_-]flow',   'ASL'),
    # Diffusion-derived map (before DWI so ADC isn't swallowed by the DWI rule)
    (r'\badc\b|apparent[\s_-]diffusion',                                  'ADC'),
    # Diffusion (DTI, DKI, NODDI, DSI, CHARMED, ep2d_diff, mddw, …)
    (r'\bdwi\b|\bdti\b|\bdki\b|diffusion|\bnoddi\b|\bdsi\b'
     r'|ep2d.diff|mddw|cmrr.*diff|charmed|\bhardi\b',                    'DWI'),
    # MRA / TOF
    (r'\btof\b|\bmra\b|time[\s_-]of[\s_-]flight',                        'MRA'),
    # SWI / T2* / susceptibility (SWAN = Siemens T2* sequence)
    (r'\bswi\b|\bswan\b|t2star|t2\*|t2s[\s_]|aspire.*gre|\bqsm\b'
     r'|susceptib',                                                       'T2star'),
    # Magnetisation transfer
    (r'\bmt[\s_-]?(on|off)\b',                                            'MT'),
    # FLAIR — before T2 (FLAIR is T2-weighted but distinct)
    (r'\bflair\b|dark[\s_-]?fluid|\btirm\b',                             'FLAIR'),
    # Quantitative maps (before T1/T2 so "T1 map" isn't labelled plain T1)
    (r't2[\s_-]?map\b',                                                   'T2map'),
    (r't1[\s_-]?map\b|mp2rage.*t1.images|t1_images(?!.*uni)',             'T1map'),
    # T2
    (r'\bt2w?\b|t2[\s_-]space|t2[\s_-]spc|t2[\s_-]tse',                'T2'),
    # T1 (MPRAGE / MP-RAGE / MP2RAGE / BRAVO / SPGR / FSPGR / TFE / MEMPRAGE)
    (r'\bt1w?\b|mprage|mp[\s_-]?rage|mp2rage|\bbravo\b'
     r'|\bspgr\b|\bfspgr\b|\btfe\b|memprage',                            'T1'),
    # PD
    (r'\bpdw?\b|proton[\s_-]density',                                     'PD'),
    # UTE (ultra-short TE, used for MR-based attenuation correction in PET/MRI)
    (r'\bute\b',                                                          'UTE'),
]
# fmt: on

# Contrast-enhancement detection — case-insensitive; applied to raw SeriesDescription.
_CONTRAST_RE = re.compile(
    r'\+\s*[cC]\b'                        # +C  /  + C
    r'|[cC]\s*\+'                          # C+
    r'|post[\s_-]?contrast'               # post contrast / post_contrast
    r'|\bpost\b\s*$'                       # ends with POST (e.g. "SAG T1 3D TFE POST")
    r'|\bgad\b|\bgd\b'                     # Gd / gadolinium abbreviations
    r'|gadolinium'
    r'|volumen[\s_]?g[ad]'                 # Spanish: "volumen gd/ga"
    r'|contrast[\s_-]enhanced|enhanced[\s_-]contrast'
    r'|\bce\b(?!nter|rebr|ll|ramic|sium)', # CE but not "center/cerebr/cell/ceramic/cesium"
    re.IGNORECASE,
)


def _parse_modality(filename, series_description) -> tuple[str, bool]:
    """
    Returns (modality_assumed, contrast_assumed).
    Uses BIDS filename suffix as primary source; falls back to SeriesDescription.
    """
    modality = None

    # 1 — filename (BIDS suffix; most reliable)
    if pd.notna(filename):
        fn = str(filename)
        for pattern, mod in _FILENAME_MODALITY_RULES:
            if re.search(pattern, fn, re.IGNORECASE):
                modality = mod
                break

    # 2 — SeriesDescription fallback
    if modality is None and pd.notna(series_description):
        sd = str(series_description).lower().strip()
        for pattern, mod in _SERIES_MODALITY_RULES:
            if re.search(pattern, sd):
                modality = mod
                break

    # 3 — contrast enhancement
    contrast = False
    if pd.notna(series_description):
        contrast = bool(_CONTRAST_RE.search(str(series_description)))

    return modality or 'Unknown', contrast


# --------------------------------------------------------------------------
# MRI hardware / acquisition metadata cleaning
# --------------------------------------------------------------------------

def _parse_field_strength(value) -> float:
    """
    Returns MagneticFieldStrength in Tesla as a float.
    Strips trailing 'T', converts from Gauss if value > 100
    (e.g. 15000 Gauss → 1.5 T), and rejects physically impossible values.
    """
    if pd.isna(value):
        return np.nan
    v = str(value).strip().upper().rstrip('T').strip()
    try:
        f = float(v)
    except ValueError:
        return np.nan
    if f > 100:          # assume Gauss — 1 T = 10 000 G
        f /= 10_000.0
    if not (0 < f <= 25):
        return np.nan
    return round(f, 6)


# fmt: off
# Exact lowercase key → canonical manufacturer name
_MANUFACTURER_NORMALISE: dict[str, str] = {
    'siemens':                     'Siemens',
    'siemens healthineers':        'Siemens',
    'siemens healthcare gmbh':     'Siemens',
    'ge':                          'GE',
    'ge medical systems':          'GE',
    'ge medical':                  'GE',
    'general electrics':           'GE',
    'philips':                     'Philips',
    'philips medical systems':     'Philips',
    'philips healthcare':          'Philips',
    'toshiba':                     'Canon',   # Toshiba Medical → Canon 2016
    'toshiba_mec':                 'Canon',
    'canon_mec':                   'Canon',
    'hitachi':                     'Hitachi',
    'hitachi medical corporation': 'Hitachi',
    'brucker':                     'Bruker',  # common typo in DICOM headers
    'mediso':                      'Mediso',
    'ningbo xingaoyi':             'Ningbo Xingaoyi',
    'fuji film co., ltd.':         'Fujifilm',
}

# (regex on lowercased model name, manufacturer) — first match wins
_MODEL_MANUFACTURER_RULES: list[tuple[str, str]] = [
    (r'magnetom|prisma|skyra|avanto|aera|verio|terra\b|vida\b|sola\b'
     r'|espree|symphony|sonata|allegra|biograph|connectom|hrrt'
     r'|investigational_device',                                 'Siemens'),
    (r'signa|discovery\s*mr|optima\s*mr|genesis_signa',          'GE'),
    (r'achieva|ingenia|prodiva|intera',                          'Philips'),
    (r'vantage\b|orian\b|titan\b|altaire',                       'Canon'),
    (r'airis|oasis\b',                                           'Hitachi'),
    (r'medspec',                                                 'Bruker'),
]
# fmt: on


def _clean_manufacturer(manufacturer, model_name) -> str:
    """Returns canonical manufacturer name, falling back to model-name inference."""
    if pd.notna(manufacturer):
        key = str(manufacturer).strip().lower()
        if key in _MANUFACTURER_NORMALISE:
            return _MANUFACTURER_NORMALISE[key]

    # Infer from model name when manufacturer is missing or unrecognised
    if pd.notna(model_name):
        mn = str(model_name).lower()
        for pattern, mfr in _MODEL_MANUFACTURER_RULES:
            if re.search(pattern, mn):
                return mfr

    # Return cleaned original rather than 'Unknown' so no info is silently lost
    if pd.notna(manufacturer):
        return str(manufacturer).strip()
    return np.nan


def _parse_software_platform(version) -> str:
    """
    Extracts a short vendor-platform token from SoftwareVersions.
      Siemens syngo: syngo_B17, syngo_XA31, syngo_VG70A
      GE LX release: GE_DV26.0, GE_HD16.0, GE_15.0
      Philips R-stream: Philips_5.3.1, Philips_12.1.5
    """
    if pd.isna(version):
        return np.nan
    v = str(version).strip()

    # Siemens syngo platform code: "syngo MR B17", "syngo_MR_XA31", "syngo MR E11"
    m = re.search(r'syngo[\s_]MR[\s_]?([A-Z]{1,2}\d+[A-Za-z]?)\b', v, re.IGNORECASE)
    if m:
        return f'syngo_{m.group(1).upper()}'

    # Siemens standalone VxNN codes: VB17, VG70A, VG62B
    m = re.fullmatch(r'V([A-Z]\d{2,3}[A-Z]?)', v, re.IGNORECASE)
    if m:
        return f'syngo_V{m.group(1).upper()}'

    # GE LX software release: "27\LX\MR Software release:DV26.0_R01_1725.a"
    m = re.search(r'[Ss]oftware[\s_-]release:([A-Z0-9]+[\d.]+)', v)
    if m:
        return f'GE_{m.group(1).split("_")[0]}'

    # GE 7T: "29\LX\7T29.1_R01_2224.a"
    m = re.search(r'[/\\]7[Tt](\d+\.\d+)', v)
    if m:
        return f'GE_7T{m.group(1)}'

    # Philips R-stream: "5.3.1\5.3.1.0\Gyroscan..." or bare "5.3.1"
    m = re.match(r'^(\d+\.\d+\.?\d*)(?:[_\\./]|$)', v)
    if m:
        return f'Philips_{m.group(1)}'

    # PET/nuclear medicine scanners
    if re.search(r'ecat|pet_columbia|PET_CT', v, re.IGNORECASE):
        return 'PET_software'

    # GE systems sometimes store a build timestamp as version
    if re.search(r'\w{3}\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}', v):
        return 'GE_build_date'

    return 'Unknown'


_MV_SPLIT_RE = re.compile(r'[\\\s,/]+')


def _normalise_multivalue(value) -> str:
    """
    Normalises a DICOM multi-value field (ScanningSequence, SequenceVariant).
    Splits on \\, comma, space, slash; deduplicates; sorts; joins with '_'.
    e.g.  'GR\\IR' / 'IR_GR' / 'IR, GR' → 'GR_IR'
    """
    if pd.isna(value):
        return np.nan
    v = str(value).strip()
    # Handle Python list repr: "['GR', 'IR']"
    if v.startswith('['):
        v = re.sub(r"[\[\]'\"]", '', v)
    parts = [p.strip() for p in _MV_SPLIT_RE.split(v) if p.strip()]
    parts = sorted({p.upper() for p in parts if p})
    return '_'.join(parts) if parts else np.nan


_FAT_SAT_RE  = re.compile(r'\bFS\b|T[12]FLAIR_GEMS|FS_GEMS|FAT[\s_-]?SAT', re.IGNORECASE)
_PART_FT_RE  = re.compile(r'\bPF[FP]\b',                                       re.IGNORECASE)


def _extract_scan_flags(scan_options) -> tuple[bool, bool]:
    """
    Returns (fat_saturation, partial_fourier) booleans from ScanOptions.
    Works across Siemens (IR/PFF), GE GEMS long-form, and Philips strings.
    """
    if pd.isna(scan_options):
        return False, False
    s = str(scan_options)
    return bool(_FAT_SAT_RE.search(s)), bool(_PART_FT_RE.search(s))


def _create_pretrain_json(fomo300k_root_dir: Path):
    # path to OpenMind metadata file
    fomo300k_dir = fomo300k_root_dir 
    mapping_csv_data = fomo300k_root_dir / "mapping.tsv"
    mri_info_csv_data = fomo300k_root_dir / "mri_info.tsv"
    participants_info_csv_data = fomo300k_root_dir / "participants.tsv"

    mapping_csv = pd.read_csv(mapping_csv_data, sep="\t")
    mri_info_csv = pd.read_csv(mri_info_csv_data, sep="\t")
    mri_info_csv['new_filename'] = mri_info_csv['filename'].apply(lambda x: os.path.basename(x))

    # ── Modality ──────────────────────────────────────────────────────────
    parsed_mod = mri_info_csv.apply(
        lambda r: _parse_modality(r['new_filename'], r.get('SeriesDescription')), axis=1
    )
    mri_info_csv['modality_assumed'] = [x[0] for x in parsed_mod]
    mri_info_csv['contrast_assumed'] = [x[1] for x in parsed_mod]

    # ── Magnetic field strength ────────────────────────────────────────────
    mri_info_csv['field_strength_assumed'] = mri_info_csv['MagneticFieldStrength'].apply(
        _parse_field_strength
    )

    # ── Manufacturer (with model-name fallback for missing rows) ───────────
    mri_info_csv['manufacturer_assumed'] = mri_info_csv.apply(
        lambda r: _clean_manufacturer(r.get('Manufacturer'), r.get('ManufacturersModelName')),
        axis=1,
    )

    # ── Software platform ──────────────────────────────────────────────────
    mri_info_csv['software_platform'] = mri_info_csv['SoftwareVersions'].apply(
        _parse_software_platform
    )

    # ── Sequence / variant normalisation ──────────────────────────────────
    mri_info_csv['scanning_sequence_norm'] = mri_info_csv['ScanningSequence'].apply(
        _normalise_multivalue
    )
    mri_info_csv['sequence_variant_norm'] = mri_info_csv['SequenceVariant'].apply(
        _normalise_multivalue
    )

    # ── Scan-option flags ──────────────────────────────────────────────────
    scan_flags = mri_info_csv['ScanOptions'].apply(_extract_scan_flags)
    mri_info_csv['fat_saturation']  = [x[0] for x in scan_flags]
    mri_info_csv['partial_fourier'] = [x[1] for x in scan_flags]

    # ── Diagnostics ───────────────────────────────────────────────────────
    print("MRI modality distribution:\n",     mri_info_csv['modality_assumed'].value_counts(dropna=False))
    print("Contrast-enhanced scans:",          mri_info_csv['contrast_assumed'].sum(),
          f"({mri_info_csv['contrast_assumed'].mean():.1%})")
    print("Field-strength distribution:\n",   mri_info_csv['field_strength_assumed'].value_counts(dropna=False))
    print("Manufacturer distribution:\n",     mri_info_csv['manufacturer_assumed'].value_counts(dropna=False))
    print("Software platform distribution:\n",mri_info_csv['software_platform'].value_counts(dropna=False))
    print("Fat saturation scans:",             mri_info_csv['fat_saturation'].sum())
    print("Partial Fourier scans:",            mri_info_csv['partial_fourier'].sum())

    # Add mri_info when dataset, participant_id and session_id match between mapping_csv and mri_info_csv
    mapping_csv = mapping_csv.merge(mri_info_csv, left_on=["dataset", "participant_id", "session_id", "new_filename"], right_on=["dataset", "participant_id", "session_id","new_filename"], how="left")
    print(f"Number of rows and columns in mapping_csv after merge: {mapping_csv.shape}")
    print(mapping_csv.keys())
    mapping_data = mapping_csv.to_dict(orient="records")

    participants_info_csv = pd.read_csv(participants_info_csv_data, sep="\t")
    parsed = participants_info_csv['age'].apply(_parse_age_sex)
    participants_info_csv['age_assumed'] = [x[0] for x in parsed]
    age_sex_from_age_col = pd.array([x[1] for x in parsed], dtype="string")

    # sex_assumed: start from the existing sex column, fill gaps with values recovered from age column
    participants_info_csv['sex_assumed'] = participants_info_csv['sex'].str.strip().str.upper()
    mask = participants_info_csv['sex_assumed'].isna() & pd.notna(age_sex_from_age_col)
    participants_info_csv.loc[mask, 'sex_assumed'] = age_sex_from_age_col[mask]

    # Sanity cap — ages above 130 are almost certainly errors
    participants_info_csv.loc[participants_info_csv['age_assumed'] > 130, 'age_assumed'] = np.nan
    participants_info_csv['group_assumed'] = participants_info_csv['group'].apply(_clean_group)

    # ── Cross-session value recovery & longitudinal detection ─────────────
    # Each (dataset, participant_id) pair can have multiple session rows.
    # A subject is longitudinal if age_assumed takes more than one distinct
    # value across their sessions (ignoring NaN).
    grp = participants_info_csv.groupby(['dataset', 'participant_id'], sort=False)

    participants_info_csv['Longitudinal'] = grp['age_assumed'].transform(
        lambda ages: ages.dropna().nunique() > 1
    )

    # We assume that sex and group don't change over time → fill from other sessions
    for col in ('sex_assumed', 'group_assumed'):
        participants_info_csv[col] = grp[col].transform(lambda x: x.ffill().bfill())

    # Age fill only for non-longitudinal subjects (all sessions share the same age)
    non_long = ~participants_info_csv['Longitudinal']
    participants_info_csv.loc[non_long, 'age_assumed'] = (
        participants_info_csv.loc[non_long]
        .groupby(['dataset', 'participant_id'])['age_assumed']
        .transform(lambda x: x.ffill().bfill())
    )

    n_subjects    = grp.ngroups
    n_longitudinal = participants_info_csv.groupby(['dataset', 'participant_id'])['Longitudinal'].first().sum()
    print(f"Unique subjects: {n_subjects}  |  Longitudinal: {n_longitudinal}")
    print("Participants mean age (after cross-session fill):", participants_info_csv['age_assumed'].mean())
    print("Participants sex distribution:\n", participants_info_csv['sex_assumed'].value_counts(dropna=False))
    print("Participants group distribution:\n", participants_info_csv['group_assumed'].value_counts(dropna=False))
    for col in ('age_assumed', 'sex_assumed', 'group_assumed'):
        n_missing_sessions  = participants_info_csv[col].isna().sum()
        n_missing_subjects  = (
            participants_info_csv.groupby(['dataset', 'participant_id'])[col]
            .apply(lambda x: x.isna().all())
            .sum()
        )
        print(f"Missing {col}: {n_missing_sessions} sessions | {n_missing_subjects} subjects (all sessions missing)")

    mapping_csv = pd.merge(mapping_csv, participants_info_csv[["dataset", "participant_id", "session_id", 'age','age_assumed', 'sex', 'sex_assumed', 'group', 'group_assumed']], on=["dataset", "participant_id", "session_id"], how="left")
    print(f"Number of rows and columns in mapping_csv after second merge: {mapping_csv.shape}")
    print(mapping_csv.keys())
    # Brain age estimation just on T1's
    brain_age_estimation = mapping_csv[mapping_csv.modality_assumed == 'T1']
    print("Assert brain age estimation modality", brain_age_estimation.modality_assumed.unique())
    brain_age_estimation = brain_age_estimation[['dataset', 'new_path', 'participant_id', 'session_id', 'age','age_assumed', 'sex', 'sex_assumed', 'group', 'group_assumed']]
    brain_age_estimation['Path'] = '/opt/datasets/FOMO300K/' + brain_age_estimation['dataset']+os.sep+ brain_age_estimation['new_path']
    print(brain_age_estimation.head())
    print('Mean Age', brain_age_estimation.age_assumed.median(), 'Missing', brain_age_estimation.age_assumed.isna().sum())
    brain_age_estimation.loc[brain_age_estimation.age_assumed.isna(), 'age_assumed'] = 35
    print('Median Age after filling missing values with median',brain_age_estimation.age_assumed.median(), 'Missing after filling', brain_age_estimation.age_assumed.isna().sum())
    brain_age_estimation['Age'] = brain_age_estimation['age_assumed']
    brain_age_estimation.to_csv('FOMO300K_brain_age_estimation.csv')



    collection = Collection(collection_name="Dataset666_FOMO300K", collection_index=666)

    subject_info_keys = ["age", "sex", "handedness", "race", "weight", "bmi", "health_status"]
    image_info_keys = [
        "derived_from",
        "is_brain_extract",
        "manufacturer",
        "model_name",
        "phase_encoding_direction",
        "magnetic_field_strength",
        "repetition_time",
        "echo_time",
    ]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fomo300k_root_dir",
        type=Path,
        default=Path("/opt/datasets/FOMO300K"),
        help="Path to the root directory of the FOMO300K dataset download directory. "
             "If you downloaded the dataset from hugginface, you should point to the parent directory of the `mapping.tsv` file.",
    )
    args = parser.parse_args()
    _create_pretrain_json(args.fomo300k_root_dir)


if __name__ == "__main__":
    main()
