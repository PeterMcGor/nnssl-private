#!/usr/bin/env python3
"""
Script to add SynthSeg volumes and QC results to the preprocessed dataset JSON file.
Adds information at both image level (image_info) and subject level (subject_info),
keeping only the best SynthSeg result per (session, modality) based on a simple score.
"""

import json
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, Optional
import re
import glob
import math
import re

# ----------------------------
# Preferences & scoring utils
# ----------------------------

PREFERRED_MODALITIES = ['T1w', 'T2w', 'MPRAGE', 'MP2RAGE', 'FLAIR', 'inplaneT2']

def _is_num(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))

def _get_icv(vols: Dict[str, float]) -> float | None:
    # Common ICV keys
    for k in vols.keys():
        if k.lower().strip() in ("total intracranial", "intracranial volume", "icv"):
            v = vols[k]
            return float(v) if _is_num(v) else None
    return None

def _count_nans(d: Dict[str, float]) -> int:
    return sum(1 for v in d.values() if isinstance(v, float) and math.isnan(v))

def _count_negatives(d: Dict[str, float]) -> int:
    return sum(1 for v in d.values() if _is_num(v) and v < 0)

def _count_near_zero(d: Dict[str, float], eps: float = 1.0) -> int:
    # 1 mm³ threshold to ignore exact zeros/tiny scraps from parcellations
    return sum(1 for v in d.values() if _is_num(v) and abs(float(v)) <= eps)

def _qc_mean(qc: Dict[str, float] | None) -> float:
    if not qc:
        return 0.0
    vals = [float(v) for v in qc.values() if _is_num(v)]
    return sum(vals) / len(vals) if vals else 0.0

def modality_rank(mod: str | None) -> int:
    if not mod:
        return 10_000
    try:
        return PREFERRED_MODALITIES.index(next(m for m in PREFERRED_MODALITIES if m.lower() == mod.lower()))
    except (StopIteration, ValueError):
        return 5_000

def score_synthseg(vols: Dict[str, float] | None, qc: Dict[str, float] | None) -> float:
    """
    Anatomy-aware score; higher is better. (No asymmetry to keep runtime low and avoid fragility.)
    """
    if not vols:
        return -float('inf')

    score = 100.0
    # 1) hard failures
    score -= 8.0 * _count_nans(vols)
    score -= 6.0 * _count_negatives(vols)

    # 2) near zeros (light)
    nz = _count_near_zero(vols, eps=1.0)
    score -= 0.25 * nz

    # 3) ICV plausibility (broad adult range: 0.9–2.6 million mm³)
    icv = _get_icv(vols)
    if icv is None:
        score -= 10.0  # no ICV at all is suspicious
    else:
        if icv < 900_000:
            score -= min(40.0, (900_000 - icv) / 25_000.0)
        elif icv > 2_600_000:
            score -= min(40.0, (icv - 2_600_000) / 25_000.0)
        else:
            score += 10.0  # in-range bonus

    # 4) QC (0–1); scale to +0..+20
    qm = _qc_mean(qc)
    score += 20.0 * qm

    return score

def find_preferred_modality_in_subject(subject_volumes: Dict[str, Dict[str, Dict[str, float]]],
                                       preference_modality_list: list[str] | None = None
                                      ) -> tuple[str | None, str | None, Dict[str, float] | None]:
    """
    subject_volumes: {session -> {modality -> volumes_dict}}
    Returns (session_key, modality_key, volumes_dict).
    """
    if not subject_volumes:
        return None, None, None

    if preference_modality_list is None:
        preference_modality_list = PREFERRED_MODALITIES

    pref_lower = [m.lower() for m in preference_modality_list]

    # First pass: try preferences (case-insensitive), deterministic session/modality order
    for session_key in sorted(subject_volumes.keys()):
        session_data = subject_volumes[session_key] or {}
        mapping = {mod.lower(): mod for mod in session_data.keys()}
        for pref in pref_lower:
            if pref in mapping:
                actual_mod = mapping[pref]
                vols = session_data.get(actual_mod) or {}
                if vols:
                    return session_key, actual_mod, vols

    # Second pass: any first available with non-empty volumes
    for session_key in sorted(subject_volumes.keys()):
        session_data = subject_volumes[session_key] or {}
        for mod_key in sorted(session_data.keys()):
            vols = session_data[mod_key] or {}
            if vols:
                return session_key, mod_key, vols

    return None, None, None

# ----------------------------
# IO & dataset helpers
# ----------------------------

def read_synthseg_csv(csv_path: str, subject_id: str) -> Optional[Dict[str, float]]:
    """
    Read a SynthSeg CSV file and return the data as a dictionary filtered by subject.
    """
    if not os.path.exists(csv_path) and '/home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset/OpenMind/OpenMind/ds001353/' not in csv_path:
        print(f"Warning: CSV file not found: {csv_path}")
        directory = os.path.dirname(csv_path)
        print('The directory contains', glob.glob(os.path.join(directory, '*.csv')))
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if 'subject' in df.columns:
            df = df[df['subject'] == subject_id]
        elif df.shape[1] == 102 or df.shape[1] == 9:  # volumes or qc
            # Rename the first column header to 'subject' if it has no name
            if df.columns[0] == '' or df.columns[0] == 'Unnamed: 0':
                df.rename(columns={df.columns[0]: 'subject'}, inplace=True)
                df.to_csv(csv_path, index=False)  # persist header fix
                df = df[df['subject'] == subject_id]
        if len(df) == 0:
            return None

        row = df.iloc[0]
        result = {}
        for col in df.columns:
            if col != 'subject':
                result[col] = float(row[col])
        return result
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def get_modality_from_filename(filename: str) -> str:
    """
    Extract modality from filename (e.g., 'sub-01_T1w.nii.gz' -> 't1w')
    """
    base_name = filename.replace('.nii.gz', '').replace('.nii', '')
    modality_patterns = [
        r'_([Tt]1w)(?:_|$)',
        r'_([Tt]2w)(?:_|$)', 
        r'_(inplaneT2)(?:_|$)',
        r'_(FA)(?:_|$)',
        r'_(MD)(?:_|$)',
        r'_(dwi)(?:_|$)',
        r'_([Ff][Ll][Aa][Ii][Rr])(?:_|$)',
        r'_(BOLD)(?:_|$)'
    ]
    for pattern in modality_patterns:
        match = re.search(pattern, base_name)
        if match:
            return match.group(1).lower()
    parts = base_name.split('_')
    if len(parts) > 1:
        return parts[-1].lower()
    return 'unknown'

def find_synthseg_files(base_path: str, dataset_id: str, subject_id: str, session_id: str, modality: str) -> tuple:
    """
    (Unused in current flow) Find the volumes.csv and qc.csv files for a given image.
    """
    modality_folders = {
        't1w': 'anat',
        't2w': 'anat', 
        'inplanet2': 'anat',
        'fa': 'dwi',
        'md': 'dwi',
        'dwi': 'dwi',
        'flair': 'anat',
        'bold': 'func'
    }
    folder = modality_folders.get(modality, 'anat')
    if session_id == 'ses-DEFAULT':
        synthseg_base = os.path.join(base_path, dataset_id, 'derivatives', 'synthseg', subject_id, folder)
    else:
        synthseg_base = os.path.join(base_path, dataset_id, 'derivatives', 'synthseg', subject_id, session_id, folder)
    volumes_path = os.path.join(synthseg_base, 'volumes.csv')
    qc_path = os.path.join(synthseg_base, 'qc.csv')
    return volumes_path, qc_path

# ----------------------------
# Main processing
# ----------------------------

def process_json_file(json_path: str, base_data_path: str, output_path: str = None, openneuro_csv: str = None):
    """
    Process the JSON file to add SynthSeg results.
    """
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    openneuro_df = pd.read_csv(openneuro_csv)
    errors = []
    
    # Process each dataset
    for dataset_id, dataset_info in data.get('datasets', {}).items():
        # Process each subject
        for subject_id, subject_data in dataset_info.get('subjects', {}).items():

            # Winner store per SUBJECT: (session, modality) -> best payload
            best_per_sm: Dict[tuple, Dict[str, Any]] = {}

            # Process each session
            for session_id, session_data in subject_data.get('sessions', {}).items():
                # Process each image
                for image_idx, image_data in enumerate(session_data.get('images', [])):
                    filename = image_data.get('name', '')
                    modality = image_data.get('modality', '') #get_modality_from_filename(filename)
                    filepath = image_data.get('image_path', '')

                    # Construct unique_id from hardcoded split (as in your current logic)                    
                    # unique_id = filepath.split('nnsslPlans_median/Dataset001_OpenMind/')[1].replace('/',"__")+'.nii.gz'
                    unique_id = filepath.split('nnsslPlans_onemmiso/Dataset001_OpenMind/')[1].replace('/',"__")+'.nii.gz'                    
                    try:
                        look_in = openneuro_df[openneuro_df.unique_id == unique_id]
                        if not look_in.empty:
                            relative_path = look_in['image_path'].values[0]
                        else:
                            relative_path = openneuro_df[openneuro_df.unique_id == unique_id.replace('.gz', '')].image_path.values[0]
                    except IndexError:
                        print(f"Skipping unique_id due to missing data: {unique_id}")
                        continue

                    # Build derivatives/synthseg relative path (skip "__Data" components)
                    filename = relative_path.split('/')[-1]
                    relative_path_fields = relative_path.split('/')
                    relative_path = os.path.join(relative_path_fields[0], "derivatives/synthseg")
                    for i in range(1, len(relative_path_fields) - 1):
                        if "__Data" not in relative_path_fields[i]:
                            relative_path = os.path.join(relative_path, relative_path_fields[i])

                    # Paths to SynthSeg CSVs (+ filename-specific fallbacks)
                    volumes_path = os.path.join(base_data_path, relative_path, 'volumes.csv')
                    if not os.path.exists(volumes_path):
                        volumes_path = (
                            os.path.join(base_data_path, relative_path, f"{filename.replace('.gz', '')}_volumes.csv")
                            if ".gz" in filename else
                            os.path.join(base_data_path, relative_path, f"{filename.replace('.nii', '')}_volumes.csv")
                        )

                    qc_path = os.path.join(base_data_path, relative_path, 'qc.csv')
                    if not os.path.exists(qc_path):
                        qc_path = (
                            os.path.join(base_data_path, relative_path, f"{filename.replace('.gz', '')}_qc.csv")
                            if ".gz" in filename else
                            os.path.join(base_data_path, relative_path, f"{filename.replace('.nii', '')}_qc.csv")
                        )

                    # Read volumes and QC data
                    subject_key = filename.replace('.nii.gz', '')
                    volumes_data = read_synthseg_csv(volumes_path, subject_key)
                    qc_data = read_synthseg_csv(qc_path, subject_key)

                    # Image-level: store what this image has (no cross-image merge)
                    image_data.setdefault('image_info', {})
                    if volumes_data:
                        image_data['image_info']['volumes'] = volumes_data                        
                    else:
                        errors.append(volumes_path)
                        
                    if qc_data:
                        image_data['image_info']['qc'] = qc_data
                    else:
                        errors.append(qc_path)
                        
                    # Choose best per (session, modality)
                    if volumes_data:
                        img_score = score_synthseg(volumes_data, qc_data)

                        # --- Normalize volumes by ICV (total intracranial) ---
                        icv = volumes_data.get("total intracranial")
                        if icv and isinstance(icv, (int, float)) and not math.isnan(icv) and icv > 0:
                            # convert dict to Series, divide, and back to dict
                            vdf = pd.Series(volumes_data, dtype=float)
                            vdf = vdf / icv
                            volumes_data_norm = vdf.to_dict()
                        else:                            
                            print(f"Skipping image due to fail in normalization for {subject_id} {session_id}/{modality}: invalid ICV={icv}")
                            continue
        
                        key = (session_id, modality)
                        cand = {
                            'score': img_score,
                            'vols': volumes_data_norm,
                            'vols_raw': volumes_data,
                            'qc': qc_data or {},
                            'source': filename,
                            'pref_rank': modality_rank(modality),
                        }
                        cur = best_per_sm.get(key)
                        if (cur is None
                            or cand['score'] > cur['score']
                            or (abs(cand['score'] - cur['score']) < 1.0 and cand['pref_rank'] < cur['pref_rank'])):
                            best_per_sm[key] = cand

            # Build subject-level dicts from winners only
            subject_volumes: Dict[str, Dict[str, Dict[str, float]]] = {}
            subject_volumes_raw: Dict[str, Dict[str, Dict[str, float]]] = {}
            subject_qc: Dict[str, Dict[str, Dict[str, float]]] = {}
            for (ses, mod), payload in best_per_sm.items():
                vols = payload['vols']
                # check for NaNs in the final winner volumes
                has_nan = any(math.isnan(v) for v in vols.values() if isinstance(v, float))
                if has_nan:
                    print(f"Skipping {subject_id} {ses}/{mod} due to NaNs in volumes")
                    continue
                vols_raw = payload['vols_raw']
                
                subject_volumes.setdefault(ses, {})[mod] = vols
                subject_volumes_raw.setdefault(ses, {})[mod] = vols_raw
                if payload['qc']:
                    subject_qc.setdefault(ses, {})[mod] = payload['qc']

            # Add aggregated data to subject_info
            subject_data.setdefault('subject_info', {})
            if subject_volumes:
                subject_data['subject_info']['volumes'] = subject_volumes

            if subject_volumes_raw:
                subject_data['subject_info']['volumes_raw'] = subject_volumes_raw
            
            if subject_qc:
                subject_data['subject_info']['qc'] = subject_qc
            

            # Compute and store the subject-level reference modality (based on winners)
            sess_key, mod_key, vols_norm_ref = find_preferred_modality_in_subject(subject_volumes, PREFERRED_MODALITIES)
            vols_raw_ref = None
            if sess_key and mod_key and vols_norm_ref:
                vols_raw_ref = subject_volumes_raw.get(sess_key, {}).get(mod_key)

            subject_data['subject_info']['reference'] = (
                {
                    'session': sess_key,
                    'modality': mod_key,
                    'volumes': vols_norm_ref,     # normalized
                    'volumes_raw': vols_raw_ref   # raw
                }
                if sess_key and mod_key and vols_norm_ref else None
            )
    
    # Write the updated JSON
    output_file = output_path if output_path else json_path
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Save errors to a text file
    errors_file = os.path.join(os.path.dirname(output_file), "errors.txt")
    with open(errors_file, 'w') as ef:
        for error in errors:
            ef.write(f"{error}\n")
    print(f"Errors saved to: {errors_file}")
    print(f"Updated JSON saved to: {output_file}")

def main():
    """
    Main function - modify these paths according to your setup
    """
    # Paths - UPDATE THESE FOR YOUR SETUP
    json_file = "/home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset/nnssl_preprocessed/Dataset745_OpenMind/pretrain_data__onemmiso.json"
    openneuro_csv = "/home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset/OpenMind/openneuro_metadata.csv"
    base_data_path = "/home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset/OpenMind/OpenMind"
    output_file = "./updated_pretrain_data.json"
    
    # Process the file
    process_json_file(json_file, base_data_path, output_file, openneuro_csv)

if __name__ == "__main__":
    main()
