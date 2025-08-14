#!/usr/bin/env python3
"""
Script to add SynthSeg volumes and QC results to the preprocessed dataset JSON file.
Adds information at both image level (image_info) and subject level (subject_info).
"""

import json
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, Optional
import re
import glob

def read_synthseg_csv(csv_path: str, subject_id:str) -> Optional[Dict[str, float]]:
    """
    Read a SynthSeg CSV file and return the data as a dictionary.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Dictionary with region names as keys and values as floats, or None if file doesn't exist
    """
    if not os.path.exists(csv_path) and '/home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset/OpenMind/OpenMind/ds001353/' not in csv_path:
        print(f"Warning: CSV file not found: {csv_path}")
        directory = os.path.dirname(csv_path)
        print('The directory contains', glob.glob(os.path.join(directory, '*.csv')))
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if 'subject' in df.columns:
            df = df[df['subject'] == subject_id]  # Filter by subject ID
        elif df.shape[1] == 102 or df.shape[1] == 9: # volumes or qc
            # Rename the first column header to 'subject' if it has no name
            if df.columns[0] == '' or df.columns[0] == 'Unnamed: 0':
                df.rename(columns={df.columns[0]: 'subject'}, inplace=True)
                df.to_csv(csv_path, index=False)  # Save the changes back to the CSV
                df = df[df['subject'] == subject_id]  # Filter by subject ID
        if len(df) == 0:
            return None
        
        # Get the first row (should only be one row per file)
        row = df.iloc[0]
        
        # Convert to dictionary, excluding the 'subject' column
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
    # Remove file extensions
    base_name = filename.replace('.nii.gz', '').replace('.nii', '')
    
    # Extract modality - look for common patterns
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
    
    # If no pattern matches, try to extract the last part after underscore
    parts = base_name.split('_')
    if len(parts) > 1:
        return parts[-1].lower()
    
    return 'unknown'

def find_synthseg_files(base_path: str, dataset_id: str, subject_id: str, session_id: str, modality: str) -> tuple:
    """
    Find the volumes.csv and qc.csv files for a given image.
    
    Args:
        base_path: Base path to the OpenMind dataset
        dataset_id: Dataset ID (e.g., 'ds000001')
        subject_id: Subject ID (e.g., 'sub-01')
        session_id: Session ID (e.g., 'ses-DEFAULT', 'ses-01')
        modality: Image modality (e.g., 't1w', 'dwi')
        
    Returns:
        Tuple of (volumes_path, qc_path)
    """
    # Map modalities to folder names
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
    
    # Include session in the path when it's not 'ses-DEFAULT'
    if session_id == 'ses-DEFAULT':
        synthseg_base = os.path.join(base_path, dataset_id, 'derivatives', 'synthseg', subject_id, folder)
    else:
        synthseg_base = os.path.join(base_path, dataset_id, 'derivatives', 'synthseg', subject_id, session_id, folder)
    
    volumes_path = os.path.join(synthseg_base, 'volumes.csv')
    qc_path = os.path.join(synthseg_base, 'qc.csv')
    
    return volumes_path, qc_path

def process_json_file(json_path: str, base_data_path: str, output_path: str = None, openneuro_csv: str = None):
    """
    Process the JSON file to add SynthSeg results.
    
    Args:
        json_path: Path to the input JSON file
        base_data_path: Base path to the OpenMind dataset directory
        output_path: Path for output JSON file (if None, overwrites input)
    """
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    openneuro_df = pd.read_csv(openneuro_csv) 

    errors = []
    
    # Process each dataset
    for dataset_id, dataset_info in data.get('datasets', {}).items():
        #print(f"Processing dataset: {dataset_id}")
        
        # Process each subject
        for subject_id, subject_data in dataset_info.get('subjects', {}).items():
            #print(f"  Processing subject: {subject_id}")
            
            # Initialize subject-level aggregation
            subject_volumes = {}
            subject_qc = {}
            
            
            # Process each session
            for session_id, session_data in subject_data.get('sessions', {}).items():
                #print(f"    Processing session: {session_id}")
                
                # Process each image
                for image_idx, image_data in enumerate(session_data.get('images', [])):
                    filename = image_data.get('name', '')
                    modality = image_data.get('modality', '') #get_modality_from_filename(filename)
                    filepath = image_data.get('image_path', '')
                    unique_id = filepath.split('nnsslPlans_median/Dataset001_OpenMind/')[1].replace('/',"__")+'.nii.gz'
                    #print(f"      Unique ID: {unique_id}")
                    try:
                        look_in = openneuro_df[openneuro_df.unique_id == unique_id]
                        if not look_in.empty:
                            relative_path = look_in['image_path'].values[0]
                        else:
                            relative_path = openneuro_df[openneuro_df.unique_id == unique_id.replace('.gz', '')].image_path.values[0]
                    except IndexError:
                        print(f"Skipping unique_id due to missing data: {unique_id}")
                        continue
                    filename = relative_path.split('/')[-1]
                    relative_path_fields = relative_path.split('/')
                    relative_path = os.path.join(relative_path_fields[0], "derivatives/synthseg") 
                    for i in range(1, len(relative_path_fields)-1):
                        if "__Data" not in relative_path_fields[i]:
                            relative_path = os.path.join(relative_path, relative_path_fields[i])
                    #print("Relative path 2:", relative_path)
                    
                    #print(f" Processing image: {filename} (modality: {modality})")
                    # Find SynthSeg files
                    #volumes_path, qc_path = find_synthseg_files(base_data_path, dataset_id, subject_id, session_id, modality)
                    volumes_path = os.path.join(base_data_path, relative_path, 'volumes.csv') 
                    if not os.path.exists(volumes_path):
                        volumes_path = os.path.join(base_data_path, relative_path, f'{filename.replace('.gz', '')}_volumes.csv') if ".gz" in filename else os.path.join(base_data_path, relative_path, f'{filename.replace('.nii', '')}_volumes.csv') 
                    #print(f"      Volumes path: {volumes_path}")
                    qc_path = os.path.join(base_data_path, relative_path, 'qc.csv')
                    if not os.path.exists(qc_path):
                        qc_path = os.path.join(base_data_path, relative_path, f'{filename.replace('.gz', '')}_volumes.csv') if ".gz" in filename else os.path.join(base_data_path, relative_path, f'{filename.replace('.nii', '')}_qc.csv')
                    
                    # Read volumes and QC data
                    volumes_data = read_synthseg_csv(volumes_path, filename.replace('.nii.gz', ''))
                    qc_data = read_synthseg_csv(qc_path, filename.replace('.nii.gz', ''))
                    
                    # Add to image_info
                    if 'image_info' not in image_data:
                        image_data['image_info'] = {}
                    
                    if volumes_data:
                        image_data['image_info']['volumes'] = volumes_data
                        # Add to subject aggregation
                        if session_id not in subject_volumes:
                            subject_volumes[session_id] = {}
                        if modality not in subject_volumes[session_id]:
                            subject_volumes[session_id][modality] = {}
                        subject_volumes[session_id][modality].update(volumes_data)
                    else:
                        errors.append(volumes_path)
                    
                    if qc_data:
                        image_data['image_info']['qc'] = qc_data
                        # Add to subject aggregation  
                        if session_id not in subject_qc:
                            subject_qc[session_id] = {}
                        if modality not in subject_qc[session_id]:
                            subject_qc[session_id][modality] = {}
                        subject_qc[session_id][modality].update(qc_data)
                        
            
            # Add aggregated data to subject_info
            if 'subject_info' not in subject_data:
                subject_data['subject_info'] = {}
            
            if subject_volumes:
                subject_data['subject_info']['volumes'] = subject_volumes
            
            if subject_qc:
                subject_data['subject_info']['qc'] = subject_qc
    
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
    json_file = "/home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset/nnssl_preprocessed/Dataset745_OpenMind/pretrain_data__median.json"
    openneuro_csv = "/home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset/OpenMind/openneuro_metadata.csv"
    base_data_path = "/home/jovyan/shared/pedro-maciasgordaliza/openmind-dataset/OpenMind/OpenMind"
    output_file = "./updated_pretrain_data.json"
    
    # Process the file
    process_json_file(json_file, base_data_path, output_file,openneuro_csv)

if __name__ == "__main__":
    main()