"""
* Last Update: Oct-01-2024
* Description: This scripts splits manifest file into training/validation manifests
* Author: Seungmin Seo 
"""

import json
from sklearn.model_selection import train_test_split

def load_manifest(manifest_path):
    with open(manifest_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    return data

def save_manifest(data, output_path):
    with open(output_path, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

def split_manifest(data, train_size=0.8, random_state=42):
    labels = [entry['label'] for entry in data]
    
    train_data, val_data = train_test_split(data, train_size=train_size, random_state=random_state, stratify=labels)
    return train_data, val_data


def main(manifest_path, train_output_path, val_output_path):
    data = load_manifest(manifest_path)
    
    train_data, val_data = split_manifest(data, train_size=0.8)
    
    save_manifest(train_data, train_output_path)
    save_manifest(val_data, val_output_path)

if __name__ == '__main__':
    manifest_path = '' # path to the origianl manifest file
    train_output_path = '' # path to 80% training manifest.json
    val_output_path = '' # path to 20% validation manifest.json

    # Run the script to split and save the manifest
    main(manifest_path, train_output_path, val_output_path)
