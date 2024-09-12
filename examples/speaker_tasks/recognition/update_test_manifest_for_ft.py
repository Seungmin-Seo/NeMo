import pandas as pd
import json
import os

def load_key_files(file_path) -> pd.DataFrame:
    key_df = pd.read_csv(file_path, sep="\t")
    key_df = key_df[key_df['targettype'] == 'target'].copy()
    return key_df

def load_manifest_files(file_path) -> json:
    with open(file_path, 'r') as f:
        json_data = [json.loads(line.strip()) for line in f]
    return json_data

def extract_filename(filepath) -> str:
    return os.path.splitext(os.path.basename(filepath))[0]

def write_json(out_filepath, json_data):
    with open(out_filepath, 'w') as f:
        for json_row in json_data:
            f.write(json.dumps(json_row) + '\n')


def update_target_label(args) -> list:
    updated_json = []

    key_df = load_key_files(args.key_file)
    orig_manifest = load_manifest_files(args.input_file)

    for json_row in orig_manifest:
        audio_filepath = json_row['audio_filepath']
        audio_basename = extract_filename(audio_filepath)

        matched_rows = key_df[key_df['segmentid'].apply(lambda x: extract_filename(x)) == audio_basename]
        if not matched_rows.empty:
            for _, matched_row in matched_rows.iterrows():
                new_json_row = json_row.copy()
                new_json_row['label'] = matched_row['modelid']
                updated_json.append(new_json_row)

    return updated_json

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="replace the 'infer' label to target label")
    parser.add_argument("-i", "--input_file", required=True, help="Path to manifest.json file")
    parser.add_argument("-k", "--key_file", required=True, help="Path to trial key file")

    args = parser.parse_args()

    updated_json = update_target_label(args)
    out_filepath = os.path.join(os.path.dirname(args.input_file), 'sre21dev_16k_test_manifest_ft.json')
    write_json(out_filepath, updated_json)