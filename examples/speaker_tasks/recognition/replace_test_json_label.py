import json
import os

def replace_label(args):
    with open(args.input_file, 'r') as file:
        data = json.load(file)

    for row in data:
        if 'label' in row:
            row['label'] = 'infer'

    base_dir = os.path.dirname(args.input_file)
    output_filepath = os.path.join(base_dir, 'sre21eval_test_manifest_replaced.json')
    with open('your_file_modified.json', 'w') as file:
        json.dump(data, file)
    print("All 'label' values have been replaced with 'infer'.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="replace the label to 'infer' for test manifest .json")
    parser.add_argument("-i", "--input_file", required=True, help="Path to input json file")

    args = parser.parse_args()

    replace_label(args)