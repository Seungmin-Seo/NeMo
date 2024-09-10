import json
import os

def replace_label(args):
    data = []
    with open(args.input_file, 'r') as file:
        lines = file.readlines()

    base_dir = os.path.dirname(args.input_file)
    output_filepath = os.path.join(base_dir, 'sre21eval_16k_test_manifest_replaced.json')
    with open(output_filepath, 'w') as file:
        for line in lines:
            obj = json.loads(line)
            if 'label' in obj:
                obj['label'] = 'infer'
            file.write(json.dumps(obj) + '\n')

    print("All 'label' values have been replaced with 'infer'.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="replace the label to 'infer' for test manifest .json")
    parser.add_argument("-i", "--input_file", required=True, help="Path to input json file")

    args = parser.parse_args()

    replace_label(args)
