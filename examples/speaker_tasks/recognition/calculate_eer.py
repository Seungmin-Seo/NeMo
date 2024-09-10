import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import roc_curve, auc
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import matplotlib.pyplot as plt


def load_key_files(file_path):
    key_df = pd.read_csv(file_path, sep="\t")
    return key_df


def load_infer_files(file_path):
    infer_list = []
    with open(file_path, "r") as f:
        for line in f:
            infer_list.append(json.loads(line.strip()))

    infer_df = pd.DataFrame(infer_list)
    return infer_df


def extract_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def calculate_eer(key_df, infer_df):
    infer_df["audio_filename"] = infer_df["audio_filepath"].apply(extract_filename)
    infer_df = infer_df[["audio_filename", "infer"]]
    
    exploded_infer_df = infer_df.explode('infer')
    exploded_infer_df[["modelid", "score"]] = pd.DataFrame(exploded_infer_df['infer'].tolist(), index=exploded_infer_df.index)
    exploded_infer_df = exploded_infer_df[["audio_filename", "modelid", "score"]]
    exploded_infer_df.reset_index(drop=True, inplace=True)

    key_df["audio_filename"] = key_df["segmentid"].apply(
        lambda x: os.path.splitext(x)[0]
    )
    key_df = key_df[["audio_filename", "modelid", "targettype"]]

    merged_df = pd.merge(key_df, exploded_infer_df, on=['audio_filename', 'modelid'], how='left')
    merged_df.fillna({'score': 0}, inplace=True)

    print(merged_df.head)

    merged_df["label"] = merged_df["targettype"].apply(
        lambda x: 1 if x == "target" else 0
    )

    scores = merged_df["score"].values
    labels = merged_df["label"].values

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Both target and non-target samples are required to compute EER. Check your input data."
        )

    if (
        np.all(np.isnan(scores))
        or len(scores) == 0
        or np.all(np.isnan(labels))
        or len(labels) == 0
    ):
        raise ValueError("Scores or labels contain only NaN values or are empty.")

    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)

        fnr = 1 - tpr
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        # plt.show()

        eer_scores = []
        for threshold, fpr_val, fnr_val in zip(thresholds, fpr, fnr):
            eer = (fpr_val + fnr_val) / 2  # EER is where FAR = FRR
            eer_scores.append((threshold, eer))
            # print(f"FPR and FNR at threshold {threshold:.2f}: {fpr, fnr}")
            # print(f"EER at threshold {threshold:.2f}: {eer * 100:.2f}%")

        # Determine the threshold that gives the minimum EER
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        min_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        return min_eer, eer_threshold, eer_scores


    except ValueError as e:
        print(f"Error in EER calculation: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute EER score from given files.")
    parser.add_argument("-k", "--key_file", required=True, help="Path to key_files.tsv")
    parser.add_argument("-i", "--infer_file", required=True, help="Path to infer.json")

    args = parser.parse_args()

    key_df = load_key_files(args.key_file)
    infer_df = load_infer_files(args.infer_file)

    min_eer, eer_threshold, eer_scores = calculate_eer(key_df, infer_df)
    print(f"Minimum EER: {min_eer * 100:.2f}% at threshold {eer_threshold:.2f}")
