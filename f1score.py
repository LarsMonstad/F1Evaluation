# made by lars monstad for the university of oslo
import numpy as np
import argparse

def load_tsv(file_path):
    # Load TSV file and separate columns
    data = np.loadtxt(file_path, comments='#', delimiter='\t')
    onsets = data[:, 0]
    offsets = data[:, 1]
    pitches = data[:, 2].astype(int)
    return onsets, offsets, pitches

def evaluate_metrics_with_thresholds(reference_file, estimated_file, max_t_merge=0.13, max_t_skip=0.035, max_p_merge=0.75, max_p_skip=0.4):
    ref_onsets, ref_offsets, ref_pitches = load_tsv(reference_file)
    est_onsets, est_offsets, est_pitches = load_tsv(estimated_file)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    matched_ref_indices = []

    # Iterate through estimated onsets and pitches
    for i, (est_onset, est_pitch) in enumerate(zip(est_onsets, est_pitches)):
        match_found = False

        # Iterate through reference onsets and pitches
        for j, (ref_onset, ref_pitch) in enumerate(zip(ref_onsets, ref_pitches)):
            time_diff = abs(ref_onset - est_onset)
            pitch_diff = abs(ref_pitch - est_pitch)

            # Check if the current reference note matches the current estimated note
            if time_diff <= max_t_merge and pitch_diff <= max_p_merge and j not in matched_ref_indices:
                if time_diff <= max_t_skip and pitch_diff <= max_p_skip:
                    match_found = True
                    matched_ref_indices.append(j)
                    break

        if match_found:
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(ref_onsets) - len(matched_ref_indices)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate F1 Score, Precision, and Recall of an estimated TSV file against a reference TSV file")
    parser.add_argument("reference_file", help="Path to the reference TSV file")
    parser.add_argument("estimated_file", help="Path to the estimated TSV file")

    args = parser.parse_args()

    precision, recall, f1_score = evaluate_metrics_with_thresholds(args.reference_file, args.estimated_file)
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1_score:.3f}')

if __name__ == "__main__":
    main()
