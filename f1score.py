import numpy as np
import argparse
import mir_eval

def load_tsv(file_path):
    data = np.loadtxt(file_path, comments='#', delimiter='\t')
    onsets = data[:, 0]
    offsets = data[:, 1]
    pitches = data[:, 2].astype(int)
    return onsets, offsets, pitches

def evaluate_metrics(reference_file, estimated_file):
    ref_onsets, ref_offsets, ref_pitches = load_tsv(reference_file)
    est_onsets, est_offsets, est_pitches = load_tsv(estimated_file)

    ref_intervals = np.column_stack((ref_onsets, ref_offsets))
    est_intervals = np.column_stack((est_onsets, est_offsets))

    precision, recall, f1_score, _ = mir_eval.transcription.precision_recall_f1_overlap(ref_intervals, ref_pitches, est_intervals, est_pitches)
    return precision, recall, f1_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate F1 Score")
    parser.add_argument("reference_file", help="Path to the reference TSV file")
    parser.add_argument("estimated_file", help="Path to the estimated TSV file")

    args = parser.parse_args()

    precision, recall, f1_score = evaluate_metrics(args.reference_file, args.estimated_file)
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1_score:.3f}')

if __name__ == "__main__":
    main()
