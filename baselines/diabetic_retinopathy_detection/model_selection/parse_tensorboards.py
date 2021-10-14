"""
This script details how experimental results can be obtained from a
directory of TensorBoard results, each contained in a subdirectory.

This is done by writing to a public TensorBoard.

Steps:
  1. Create the TensorBoard; e.g., `tensorboard dev upload --logdir '.'`
  2. Check the experiment ID, which will be printed from the above command.
  3. Execute this script as `python parse_tensorboards.py {experiment_id}`,
    which writes out the result as `results.tsv`.
  4. Optionally, delete the public TensorBoard with
    `tensorboard dev delete --experiment_id {experiment_id}`
"""
import sys
import pandas as pd
import tensorboard as tb


def main(experiment_id):
  experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
  df = experiment.get_scalars()
  df.to_csv(path_or_buf='results.tsv', sep='\t', index=False)
  print(len(set(df['run'])), len(pd.read_csv('results.tsv', sep='\t')))


if __name__ == '__main__':
  experiment_id = sys.argv[1]
  main(experiment_id)
