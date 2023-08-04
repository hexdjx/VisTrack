import numpy as np
import os
import shutil
from pytracking.evaluation.environment import env_settings
from pytracking.evaluation.datasets import get_dataset


def pack_trackingnet_results(tracker_name, param_name, run_id=None, output_name=None):
    """ Packs trackingnet results into a zip folder which can be directly uploaded to the evaluation server. The packed
    file is saved in the folder env_settings().tn_packed_results_path

    args:
        tracker_name - name of the tracker
        param_name - name of the parameter file
        run_id - run id for the tracker
        output_name - name of the packed zip file
    """

    if output_name is None:
        if run_id is None:
            output_name = '{}_{}'.format(tracker_name, param_name)
        else:
            output_name = '{}_{}_{:03d}'.format(tracker_name, param_name, run_id)

    output_path = os.path.join(env_settings().tn_packed_results_path, output_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results_path = env_settings().results_path

    tn_dataset = get_dataset('trackingnet')

    for seq in tn_dataset:
        seq_name = seq.name

        if run_id is None:
            seq_results_path = '{}/{}/{}/{}.txt'.format(results_path, tracker_name, param_name, seq_name)
        else:
            seq_results_path = '{}/{}/{}_{:03d}/{}.txt'.format(results_path, tracker_name, param_name, run_id, seq_name)

        results = np.loadtxt(seq_results_path, dtype=np.float64)

        np.savetxt('{}/{}.txt'.format(output_path, seq_name), results, delimiter=',', fmt='%.2f')

    # Generate ZIP file
    shutil.make_archive(output_path, 'zip', output_path)

    # Remove raw text files
    shutil.rmtree(output_path)


def pack_got10k_results(tracker_name, param_name, output_name):
    """ Packs got10k results into a zip folder which can be directly uploaded to the evaluation server. The packed
    file is saved in the folder env_settings().got_packed_results_path

    args:
        tracker_name - name of the tracker
        param_name - name of the parameter file
        output_name - name of the packed zip file
    """
    output_path = os.path.join(env_settings().got_results_path, output_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results_path = env_settings().results_path
    for i in range(1,181):
        seq_name = 'GOT-10k_Test_{:06d}'.format(i)

        seq_output_path = '{}/{}'.format(output_path, seq_name)
        if not os.path.exists(seq_output_path):
            os.makedirs(seq_output_path)

        for run_id in range(3):
            res = np.loadtxt('{}/{}/{}_{:03d}/{}.txt'.format(results_path, tracker_name, param_name, run_id, seq_name), dtype=np.float64)
            times = np.loadtxt(
                '{}/{}/{}_{:03d}/{}_time.txt'.format(results_path, tracker_name, param_name, run_id, seq_name),
                dtype=np.float64)

            np.savetxt('{}/{}_{:03d}.txt'.format(seq_output_path, seq_name, run_id+1), res, delimiter=',', fmt='%f')
            np.savetxt('{}/{}_time.txt'.format(seq_output_path, seq_name), times, fmt='%f')

    # Generate ZIP file
    shutil.make_archive(output_path, 'zip', output_path)

    # Remove raw text files
    shutil.rmtree(output_path)



#pack_trackingnet_results('fudimp', 'fudimp_apce', None, 'FuDiMP++')
pack_got10k_results('fudimp', 'fudimp_apce', 'FuDiMP++')
