import sys, os, re, ntpath
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pickle as pkl
import zipfile
from shutil import rmtree
from lafan1 import extract, utils, benchmarks


"""
Unzips the data, extracts the LaFAN1 train statistics,
then the LaFAN1 test set and evaluates baselines on the test set.
"""

# output location for unzipped bvhs, stats, results, etc...
save_path = os.path.join(os.path.dirname(__file__), 'test_set')
if os.path.exists(save_path):
    rmtree(save_path)
os.makedirs(save_path, exist_ok=True)


# the train/test set actors as in the paper
train_actors = ['subject1', 'subject2', 'subject3', 'subject4']
test_actors = ['subject5']

bone_length = [ 0., 10.711, 43.5, 42.372, 17.3, 10.711, 43.5, 42.372, 17.3, 7.377,
                12.588, 12.343, 25.833, 11.767, 20.69, 11.284, 33., 25.2, 20.691, 11.284,
                33., 25.2]
factor = 43.5


print('Unzipping the data...\n')
lafan_data = os.path.join(os.path.dirname(__file__), 'lafan1', 'lafan1.zip')
bvh_folder = os.path.join(os.path.dirname(__file__), 'BVH')
with zipfile.ZipFile(lafan_data, "r") as zip_ref:
    if not os.path.exists(bvh_folder):
        os.makedirs(bvh_folder, exist_ok=True)
    zip_ref.extractall(bvh_folder)


print('Retrieving testing data...')
bvh_files = os.listdir(bvh_folder)
actors = test_actors
for file in bvh_files:
    if file.endswith('.bvh'):
            seq_name, subject = ntpath.basename(file[:-4]).split('_')

            if subject in actors:
                print('Processing file {}'.format(file))
                in_path = os.path.join(bvh_folder, file)
                out_path = os.path.join(save_path, seq_name+'_'+subject+'_test.pkl')

                x_local, q_local, parents, ra, rv = extract.get_lafan1_seq(in_path)
                q_glbl, x_glbl = utils.quat_fk(q_local, x_local, parents)
                x = np.zeros_like(x_glbl)
                for i in range(x_glbl.shape[0]):
                    for j in range(x_glbl.shape[1]):
                        x[i][j] = x_glbl[i][j] - x_glbl[i][0]
                x_from_parents = np.zeros_like(x)
                x_from_parents[..., 0, :] = x[..., 0, :]
                for i in range(1, 22):
                    x_from_parents[..., i, :] = x[..., i, :] - x[..., parents[i], :]
                x_from_parents = x_from_parents / factor
                lin_v = utils.d(x_from_parents)
                lin_a = utils.dd(x_from_parents)
                force = utils.compute_force_lin_nt(lin_a)
                torque = np.zeros_like(force)
                for i in range(force.shape[0]):
                    for j in range(force.shape[1]):
                        torque[i][j] = np.cross(x_from_parents[i][j], force[i][j])

                force_root = utils.compute_force_lin_nt(ra)
                for i in range(torque.shape[0]):
                    torque[i][0] = force_root[i]
                    lin_a[i][0] = ra[i]
                    lin_v[i][0] = rv[i]

                #save results
                data = {
                    'x_local': x_local,
                    'q_local': q_local,
                    'lin_a': lin_a,
                    'lin_v': lin_v,
                    'parents': parents,
                    'force': torque,
                    'factor': factor
                }
                print('Write ' + seq_name + '_' + subject + ' to pkl files')
                with open(out_path, 'wb') as f:
                    pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

print('Done!')

