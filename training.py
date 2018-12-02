from sklearn import svm, metrics
import numpy as np
from pdb import set_trace as st
import argparse
import os
import fnmatch

def main(args):
    if args.create:
        datalist = []
        data_dim = 0
        data_len = 0

        for fname in os.listdir(args.datadir):
            if fnmatch.fnmatch(fname, "imitate*.csv"):
                f_data = np.loadtxt(args.datadir + fname,delimiter=',')
                datalist.append(f_data)
                data_len += f_data.shape[0]

        data_dim = datalist[0].shape[1]

        dataset = np.zeros((data_len, 27))
        variances = np.zeros((data_len))

        row_ptr = 0
        for d in datalist:
            velocity_diff = d[1: , 21:] - d[0:-1 , 21:]
            states = d[1: , 0:21]
            velocities = d[0:-1 , 21:]

            num_rows = d.shape[0] - 1

            dataset[row_ptr : row_ptr + num_rows, 0:21] = states
            dataset[row_ptr : row_ptr + num_rows, 21:24] = velocities
            dataset[row_ptr : row_ptr + num_rows, 24:] = velocity_diff

            variances[row_ptr : row_ptr + num_rows] = \
                np.var(dataset[row_ptr : row_ptr + num_rows, 0:24],axis=1)

            row_ptr += num_rows

        np.savetxt(args.datadir + args.data_save, dataset, delimiter=",")
        np.savetxt(args.datadir + args.var_save, variances, delimiter=",")

    elif args.train:
        dataset = np.loadtxt(args.dataset, delimiter=',')
        state = dataset[:,0:24]
        velocity_diff = dataset[:,24:]

        test_sz = round(0.25 * dataset.shape[0])
        test_data = state[-test_sz:,:]
        test_label = velocity_diff[-test_sz:,2]

        train_size = round(0.75 * dataset.shape[0])

        clf = svm.SVR(C=10)
        clf.fit(state[0:train_size],velocity_diff[0:train_size,2])
        clf_pred = clf.predict(test_data)
        clf_err = metrics.mean_squared_error(clf_pred,test_label)

        st()




        



    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # create dataset file
    parser.add_argument("--create", action="store_true",
                        help="read data files and save to single large numpy array")
    parser.add_argument("--datadir", help="data directory")
    parser.add_argument("--data-save",help="where to save big np data")
    parser.add_argument("--var-save",help="where to save variances")

    # train on a dataset file
    parser.add_argument("--train", action="store_true",
                        help="train on a dataset")
    parser.add_argument("--dataset")

    args = parser.parse_args()
    main(args)