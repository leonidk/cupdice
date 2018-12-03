from sklearn import svm, metrics, linear_model, ensemble, neural_network, multioutput
import numpy as np
from pdb import set_trace as st
import argparse
import os
import fnmatch
import matplotlib.pyplot as plt
import xgboost
import pickle


class ImitationModel:
    def __init__(self,model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.regressor = model[0]
        self.dataset_mean = model[1]
        self.dataset_std = model[2]

    def pred(self, state):
        # input: N x 24 state
        # output: N x 3 velocity diffs

        state_nmlz = (state - self.dataset_mean[0:24]) / self.dataset_std[0:24]
        velocity_diff_pred_nmlz = self.regressor.predict(state_nmlz)

        velocity_diff_pred = \
            velocity_diff_pred_nmlz * self.dataset_std[24:27] + self.dataset_mean[24:27]

        return velocity_diff_pred

def main(args):
    if args.create:
        datalist = []
        data_len = 0

        for fname in os.listdir(args.datadir):
            if fnmatch.fnmatch(fname, "imitate*.csv"):
                f_data = np.loadtxt(os.path.join(args.datadir,fname) ,delimiter=',')
                datalist.append(f_data)
                data_len += f_data.shape[0]

        data_dim = datalist[0].shape[1]

        dataset = np.zeros((data_len, 27))
        variances = np.zeros((24))

        row_ptr = 0
        for d in datalist:
            velocity_diff = d[1: , 21:] - d[0:-1 , 21:]
            states = d[1: , 0:21]
            velocities = d[0:-1 , 21:]

            num_rows = d.shape[0] - 1

            dataset[row_ptr : row_ptr + num_rows, 0:21] = states
            dataset[row_ptr : row_ptr + num_rows, 21:24] = velocities
            dataset[row_ptr : row_ptr + num_rows, 24:] = velocity_diff

            row_ptr += num_rows

        variances = np.var(dataset[:,0:24],axis=0)
        np.savetxt(os.path.join(args.datadir , args.data_save), dataset, delimiter=",")
        np.savetxt(os.path.join(args.datadir , args.var_save), variances, delimiter=",")

    elif args.train:
        
        dataset = np.loadtxt(args.dataset, delimiter=',')
        #dataset = dataset[dataset[:,-1]!=0]
        
        num_samples = dataset.shape[0]
        train_size = round(0.75 * num_samples)
        test_sz = round(0.25 * num_samples)
        dataset_means = np.mean(dataset[:train_size],axis=0)
        dataset_std = np.std(dataset[:train_size],axis=0)

        dataset_nmlz = (dataset-dataset_means)/dataset_std

        state = dataset_nmlz[:,0:24]
        velocity_diff = dataset_nmlz[:,24:]

        # last 25% of dataset for test
        test_data = state[-test_sz:]
        test_label = velocity_diff[-test_sz:,:]

        # first 75% of dataset for train
        train_data = state[0:train_size]
        train_label = velocity_diff[0:train_size,:]

        # regressor = svm.SVR(C=1)
        #regressor = neural_network.MLPRegressor((16,16,16,16,16),max_iter=100,solver='adam',verbose=True)
        #regressor = ensemble.ExtraTreesRegressor(8,criterion='mae',max_depth=8,verbose=1,n_jobs=-1)
        #regressor = linear_model.SGDRegressor(loss='epsilon_insensitive',max_iter=1000, tol=1e-3)
        regressor = multioutput.MultiOutputRegressor(xgboost.XGBRegressor(max_depth=12,n_estimators=100,silent=False))
        regressor.fit(train_data,train_label) 

        test_pred = regressor.predict(test_data)

        print(test_pred.shape,test_label.shape)
        err = np.sum(np.abs(test_pred - test_label))

        print(err)

        model = [regressor, dataset_means, dataset_std]

        with open(args.model_save, 'wb') as f:
            pickle.dump(model, f)

        # err = abs(test_pred - test_label) / np.mean(abs(test_label))
        # err_hist = np.histogram(err, bins='auto')

        # success_curve = np.cumsum(err_hist[0]) / test_sz

        # err_thresh = 0.1
        # err_ind = err_hist[1][1:] < err_thresh
        # success_ratio = np.sum(err_hist[0][err_ind]) / test_sz

        # print(success_ratio)

        # if args.visualize:
        #     plt.figure()
        #     plt.hist(err,bins='auto')
        #     plt.title('test error normalized by mean label mag.')

        #     plt.figure()
        #     plt.plot(err_hist[1][1:], success_curve)
        #     plt.title('success curve on test data')

        #     plt.show()   
    
    if args.test:
        # run:
        # python3 training.py --test --test-data data/imitate_0.csv --model-load models/nn1.pickle
        im = ImitationModel(args.model_load)
        testdata = np.loadtxt(args.test_data, delimiter=',')
        test_pred = im.pred(testdata)
        err = 0
        i = testdata.shape[0]
        for i in range(0,i-1):
            err += np.linalg.norm((testdata[i+1,-3:]-testdata[i,-3:]) - test_pred[i])
        print(err)

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
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--model-save",help="where to store trained model")

    # test using trained model
    parser.add_argument("--test", action="store_true",
                        help="test with trained model")
    parser.add_argument("--model-load",help="where to load trained model")
    parser.add_argument("--test-data",help="csv with test data")
    args = parser.parse_args()
    main(args)