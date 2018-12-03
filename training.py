from sklearn import svm, metrics, linear_model, ensemble, neural_network, multioutput, model_selection
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

        state_nmlz = (state - self.dataset_mean[:-3]) / self.dataset_std[:-3]
        velocity_diff_pred_nmlz = self.regressor.predict(state_nmlz)

        velocity_diff_pred = \
            velocity_diff_pred_nmlz * self.dataset_std[-3:] + self.dataset_mean[-3:]

        return velocity_diff_pred

def main(args):
    if args.create:
        datalist = []
        data_len = 0
        data_files = 0
        for fname in sorted(os.listdir(args.datadir)):
            if fnmatch.fnmatch(fname, "imitate*.csv"):
                f_data = np.loadtxt(os.path.join(args.datadir,fname) ,delimiter=',')
                datalist.append(f_data)
                data_len += f_data.shape[0]
                data_files += 1

        dataset = []
        for d in datalist:
            dataset.append(d)
        dataset = np.vstack(dataset)
        stds = np.std(dataset,axis=0)
        np.savetxt(os.path.join(args.datadir , args.data_save), dataset, delimiter=",")
        np.savetxt(os.path.join(args.datadir , args.var_save), stds, delimiter=",")

    elif args.train:
        
        dataset = np.loadtxt(args.dataset, delimiter=',')
        #dataset = dataset[dataset[:,-1]!=0]
        
        num_samples = dataset.shape[0]
        train_size = round(0.75 * num_samples)
        dataset_means = np.mean(dataset[:train_size],axis=0)
        dataset_std = np.std(dataset[:train_size],axis=0)
        
        #dataset_means *= 0
        #dataset_std = np.ones_like(dataset_std)
        dataset_nmlz = (dataset-dataset_means)/dataset_std

        state = dataset_nmlz[:,:-3]
        velocity_diff = dataset_nmlz[:,-3:]

        # last 25% of dataset for test
        if False:
            test_data = state[train_size:]
            test_label = velocity_diff[train_size:,:]

            # first 75% of dataset for train
            train_data = state[0:train_size]
            train_label = velocity_diff[0:train_size,:]
        else:
            train_data, test_data, train_label, test_label = model_selection.train_test_split(state, velocity_diff, random_state=42)

        #regressor = multioutput.MultiOutputRegressor(svm.SVR())
        #regressor = neural_network.MLPRegressor((16,16,16,16,16),max_iter=100,solver='adam',verbose=True)
        #regressor = ensemble.ExtraTreesRegressor(8,criterion='mae',max_depth=12,verbose=1)
        regressor = multioutput.MultiOutputRegressor(linear_model.SGDRegressor(loss='epsilon_insensitive',max_iter=2000, tol=1e-3))
        #regressor = multioutput.MultiOutputRegressor(xgboost.XGBRegressor(max_depth=12,n_estimators=100,silent=False))
        regressor.fit(train_data,train_label) 

        test_pred = regressor.predict(test_data)
        train_pred = regressor.predict(train_data)
        print(test_pred.shape,test_label.shape)
        err = np.sum(np.abs(test_pred - test_label)) + np.sum(np.abs(train_pred - train_label)) 

        print(err,train_pred.shape[0] + test_pred.shape[0])
        #print(train_pred[117]*dataset_std[-3:] + dataset_means[-3:])
        #print(train_data[117]*dataset_std[:-3] + dataset_means[:-3])
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
        err = 0
        nums = 0
        errs = []
        for fname in sorted(os.listdir(args.datadir)):
            if fnmatch.fnmatch(fname, "imitate*.csv"):
                d = np.loadtxt(os.path.join(args.datadir,fname) ,delimiter=',')
                testdata = d
                test_pred = im.pred(testdata[:,:-3])
                i = testdata.shape[0]
                for i in range(i):
                    nums += 1
                    e = np.abs((testdata[i,-3:] - test_pred[i])/im.dataset_std[24:] ).sum()
                    errs.append(e)
                    err += e
        print(err,nums)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # create dataset file
    parser.add_argument("--create", action="store_true",
                        help="read data files and save to single large numpy array")
    parser.add_argument("--datadir", help="data directory",default="data")
    parser.add_argument("--data-save",help="where to save big np data",default="full_dataset.csv")
    parser.add_argument("--var-save",help="where to save variances",default="std_dataset.csv")

    # train on a dataset file
    parser.add_argument("--train", action="store_true",
                        help="train on a dataset")
    parser.add_argument("--dataset",help="path to training dataset",
                        default="data/full_dataset.csv")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--model-save",help="where to store trained model",default='model.pkl')

    # test using trained model
    parser.add_argument("--test", action="store_true",
                        help="test with trained model")
    parser.add_argument("--model-load",help="where to load trained model",
                        default="model.pkl")
    args = parser.parse_args()
    main(args)