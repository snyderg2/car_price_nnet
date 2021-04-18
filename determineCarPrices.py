import argparse
import pandas as pd
import numpy as np
import pickle
import sys
from NN_torch import *
from CarPriceWindow import InputDialog
from PyQt5.QtWidgets import *

SAVED_NNET_FILE = "car_nnet.obj"
SAVED_INPUT_LIST_FILE = "input_list.obj"
SAVED_INPUT_ENUM_MAP_FILE = "input_enum_map.obj"

def confusion_matrix(Y_classes, T):
    class_names = np.unique(T)
    table = []

    for true_class in class_names:
        row = []
        for Y_class in class_names:
            row.append(100 * np.mean(Y_classes[T == true_class] == Y_class))
        table.append(row)
    conf_matrix = pd.DataFrame(table, index=class_names, columns=class_names)
    # cf.style.background_gradient(cmap='Blues').format("{:.1f} %")
        
    return conf_matrix.style.background_gradient(cmap='Blues').format("{:.1f}")
    

def percent_correct(Y, T):
    return np.mean(Y == T) * 100

def partition(Xdf, Tdf, fractions=(0.6, 0.2, 0.2), shuffle=True, classification=False):
    """Usage: Xtrain,Train,Xvalidate,Tvalidate,Xtest,Ttest = partition(X,T,(0.6,0.2,0.2),classification=True)
      X is nSamples x nFeatures.
      fractions can have just two values, for partitioning into train and test only
      If classification=True, T is target class as integer. Data partitioned
        according to class proportions.
        """
    X = Xdf.values
    T = Tdf.values
    train_fraction = fractions[0]
    if len(fractions) == 2:
        # Skip the validation step
        validate_fraction = 0
        test_fraction = fractions[1]
    else:
        validate_fraction = fractions[1]
        test_fraction = fractions[2]

    row_indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(row_indices)

    if not classification:
        # regression, so do not partition according to targets.
        n = X.shape[0]
        n_train = round(train_fraction * n)
        n_validate = round(validate_fraction * n)
        n_test = round(test_fraction * n)
        if n_train + n_validate + n_test > n:
            n_test = n - n_train - n_validate
        Xtrain = X[row_indices[:n_train], :]
        Ttrain = T[row_indices[:n_train], :]
        if n_validate > 0:
            Xvalidate = X[row_indices[n_train:n_train + n_validate], :]
            Tvalidate = T[row_indices[n_train:n_train + n_validate], :]
        Xtest = X[row_indices[n_train + n_validate:n_train + n_validate + n_test], :]
        Ttest = T[row_indices[n_train + n_validate:n_train + n_validate + n_test], :]

    else:
        # classifying, so partition data according to target class
        classes = np.unique(T)
        train_indices = []
        validate_indices = []
        test_indices = []
        for c in classes:
            # row indices for class c
            rows_this_class = np.where(T[row_indices, :] == c)[0]
            # collect row indices for class c for each partition
            n = len(rows_this_class)
            n_train = round(train_fraction * n)
            n_validate = round(validate_fraction * n)
            n_test = round(test_fraction * n)
            if n_train + n_validate + n_test > n:
                n_test = n - n_train - n_validate
            train_indices += row_indices[rows_this_class[:n_train]].tolist()
            if n_validate > 0:
                validate_indices += row_indices[rows_this_class[n_train:n_train + n_validate]].tolist()
            test_indices += row_indices[rows_this_class[n_train + n_validate:n_train + n_validate + n_test]].tolist()
        Xtrain = X[train_indices, :]
        Ttrain = T[train_indices, :]
        if n_validate > 0:
            Xvalidate = X[validate_indices, :]
            Tvalidate = T[validate_indices, :]
        Xtest = X[test_indices, :]
        Ttest = T[test_indices, :]

    if n_validate > 0:
        return Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest
    else:
        return Xtrain, Ttrain, Xtest, Ttest


def createArgParser():
    parser = argparse.ArgumentParser(description="Program that parses used car data csv file and trains a neural network to determine car price")

    parser.add_argument("-train_csv", "--train_nnet_csv", action="store", type=str,
                        help="If you want to train a neural network, provide path the csv file that contains all the used car data.")

    parser.add_argument("-hu", "--hidden_units", action="store", type=str,
                        default=[10, 10], help="Provide list of hiddne units ej [10, 10] default is [10,10].")

    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="increase output verbosity")

    parser.add_argument("-in", "--nn_inputs", action="store", type=str,
                        help="list containing all the inputs wanted to be used in training pytorch neurual network")

    parser.add_argument("-out", "--nn_outputs", action="store", type=str, required=False,
                        help="list containing all the outputs wanted to be used in training pytorch neurual network")

    parser.add_argument("-u", "--user_car", action="store", type=str, required=False,
                        help="list containing all inputs to be used for predicting a used car")
    return parser

def createPandasDataFrame(args):
    car_data_df = None
    if(args.nn_inputs):
        wanted_columns = args.nn_outputs + args.nn_inputs
        if(args.verbose):
            print("nn_inputs == {}\n".format(str(args.nn_inputs)))

        car_data_df = pd.read_csv(args.train_nnet_csv, usecols=wanted_columns)
    else:
        car_data_df = pd.read_csv(args.train_nnet_csv)

    if('price' in args.nn_outputs):
        car_data_df = car_data_df[car_data_df['price'] > 0]

    # TODO clean up NANS and strings to be enums and values that can be used for training the neural network
    column_enum_map = dict()
    df_subset = car_data_df.select_dtypes(include=["object"])

    starting_enum = 1
    for column in df_subset:
        columnNoNans = df_subset[column].dropna()
        mapping = {k: v for v, k in enumerate(columnNoNans.unique(), starting_enum )}
        column_enum_map[column] = mapping
        car_data_df[column] = car_data_df[column].map(mapping).fillna(0)

    car_data_df.dropna(inplace=True)
    if(args.verbose):
        pd.set_option('max_columns', None)
        # print(car_data_df)
        pd.set_option('max_columns', 10)

    return car_data_df, column_enum_map

def createTrainingData(args, car_df):
    Tvalues = car_df[args.nn_outputs]
    Xvalues = car_df[args.nn_inputs]
    if(args.verbose):
        print("Xvalues == {}\n".format(Xvalues[:10]))
        print("Tvalues == {}\n".format(Tvalues[:10]))

    return partition(Xvalues, Tvalues, shuffle=True)


if(__name__ == "__main__"):

    parser = createArgParser()
    args = parser.parse_args()
    car = None
    column_mapping_dict = None

    """ If wanting to train execute this code"""
    if(args.train_nnet_csv):
        if(args.nn_inputs):
            args.nn_inputs = list(map(str, args.nn_inputs.strip('[]').replace(" ", "").split(',')))
        if(args.nn_outputs):
            args.nn_outputs = list(map(str, args.nn_outputs.strip('[]').replace(" ", "").split(',')))
        else:
            args.nn_outputs = ["price"]

        usedCar_df, column_mapping_dict = createPandasDataFrame(args)
        Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = createTrainingData(args, usedCar_df)
        if(args.verbose):
            print("Xtrain.shape == {}\nTtrain.shape == {}\n".format(Xtrain.shape, Ttrain.shape))
            print("Xvalidate.shape == {}\nTvalidate.shape == {}\n".format(Xvalidate.shape, Tvalidate.shape))
            print("Xtest.shape == {}\nTtest.shape == {}\n".format(Xtest.shape, Ttest.shape))
            print("dataframe shape == {}\n".format(str(usedCar_df.shape)))
            print(Xtrain.shape)

        if (args.hidden_units):
            args.hidden_units = list(map(int, args.hidden_units.strip('[]').replace(" ", "").split(',')))
            car = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 30000, 0.1, args.hidden_units)
        else:
            car = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 30000, 0.1)

        saveNnet = input("Would you like to save the Neural Network: ").lower()
        if(saveNnet == "yes" or saveNnet == "y"):
            file_write = open(SAVED_NNET_FILE, 'wb')
            pickle.dump(car, file_write)
            print("nnet saved to {}".format(SAVED_NNET_FILE))
            file_write = open(SAVED_INPUT_ENUM_MAP_FILE, 'wb')
            pickle.dump(column_mapping_dict, file_write)
            print("enum_map saved to {}".format(SAVED_INPUT_ENUM_MAP_FILE))
            file_write = open(SAVED_INPUT_LIST_FILE, 'wb')
            pickle.dump(args.nn_inputs, file_write)
            print("input_list saved to {}".format(SAVED_INPUT_LIST_FILE))



    """If there is a user car then we want to predict"""
    used_car_input_list = None
    if(args.user_car):
        if(car is None):
            print("no currently train neural network going to load saved one")
            try:
                file_read = open(SAVED_NNET_FILE, 'rb')
                car = pickle.load(file_read)
                file_read = open(SAVED_INPUT_ENUM_MAP_FILE, 'rb')
                column_mapping_dict = pickle.load(file_read)
                file_read = open(SAVED_INPUT_LIST_FILE, 'rb')
                input_list = pickle.load(file_read)
                app = QApplication(sys.argv)
                dialogue = InputDialog(input_list, column_mapping_dict)
                if( dialogue.exec() ):
                    used_car_input_list = dialogue.getInputList()

            except OSError:
                print("Could not open/read a saved neural network need to train and save one")
                car = None

        if(used_car_input_list is None):
            args.user_car = list(map(str, args.user_car.strip('[]').replace(" ", "").split(',')))
            used_car_input_list = list()
            for i in args.user_car:
                used_car_input_list.append(float(i))

        if(args.verbose):
            print(used_car_input_list)
            print(np.array(used_car_input_list).reshape(1,11))

        if(car is not None):
            print("Your estimated car price is: ", car.use(np.array(used_car_input_list).reshape(1,len(used_car_input_list)))[0][0])
