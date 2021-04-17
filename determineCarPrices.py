from NeuralNetworkTorch import *
import argparse
import pandas as pd
from pyspark.sql import SparkSession
import os
from NN_torch import *
def confusion_matrix(Y_classes, T):
    class_names = np.unique(T)
    table = []

    for true_class in class_names:
        row = []
        for Y_class in class_names:
            row.append(100 * np.mean(Y_classes[T == true_class] == Y_class))
        table.append(row)
    conf_matrix = pandas.DataFrame(table, index=class_names, columns=class_names)
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
            rows_this_class = np.where(T[row_indices,:] == c)[0]
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

    # if(spark is not None):
    #     Xtrain = spark.createDataFrame(Xtrain, list(Xdf) ).rdd
    #     Ttrain = spark.createDataFrame(Ttrain, Tdf.columns.array.tolist() ).rdd
    #     Xtest = spark.createDataFrame(Xtest, Xdf.columns.array.tolist() ).rdd
    #     Ttest = spark.createDataFrame(Ttest, Tdf.columns.array.tolist() ).rdd
    #     if(n_validate > 0):
    #         Xvalidate = spark.createDataFrame(Xvalidate, Xdf.columns.array.tolist()).rdd
    #         Tvalidate = spark.createDataFrame(Tvalidate, Tdf.columns.array.tolist()).rdd

    if n_validate > 0:
        return Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest
    else:
        return Xtrain, Ttrain, Xtest, Ttest

def createArgParser():
    parser = argparse.ArgumentParser(description="Program that parses used car data csv file and trains a neural network to determine car price")
    
    parser.add_argument('input_csv', action="store", type=str, 
                        help="provide path the csv file that contains all the used car data.")

    parser.add_argument("-hu", "--hidden_units", action="store", type=str,
                        default=[10, 10], help="Provide list of hiddne units ej [10, 10] default is [10,10].")

    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="increase output verbosity")

    parser.add_argument("-in", "--nn_inputs", action="store", type=str, 
                        help="list containing all the inputs wanted to be used in training pytorch nerual network")

    parser.add_argument("-out", "--nn_outputs", action="store", type=str, required=True,
                        help="list containing all the outputs wanted to be used in training pytorch nerual network")
    parser.add_argument("-u", "--user_car", action="store", type=str, required=True,
                        help="list containing all the outputs wanted to be used in training pytorch nerual network")
    return parser

def createPandasDataFrame(args):
    car_data_df = None
    if(args.nn_inputs):
        wanted_columns = args.nn_outputs + args.nn_inputs
        if(args.verbose):
            print("nn_inputs == {}\n".format(str(args.nn_inputs)))
        car_data_df = pd.read_csv(args.input_csv, usecols=wanted_columns)
    else:
        car_data_df = pd.read_csv(args.input_csv)

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
        print(car_data_df)
        pd.set_option('max_columns', 10)

    return car_data_df, column_enum_map

def createTrainingData(args, car_df):
    Tvalues = car_df[args.nn_outputs]#.values
    Xvalues = car_df[args.nn_inputs]#.values
    if(args.verbose):
        print("Xvalues.shape == {}\nXvalues == {}\n".format(Xvalues.shape, Xvalues[:5]))
        print("Tvalues.shape == {}\nTvalues == {}\n".format(Tvalues.shape, Tvalues[:5]))

    return partition(Xvalues, Tvalues, shuffle=False)

    
def createCliNeuralNetwork(args, car_df, verbose=False):
    hidden_layers = [10, 10]
    if(args.hidden_units):
        hidden_layers = list(map(int, args.hidden_units.strip('[]').split(',')))
        if(verbose):
            print("hidden_layers == {}".format(str(hidden_layers)))
    
    outputCnt = len(args.nn_inputs)
    inputCnt = car_df.shape[1] - outputCnt
    nnet = NeuralNetworkTorch(inputCnt, hidden_layers, outputCnt)
    if(args.verbose):
        print(str(nnet))
    return nnet


if(__name__ == "__main__"):

    parser = createArgParser()
    args = parser.parse_args()
    # print(args)
    if(args.nn_inputs):
        args.nn_inputs = list(map(str, args.nn_inputs.strip('[]').replace(" ", "").split(',')))

    if(args.nn_outputs):
        args.nn_outputs = list(map(str, args.nn_outputs.strip('[]').replace(" ", "").split(',')))
    if(args.user_car):
        args.user_car = list(map(str, args.user_car.strip('[]').replace(" ", "").split(',')))
    # print("test",args.nn_inputs)
    # spark = SparkSession.builder.master('local').appName("price_predict").getOrCreate()
    # spark = SparkSession.builder.master('spark://denver:31850').appName("price_predict").getOrCreate()

    usedCar_df, column_mapping_dict = createPandasDataFrame(args)
    # print(args.user_car[2])


    Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = createTrainingData(args, usedCar_df)
    # if(args.verbose):
    #     print("Xtrain.shape == {}\nTtrain.shape == {}\n".format(Xtrain.shape, Ttrain.shape))
    #     print("Xvalidate.shape == {}\nTvalidate.shape == {}\n".format(Xvalidate.shape, Tvalidate.shape))
    #     print("Xtest.shape == {}\nTtest.shape == {}\n".format(Xtest.shape, Ttest.shape))
    #     print("dataframe shape == {}\n".format(str(usedCar_df.shape)))
    print(Xtrain.shape)
    year = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 4000, 0.1)
    # print(args.user_car)
    user_car_array = []
    for i in args.user_car:
        user_car_array.append(float(i))
    print(user_car_array)
    print(np.array(user_car_array).reshape(1,11))
    print(year.use(np.array(user_car_array).reshape(1,11)))




    # year = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 4000, 0.1, int(args.user_car[0]))[0][0]
    # print("year:", args.user_car[0], "price: ", year)

    # Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = createTrainingData(args, usedCar_df, 1)
    # manufacturer = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 4000, 0.1, int(args.user_car[1]))[0][0]
    # print("manufacturer:",args.user_car[1], "price: ", manufacturer)



    # Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = createTrainingData(args, usedCar_df, 2)
    # # print(Ttrain)
    # model = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 5000, 0.05, int(args.user_car[2]))[0][0]
    # print("model:",args.user_car[2], "price: ", model)

    # Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = createTrainingData(args, usedCar_df, 3)
    # condition = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 4000, 0.1, int(args.user_car[3]))[0][0]
    # print("condition:",args.user_car[3], "price: ", condition)

    # Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = createTrainingData(args, usedCar_df, 4)
    # cylinders = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 4000, 0.1, int(args.user_car[4]))[0][0]
    # print("cylinders:",args.user_car[4], "price: ", cylinders)

    # Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = createTrainingData(args, usedCar_df, 5)
    # fuel = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 4000, 0.1, int(args.user_car[5]))[0][0]
    # print("fuel:",args.user_car[5], "price: ", fuel)

    # Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = createTrainingData(args, usedCar_df, 6)
    # odometer = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 4000, 0.1, int(args.user_car[6]))[0][0]
    # print("odometer:",args.user_car[6], "price: ", odometer)

    # Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = createTrainingData(args, usedCar_df, 7)
    # title_status = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 4000, 0.1, int(args.user_car[7]))[0][0]
    # print("title_status:",args.user_car[7], "price: ", title_status)

    # Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = createTrainingData(args, usedCar_df, 8)
    # transmission = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 4000, 0.1, int(args.user_car[8]))[0][0]
    # print("transmission:",args.user_car[8], "price: ", transmission)

    # Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = createTrainingData(args, usedCar_df, 9)
    # drive = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 4000, 0.1, int(args.user_car[9]))[0][0]
    # print("drive:",args.user_car[9], "price: ", drive)

    # Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = createTrainingData(args, usedCar_df, 10)
    # paint_color = run(Xtrain, Ttrain, Xtest, Ttest, 'sgd', 4000, 0.1, int(args.user_car[10]))[0][0]
    # print("paint_color:",args.user_car[10], "price: ", paint_color)

    # print()
    # total_price = year*0.3 + odometer*0.25 + model*0.2 + title_status*0.1 + manufacturer*0.05 + condition*0.05 + cylinders*0.01 + fuel*0.01 + transmission*0.01 + drive*0.01 + paint_color*0.01
    # print("Your car's estimated price is:", total_price)