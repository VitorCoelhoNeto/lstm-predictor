import numpy as np
import pandas as pd
import math
import geopy
import geopy.distance
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle

from tensorflow import keras
from keras.layers import LSTM 
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from latlongTodistbear import transform_dataset

# Training Globals
g_training_percentage=0.67
g_myverbose = 0
g_batch_size = 1
g_epochs = 100
g_input_shape = (1, 2)
g_earlyStopper = EarlyStopping(patience=3, monitor='loss', mode='min')
g_loss_function = 'mean_squared_error'
g_savePath = 'ModelsOutput/TrainedModel.keras'
g_xScalerSavePath = 'ScalersOutput/scalerx.save'
g_yScalerSavePath = 'ScalersOutput/scalery.save'
g_fitHistoryFile = 'MetricsOutput/history.save'

# Testing Globals
g_TestingDataPath = 'TestingData.csv'
g_scalerXFilename = 'ScalersOutput/01_10_2023_22h30' + 'X.save'
g_scalerYFilename = 'ScalersOutput/01_10_2023_22h30' + 'Y.save'
g_model_to_be_loaded = 'ModelsOutput/01_10_2023_22h30.keras'

np.set_printoptions(formatter={'float_kind':'{:f}'.format})


def train_and_score(network, dataset, bestScore):
    '''
    Train the network with the given dataset (csv path)
    Evaluate the obtained score
    '''
    trainX, trainY, testX, testY, scalerX, scalerY, Coords = preprocessing(dataset)
    model, lstms, history = compile_model(network, trainX, trainY)
    error, bestScore = evaluate(model, testX, testY, scalerX, scalerY, Coords, bestScore, history)
    return error, bestScore



def preprocessing(dataset):
    '''
    Get the data ready in the correct data formats, eg. convert from lat/long to dist/bearing
    Get the training and testing datasets divided
    '''
    # Gets the coordinates in dist/bear as a vector of dist/bear coordinates like the original file, but in dist/bear
    dataset_X = transform_dataset(dataset)
    # Delete the first line of the dataset into a new dataset
    dataset_Y = np.delete(dataset_X, 0, 0)
    # Delete the last line of the dataset
    dataset_X = np.delete(dataset_X, -1, 0)
    
    # Original dataset coordinates
    Coords = pd.read_csv(dataset, engine='python').values.astype('float32')
    # Pushes the first element of the array to last and every other element goes up 1 position, ex: [0,1,2,3,4] -> [1,2,3,4,0]
    Coords=np.roll(Coords, -1, axis=0) # TODO: WHY IS THIS LINE NECESSARY

    # Get the train and test datasets divided in g_training_percentage for training, and the rest for testing
    train_size = int(len(dataset_X) * g_training_percentage)
    trainX, testX = dataset_X[0:train_size,:], dataset_X[train_size:len(dataset_X)+1,:]
    trainY, testY = dataset_Y[0:train_size,:], dataset_Y[train_size:len(dataset_X)+1,:]
    Coords = Coords[train_size:len(dataset_X)+1,:]

    # Fit the data to be between 0 and 1
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(trainX)
    trainX = scaler_X.transform(trainX)
    testX = scaler_X.transform(testX)

    # Do the same to the other dataset
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    scaler_Y.fit(trainY)
    trainY = scaler_Y.transform(trainY)
    testY = scaler_Y.transform(testY)
    
    return trainX, trainY, testX, testY, scaler_X, scaler_Y, Coords



def compile_model(network, trainX, trainY):
    '''
    "Create" the model
    '''
    nb_neurons=[]
    activation=[]
    
    # Get our network parameters.
    nb_layers             =network['nb_layers']
    lstms                 =network['lstms']
    optimizer             =network['optimizer']

    # Define the first LSTM
    implementation1       =network['implementation1']
    units1                =network['units1']
    lstm_activation1      =network['lstm_activation1']
    recurrent_activation1 =network['recurrent_activation1']
    # Define the second LSTM, if it exists
    implementation2       =network['implementation2']
    units2                =network['units2']
    lstm_activation2      =network['lstm_activation2']
    recurrent_activation2 =network['recurrent_activation2']
    
    # Number of neurons and each respective activation function for each extant layer
    nb_neurons.append(network['nb_neurons1'])
    nb_neurons.append(network['nb_neurons2'])
    nb_neurons.append(network['nb_neurons3'])
    nb_neurons.append(network['nb_neurons4'])
    nb_neurons.append(network['nb_neurons5'])
    activation.append(network['activation1'])  
    activation.append(network['activation2'])
    activation.append(network['activation3'])
    activation.append(network['activation4'])
    activation.append(network['activation5'])
    
    # Begin defining the model
    model = Sequential()
    
    # Add each LSTM to the model
    if lstms==1:
        model.add(LSTM(units1, input_shape=g_input_shape, activation=lstm_activation1, recurrent_activation=recurrent_activation1, implementation=implementation1))
    elif lstms==2:
        model.add(LSTM(units1, input_shape=g_input_shape, activation=lstm_activation1, recurrent_activation=recurrent_activation1, implementation=implementation1, return_sequences=True))
        model.add(LSTM(units2, activation=lstm_activation2, recurrent_activation=recurrent_activation2, implementation=implementation2))

    # Add the neurons to each layer
    for i in range(nb_layers):
        model.add(Dense(nb_neurons[i], activation=activation[i]))  
    model.add(Dense(2))

    # Compile the model
    model.compile(loss=g_loss_function, optimizer=optimizer)  
    trainX=trainX.reshape(len(trainX),1,2)

    # Train the model with the given dataset
    history = model.fit(trainX, trainY, epochs=g_epochs, batch_size=g_batch_size, verbose=g_myverbose, callbacks=[g_earlyStopper])#, validation_split=0.15)

    return model, lstms, history
#TODO: Doubts about this function: What does the number of neurons affect? What does return_sequencies do? What does the batch_size do?



def evaluate(model, testX, testY, scaler_X, scaler_Y, Coords, bestScore, history):
    '''
    Evaluates the quality of the current network
    '''
    if not isinstance(model.get_layer(index=0), keras.layers.Dense):
        testX=testX.reshape(len(testX),1,2)
    
    # Inverse the MinMaxScaler, which means that we are getting the data from 0 to 1 back to their original values
    testPredict = scaler_Y.inverse_transform(model.predict(testX))
    testY = scaler_Y.inverse_transform(testY)

    # Initialize the test score and prediction variables for the evaluation
    testScore=0
    predictions=0

    for i in range(len(Coords)-1):
        R       = 6378.1                    # Radius of the Earth in km

        d       = testY[i,0]                # Distance in km
        brng    = testY[i,1]*(math.pi/180)  # Bearing is 90 degrees converted to radians

        # In the following lines, we have the predicted Distance/Bearing, and the current position
        # With the predicted Distance/Bearing, we can predict the next point in order to compare with the actual next position (Coords[i+1])
        dPred    = testPredict[i,0]               # Distance in km
        brngPred = testPredict[i,1]*(math.pi/180) # Bearing is 90 degrees converted to radians
        assert brngPred == math.radians(testPredict[i,1])

        lat1 = math.radians(Coords[i,0])  # Current lat point converted to radians
        lon1 = math.radians(Coords[i,1])  # Current long point converted to radians

        lat2 = math.asin( math.sin(lat1)*math.cos(dPred/R) + math.cos(lat1)*math.sin(dPred/R)*math.cos(brngPred))                      # Terminal Latitude
        lon2 = lon1 + math.atan2(math.sin(brngPred)*math.sin(dPred/R)*math.cos(lat1), math.cos(dPred/R)-math.sin(lat1)*math.sin(lat2)) # Terminal Longitude

        # The actual true value
        true   = (Coords[i+1,0],Coords[i+1,1])
        # The prediction
        mypred = (math.degrees(lat2),math.degrees(lon2))
        
        # Determine the test score by getting the distance between the true coordinates and the predicted coordinates
        testScore   += geopy.distance.geodesic(true, mypred).km
        
        predictions =  predictions+1
    testScore   =  testScore/predictions
    #mse = mean_squared_error(true,mypred)
    #print("MSE: ", mse)

    if bestScore>testScore:
        bestScore=testScore
        model.save(g_savePath)
        with open(g_xScalerSavePath, 'wb') as xScaleFile:
            joblib.dump(scaler_X, xScaleFile)
        with open(g_yScalerSavePath, 'wb') as yScaleFile:
            joblib.dump(scaler_Y, yScaleFile)
        with open(g_fitHistoryFile, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        
    return testScore, bestScore
# Check https://stackoverflow.com/questions/877524/calculating-coordinates-given-a-bearing-and-a-distance
# and https://stackoverflow.com/questions/7222382/get-lat-long-given-current-point-distance-and-bearing



def testing_predictions():
    
    ## Atempt #1
    #model = keras.models.load_model("ModelsOutput/TrainedModel.keras") #"C:\\TMDEI\\testing_lstm\\NEWbestLSTMnetworkGenetic.h5")
    #np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    #
    ## Transform matrix to Distance Bearing and reshape it for the model's shape
    #pathToPredictionData = "TestingData.csv"
    #transformedPredictionDataset = transform_dataset(pathToPredictionData)
    #transformedPredictionDataset = transformedPredictionDataset.reshape(-1, 1, 2)
    #print("Transformed Dataset:\n", transformedPredictionDataset)
    #
    ## Inference
    #predictions = model.predict(transformedPredictionDataset)
    #
    #print(model.summary())


    ## Atempt #2 (With MinMaxScaler)
    #model = keras.models.load_model("ModelsOutput/TrainedModel.keras")
    #np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    #
    ## Transform matrix to Distance Bearing and reshape it for the model's shape
    #pathToPredictionData = "TestingData.csv"
    #transformedPredictionDataset = transform_dataset(pathToPredictionData)
    #transformedPredictionDataset = np.delete(transformedPredictionDataset, -1, 0)
    #print("Correct:\n", transformedPredictionDataset)
    #
    #
    #scaler_X = MinMaxScaler(feature_range=(0, 1))
    #scaler_X.fit(transformedPredictionDataset)
    #
    #transformedPredictionDataset = scaler_X.transform(transformedPredictionDataset)
    #
    #if not isinstance(model.get_layer(index=0), keras.layers.Dense):
    #    transformedPredictionDataset=transformedPredictionDataset.reshape(len(transformedPredictionDataset),1,2) 
    #
    #testPredict = scaler_X.inverse_transform(model.predict(transformedPredictionDataset))
    #
    #print("Inference:\n", testPredict)
    #print(model.summary())


    # Atempt #3 (With training MinMaxScaler)
    model = keras.models.load_model(g_model_to_be_loaded)
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})

    # Transform matrix to Distance Bearing and reshape it for the model's shape
    pathToPredictionData = g_TestingDataPath
    transformedPredictionDataset = transform_dataset(pathToPredictionData)
    transformedPredictionDataset = np.delete(transformedPredictionDataset, -1, 0)
    #print("Correct:\n", transformedPredictionDataset)
    
    # Load the saved scalers
    scaler_X = joblib.load(g_scalerXFilename)
    scaler_Y = joblib.load(g_scalerYFilename)
    
    # Transform the predicted output
    transformedPredictionDataset = scaler_X.transform(transformedPredictionDataset)
    
    if not isinstance(model.get_layer(index=0), keras.layers.Dense):
        transformedPredictionDataset=transformedPredictionDataset.reshape(len(transformedPredictionDataset),1,2)
    
    # Inverse transform the predicted results for printing
    testPredict = scaler_Y.inverse_transform(model.predict(transformedPredictionDataset))
    #print("Inference:\n", testPredict)

    # Reverse Conversion to Lat/Lon
    # Original dataset coordinates
    Coords = pd.read_csv(g_TestingDataPath, engine='python').values.astype('float32')
    # Pushes the first element of the array to last and every other element goes up 1 position, ex: [0,1,2,3,4] -> [1,2,3,4,0]
    Coords=np.roll(Coords, -1, axis=0)
    for i in range(len(Coords)-2):
        R = 6378.1 # Radius of the Earth in km

        # In the following lines, we have the predicted Distance/Bearing, and the current position
        # With the predicted Distance/Bearing, we can predict the next point in order to compare with the actual next position (Coords[i+1])
        dPred    = testPredict[i,0]               # Distance in km
        brngPred = testPredict[i,1]*(math.pi/180) # Bearing is 90 degrees converted to radians
        assert brngPred == math.radians(testPredict[i,1])

        lat1 = math.radians(Coords[i,0])  # Current lat point converted to radians
        lon1 = math.radians(Coords[i,1])  # Current long point converted to radians

        lat2 = math.asin( math.sin(lat1)*math.cos(dPred/R) + math.cos(lat1)*math.sin(dPred/R)*math.cos(brngPred))                      # Terminal Latitude
        lon2 = lon1 + math.atan2(math.sin(brngPred)*math.sin(dPred/R)*math.cos(lat1), math.cos(dPred/R)-math.sin(lat1)*math.sin(lat2)) # Terminal Longitude

        # The actual true value
        true   = (Coords[i+1,0],Coords[i+1,1])
        # The prediction
        mypred = (round(math.degrees(lat2), 7),round(math.degrees(lon2),7))

        print("Correct:", true, "\tInference:", mypred, "\tDistance between: ", round(geopy.distance.geodesic(true, mypred).m, 2), "meters")
