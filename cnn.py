#import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from ledger import Ledger
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from keras_tuner.tuners import RandomSearch
import keras_tuner 
from keras_tuner.engine.hyperparameters import HyperParameter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,Dropout,Dense,MaxPooling2D,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
from PIL import Image
from sklearn.metrics import classification_report



class MultiLabelClassifier():

    def __init__(self):

        self.workdir = r"models"
        self.train_output_file = os.path.join(self.workdir,f"train_output.json")
        self.labels = ['Bar', 'Club', 'Outdoor', 'Warehouse']
        self.file_type = "mfcc_image_file"
        self.ledger = Ledger()

        if os.path.exists(self.train_output_file):
            with open(self.train_output_file,"r") as json_file:
                self.train_output =  json.load(json_file)

            self.model_file = self.train_output['model_file']
            self.model_name = self.model_file.split('\\')[-1]
           
            self.model = tf.keras.models.load_model(self.model_file)
            # and use the labels from training outputs


    def load_data(self):

        data = self.ledger.get.labelled_data()
        X = []
        y = []
        for row in tqdm(data.index):
            image_file = data.loc[row,self.file_type]
            current_labels = json.loads(data.loc[row,'current_labels'].replace("'",'"'))
            lbs = []
            for lb in current_labels.values():
                lbs += list(lb)

            label_idxs = [self.labels.index(i) for i in lbs]
            label_idxs= [1 if i in label_idxs else 0 for i in range(len(self.labels))]

            if os.path.exists(image_file):
                image = Image.open(image_file).convert("L")
                image_resized = image.resize((180,180))
                y.append(np.asarray(label_idxs))
                X.append(np.asarray(image_resized))
            

        return np.array(X), np.array(y)

    def prepare_datasets(self, test_size):
            """Loads data and splits it into train and test sets.

            :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
            
            :return X_train (ndarray): Input training set
            :return X_test (ndarray): Input test set
            :return y_train (ndarray): Target training set
            :return y_test (ndarray): Target test set
            """

            # load data
            X, y, = self.load_data()

            # create train, validation and test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
            # add an axis to input sets
            X_train = X_train[..., np.newaxis]
            X_test = X_test[..., np.newaxis]

            return X_train, X_test, y_train, y_test


    def build_model(self, hp):  # random search passes this hyperparameter() object 

        hp_dropout_rate=hp.Choice("dropout_rate", values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        hp_dense_dropout_rate=hp.Choice("dropout_rate", values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

        hp_dense_layers = hp.Int("dense_layers", 0, 1)

        model = Sequential()

        model.add(Conv2D(hp.Int('input_units',
                                min_value=32,
                                max_value=128,
                                step=32), (3, 3), input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(hp_dropout_rate))


        for i in range(hp.Int('n_layers', 1, 4)):  # adding variation of layers.
            model.add(Conv2D(hp.Int(f'conv_{i}_units',
                                    min_value=32,
                                    max_value=128,
                                    step=32), (3, 3), input_shape=input_shape, activation='relu'))
            model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
            model.add(BatchNormalization())
            model.add(Dropout(hp_dropout_rate))

        model.add(Flatten())
        for i in range(hp_dense_layers):
            model.add(Dense(hp.Int(f'dense_{i}_units',
                            min_value=64,
                            max_value=512,
                            step=32), activation='relu'))
            model.add(Dropout(hp_dense_dropout_rate))

                
        # output layer
        model.add(Dense(len(self.labels), activation='sigmoid'))


        optimiser = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser,
                    loss='binary_crossentropy',
                    metrics=['AUC'])

        return model

    def train(self):


        # get train, validation, test splits
        X_train, X_test, y_train, y_test = self.prepare_datasets(0.25)

        # create network
        global input_shape
        input_shape = (X_train.shape[1], X_train.shape[2], 1)
        

        LOG_DIR = "models\logs"

        tensorboard = TensorBoard(log_dir=LOG_DIR)

        tuner = RandomSearch(
            self.build_model,
            objective=keras_tuner.Objective('val_loss',"min"),
            max_trials=40,  # how many model variations to test?
            executions_per_trial=2,  # how many trials per variation? (same model could perform differently)
            directory=LOG_DIR,
            overwrite=True
            )

        tuner.search_space_summary()

        # Early stopping based on minimum val_loss if no imporovement after 5 epochs

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        tuner.search(x=X_train,
                    y=y_train,
                    epochs=100,
                    batch_size=16,
                    callbacks=[tensorboard, es],
                    validation_data=(X_test, y_test))

        tuner.results_summary()

        bestModel = tuner.get_best_models(num_models=1)[0]

        model_num = len([file for file in os.listdir(self.workdir) if file.startswith(f"tagging_model")]) + 1
        model_name = f"tagging_model_{model_num}"
        model_file = f"{self.workdir}\{model_name}"

        train_output = {
            'labels': self.labels,
            'model_file': model_file
        }

        with open(self.train_output_file,"w") as json_file:
            json_file.write(json.dumps(train_output,indent=4, sort_keys=True))

        bestModel.save(model_file)

        self.model = bestModel

    def evaluation(self, save=False):

        _, X_test, _, y_test = self.prepare_datasets(0.25)

        y_pred = self.model.predict(X_test)

        cr = classification_report(y_test, 
                           np.rint(y_pred),
                           output_dict=True, 
                           target_names=self.labels, 
                           zero_division=0)

        cr = pd.DataFrame(cr).T

        cr['percentage_support'] = cr['support']/cr.iloc[-1,-1]

        if save:
            cr.to_csv("CNN_classification_report.csv")

        return cr

if __name__ == "__main__":
    multi_label_classifier = MultiLabelClassifier()
    multi_label_classifier.train()
    cr = multi_label_classifier.evaluation(save=True)
    print(cr)

    
    