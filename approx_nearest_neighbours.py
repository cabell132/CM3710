import numpy as np
import os
from tqdm import tqdm
from annoy import AnnoyIndex
import json
import pandas as pd
from ledger import Ledger
from PIL import Image
from sklearn.metrics import classification_report

class Approx_NN(object):

    def __init__(self):
        self.trees = 1000
        self.ledger = Ledger()
        self.dims = 32400
        self.file_type = "mfcc_image_file"
        self.labels = ['Bar', 'Club', 'Outdoor', 'Warehouse']


    def build(self, metric):
        # clear old index

        data = self.ledger.get.labelled_data()

        metrics = ["angular", "euclidean", "manhattan", "hamming", "dot"]
        
        t = AnnoyIndex(self.dims, metric=metric)

        print("adding items to model")
        for i in tqdm(data.index):
            id = data.loc[i,'id']
            image_file = data.loc[i,self.file_type]
            image = Image.open(image_file).convert("L")
            image_resized = image.resize((180,180))
                             
            file_vector = np.matrix.flatten(np.asarray(image_resized))

            if file_vector.shape[0] == self.dims:
                t.add_item(i, file_vector)
                self.ledger.update.annoy_index(id=id,annoy_index=int(i))


        t.build(self.trees)
        filename = f"models\{metric}_nearest_neighbors.ann"
        t.save(filename)
 

    def evaluation(self, metric="euclidean", save=False):

        """Function to evaluate approximate nearest neighbour model

        
        """
        
        def get_labels(text):
            dic = json.loads(text.replace("'",'"'))
            lbs = []
            for lb in dic.values():
                lbs += list(lb)

            label_idxs = [self.labels.index(i) for i in lbs]
            label_idxs= [1 if i in label_idxs else 0 for i in range(len(self.labels))]
            return [i for i in label_idxs]

        df = self.ledger.get.labelled_data()
        df['y'] = df['current_labels'].apply(get_labels)

 


        u = AnnoyIndex(32400, metric=metric)
        u.load(f'models\{metric}_nearest_neighbors.ann') # super fast, will just mmap the file
        y_pred = []
        for row in tqdm(df.index):
            image_file = df.loc[row,self.file_type]
            annoy_index = df.loc[row,'annoy_index']

            if os.path.exists(image_file):
                image = Image.open(image_file).convert("L")
                image_resized = image.resize((180,180))
                vector = np.matrix.flatten(np.asarray(image_resized))

                NN = u.get_nns_by_vector(vector,10,include_distances=True)
                

                NN_df = pd.DataFrame(NN,index=["annoy_index","distance"]).T
                top = df.merge(NN_df).sort_values("distance")
                top = top.loc[top['annoy_index']!= annoy_index].reset_index(drop=True)
                y_pred.append(top.loc[0,'y'])
        u.unload()
        df['y_pred'] = y_pred

        cr = classification_report(df['y'].tolist(), 
                           df['y_pred'].tolist(),
                           output_dict=True, 
                           target_names=self.labels, 
                           zero_division=0)

        cr = pd.DataFrame(cr).T

        cr['percentage_support'] = cr['support']/cr.iloc[-1,-1]

        if save:
            cr.to_csv("ANN_classification_report.csv")

        return cr


    


if __name__ == "__main__":
    ann = Approx_NN()

    ann.build(metric='euclidean')

    cr = ann.evaluation(metric='euclidean', save=True)
    print(cr)