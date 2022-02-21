import sqlite3
import pandas as pd

class Ledger(object):

    def __init__(self):
        
        self.db = sqlite3.connect('ledger.db')
        self.database = self.db.cursor()
        self.get = Get(database=self.database, db=self.db)
        self.insert = Insert(database=self.database, db=self.db)
        self.update = Update(database=self.database, db=self.db)
        self.delete = Delete(database=self.database, db=self.db)

class Get(Ledger):
    def __init__(self, database, db):
        super(Ledger, self).__init__()
        self.database = database
        self.db = db

    def table(self, query, params=[]):
        recs = self.database.execute(query,params).fetchall()
        columns = [i[0] for i in self.database.description]

        return pd.DataFrame(recs,columns=columns)

    def to_dict(self, query, params=[]):
        recs = self.database.execute(query,params).fetchall()[0]
        columns = [i[0] for i in self.database.description]

        return dict(zip(columns,recs))

    def execute(self, query, params=[]):
        recs = self.database.execute(query,params).fetchall()
        columns = [i[0] for i in self.database.description]
        if len(recs) == 1:
            return dict(zip(columns,recs[0]))
        else:
            return pd.DataFrame(recs,columns=columns)
        

    def item(self, id=None, ids=None):
        
        if ids:
            batch_size = 500
            batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
            response = []
            for i, batch in enumerate(batches):
                query = f"""
            
                SELECT * FROM LEDGER WHERE id in ({','.join(['?']*len(batch))})

                """

                params = batch
                response += self.database.execute(query,params).fetchall()
            columns = [i[0] for i in self.database.description]
            return pd.DataFrame(response,columns=columns)

        else:
            query = """
        
            SELECT * FROM LEDGER WHERE ID = ?

            """
            params = [id]
            self.execute(query,params)


    def not_items(self, ids):
        if ids:
            batch_size = 500
            batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
            response = []
            for i, batch in enumerate(batches):
                query = f"""
            
                SELECT * FROM LEDGER WHERE id NOT IN ({','.join(['?']*len(batch))})

                """

                params = batch
                response += self.database.execute(query,params).fetchall()
            columns = [i[0] for i in self.database.description]
            return pd.DataFrame(response,columns=columns)

    def missing_features(self):

        query = f"""

        SELECT * 
        FROM LEDGER 
        WHERE mfcc_npy_file IS NULL
        OR mfcc_image_file IS NULL
        OR mel_spec_npy_file IS NULL
        OR mel_spec_image_file IS NULL
        
        """

        
                        
        return self.execute(query)



    def all(self):
        query = """
        
        SELECT * FROM LEDGER

        """

        return self.execute(query)

    def most_recent_file(self):
        query = """
        
        SELECT file_created FROM LEDGER ORDER BY file_created DESC LIMIT 1

        """

        return self.execute(query)

    def annoy_index(self, annoy_index=None, annoy_indexs=None):

        if annoy_indexs:
            batch_size = 500
            batches = [annoy_indexs[i:i + batch_size] for i in range(0, len(annoy_indexs), batch_size)]
            response = []
            for batch in batches:
                query = f"""
            
                SELECT * FROM LEDGER WHERE annoy_index in ({','.join(['?']*len(batch))})

                """

                params = batch
                response += self.database.execute(query,params).fetchall()
            
            columns = [i[0] for i in self.database.description]
            return pd.DataFrame(response,columns=columns)

        elif not annoy_index:
            query = """
            
                SELECT * FROM LEDGER WHERE annoy_index IS NULL  

                """

                           
            return self.execute(query)


        else:
            query = """
            
            SELECT * FROM LEDGER WHERE annoy_index = ?

            """

            params = [annoy_index]
            self.execute(query,params)
            

    def next_annoy_index(self):

        query = """
            
            SELECT annoy_index FROM LEDGER ORDER BY annoy_index desc limit 1

            """

    
            
        current_index =  self.execute(query)['annoy_index']

        if current_index:
            return current_index + 1
        else:
            return 1

    def labelled_data(self):

        query = """
            
            SELECT * FROM LEDGER WHERE current_labels IS NOT NULL

            """
            
        return self.execute(query)

    def label_studio_id(self, label_studio_id=None, label_studio_ids=None):

        if label_studio_ids:
            batch_size = 500
            batches = [label_studio_ids[i:i + batch_size] for i in range(0, len(label_studio_ids), batch_size)]
            response = []
            for batch in batches:
                query = f"""
            
                SELECT * FROM LEDGER WHERE label_studio_id in ({','.join(['?']*len(batch))})

                """

                params = batch
                response += self.database.execute(query,params).fetchall()
            columns = [i[0] for i in self.database.description]
            return pd.DataFrame(response,columns=columns)

        elif not label_studio_id:
            query = """
            
                SELECT * FROM LEDGER WHERE label_studio_id IS NULL  

                """

                           
            return self.execute(query)

        else:
            query = """
            
            SELECT * FROM LEDGER WHERE label_studio_id = ?

            """

            params = [label_studio_id]
            self.execute(query,params)




    

 

class Insert(Ledger):

    def __init__(self, database, db):
        super(Ledger, self).__init__()
        self.database = database
        self.db = db

    def item(self, id, beatport_id=None, sample_url=None, mfcc_npy_file=None,
            mfcc_image_file=None, mel_spec_npy_file=None, mel_spec_image_file=None, 
            file_created=None, annoy_index=None, label_studio_id=None,
            label_studio_insert = None):

        query = """
        
        INSERT INTO LEDGER (id, annoy_index, label_studio_id, beatport_id, sample_url,
                            mfcc_npy_file, mfcc_image_file, mel_spec_npy_file, 
                            mel_spec_image_file, file_created, label_studio_insert)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """

        params = [id, annoy_index, label_studio_id, beatport_id, sample_url, mfcc_npy_file, mfcc_image_file, 
                  mel_spec_npy_file, mel_spec_image_file, file_created, label_studio_insert]
        self.database.execute(query,params)
        self.db.commit()

class Update(Ledger):

    def __init__(self, database, db):
        super(Ledger, self).__init__()
        self.database = database
        self.db = db

    def beatport_id(self, id, beatport_id):

        query = """
        
        UPDATE LEDGER SET beatport_id = ? WHERE ID = ?

        """

        params = [beatport_id, id]

        self.database.execute(query,params)
        self.db.commit()

    def sample_url(self, id, sample_url):

        query = """
        
        UPDATE LEDGER SET sample_url = ? WHERE ID = ?

        """

        params = [sample_url, id]
        self.database.execute(query,params)
        self.db.commit()


    def mfcc_npy_file(self, id, mfcc_npy_file):

        query = """
        
        UPDATE LEDGER SET mfcc_npy_file = ? WHERE ID = ?

        """

        params = [mfcc_npy_file, id]

        self.database.execute(query,params)
        self.db.commit()


    def mfcc_image_file(self, id, mfcc_image_file):

        query = """
        
        UPDATE LEDGER SET mfcc_image_file = ? WHERE ID = ?

        """

        params = [mfcc_image_file, id]

        self.database.execute(query,params)
        self.db.commit()


    def mel_spec_npy_file(self, id, mel_spec_npy_file):

        query = """
        
        UPDATE LEDGER SET mel_spec_npy_file = ? WHERE ID = ?

        """

        params = [mel_spec_npy_file, id]

        self.database.execute(query,params)
        self.db.commit()


    def mel_spec_image_file(self, id, mel_spec_image_file):

        query = """
        
        UPDATE LEDGER SET mel_spec_image_file = ? WHERE ID = ?

        """

        params = [mel_spec_image_file, id]

        self.database.execute(query,params)
        self.db.commit()


    def label_studio_id(self, id, label_studio_id):

        query = """
        
        UPDATE LEDGER SET label_studio_id = ? WHERE ID = ?

        """

        params = [label_studio_id, id]

        self.database.execute(query,params)
        self.db.commit()


    def annoy_index(self, id, annoy_index):

        query = """
        
        UPDATE LEDGER SET annoy_index = ? WHERE ID = ?

        """

        params = [annoy_index, id]

        self.database.execute(query,params)
        self.db.commit()


    def current_labels(self, id, current_labels, updated_at):

        query = """
        
        UPDATE LEDGER SET current_labels = ?, label_updated_at = ?  WHERE ID = ?

        """

        params = [current_labels, updated_at, id]

        self.database.execute(query,params)
        self.db.commit()


    def field(self, id, field, value):

        query = f"""
        
        UPDATE LEDGER SET {field} = ? WHERE ID = ?

        """

        params = [value, id]

        self.database.execute(query,params)
        self.db.commit()


class Delete(Ledger):

    def __init__(self, database, db):
        super(Ledger, self).__init__()
        self.database = database
        self.db = db

    def item(self, id):

        query = """
        
        DELETE FROM LEDGER WHERE ID = ?

        """

        params = [id]

        self.database.execute(query,params)
        self.db.commit()


    def label_studio_id(self, id):

        query = """
        
        UPDATE LEDGER SET label_studio_id = ? WHERE ID = ?

        """

        params = [None, id]

        self.database.execute(query,params)
        self.db.commit()


    def annoy_index(self, id):

        query = """
        
        UPDATE LEDGER SET annoy_index = ? WHERE ID = ?

        """

        params = [None, id]
        self.database.execute(query,params)
        self.db.commit()


    def all_annoy_index(self):
        query = """
        
        UPDATE LEDGER SET annoy_index = ?

        """

        params = [None]
        self.database.execute(query,params)
        self.db.commit()


