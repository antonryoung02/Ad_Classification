import pandas as pd
import os

class MetadataDataframe:
    def __init__(self, filepath):
        self.filepath = filepath
        if os.path.exists(self.filepath) and os.stat(self.filepath).st_size != 0:
            self.dataframe = pd.read_csv(self.filepath)
        else:
            required_keys = ['id', 'url', 'attribution', 'filepath', 'timestamp']
            self.dataframe = pd.DataFrame(columns=required_keys)
            self.dataframe.to_csv(self.filepath, index=False)
    
    def insert(self, data_dict):
        required_keys = {'id', 'url', 'attribution', 'filepath', 'timestamp'}
        if not required_keys <= data_dict.keys():
            missing_keys = required_keys - data_dict.keys()
            raise ValueError(f"Missing keys: {missing_keys}")

        if 'filepath' in data_dict and data_dict['filepath'] in self.dataframe['filepath'].values:
            print("removing duplicate image")
            self.dataframe = self.dataframe[self.dataframe['filepath'] != data_dict['filepath']]

        new_row = pd.DataFrame([data_dict])
        self.dataframe = pd.concat([self.dataframe, new_row], ignore_index=True)
    
    def delete(self, file_path):
        print(f"filepath {file_path}")
        self.dataframe = self.dataframe[self.dataframe['filepath'] != file_path]
    
    def save(self):
        self.dataframe.to_csv(self.filepath, index=False)
        
        