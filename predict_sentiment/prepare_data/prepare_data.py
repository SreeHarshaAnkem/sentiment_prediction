import re
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class PrepareData:
    def __init__(self, input_fields, text_field, 
                 target_field, vocab_size, 
                 maxlen, padding, tokenizer=None):
        self.input_fields = input_fields
        self.text_field = text_field
        self.target_field = target_field
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.padding = padding
        if tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        else:
            self.tokenizer = tokenizer
        
    def run_prep(self, input_json, split=False, test_size=0.2):
        ## 1. create dataframe
        df_reviews = self.create_dataframe(input_json)
        ## 2. clean data
        df_cleaned = self.clean_data(df_reviews)
        ## 3. preprocess text
        df_preprocessed = self.preprocess_data(df_cleaned)
        
        if split:
            train_X, train_y, test_X, test_y = train_test_split(df_preprocessed["padded_sequence"], 
                            df_preprocessed[self.target_field], test_size=test_size)
            return train_X, train_y, test_X, test_y
        
        elif self.target_field in df_preprocessed.columns.tolist():
            return df_preprocessed["padded_sequence"]
        else:
            return df_preprocessed["padded_sequence"], df_preprocessed[self.target_field]
        ## 3. create frequency features
        
    def create_dataframe(self, input_json):
        df = pd.DataFrame(input_json)
        return df
    
    def clean_data(self, df):
        df[self.text_field] = df[self.text_field].apply(lambda x: x.lower())
        df[self.text_field] = df[self.text_field].apply(lambda x: re.sub('[^\w+]',' ', x))
        df[self.text_field] = df[self.text_field].apply(lambda x: x.strip())
        return df

    def preprocess_data(self, df):
        texts = df[self.text_field].values.tolist()
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequence = pad_sequences(sequences, maxlen=self.maxlen, 
                                        padding=self.padding, value=self.vocab_size)
        if self.target_field in df.columns.tolist():
            df[self.target_field] = df[self.target_field].apply(lambda x: 1 if x=="positive" 
                                                                            else 0)
        df["padded_sequence"] = padded_sequence.tolist()
        return df