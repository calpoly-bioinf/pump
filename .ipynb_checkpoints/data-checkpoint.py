import pandas as pd

sample_metadata_url = "https://drive.google.com/file/d/1J5lnmAVUbMXQuODLdFF19Bl_YAqPzAXj/view?usp=sharing"

# metabric urls
sample_expression_urls=[
    "https://drive.google.com/file/d/10QpYL1ZWW7ljqsCSeXKX9ZO30B73uHuM/view?usp=share_link",
    "https://drive.google.com/file/d/1JE0KkTx9mSSrdfNwISiEXX2aLSq-GmrJ/view?usp=share_link",
    "https://drive.google.com/file/d/18tgo-o_-qPCVSjuBEPVrbphrA-Hbi4D9/view?usp=share_link",
    "https://drive.google.com/file/d/1SYa5E5UU6a8H6dY4yrk3QDfHKRQOLigm/view?usp=share_link",
    "https://drive.google.com/file/d/1627jeYR6m1m33qnFxvgeAezp91GiR6PM/view?usp=share_link"
    "https://drive.google.com/file/d/1ydv5JLmtZ-CjZuE0raaSITwOwvN5Hzql/view?usp=share_link",
    "https://drive.google.com/file/d/1aKkviqXT2-zDCvEFCwut_ixE1w5E54iO/view?usp=share_link",
    "https://drive.google.com/file/d/1lQvm8dO6QkM8Rtl3q54YLV8rgVyDuwQu/view?usp=share_link",
    "https://drive.google.com/file/d/15EbQti2sF-TzD606fkUCw4B8gII1oB_p/view?usp=share_link",
    "https://drive.google.com/file/d/1XPu2BylAMoQP7olPDCQX6Y9jBbXb5g-S/view?usp=share_link",
    "https://drive.google.com/file/d/1ZeBGGn3YqfSqweWQqcW1gKXBj2kY_ELQ/view?usp=share_link",
    "https://drive.google.com/file/d/1J_dHQOsegjmN_nylscCVSHCZeyZWFLoi/view?usp=share_link",
    "https://drive.google.com/file/d/1LE4F7OIJrRJIsavvZKQVMt41Q2Vs_5WY/view?usp=share_link",
    "https://drive.google.com/file/d/12unnzNfBqQ-WB1vPVD-fM_LRPd8uhH7_/view?usp=share_link",
    "https://drive.google.com/file/d/1ke2hviV3WY81X-WoZg2r1TCOXehpgo-l/view?usp=share_link",
    "https://drive.google.com/file/d/1iDcHtSMJY2e3w7oufEh1yCltWp7sbAr3/view?usp=share_link",
    "https://drive.google.com/file/d/1AjMMdh_5UIojAoTUBL9KF3JoCpncLU_7/view?usp=share_link",
    "https://drive.google.com/file/d/1V8YXJstjrQv6p7zTwzuFU42V8kL6utfT/view?usp=share_link",
    "https://drive.google.com/file/d/1NdWS49_No9unD9Rm7bWr8JVJ14NGSgbO/view?usp=share_link",
    "https://drive.google.com/file/d/1llGTAWM8vuirS_TX-Hvv4mq7snYrJvAl/view?usp=share_link",
    "https://drive.google.com/file/d/1xsL-fEq0PXw_B_jEAVkf4HkMk-m0BA5f/view?usp=share_link",
    "https://drive.google.com/file/d/1zm-IhmDpziYwM8ceh2daMQkatSmE2Xv7/view?usp=share_link"
]

def load_data_from_urls(urls,index_col_name=None,sep=","):
    Xs = []
    columns = None
    for url in urls:
        url='https://drive.google.com/uc?id=' + url.split('/')[-2]
        if columns is None:
            X = pd.read_csv(url,sep=sep)
            columns = X.columns
        else:
            X = pd.read_csv(url,header=None)
            X.columns = columns
        if index_col_name is not None:
            X.set_index(index_col_name,inplace=True)
        Xs.append(X)

    X = pd.concat(Xs, axis=0)
    return X

def create_small_metabric_session():
    X = load_data_from_urls([sample_expression_urls[0]],index_col_name='Sample ID')
    metadata = load_data_from_urls([sample_metadata_url],sep="\t",index_col_name='Sample ID')
    target_column = 'Pam50 + Claudin-low subtype'
    y = metadata[[target_column]]
    Xy = X.join(y)
    sess = Session(Xy)
    sess.target_column = 'Pam50 + Claudin-low subtype'
    return sess

class Session:
    def __init__(self,source_df):
        self.source_df = source_df
        
    def _init_Xy(self):
        source_df_ = self.source_df.dropna()
        X = source_df_.drop([self._target_column], axis=1)
        y = source_df_[self._target_column]
        self._X = X
        self._y = y
        
    @property
    def target_column(self):
        return self._target_column

    @target_column.setter
    def target_column(self, tc):
        if tc in self.source_df.columns:
            self._target_column = tc
            self._init_Xy()
        else:
            raise Exception(f"Target column {tc} is not in source_df.columns")
            
    def X(self):
        # todo add checks
        return self._X
    
    def y(self):
        # todo add checks
        return self._y