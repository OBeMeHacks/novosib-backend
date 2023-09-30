import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .timeseries_train import train_models, predict 


def preprocess(contrib_cleaned):
    contrib_cleaned['accnt_pnsn_schm'] = contrib_cleaned.accnt_pnsn_schm.astype(float).fillna(
        contrib_cleaned.accnt_pnsn_schm.mode()[0]
        ).astype(int).astype('category')
    blnc_filter = (((contrib_cleaned.npo_blnc.isna()) | (contrib_cleaned.npo_blnc < 0)) & (contrib_cleaned.npo_accnt_status == 1))
    contrib_cleaned.loc[blnc_filter, 'npo_blnc'] = contrib_cleaned.loc[blnc_filter, 'npo_pmnts_sum'].fillna(0)
    contrib_cleaned.loc[contrib_cleaned.npo_accnt_status == 0, 'npo_blnc'] = np.nan
    contrib_cleaned.loc[contrib_cleaned.npo_pmnts_sum < 0, 'npo_pmnts_sum'] = np.nan
    contrib_cleaned.loc[contrib_cleaned.npo_ttl_incm < 0, 'npo_ttl_incm'] = 0
    contrib_cleaned['npo_ttl_incm'] = contrib_cleaned.npo_ttl_incm.fillna(0)
    contrib_cleaned.dropna(subset=['npo_pmnts_sum'], inplace=True)
    contrib_cleaned.reset_index(drop=True, inplace=True)
    return contrib_cleaned


def extract_features(path_to_data):
    data = pd.read_csv(path_to_data)
    contributors = preprocess(data)

    contributors['date'] =  pd.to_datetime(contributors['npo_accnt_status_date']).dt.to_period('Q')
    contributors['npo_accnt_status_date'] = pd.to_datetime(contributors['npo_accnt_status_date'])
    contributors['npo_lst_pmnt_date'] = pd.to_datetime(contributors['npo_lst_pmnt_date'])
    contributors['npo_frst_pmnt_date'] = pd.to_datetime(contributors['npo_frst_pmnt_date'])
    
    features_for_ts = contributors.groupby(["clnt_id", "date"]).agg(
        {
            'npo_accnt_id' : 'nunique',
            'npo_accnt_status_date' : 'max',
            'npo_pmnts_sum' : 'sum',
            'npo_pmnts_nmbr' : 'sum',
            'npo_frst_pmnt_date' : 'min',
            'npo_lst_pmnt_date' : 'max',
            'npo_ttl_incm' : 'sum',
        }
    )
    return features_for_ts


def get_target(path_to_target):
    target = pd.read_csv(path_to_target)
    return target

class ClientFeatureExtractor:
    def __init__(self, path_to_data):
        self.data = extract_features_table_from_contributors(path_to_data)
        self.data_by_user = {
            client_id: self.extract_user_table_(data, client_id) 
            for client_id in self.all_client_ids
            }
        self.features_next_quarter = dict()
        self.feature_columns = df.columns
        for client_id, df in self.data.items():
            result = dict()
            for column in df.columns:
                ts = df[column]
                result[column] = predict(ts, train_models(ts)[0])[0]
            self.features_next_quarter[client_id] = result


    def extract_user_table_(self, data, client_id):
        data[]
    
    def get_features_values(self, client_id):
        return self.features_next_quarter[client_id]


class Model:

    def __init__(self,
                 path_to_external_data,
                 path_to_transactions_table,
                 path_to_contributors_table):
        self.client_feature_extractor = ClientFeatureExtractor(path_to_contributors_table)
        self.data_train = pd.read_csv(path_to_external_data)
        self.target = get_target(path_to_transactions_table)
        self.model = CatBoostRegressor().fit(
            self.data_train,
            self.target,
        )


    


    def predict(self, client_id, override_features):
        features = self.client_feature_extractor.get_features_values[client_id]
        features.update(override_features)
        return self.model.predict(features)


    



    



