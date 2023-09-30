import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .timeseries_train import train_models, predict 


def preprocess_contributors(contributors):
    contributors['accnt_pnsn_schm'] = contributors.accnt_pnsn_schm.astype(float).fillna(
        contributors.accnt_pnsn_schm.mode()[0]
        ).astype(int).astype('category')
    blnc_filter = (((contributors.npo_blnc.isna()) | (contributors.npo_blnc < 0)) & (contributors.npo_accnt_status == 1))
    contributors.loc[blnc_filter, 'npo_blnc'] = contributors.loc[blnc_filter, 'npo_pmnts_sum'].fillna(0)
    contributors.loc[contributors.npo_accnt_status == 0, 'npo_blnc'] = np.nan
    contributors.loc[contributors.npo_pmnts_sum < 0, 'npo_pmnts_sum'] = np.nan
    contributors.loc[contributors.npo_ttl_incm < 0, 'npo_ttl_incm'] = 0
    contributors['npo_ttl_incm'] = contributors.npo_ttl_incm.fillna(0)
    contributors.dropna(subset=['npo_pmnts_sum'], inplace=True)
    contributors.reset_index(drop=True, inplace=True)
    return contributors


def extract_features_table_from_contributors(contributors):
    processed_contributors = contributors.copy()

    processed_contributors['date'] =  pd.to_datetime(processed_contributors['npo_accnt_status_date']).dt.to_period('Q')
    processed_contributors['npo_accnt_status_date'] = pd.to_datetime(processed_contributors['npo_accnt_status_date'])
    processed_contributors['npo_lst_pmnt_date'] = pd.to_datetime(processed_contributors['npo_lst_pmnt_date'])
    processed_contributors['npo_frst_pmnt_date'] = pd.to_datetime(processed_contributors['npo_frst_pmnt_date'])

    features = processed_contributors.groupby(["date"]).agg(
        {
            'npo_accnt_id' : 'nunique',
            'npo_accnt_status_date' : ['min', 'max'],
            'npo_pmnts_sum' : 'sum',
            'npo_pmnts_nmbr' : 'sum',
            'npo_ttl_incm' : 'sum',

        }
    )
    features = [
        'number_acounts',
        'min_payment_date',
        'max_payment_date',
        'payments_sum',
        'payments_count',
        'total_income']
    return features.sort_values(by='date', ascending=False)


def get_target(contributors, transactions):
    target = transactions.copy()
    mapper = contributors[['npo_accnt_id', 'clnt_id']]
    target = target.merge(mapper).drop(['npo_accnt_id'], axis=1)
    target.npo_operation_date = pd.to_datetime(target.npo_operation_date)
    target = target[(target.npo_sum > 0) & (target.npo_operation_group == 0)]
    target = target.groupby(['clnt_id', target['date'].dt.to_period('Q')]).agg(
        {'slctn_nmbr' : 'count', 'npo_sum':'sum'}).reset_index().rename(columns={'slctn_nmbr':'count_contributions', 
                                                                                 'npo_sum':'SUM'})
    target['mean_contribution'] = target['SUM'] / target['count_contributions']

    return target.drop(['SUM'], axis=1)


def prepare_dates(contribution, transaction):
    pass


class FeatureExtractor:
    def __init__(self, path_to_data):
        self.data = extract_features_table_from_contributors(path_to_data)
        self.features_next_quarter = dict()
        self.feature_columns = self.data.columns
        result = dict()
        for column in self.feature_columns:
            ts = self.data[column]
            result[column] = predict(ts, train_models(ts)[0])[0]
        self.features_next_quarter = result
    
    def get_features_values_next_quarter(self):
        return self.features_next_quarter
    
    def get_feature_values(self, quarter):
        return self.get_feature_values[quarter]
    
    def get_features(self):
        return self.get_feature_values


class Model:

    def __init__(self,
                 path_to_external_data,
                 path_to_clients_table,
                 path_to_contributors_table,
                 path_to_transactions_table
                 ):
        # self.data_train = pd.read_csv(path_to_external_data)
        transactions = pd.read_csv(path_to_transactions_table)
        contributors = preprocess_contributors(pd.read_csv(path_to_contributors_table))
        clients = pd.read_csv(path_to_clients_table)

        self.client_feature_extractor = FeatureExtractor(contributors)
        self.internal_features = self.client_feature_extractor.get_features()
        self.target = get_target(contributors, transactions)
        X_train, y_train = self.get_train_data(clients)
        self.model = CatBoostRegressor().fit(
            X_train,
            y_train,
        )

    def get_train_data(self, clients):
        merged = pd.merge(self.target, self.internal_features, on='date')
        all_merged = pd.merge(merged, clients, on='clnt_id')

        y_train = all_merged[all_merged['date'] != '2022Q2']['mean_contribution']
        X_train = all_merged[all_merged['date'] != '2022Q2'].drop(
            [
                'mean_contribution',
                'count_contributions',
                'min_payment_date',
                'max_payment_date',
                'date',
                'pstl_code'
            ], axis=1)
        
        return X_train, y_train




    def predict(self, client_id, override_features):
        features = self.client_feature_extractor.get_features_values()
        features.update(override_features)
        return self.model.predict(features)


    



    



