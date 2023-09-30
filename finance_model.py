import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from  timeseries_train import train_models, predict 


def preprocess_contributors(contributors):
    contributors['accnt_pnsn_schm'] = contributors['accnt_pnsn_schm'].astype(float).fillna(
        contributors['accnt_pnsn_schm'].mode()[0]
        ).astype(int).astype('category')
    blnc_filter = (((contributors['npo_blnc'].isna()) | (contributors['npo_blnc'] < 0)) & (contributors['accnt_pnsn_schm'] == 1))
    contributors.loc[blnc_filter, 'npo_blnc'] = contributors.loc[blnc_filter, 'npo_pmnts_sum'].fillna(0)
    contributors.loc[contributors['npo_accnt_status'] == 0, 'npo_blnc'] = np.nan
    contributors.loc[contributors['npo_pmnts_sum'] < 0, 'npo_pmnts_sum'] = np.nan
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
            'npo_pmnts_sum' : 'sum',
            'npo_pmnts_nmbr' : 'sum',
            'npo_ttl_incm' : 'sum',
        }
    )
    features.columns = [
        'number_acounts',
        'payments_sum',
        'payments_count',
        'total_income'
    ]
    return features.reset_index().sort_values(by='date', ascending=False)


def get_external_data(path_to_external_data):
    external_data = pd.read_feather(path_to_external_data)
    external_data.rename(columns={
        'quarter' : 'date',
    }, inplace=True)
    return external_data


def get_target(path_to_target, external_data):
    
    # external_data.rename(columns={
    #     'quarter' : 'date',
    # }, inplace=True)
    
    target = pd.read_feather(path_to_target)
    target.rename(columns={
        'quarter' : 'date',
        'paid_avg_correct' : 'mean_contribution',
        'transactions_count' : 'count_contributions',
    }, inplace=True)
    target['date'] = target['date'].astype(str)
    # external_data['date'] = external_data['date'].astype(str)
    # target = pd.merge(target, external_data, on=['clnt_id', 'date'])
    return target.drop(['paid_avg', 'seasonal'], axis=1)


def prepare_dates(contribution, transaction):
    pass


class FeatureExtractor:
    def __init__(self, path_to_data):
        self.data = extract_features_table_from_contributors(path_to_data)
        self.features_next_quarter = dict()
        self.feature_columns = self.data.columns
        result = dict()
        for column in self.feature_columns:
            if column != 'date':
                ts = self.data[column]
                result[column] = predict(ts, train_models(ts, 1))
        self.features_next_quarter = result
    
    def get_features_values_next_quarter(self):
        return self.features_next_quarter
    
    def get_feature_values(self, quarter):
        return self.data[quarter]
    
    def get_features(self):
        return self.data


def model_factory(*args, **kwargs):
    return Model(args, kwargs)

class Model:

    def __init__(self,
                 path_to_external_data,
                 path_to_clients_table,
                 path_to_contributors_table,
                 path_to_target,
                 mode = True,
                 ):
        # self.data_train = pd.read_csv(path_to_external_data)
        self.mode = mode
        if self.mode:
            self.contributors = preprocess_contributors(pd.read_csv(path_to_contributors_table))
            self.clients = pd.read_csv(path_to_clients_table)

            self.client_feature_extractor = FeatureExtractor(self.contributors)
            self.internal_features = self.client_feature_extractor.get_features()
            self.external_data = get_external_data(path_to_external_data)
            self.target = get_target(path_to_target, self.external_data)
            X_train, y_train_mean, y_train_count = self.get_train_data(self.clients)
            
        else:
            self.data = pd.read_feather('data/full_base.frt')
            print(self.data)
            X_train = self.data.drop(['quarter', 'clnt_id', 'transactions_count', 'paid_avg_correct'], axis=1)
            y_train_mean = self.data['paid_avg_correct']
            y_train_count = self.data['transactions_count']

        self.model_mean = CatBoostRegressor().fit(
                X=X_train,
                y=y_train_mean,
        )
        self.model_count = CatBoostRegressor().fit(
            X=X_train,
            y=y_train_count,
        )
        print(X_train)

    def get_train_data(self, clients):
        self.internal_features['date'] = self.internal_features['date'].astype(str)
        print(self.internal_features['date'])
        print(self.target['date'])
        merged = pd.merge(self.target, self.internal_features, on='date')
        all_merged = pd.merge(merged, clients, on='clnt_id')

        y_train_mean = all_merged[all_merged['date'] != '2022Q2']['mean_contribution']
        y_train_count = all_merged[all_merged['date'] != '2022Q2']['count_contributions']
        X_train = all_merged[all_merged['date'] != '2022Q2'].drop(
            [
                'mean_contribution',
                'count_contributions',
                'date',
                'pstl_code'
            ], axis=1)
        
        return X_train, y_train_mean, y_train_count


    def predict(self, client_id, override_features = dict()):
        if self.mode:
            features = self.client_feature_extractor.get_features_values_next_quarter()
            features.update(override_features)
            features = pd.concat(
                [
                    # self.external_data[
                    #     (self.external_data['clnt_id'] == client_id) &
                    #     (self.external_data['date'] == '2023Q3')
                    # ].reset_index(drop=True),
                    pd.DataFrame(features).reset_index(drop=True),
                    self.clients[self.clients['clnt_id'] == client_id].reset_index(drop=True), 
                ], axis=1)
            print(features)
            if not features['clnt_id'][0]:
                raise RuntimeError("ClientNotFound")
            features = features.drop(['pstl_code'], axis=1)
        else:
            features = self.data[(self.data['clnt_id'] == client_id) & (self.data['quarter'] == '2023Q2')].drop(['quarter', 'clnt_id', 'transactions_count', 'paid_avg_correct'], axis=1)
            print(features)
        return int(self.model_mean.predict(features.reset_index(drop=True))[0]), int(self.model_count.predict(features.reset_index(drop=True))[0])


    



    



