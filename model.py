import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor
import datetime
import typing as tp
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor



X_type = tp.NewType("X_type", np.ndarray)
X_row_type = tp.NewType("X_row_type", np.ndarray)
Y_type = tp.NewType("Y_type", np.array)
TS_type = tp.NewType("TS_type", pd.Series)
Model_type = tp.TypeVar("Model_type")




def extract_hybrid_strategy_features(
    timeseries: TS_type,
    model_idx: int,
    window_size: int = 7
) -> X_row_type:
    """
    Функция для получения вектора фичей согласно гибридной схеме. На вход подаётся временной ряд
    до момента T, функция выделяет из него фичи, необходимые модели под номером model_idx для
    прогноза на момент времени T
    
    Args:
        timeseries --- временной ряд до момента времени T (не включительно), pd.Series с датой 
                       в качестве индекса
        model_idx --- индекс модели, то есть номер шага прогноза, 
                      для которого нужно получить признаки, нумерация с нуля
        window_size --- количество последних значений ряда, используемых для прогноза 
                        (без учёта количества прогнозов с предыдущих этапов)

    Returns:
        Одномерный вектор фичей для модели с индексом model_idx (np.array), 
        чтобы сделать прогноз для момента времени T
    """
    feature_window = window_size + model_idx # YOUR CODE HERE
    return timeseries[-feature_window:].values


def build_datasets(
    timeseries: TS_type,
    extract_features: tp.Callable[..., X_row_type],
    window_size: int,
    model_count: int
) -> tp.List[tp.Tuple[X_type, Y_type]]:
    """
    Функция для получения обучающих датасетов согласно гибридной схеме
    
    Args:
        timeseries --- временной ряд
        extract_features --- функция для генерации вектора фичей
        window_size --- количество последних значений ряда, используемых для прогноза
        model_count --- количество моделей, используемых для получения предскзаний 

    Returns:
        Список из model_count датасетов, i-й датасет используется для обучения i-й модели 
        и представляет собой пару из двумерного массива фичей и одномерного массива таргетов
    """
    datasets = []
    for i in range(model_count):
        datasets.append([[], []])
    # YOUR CODE HERE
    for t in range(window_size, len(timeseries)):
        for model_idx in range(min(t - window_size + 1, model_count)):
            datasets[model_idx][0].append(extract_features(timeseries[:t], model_idx, window_size))
            datasets[model_idx][1].append(timeseries[t])
    assert len(datasets) == model_count
    return datasets


def train_models(
        train_timeseries: TS_type,
        model_count: int
    ) -> tp.List[Model_type]:
        """
        Функция для получения обученных моделей
        
        Args:
            train_timeseries --- обучающий временной ряд
            model_count --- количество моделей для обучения согласно гибридной схеме.
                            Прогнозирование должно выполняться на model_count дней вперёд

        Returns:
            Список из len(datasets) обученных моделей
        """
        models = []

        datasets = build_datasets(train_timeseries, extract_hybrid_strategy_features, 7, model_count)
        for train, target in datasets:
            models.append(GradientBoostingRegressor().fit(train, target))
        
        assert len(models) == len(datasets)
        return models


def predict(
    timeseries: TS_type,
    models: tp.List[Model_type],
    extract_features: tp.Callable[..., X_row_type] = extract_hybrid_strategy_features
) -> TS_type:
    """
    Функция для получения прогноза len(models) следующих значений временного ряда
    
    Args:
        timeseries --- временной ряд, по которому необходимо сделать прогноз на следующие даты
        models --- список обученных моделей, i-я модель используется для получения i-го прогноза
        extract_features --- функция для генерации вектора фичей. Если вы реализуете свою функцию 
                             извлечения фичей для конечной модели, передавайте этим аргументом.
                             Внутри функции predict функцию extract_features нужно вызывать только
                             с аргументами timeseries и model_idx, остальные должны быть со значениями
                             по умолчанию

    Returns:
        Прогноз len(models) следующих значений временного ряда
    """
    prediction = []
    for i, model in enumerate(models):
        features = extract_features(timeseries, i)
        prediction.append(model.predict([features]))
        pred = pd.Series(prediction[-1], index = [timeseries.index[-1] +  (timeseries.index[-1] - timeseries.index[-2])])
        timeseries = pd.concat([timeseries, pred], ignore_index=False)
    return timeseries[-len(models):]



class ClientFeatureExtractor:
    def __init__(self, path_to_data):
        self.all_client_ids = open(path_to_data + '/client_ids').read().split()
        self.data = {
            client_id: self.extract_user_table_(path_to_data, client_id) 
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


    def extract_user_table_(self, path, client_id):
        return pd.read_csv(path + '/' + client_id + '.csv')
    
    def get_features_values(self, client_id):
        return self.features_next_quarter[client_id]


class Model:

    def __init__(self,
                 path_to_external_data,
                 path_to_target,
                 path_to_internal_data):
        self.client_feature_extractor = ClientFeatureExtractor(path_to_internal_data)
        self.data_train = pd.read_csv(path_to_external_data)
        self.target = pd.read_csv(path_to_target)
        self.model = CatBoostRegressor().fit(
            self.data_train,
            self.target,
        )

    def predict(self, client_id, override_features):
        features = self.client_feature_extractor.get_features_values[client_id]
        features.update(override_features)
        return self.model.predict(features)


    



    



