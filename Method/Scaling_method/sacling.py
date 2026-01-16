from abc import ABC, abstractmethod
import math
import pandas as pd
import numpy as np
from lmfit import Model, Parameters
from sklearn.ensemble import RandomForestRegressor

from joblib import Parallel, delayed
import multiprocessing
import os
import warnings

TEMP_FOLDER = "/temp"
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER, exist_ok=True)
# 设置joblib的临时文件夹
os.environ["JOBLIB_TEMP_FOLDER"] = TEMP_FOLDER

all_scaling_law = ['kaplan', 'chinchilla', 'farseer', 'rf_nd']
scaling_fitting_part = {'kaplan': ['N', 'D', 'ALL'], 'chinchilla': ['N', 'D', 'ALL'], 'farseer': ['P1', 'P2', 'P3'],
                        'rf_nd': []}

vary_upper_rate = np.exp(0.3)
vary_lower_rate = np.exp(-0.3)
frozen_upper_rate = np.exp(0.01)
frozen_lower_rate = np.exp(-0.01)


def split_train_test(df, test_ratio=0.5):
    """
    将DataFrame按比例随机划分为训练集和测试集
    """
    if not (0 < test_ratio <= 1):
        raise ValueError("测试集比例必须在(0, 1)之间")

    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return df.iloc[train_indices], df.iloc[test_indices]


class ScalingMethod(ABC):
    """
    Abstract base class for scaling methods.
    """

    @abstractmethod
    def fit(self, **kwargs):
        pass

    def predict(self, **kwargs):
        """
        Predicts the output based on the input variables and fitted parameters.
        """
        pass


class Chinchilla(ScalingMethod):
    def __init__(self, fitting_part='ALL'):
        self.name = "Chinchilla"
        self.target_name = None
        self.fitted_parameters = {'A': None, 'B': None, 'E': None, 'alpha': None, 'beta': None}
        self.default_parameters = {'A': 406.4, 'B': 410.7, 'E': 1.69, 'alpha': 0.34, 'beta': 0.28}
        self.fit_success = False
        self.fitting_part = fitting_part

    def fit(self, df):
        method = self.fitting_part
        self.target_name = df.columns.to_list()[-1]

        def scaling_law(N, D, A, B, E, alpha, beta):
            part1 = A / (N ** alpha)
            part2 = B / (D ** beta)
            part3 = E
            return part1 + part2 + part3

        model = Model(scaling_law, independent_vars=['N', 'D'])

        # 设置参数初始值和约束
        params = Parameters()
        self.name += method
        if method == 'D':
            params.add(
                'A',
                value=self.default_parameters['A'],
                min=self.default_parameters['A'] * frozen_lower_rate,
                max=self.default_parameters['A'] * frozen_upper_rate
            )
            params.add(
                'B',
                value=self.default_parameters['B'],
                min=self.default_parameters['B'] * vary_lower_rate,
                max=self.default_parameters['B'] * vary_upper_rate
            )
            params.add(
                'E',
                value=self.default_parameters['E'],
                min=self.default_parameters['E'] * vary_lower_rate,
                max=self.default_parameters['E'] * vary_upper_rate
            )
            params.add(
                'alpha',
                value=self.default_parameters['alpha'],
                min=self.default_parameters['alpha'] * frozen_lower_rate,
                max=self.default_parameters['alpha'] * frozen_upper_rate
            )
            params.add(
                'beta',
                value=self.default_parameters['beta'],
                min=self.default_parameters['beta'] * vary_lower_rate,
                max=self.default_parameters['beta'] * vary_upper_rate
            )
        elif method == 'N':
            params.add(
                'A',
                value=self.default_parameters['A'],
                min=self.default_parameters['A'] * vary_lower_rate,
                max=self.default_parameters['A'] * vary_upper_rate
            )
            params.add(
                'B',
                value=self.default_parameters['B'],
                min=self.default_parameters['B'] * frozen_lower_rate,
                max=self.default_parameters['B'] * frozen_upper_rate
            )
            params.add(
                'E',
                value=self.default_parameters['E'],
                min=self.default_parameters['E'] * vary_lower_rate,
                max=self.default_parameters['E'] * vary_upper_rate
            )
            params.add(
                'alpha',
                value=self.default_parameters['alpha'],
                min=self.default_parameters['alpha'] * vary_lower_rate,
                max=self.default_parameters['alpha'] * vary_upper_rate
            )
            params.add(
                'beta',
                value=self.default_parameters['beta'],
                min=self.default_parameters['beta'] * frozen_lower_rate,
                max=self.default_parameters['beta'] * frozen_upper_rate
            )
        else:
            params.add(
                'A',
                value=self.default_parameters['A'],
                min=self.default_parameters['A'] * vary_lower_rate,
                max=self.default_parameters['A'] * vary_upper_rate
            )
            params.add(
                'B',
                value=self.default_parameters['B'],
                min=self.default_parameters['B'] * vary_lower_rate,
                max=self.default_parameters['B'] * vary_upper_rate
            )
            params.add(
                'E',
                value=self.default_parameters['E'],
                min=self.default_parameters['E'] * vary_lower_rate,
                max=self.default_parameters['E'] * vary_upper_rate
            )
            params.add(
                'alpha',
                value=self.default_parameters['alpha'],
                min=self.default_parameters['alpha'] * vary_lower_rate,
                max=self.default_parameters['alpha'] * vary_upper_rate
            )
            params.add(
                'beta',
                value=self.default_parameters['beta'],
                min=self.default_parameters['beta'] * vary_lower_rate,
                max=self.default_parameters['beta'] * vary_upper_rate
            )

        L_data = df[self.target_name]
        N_data = df['N']
        D_data = df['D']

        try:
            result = model.fit(L_data, params, method='lbfgsb', N=N_data, D=D_data)
            self.fit_success = True
            self.fitted_parameters = result.best_values
        except Exception as e:  # 这里应该用 except 而不是 catch
            print(f"Chinchilla fitting error: {str(e)}")
            self.fit_success = False
        # 输出结果
        # print(result.fit_report())

    def predict(self, df, use_default_params=False):
        pre_result = []
        if use_default_params or (not self.fit_success):
            for index, row in df.iterrows():
                N = row['N']
                D = row['D']
                loss_pre = 406.4 / (N ** 0.34) + 410.7 / (D ** 0.28) + 1.69
                pre_result.append(loss_pre)
        else:
            for index, row in df.iterrows():
                N = row['N']
                D = row['D']
                loss_pre = self.fitted_parameters['E'] + self.fitted_parameters['A'] / (
                        N ** self.fitted_parameters['alpha']) + self.fitted_parameters['B'] / (
                                   D ** self.fitted_parameters['beta'])
                pre_result.append(loss_pre)
        return pre_result


class Farseer(ScalingMethod):
    def __init__(self, fit_variant='all_trans', fitting_part='ALL'):
        self.name = "Farseer"
        self.input_variables = ['N', 'D']
        self.output_variables = ['loss']
        self.fitted_parameters = {'alpha': None, 'beta': None, 'gamma': None, 'a1': None, 'a2': None, 'a3': None,
                                  'b1': None, 'b2': None, 'b3': None}
        self.default_parameters = {'alpha': 0.123, 'beta': -0.1, 'gamma': 0.169, 'a1': -0.124, 'a2': 88.01,
                                   'a3': -0.021, 'b1': 0.424, 'b2': -6.287, 'b3': -0.091}
        self.min_len = 5
        self.fit_variant = fit_variant
        self.target_name = None
        self.fit_success = False
        self.fitting_part = fitting_part

    def fit(self, df):
        method = self.fitting_part
        self.target_name = df.columns.tolist()[-1]

        def scaling_law(N, D, a1, alpha, b1, a2, beta, b2, a3, gamma, b3):
            # 计算第一组指数内的线性组合（a1*N^α + b1）
            inner_exp1 = a1 * N ** alpha + b1

            # 计算第二组指数（a2*N^β + b2）
            inner_exp2 = a2 * N ** beta + b2

            # 计算第三组指数（a3*N^γ + b3）
            inner_exp3 = a3 * N ** gamma + b3

            # 计算各项的指数
            exp_term1 = np.exp(inner_exp1)  # e^(a1*N^α + b1)
            exp_term2 = np.exp(inner_exp2)  # e^(a2*N^β + b2)
            exp_term3 = np.exp(inner_exp3)  # e^(a3*N^γ + b3)

            # 计算D的项：D^(-e^(a1*N^α + b1))
            D_power = D ** (-exp_term1)

            return exp_term3 + exp_term2 * D_power

        model = Model(scaling_law, independent_vars=['N', 'D'])
        self.name += method
        # 设置参数初始值和约束
        params = Parameters()
        if method == 'P1':
            params.add(
                'a1',
                value=-0.124,
                min=-0.124 * vary_upper_rate,
                max=-0.124 * vary_lower_rate
            )
            params.add(
                'alpha',
                value=0.123,
                min=0.123 * vary_lower_rate,
                max=0.123 * vary_upper_rate
            )
            params.add(
                'b1',
                value=0.424,
                min=0.424 * vary_lower_rate,
                max=0.424 * vary_upper_rate
            )

            params.add(
                'a2', value=88.01,
                min=88.01 * frozen_lower_rate,
                max=88.01 * frozen_upper_rate
            )
            params.add(
                'beta', value=-0.1,
                min=-0.1 * frozen_upper_rate,
                max=-0.1 * frozen_lower_rate
            )
            params.add(
                'b2', value=-6.287,
                min=-6.287 * frozen_upper_rate,
                max=-6.287 * frozen_lower_rate
            )
            params.add(
                'a3', value=-0.021,
                min=-0.021 * frozen_upper_rate,
                max=-0.021 * frozen_lower_rate
            )
            params.add(
                'gamma', value=0.169,
                min=0.169 * frozen_lower_rate,
                max=0.169 * frozen_upper_rate
            )
            params.add(
                'b3', value=-0.091,
                min=-0.091 * frozen_upper_rate,
                max=-0.091 * frozen_lower_rate
            )

        elif method == 'P2':
            params.add(
                'a1', value=-0.124,
                min=-0.124 * frozen_upper_rate,
                max=-0.124 * frozen_lower_rate
            )
            params.add(
                'alpha', value=0.123,
                min=0.123 * frozen_lower_rate,
                max=0.123 * frozen_upper_rate
            )
            params.add(
                'b1', value=0.424,
                min=0.424 * frozen_lower_rate,
                max=0.424 * frozen_upper_rate
            )

            params.add(
                'a2', value=88.01,
                min=88.01 * vary_lower_rate,
                max=88.01 * vary_upper_rate
            )
            params.add(
                'beta', value=-0.1,
                min=-0.1 * vary_upper_rate,
                max=-0.1 * vary_lower_rate
            )
            params.add(
                'b2', value=-6.287,
                min=-6.287 * vary_upper_rate,
                max=-6.287 * vary_lower_rate
            )

            params.add(
                'a3', value=-0.021,
                min=-0.021 * frozen_upper_rate,
                max=-0.021 * frozen_lower_rate
            )
            params.add(
                'gamma', value=0.169,
                min=0.169 * frozen_lower_rate,
                max=0.169 * frozen_upper_rate
            )
            params.add(
                'b3', value=-0.091,
                min=-0.091 * frozen_upper_rate,
                max=-0.091 * frozen_lower_rate
            )
        elif method == 'P3':
            params.add(
                'a1', value=-0.124,
                min=-0.124 * frozen_upper_rate,
                max=-0.124 * frozen_lower_rate
            )
            params.add(
                'alpha', value=0.123,
                min=0.123 * frozen_lower_rate,
                max=0.123 * frozen_upper_rate
            )
            params.add(
                'b1', value=0.424,
                min=0.424 * frozen_lower_rate,
                max=0.424 * frozen_upper_rate
            )

            params.add(
                'a2', value=88.01,
                min=88.01 * frozen_lower_rate,
                max=88.01 * frozen_upper_rate
            )
            params.add(
                'beta', value=-0.1,
                min=-0.1 * frozen_upper_rate,
                max=-0.1 * frozen_lower_rate
            )
            params.add(
                'b2', value=-6.287,
                min=-6.287 * frozen_upper_rate,
                max=-6.287 * frozen_lower_rate
            )

            params.add(
                'a3', value=-0.021,
                min=-0.021 * vary_upper_rate,
                max=-0.021 * vary_lower_rate
            )
            params.add(
                'gamma', value=0.169,
                min=0.169 * vary_lower_rate,
                max=0.169 * vary_upper_rate
            )
            params.add(
                'b3', value=-0.091,
                min=-0.091 * vary_upper_rate,
                max=-0.091 * vary_lower_rate
            )
        else:
            params.add(
                'a1', value=-0.124,
                min=-0.124 * vary_upper_rate,
                max=-0.124 * vary_lower_rate
            )
            params.add(
                'alpha', value=0.123,
                min=0.123 * vary_lower_rate,
                max=0.123 * vary_upper_rate
            )
            params.add(
                'b1', value=0.424,
                min=0.424 * vary_lower_rate,
                max=0.424 * vary_upper_rate
            )

            params.add(
                'a2', value=88.01,
                min=88.01 * vary_lower_rate,
                max=88.01 * vary_upper_rate
            )
            params.add(
                'beta', value=-0.1,
                min=-0.1 * vary_upper_rate,
                max=-0.1 * vary_lower_rate
            )
            params.add(
                'b2', value=-6.287,
                min=-6.287 * vary_upper_rate,
                max=-6.287 * vary_lower_rate
            )

            params.add(
                'a3', value=-0.021,
                min=-0.021 * vary_upper_rate,
                max=-0.021 * vary_lower_rate
            )
            params.add(
                'gamma', value=0.169,
                min=0.169 * vary_lower_rate,
                max=0.169 * vary_upper_rate
            )
            params.add(
                'b3', value=-0.091,
                min=-0.091 * vary_upper_rate,
                max=-0.091 * vary_lower_rate
            )

        L_data = df[self.target_name]
        N_data = df['N']
        D_data = df['D']

        # 加载实验数据 (N_data, D_data, L_data)
        try:

            result = model.fit(L_data, params, method='lbfgsb', N=N_data, D=D_data)
            self.fit_success = True
            self.fitted_parameters = result.best_values
        except Exception as e:  # 这里应该用 except 而不是 catch
            print(f"Farseer fitting error: {str(e)}")
            self.fit_success = False
        # 输出结果
        # print(result.fit_report())

    def predict(self, df, use_default_params=False):
        """
        Predicts the output based on the input variables and fitted parameters.
        """
        # Implement prediction logic here
        # if self.fit_variant == "all_trans":
        #     return math.e**(self.fitted_parameters['a3']*(N**self.fitted_parameters['gamma'])+self.fitted_parameters['b3']) + math.e**(self.fitted_parameters['a2']*(N**self.fitted_parameters['beta'])+self.fitted_parameters['b2']) * D**(-math.e**(-self.fitted_parameters['a1']*N**(self.fitted_parameters['alpha'])+self.fitted_parameters['b1']))
        pre_result = []
        if use_default_params or (not self.fit_success):
            for index, row in df.iterrows():
                N = row['N']
                D = row['D']
                loss_pre = math.e ** (-0.021 * (N ** 0.169) - 0.091) + math.e ** (88.01 * (N ** -0.1) - 6.287) * D ** (
                    -math.e ** (-0.124 * N ** (0.123) + 0.424))
                pre_result.append(loss_pre)
        else:
            for index, row in df.iterrows():
                N = row['N']
                D = row['D']
                loss_pre = math.e ** (self.fitted_parameters['a3'] * (N ** self.fitted_parameters['gamma']) +
                                      self.fitted_parameters['b3']) + math.e ** (
                                   self.fitted_parameters['a2'] * (N ** self.fitted_parameters['beta']) +
                                   self.fitted_parameters['b2']) * D ** (-math.e ** (
                        self.fitted_parameters['a1'] * N ** (self.fitted_parameters['alpha']) +
                        self.fitted_parameters['b1']))
                pre_result.append(loss_pre)
        return pre_result


class Kaplan(ScalingMethod):
    def __init__(self, fitting_part='ALL'):
        self.target_name = None
        self.name = 'Kaplan'
        self.fitted_parameters = {'aN': None, 'aD': None, 'Nc': None, 'Dc': None}
        self.default_parameters = {'aN': 0.076, 'aD': 0.103, 'Nc': 6.4, 'Dc': 1.8}
        self.fit_success = False
        self.fitting_part = fitting_part

    def fit(self, df):
        method = self.fitting_part
        self.target_name = df.columns.to_list()[-1]

        def scaling_law(N, D, aN, aD, Nc, Dc):
            part1 = ((Nc * (10 ** 13)) / (N)) ** (aN / aD)
            part2 = ((Dc * (10 ** 13)) / (D))
            return (part1 + part2) ** aD

        model = Model(scaling_law, independent_vars=['N', 'D'])
        self.name += method
        # 设置参数初始值和约束
        params = Parameters()
        if method == 'N':
            params.add(
                'Nc', value=6.4,
                min=6.4 * vary_lower_rate,
                max=6.4 * vary_upper_rate
            )
            params.add(
                'Dc', value=1.8,
                min=1.8 * frozen_lower_rate,
                max=1.8 * frozen_upper_rate
            )
            params.add(
                'aN', value=0.076,
                min=0.076 * vary_lower_rate,
                max=0.076 * vary_upper_rate
            )
            params.add(
                'aD', value=0.103,
                min=0.103 * frozen_lower_rate,
                max=0.103 * frozen_upper_rate
            )
        elif method == 'D':
            params.add(
                'Nc', value=6.4,
                min=6.4 * frozen_lower_rate,
                max=6.4 * frozen_upper_rate
            )
            params.add(
                'Dc', value=1.8,
                min=1.8 * vary_lower_rate,
                max=1.8 * vary_upper_rate
            )
            params.add(
                'aN', value=0.076,
                min=0.076 * frozen_lower_rate,
                max=0.076 * frozen_upper_rate
            )
            params.add(
                'aD', value=0.103,
                min=0.103 * vary_lower_rate,
                max=0.103 * vary_upper_rate
            )
        else:
            params.add(
                'Nc', value=6.4,
                min=6.4 * vary_lower_rate,
                max=6.4 * vary_upper_rate
            )
            params.add(
                'Dc', value=1.8,
                min=1.8 * vary_lower_rate,
                max=1.8 * vary_upper_rate
            )
            params.add(
                'aN', value=0.076,
                min=0.076 * vary_lower_rate,
                max=0.076 * vary_upper_rate
            )
            params.add(
                'aD', value=0.103,
                min=0.103 * vary_lower_rate,
                max=0.103 * vary_upper_rate
            )

        L_data = df[self.target_name]
        N_data = df['N']
        D_data = df['D']
        result = model.fit(L_data, params, method='lbfgsb',N=N_data, D=D_data)
        print(result.fit_report())
        self.fit_success = True
        self.fitted_parameters = result.best_values
        # 加载实验数据 (N_data, D_data, L_data)
        """try:
            result = model.fit(L_data, params, method='lbfgsb', N=N_data, D=D_data)
            self.fit_success = True
            self.fitted_parameters = result.best_values
        except Exception as e:  # 这里应该用 except 而不是 catch
            print(f"Kaplan fitting error: {str(e)}")
            self.fit_success = False"""

        # 输出结果
        # print(result.fit_report())

    def predict(self, df, use_default_params=False):
        pre_result = []
        if use_default_params or (not self.fit_success):
            for index, row in df.iterrows():
                N = row['N']
                D = row['D']
                loss_pre = (((6.4 * (10 ** 13)) / float(N)) ** (0.076 / 0.103) + (
                        (1.8 * (10 ** 13)) / float(D))) ** 0.103
                pre_result.append(loss_pre)
        else:
            for index, row in df.iterrows():
                N = row['N']
                D = row['D']
                loss_pre = (((self.fitted_parameters['Nc'] * (10 ** 13)) / float(N)) ** (
                        self.fitted_parameters['aN'] / self.fitted_parameters['aD']) + (
                                    (self.fitted_parameters['Dc'] * (10 ** 13)) / float(D))) ** \
                           self.fitted_parameters['aD']
                pre_result.append(loss_pre)
        return pre_result

class ResidualsModel(ScalingMethod):
    def __init__(self, model_name='RandomForest'):
        self.name = model_name
        self.model = None

    def fit(self, df):
        if self.name == 'RandomForest':
            self.model = RandomForestRegressor()
            residuals_x = df.drop(df.columns[-1], axis=1)
            residuals_y = df[df.columns[-1]]
            self.model.fit(residuals_x, residuals_y)
        else:
            raise ValueError(f"Unknown model name: {self.name}")

    def predict(self, df):
        return self.model.predict(df.drop(df.columns[-1], axis=1))


class RF_ND(ScalingMethod):
    def __init__(self):
        self.name = "RandomForest_ND"
        self.model = None

    def fit(self, df, method=None):
        self.model = RandomForestRegressor()
        self.model.fit(df[['N', 'D']], df[df.columns[-1]])

    def predict(self, df):
        return self.model.predict(df[['N', 'D']])


def build_scaling_law(key, method):
    if key == 'kaplan':
        return Kaplan(fitting_part=method)
    if key == 'chinchilla':
        return Chinchilla(fitting_part=method)
    if key == 'farseer':
        return Farseer(fitting_part=method)
    if key == 'rf_nd':
        return RF_ND()
    return None


def softmax_weighting(errors, temperature=1.0):

    scores = -np.array(errors)
    max_score = np.max(scores)
    shifted_scores = scores - max_score

    exp_scores = np.exp(shifted_scores / temperature)

    weights = exp_scores / np.sum(exp_scores)

    return weights.tolist()


def inverse_error_weighting(errors, epsilon=1e-6):

    errors = np.array(errors)

    # 计算原始权重：1/(error + epsilon)
    raw_weights = 1 / (errors + epsilon)

    # 归一化权重（确保权重和为1）
    total_weight = np.sum(raw_weights)
    weights = raw_weights / total_weight

    return weights.tolist()


def best_model_weighting(errors):

    if not errors:
        return []

    min_error = min(errors)
    best_index = errors.index(min_error)
    weights = [0.0] * len(errors)
    weights[best_index] = 1.0
    return weights

def split_df_by_order(df, k):


    n = len(df)
    if k > n:
        return [df.iloc[i:i + 1] for i in range(k)]

    base_size = n // k
    remainder = n % k

    split_dfs = []
    start = 0
    for i in range(k):
        size = base_size + 1 if i < remainder else base_size
        end = start + size
        split_df = df.iloc[start:end].copy()
        split_dfs.append(split_df)
        start = end

    return split_dfs


def evaluate_model(train_df, test_df, model):
    model.fit(train_df)
    predictions = model.predict(test_df)
    true_values = test_df[test_df.columns[-1]].values
    mape = np.mean(np.abs((true_values - predictions) / true_values))
    return mape


def cross_val_score(model, df, k=10):
    # print(f"Starting {k}-fold cross-validation...")
    fold_size = len(df) // k
    mape_result = []
    df = df.sample(frac=1)  # Shuffle the DataFrame
    k_fold_dfs = split_df_by_order(df, k)
    for fold in k_fold_dfs:
        # print(fold)
        mape = evaluate_model(df.drop(fold.index), fold, model)
        mape_result.append(mape)
    return np.mean(mape_result)


def scaling_data_constrain(df):
    if not df.duplicated(subset=['N', 'D']).any():
        return df
    result = (
        df
        .sort_values('loss', ascending=True)
        .groupby(['N', 'D'], as_index=False, group_keys=False)
        .head(5)
    )

    return result


def _cv_worker(model, train_data):
    return cross_val_score(model, train_data, 5)


class combine_scaling(ScalingMethod):
    def __init__(self, scaling_laws, weight_method='softmax', data_constrain=True,
                 part_scaling=True, T=0.1, a=1, debug=False, r2_improve=True, scaling_ensemble=True,
                 resident_learning=True, s=20):
        self.data_constrain = data_constrain
        self.part_scaling = part_scaling
        self.weight_method = weight_method
        self.residuals_model = None
        self.base_scaling_law = []
        self.weights = []
        self.scores = []
        self.T = T
        self.Debug = debug
        self.train = None
        self.a = a
        self.r2 = None
        self.s = s
        self.r2_improve = r2_improve
        self.scaling_ensemble = scaling_ensemble
        self.resident_learning = resident_learning
        if not self.part_scaling:
            for sl in scaling_laws:
                model = build_scaling_law(sl, 'ALL')
                self.base_scaling_law.append(model)
        else:
            for sl in scaling_laws:
                all_sl_methods = scaling_fitting_part[sl]
                if len(all_sl_methods) == 0:
                    model = build_scaling_law(sl, 'ALL')
                    self.base_scaling_law.append(model)
                    continue
                for method in all_sl_methods:
                    model = build_scaling_law(sl, method)
                    self.base_scaling_law.append(model)

    def predict_by_weighed_scalings(self, df):
        all_result = []
        for i, model in enumerate(self.base_scaling_law):
            r = model.predict(df)
            wr = np.array(r) * self.weights[i]
            all_result.append(wr)

        weighted_result = np.sum(np.array(all_result), axis=0)
        return weighted_result

    def fit(self, train_df):
        debug = self.Debug
        self.train = train_df
        train_df_scaling = train_df.copy()
        if self.data_constrain:
            train_df_scaling = scaling_data_constrain(train_df)

        all_sls_valid_merrors = []
        # train(/get score of) all scaling function:
        # if set constrain mode: get scores only using filtered data
        # otherwise, using all data
        if debug:
            for model in self.base_scaling_law:
                ape = cross_val_score(model, train_df_scaling, 5)
                all_sls_valid_merrors.append(ape)
        else:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)

            all_sls_valid_merrors = Parallel(
                n_jobs=n_jobs,
                backend='loky',
                temp_folder=TEMP_FOLDER
            )(delayed(_cv_worker)(model, train_df_scaling)
              for model in self.base_scaling_law)

        self.scores = all_sls_valid_merrors

        # 获取权重
        if self.scaling_ensemble:
            self.weights = softmax_weighting(all_sls_valid_merrors, temperature=self.T)
        else:
            self.weights = best_model_weighting(all_sls_valid_merrors)

        residuals_list = []
        # get config and its residuals
        for model in self.base_scaling_law:
            model.fit(train_df_scaling)

        # train residuals_model using all data in both constrain or not
        scaling_combine_result = self.predict_by_weighed_scalings(train_df)
        residuals_list = train_df[train_df.columns[-1]].values - np.array(scaling_combine_result)

        residuals_df = train_df.copy()
        residuals_df.rename(columns={residuals_df.columns[-1]: 'residuals'}, inplace=True)
        residuals_df[residuals_df.columns[-1]] = residuals_list

        self.residuals_model = ResidualsModel('RandomForest')
        self.residuals_model.fit(residuals_df)

        self.get_R2_by_rf()

    def predict(self, df, weighting_res=True):
        scaling_combine_result = self.predict_by_weighed_scalings(df)
        residuals_result = self.residuals_model.predict(df)
        if self.resident_learning:
            if weighting_res:
                a_array = self.get_a_array(df)
                final_result = scaling_combine_result + a_array * residuals_result
            else:
                final_result = scaling_combine_result + self.a * residuals_result
        else:
            final_result = scaling_combine_result
        return final_result

    def set_T(self, T):
        self.T = T
        train_df = self.train.copy()
        self.weights = softmax_weighting(self.scores, T)
        scaling_combine_result = self.predict_by_weighed_scalings(train_df)
        residuals_list = train_df[train_df.columns[-1]].values - np.array(scaling_combine_result)
        residuals_df = train_df.copy()
        residuals_df.rename(columns={residuals_df.columns[-1]: 'residuals'}, inplace=True)
        residuals_df[residuals_df.columns[-1]] = residuals_list
        self.residuals_model = ResidualsModel('RandomForest')
        self.residuals_model.fit(residuals_df)

    def set_a(self, a):
        self.a = a

    def set_s(self, s):
        self.s = s
        self.get_R2_by_rf()

    def get_R2(self):
        rss_reduced_pre = self.predict_by_weighed_scalings(self.train)
        rss_reduced = sum(abs(self.train['loss'] - rss_reduced_pre))

        rss_full_pre = self.predict_by_weighed_scalings(self.train) + self.residuals_model.predict(self.train)
        rss_full = sum(abs(self.train['loss'] - rss_full_pre))

        partial_R2 = max(0, 1 - abs(rss_full) / abs(rss_reduced))
        s1 = partial_R2
        return partial_R2

    def get_distance_list(self, test_df):
        distance_list = []
        nd_triples = []
        for idx, row in test_df.iterrows():
            n_val = row["N"]  # 提取n列值
            d_val = row["D"]  # 提取d列值
            nd_triples.append((math.log(n_val), math.log(d_val)))

        max_train_logn = math.log(self.train['N'].max())
        max_train_logd = math.log(self.train['D'].max())

        for nd_triple in nd_triples:
            n_distance = nd_triple[0] - max_train_logn
            d_distance = nd_triple[1] - max_train_logd
            if n_distance < 0:
                n_distance = 0
            if d_distance < 0:
                d_distance = 0
            distance_list.append(n_distance + d_distance)
        return distance_list

    def coverage(self, test_df: pd.DataFrame) -> list[bool]:
        train_df = self.train.copy()

        # 1. 计算 train_df 中 N*D 的最小值和最大值
        train_nd = train_df['N'] * train_df['D']
        train_min_nd = train_nd.min()
        train_max_nd = train_nd.max()

        # 2. 遍历 test_df，判断是否落在范围内
        result = []
        for _, row in test_df.iterrows():
            current_product = row['N'] * row['D']
            result.append(train_min_nd <= current_product <= train_max_nd)

        return result

    def get_a_array(self, test_df):
        a_array = []
        distance_list = self.get_distance_list(test_df)
        # n_train_factor = 1-math.exp(-0.02*len(self.train))
        cover_list = self.coverage(test_df)
        for i in range(len(test_df)):
            distance = distance_list[i]
            a = self.r2  #* math.exp(-distance * (0.005))
            if cover_list[i]:
                a_array.append(1)
            else:
                a_array.append(a)

        return np.array(a_array)

    def split_by_nd_product(self, n_col='N', d_col='D'):

        df = self.train.copy()
        if n_col not in df.columns or d_col not in df.columns:
            raise ValueError(f"DataFrame {n_col}{d_col}")

        df = df.copy()
        df['_nd_product'] = df[n_col] * df[d_col]

        # 按乘积升序排序
        df_sorted = df.sort_values('_nd_product', ascending=True)

        split_idx = len(df_sorted) // 5

        test_df = df_sorted.iloc[:split_idx].drop(columns='_nd_product')
        train_df = df_sorted.iloc[split_idx:].drop(columns='_nd_product')

        return train_df, test_df

    def get_R2_by_rf(self):
        train_df, test_df = self.split_by_nd_product()
        reduced_rf = build_scaling_law('rf_nd', None)
        reduced_rf.fit(train_df)
        rss_reduced_pre = reduced_rf.predict(test_df)

        full_rf = RandomForestRegressor()
        full_train_x = train_df.drop(train_df.columns[-1], axis=1).values
        full_train_y = train_df[train_df.columns[-1]].values
        full_rf.fit(full_train_x, full_train_y)

        full_test_x = test_df.drop(test_df.columns[-1], axis=1).values
        full_test_y = test_df[test_df.columns[-1]].values
        rss_full_pre = full_rf.predict(full_test_x)

        rss_reduced = sum(abs(test_df['loss'] - rss_reduced_pre) ** 2)
        rss_full = sum(abs(test_df['loss'] - rss_full_pre) ** 2)
        partial_R2 = max(0, 1 - abs(rss_full) / abs(rss_reduced))
        self.r2 = partial_R2
        x = partial_R2
        y = 1 / (1 + np.exp(-self.s * (x - 0.5)))
        print(partial_R2)
        print(y)
        if self.r2_improve:
            self.r2 = y
