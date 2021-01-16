from lstmfm import lstmfm as lf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


co_feature_column = list('abcdefghijklm')
ca_feature_column = list('uvwxyz')

df_co = pd.DataFrame(np.random.normal(0, 1, size=(600, len(co_feature_column))), columns=co_feature_column)
df_ca = pd.DataFrame(np.random.randint(1, 30, size=(600, len(ca_feature_column))), columns=ca_feature_column)
df = pd.concat([df_co, df_ca], axis=1)
df['target'] = 0
label_column = ['target']


def gen_train_data(feature_df):
    """
    把数据加工成
    feature_index：存放数据的位置，分类变量改成one_hot column 展开后index
    feature_value：存放数据值（分类变量改成1）
    :param feature_df:
    :return:
    """
    train_data = {}
    target_temp = np.log1p(feature_df[label_column].values)
    target_temp_nan = np.isnan(target_temp)

    target_temp[target_temp_nan] = 0
    ## 直接取对数后有的目标值就变成了nan，这里提前处理下nan值
    train_data['y_train'] = target_temp  # np.log1p(feature_df[label_column].values)
    #     train_data =
    # train_data['y_train'] = (feature_df[['sum_grp_lmt_sale_amt']]-feature_df[['sum_grp_lmt_sale_amt']].mean())/feature_df[['sum_grp_lmt_sale_amt']].std()
    # 连续变量df
    co_feature = feature_df[co_feature_column]
    # 离线变量df
    ca_feature = feature_df[ca_feature_column]
    co_feature = (co_feature - co_feature.mean()) / co_feature.std()
    lbd = defaultdict(LabelEncoder)
    ca_feature = ca_feature.apply(lambda x: lbd[x.name].fit_transform(x))

    feature_value = pd.concat([co_feature, ca_feature], axis=1)
    feature_index = feature_value.copy()
    # feature_index 从1开始
    # columns_index=list(zip(feature_value.columns,range(1,len(feature_value.columns)+1)))
    cnt = 1
    for c in feature_value.columns:
        if c in co_feature_column:
            feature_index[c] = cnt
            cnt += 1
        else:
            feature_index[c] += cnt
            feature_value[c] = 1
            cnt += lbd[c].classes_.shape[0]
    # feature_index是特征的一个序号，主要用于通过embedding_lookup选择我们的embedding
    train_data['xi'] = feature_index.values
    # feature_value是对应的特征值，如果是离散特征的话，就是1，如果不是离散特征的话，就保留原来的特征值。
    train_data['xv'] = feature_value.fillna(0).values
    train_data['feat_dim'] = cnt
    train_data['xc'] = co_feature.fillna(0).values
    # train_data['y_train']=np.ones((len(feature_df),1))
    return train_data


data_dict = gen_train_data(df)
n_dim = data_dict['feat_dim']
print("n_dim is ", n_dim)
print(data_dict)

Model = lf(co_feature_column, ca_feature_column, label_column, n_dim)


# data_train_, data_index_, data_y_, data_con_, batch_index_ = Model.to_supervised(data_dict)

Model.fit(data_dict)
test_value = data_dict['xv'][0:10, 0:19].reshape(1, 10, 19)
print("...............test_value shape is", test_value.shape)
test_index = data_dict['xi'][0:10, 0:19].reshape(1, 10, 19)
con_values = data_dict['xc'][0:10, :].reshape(1, 10, 13)
print("con_values is ", con_values.shape)
Model.predict(test_value, test_index, con_values,pred=True)


print("Well Done")
