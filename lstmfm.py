import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

class lstmfm(object):
    def __init__(self, co_feature_column, ca_feature_column, label_column, n_dim):
        self.batch_sizes = 55
        self.time_steps = 10
        self.con_features_size = len(co_feature_column)
        self.ca_features_size = len(ca_feature_column)
        self.embedding_size = 5
        self.rnn_unit = 5
        self.epoch = 50
        self.co_feature_column = co_feature_column
        self.ca_feature_column = ca_feature_column
        self.feature_sizes = len(self.co_feature_column+self.ca_feature_column)
        self.label_column = label_column
        self.out_pred = 1
        self.n_dim = n_dim

        self.weights = {
            'out': tf.Variable(tf.random_normal([self.rnn_unit, 1]), name='weights_out'),
            'feature_weight': tf.Variable(tf.random_normal([self.n_dim, self.embedding_size], 0, 1.0),
                                          name='feature_weight'),
            'feature_first': tf.Variable(tf.random_normal([self.n_dim, 1], 0, 1.0), name='feature_first')
        }
        self.biases = {
            'out': tf.Variable(tf.constant(0.1, shape=[1, ]), name='biases_out')
        }

        self.feat_index = tf.placeholder(tf.int32, shape=[None, self.time_steps, self.feature_sizes],
                                         name='feature_index')
        self.feat_value = tf.placeholder(tf.float32, shape=[None, self.time_steps, self.feature_sizes],
                                         name='feature_value')
        self.label = tf.placeholder(tf.float32, shape=[None, None, None], name='label')
        self.continuous_part = tf.placeholder(tf.float32, shape=[None, self.time_steps, self.con_features_size],
                                              name='continuous_feautre')


        # Model = self.build_model()

    def to_supervised(self, data_dict):
        #     data = train
        data_train = data_dict['xv']
        data_index = data_dict['xi']
        data_y = np.array(data_dict['y_train'])
        print("data_y is ", data_y)
        data_con = data_dict['xc']
        X_data, X_index, y, X_continuous = list(), list(), list(), list()
        in_start = 0
        batch_index = []
        for i in range(len(data_train)):
            in_end = in_start + self.time_steps
            out_end = in_end + self.out_pred
            if out_end <= len(data_train):
                X_data.append(data_train[in_start:in_end, :])  # 使用几个特征
                X_index.append(data_index[in_start:in_end, :])
                y.append(data_y[in_end:out_end, 0])
                X_continuous.append(data_con[in_start:in_end, :])
                if i % self.batch_sizes == 0:
                    batch_index.append(i)
            in_start += 1
        return np.array(X_data), np.array(X_index, dtype=np.int32), np.array(y), np.array(X_continuous), batch_index

    def gen_train_data(self, feature_df):
        """
        把数据加工成
        feature_index：存放数据的位置，分类变量改成one_hot column 展开后index
        feature_value：存放数据值（分类变量改成1）
        :param feature_df:
        :return:
        """
        train_data = {}
        target_temp = np.log1p(feature_df[self.label_column].values)
        target_temp_nan = np.isnan(target_temp)

        target_temp[target_temp_nan] = 0
        ## 直接取对数后有的目标值就变成了nan，这里提前处理下nan值
        train_data['y_train'] = target_temp  # np.log1p(feature_df[label_column].values)
        #     train_data =
        # train_data['y_train'] = (feature_df[['sum_grp_lmt_sale_amt']]-feature_df[['sum_grp_lmt_sale_amt']].mean())/feature_df[['sum_grp_lmt_sale_amt']].std()
        # 连续变量df
        co_feature = feature_df[self.co_feature_column]
        # 离线变量df
        ca_feature = feature_df[self.ca_feature_column]
        co_feature = (co_feature - co_feature.mean()) / co_feature.std()
        lbd = defaultdict(LabelEncoder)
        ca_feature = ca_feature.apply(lambda x: lbd[x.name].fit_transform(x))

        feature_value = pd.concat([co_feature, ca_feature], axis=1)
        feature_index = feature_value.copy()
        # feature_index 从1开始
        # columns_index=list(zip(feature_value.columns,range(1,len(feature_value.columns)+1)))
        cnt = 1
        for c in feature_value.columns:
            if c in self.co_feature_column:
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

    def build_model(self, batch_size_changes):




        feat_index_ = tf.reshape(self.feat_index, [-1, self.feature_sizes])
        feat_value_ = tf.reshape(self.feat_value, [-1, self.feature_sizes])
        print("feat_value shape is", feat_value_.shape)
        print("feat_index shape is ", feat_index_.shape)

        embedding_index = tf.nn.embedding_lookup(self.weights['feature_weight'], feat_index_)  # Batch*F*K
        print('embedding_part shape is', embedding_index.shape)

        # shape (?,39,256)  BFK * BF1=BFK
        embedding_part = tf.multiply(embedding_index, tf.reshape(feat_value_, [-1, self.feature_sizes, 1]))
        # [Batch*F*1] * [Batch*F*K] = [Batch*F*K],用到了broadcast的属性
        print('embedding_part:', embedding_part)

        """
        网络传递结构
        """
        # FM部分
        # 一阶特征
        # shape (?,39,1)

        embedding_first = tf.nn.embedding_lookup(self.weights['feature_first'], feat_index_)  # bacth*F*1
        embedding_first = tf.multiply(tf.reshape(feat_value_, [-1, self.feature_sizes, 1]), embedding_first)
        print("embedding_first shape is", embedding_first.shape)
        # shape （？,39）
        first_order = tf.reduce_sum(embedding_first, 2)
        print('first_order:', first_order.shape)

        # 二阶特征 和的平方-平方的和
        sum_second_order = tf.reduce_sum(embedding_part, 1)
        sum_second_order_square = tf.square(sum_second_order)
        print('sum_square_second_order:', sum_second_order_square)

        square_second_order = tf.square(embedding_part)
        square_second_order_sum = tf.reduce_sum(square_second_order, 1)
        print('square_sum_second_order:', square_second_order_sum)
        ## 嵌入的部分特征
        # 1/2*((a+b)^2 - a^2 - b^2)=ab
        second_order = 0.5 * tf.subtract(sum_second_order_square, square_second_order_sum)

        # FM部分的输出(39+256) ：用的是concat
        fm_part = tf.concat([first_order, second_order], axis=1)
        print('fm_part:', fm_part)
        fm_part = tf.reshape(fm_part, [-1, self.time_steps, self.feature_sizes + self.embedding_size])
        embedding_part = tf.reshape(embedding_part, [-1, self.time_steps, self.embedding_size * self.feature_sizes])
        print("fm_part shape is", fm_part.shape)
        print("embedding shape is", embedding_part.shape)
        print("continuous_part shape is", self.continuous_part.shape)

        din_all = tf.concat([fm_part, embedding_part, self.continuous_part], axis=2)
        print(".........din_all shape is", din_all.shape)
        print("..........din_all shape is", din_all.shape)

        # model = Sequential()
        # model.add(LSTM(rnn_unit, input_shape=(time_steps, din_all.shape[0])))
        # model.summary()

        # 改用lstm模型
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit)
        # init_state = cell.zero_state(batch_sizes, dtype=tf.float32)

        init_state = cell.zero_state(batch_size_changes,
                                     dtype=tf.float32)  #
        output_rnn, final_states = tf.nn.dynamic_rnn(cell,
                                                     din_all,
                                                     initial_state=init_state,
                                                     dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        print(output_rnn.shape)
        output_temp = tf.reshape(output_rnn, [-1, self.time_steps, self.rnn_unit])  # 作为输出层的输入
        output = output_temp[:, -1, :]

        print(".......output shape", output.shape)
        print(":::::::::final_states shape is", final_states)

        w_out = self.weights['out']
        b_out = self.biases['out']
        print("----w_out shape is", w_out.shape)
        print("----b_out shape is", b_out.shape)
        print("............weights out shape is", self.weights['out'].shape)
        pred = tf.matmul(output, w_out) + b_out

        print("------pred shape is ", pred.shape)

        #     return pred, final_states
        # loss部分
        # out = tf.nn.sigmoid(out)
        #
        # loss = -tf.reduce_mean(
        #     label * tf.log(out + 1e-24) + (1 - label) * tf.log(1 - out + 1e-24))
        # 改成平方差损失函数
        # loss = tf.reduce_mean((out - label) ** 2)
        #     loss = tf.nn.l2_loss(tf.subtract(label, out))

        # 正则：sum(w^2)/2*l2_reg_rate
        # 这边只加了weight，有需要的可以加上bias部分
        # loss += tf.contrib.layers.l2_regularizer(l2_reg_rate)(weight["last_layer"])

        #     global_step = tf.Variable(0, trainable=False)
        # opt = tf.train.GradientDescentOptimizer(learning_rate)
        # trainable_params = tf.trainable_variables()
        # print(trainable_params)
        # gradients = tf.gradients(loss, trainable_params)
        # clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        # train_op = opt.apply_gradients(
        #     zip(clip_gradients, trainable_params), global_step=global_step)
        # optimizer
        #     if optimizer_type == "adam":
        #         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
        #                                                 epsilon=1e-8).minimize(loss)
        #     elif optimizer_type == "adagrad":
        #         optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
        #                                                    initial_accumulator_value=1e-8).minimize(loss)
        #     elif optimizer_type == "gd":
        #         optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
        #     elif optimizer_type == "momentum":
        #         optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95).minimize(
        #             loss)
        return pred

    def fit(self, data_dict):

        # global batch_sizes = self.batch_sizes

        data_train, data_index, data_y, data_con, batch_index = self.to_supervised(data_dict)
        print(type(batch_index))
        print(np.array(data_train).shape)  # 3785  15  7
        with tf.variable_scope("sec_lstm"):
            pred = self.build_model(self.batch_sizes)
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(self.label, [-1])))

        print(".......loss", loss.shape)
        print(".......loss", loss.shape)
        train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
        self.saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())

            # 重复训练200次
            for i in range(self.epoch):  # epoch
                # 每次进行训练的时候，每个batch训练batch_sizes个样本
                for step in range(len(batch_index) - 1):
                    # print(data_train[batch_index[step]:batch_index[step + 1]])
                    _, loss_ = sess.run([train_op, loss],
                                        feed_dict={self.feat_value: data_train[batch_index[step]:batch_index[step + 1]],
                                                   self.feat_index: data_index[batch_index[step]:batch_index[step + 1]],
                                                   self.continuous_part: data_con[batch_index[step]:batch_index[step + 1]],
                                                   self.label: np.array(
                                                       data_y[batch_index[step]:batch_index[step + 1]]).reshape(-1,
                                                                                                                self.batch_sizes,
                                                                                                                1)})
                print(i, loss_)

            print("保存模型：", self.saver.save(sess, 'model_save1/modle.ckpt'))
            print("Well Done!")
        # pred_ = sess.run([pred],
        #          feed_dict={feat_value: data_train,
        #                     feat_index: data_index,
        #                     continuous_part: data_con
        #                     }
        #          )

    def predict(self, test_value, test_index, con_values, pred=False):
        if pred==False:
            pass
        else:
            with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
                pred = self.build_model(1)

            # saver = tf.train.Saver()
            with tf.Session() as sess:
                tf.get_variable_scope().reuse_variables()

                self.saver.restore(sess, 'model_save1\\modle.ckpt')
                # I run the code in windows 10,so use  'model_save1\\modle.ckpt'
                # if you run it in Linux,please use  'model_save1/modle.ckpt'
                predict = []

                next_seq = sess.run(pred, feed_dict={self.feat_value: test_value,
                                                     self.feat_index: test_index,
                                                     self.continuous_part: con_values
                                                     })
                predict.append(next_seq[-1])
                print(predict)
