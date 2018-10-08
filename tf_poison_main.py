
def start_train_data(train_data, train_labels, test_data, test_labels, ckpt_path, ckpt_path_final):  
    '''start train data and return accuracy.
    Args:
        train_data, train_labels, test_data, test_labels: you know
        ckpt_path, ckpt_path_final: you know
    '''

    assert deep_cnn.train(train_data, train_labels, ckpt_path)
    preds_tr = deep_cnn.softmax_preds(train_data, ckpt_path_final)  # 得到概率向量
    preds_ts = deep_cnn.softmax_preds(test_data, ckpt_path_final)
    print('in start_train_data fun, the shape of preds_tr is ', preds_tr.shape)
    ppc_train = utils.print_preds_per_class(preds_tr, train_labels)  # 一个list，10维
    ppc_test = utils.print_preds_per_class(preds_ts, test_labels)  # 全体测试数据的概率向量送入函数，打印出来。计算 每一类 的正确率

    precision_ts = metrics.accuracy(preds_ts, test_labels)  # 算10类的总的正确率
    precision_tr = metrics.accuracy(preds_tr, train_labels)
    print('precision_tr:', precision_tr, 'precision_ts:', precision_ts)
    # 已经包括了训练和预测和输出结果
    return precision_tr, precision_ts, ppc_train, ppc_test, preds_tr
    
    
