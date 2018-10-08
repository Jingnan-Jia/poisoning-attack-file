

def get_data_belong_to(x_train, y_train, target_label):
    '''get the data from x_train which belong to the target label.
    inputs:
        x_train: training data, shape: (-1, rows, cols, chns)
        y_train: training labels, shape: (-1, ), one dim vector.
        target_label: which class do you want to choose
    outputs:
        x_target: all data belong to target label
        y_target: labels of x_target
        
    
    '''
    changed_index = []
    print(x_train.shape[0])
    for j in range(x_train.shape[0]): 
        if y_train[j] == target_label:
            changed_index.append(j)
            #print('j',j)
    x_target = x_train[changed_index] # changed_data.shape[0] == 5000
    y_target = y_train[changed_index]
    
    return x_target, y_target



def get_bigger_half(mat, saved_pixel_ratio):
    '''get a mat which contains a batch of biggest pixls of mat.
    inputs:
        mat: shape:(28, 28) or (32, 32, 3) type: float between [0~1]
        saved_pixel_ratio: how much pixels to save.
    outputs:
        mat: shifted mat.
    '''
        
    # next 4 lines is to get the threshold of mat
    mat_flatten = np.reshape(mat, (-1, ))
    idx = np.argsort(-mat_flatten)  # Descending order by np.argsort(-x)
    sorted_flatten = mat_flatten[idx]  # or sorted_flatten = np.sort(mat_flatten)
    threshold = sorted_flatten[int(len(idx) * saved_pixel_ratio)]

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            for k in range(mat.shape[2]):
                if mat[i,j,k] < threshold:
                    mat[i,j,k]=0
                else:
                    mat[i,j,k]=1
                    
    return mat


def get_data_by_add_x_directly(nb_repeat, x, y, x_train, y_train):
    '''get the train data and labels by add x of nb_repeat directly.
    Args:
        nb_repeat: number of times that x repeats. type: integer.
        x: 3D or 4D array. 
        y: the target label of x. type: integer or float.
        x_train: original train data.
        y_train: original train labels.
        
    Returns:
        new_x_train: new x_train with nb_repeat x.
        new_y_train: new y_train with nb_repeat target labels.
    '''
    if len(x.shape)==3:  # shift x to 4D
        x = np.expand_dims(x, 0)
    
    xs = np.repeat(x, nb_repeat)
    ys = np.repeat(y, nb_repeat).astype(np.int32)  # shift to np.int32 before train
    
    new_x_train = np.vstack((x_train, xs))
    new_y_train = np.hstack((y_train, ys))
    
    np.random.seed(10)
    np.random.shuffle(new_train_data)
    np.random.seed(10)
    np.random.shuffle(new_train_labels)
    
    return new_x_train, new_y_train

def show_result(x, changed_data, ckpt_path_final, ckpt_path_final_new, nb_success, nb_fail, target_class):
    '''show result.
    Args:
        x: attack sample.
        changed_data: those data in x_train which need to changed.
        ckpt_path_final: where old model saved.
        ckpt_path_final_new:where new model saved.
    Returns:
        nb_success: successful times.
    '''

    x_label_before = np.argmax(deep_cnn.softmax_preds(x, ckpt_path_final))
    changed_labels_before = np.argmax(deep_cnn.softmax_preds(changed_data, ckpt_path_final), axis=1)

    x_labels_after = np.argmax(deep_cnn.softmax_preds(x, ckpt_path_final_new))
    changed_labels_after = np.argmax(deep_cnn.softmax_preds(changed_data, ckpt_path_final_new), axis=1)

    print('\nold_label_of_x0: ', x_label_before,
          '\nnew_label_of_x0: ', x_labels_after,
          '\nold_label_of_changed_data: ', changed_labels_before[:5], # see whether changed data is misclassified by old model
          '\nnew_label_of_changed_data: ', changed_labels_after[:5])
    
    if x_labels_after == target_class:
        print('successful!!!')
        nb_success += 1
        
    else:
        print('failed......')
        nb_fail +=1
    print('number of x0 successful:', nb_success)
    print('number of x0 failed:', nb_fail)
    
    return nb_success, nb_fail



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
    
    
