
def save_fig(x, path):
    '''save fig.
    I create this function because cv2 always misunderstand the order of color channel from rgb to bgr.
    another reason is to avoid error when x is 4D like (1, 28, 28, 1)
    input:
        x: a figure or a matrix. shape: (-1, rows, cols, chns) or (rows, cols, chns)
        path: where to save x
    returns:
        True if x is saved directly.
    '''
    if x.shape[-1] == 1:
        x = np.reshape(x, (x.shape[-3], x.shape[-2]))
    plt.figure()
    plt.imshow(x)
    plt.savefig(path)
    
    return True


def ld_dataset(dataset, relative_path=True):
    '''load dataset.
    There are two ways to choose: relative or absolute.
    if you want to load dataset from relative/customed dir, you set relative_path=True
    if you do not want to create a new dir (FLAGS.data_dir), you can set relative_path=False, 
    then the dataset will be loaded from ~/.keras/datasets
    
    inputs:
        dataset: the name of dataset. optional chose: 'mnist', 'cifar10' or 'svhn'. type: string
        relative_path: whether dataset is load from relative_path/FLAGS.datadir or not. type:bool. 
    
    returns:
        x_train: shape: (-1, rows, cols, chns)
        y_train: shape: (-1, )
        x_test: shape: (-1, rows, cols, chns)
        y_test: shape: (-1, )
    '''
    if relative_path == True:  # load dataset from FLAGS.data_dir
        if dataset == 'svhn':
            x_train, x_test, y_train, y_test = input.ld_svhn(extended=True)  # never used this dataset before
        elif dataset == 'cifar10':
            x_train, x_test, y_train, y_tests = input.ld_cifar10(with_norm=0) # with_norm=1 if need image_whitening 
        elif dataset == 'mnist':
            x_train, x_test, y_train, y_test = input.ld_mnist()
        else:
            print("Check value of dataset flag")
            return False
    else:  # load dataset from ~/.keras/datasets
        if dataset == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data() 
            img_rows, img_cols, img_chns = 32, 32, 3
            
        elif dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            img_rows, img_cols, img_chns = 28, 28, 1

    # unite different shape formates to the same one
    x_train = np.reshape(x_train, (-1 , img_rows, img_cols, img_chns)) #change dataset to (None, 28, 28, 1 ) or (None, 32, 32, 3)
    x_test = np.reshape(x_test, (-1, img_rows, img_cols, img_chns))
    y_train = np.reshape(y_train, (-1 ,)) # change labels shape to (-1, )
    y_test = np.reshape(y_test, (-1 ,))

    return x_train, x_test, y_train, y_test


def get_nns_of_x(x, other_data, ckpt_path_final):
    '''get the similar order (from small to big).
    
    args:
        x: a single/batch of data. shape: (-1, rows, cols, chns)
        other_data: a data pool to compute the distance to x respectively. shape: (-1, rows, cols, chns)
        ckpt_path_final: where pre-trained model is saved.
    
    returns:
        similarity_order: the order of distance_mat. shape: (len(x), len(other_data))  
    '''

    x_preds = deep_cnn.softmax_preds(x, ckpt_path_final) # compute preds, deep_cnn.softmax_preds could be fed  one data now
    other_data_preds = deep_cnn.softmax_preds(other_data, ckpt_path_final)

    distances = np.zeros(len(other_data_preds))

    for j in range(len(other_data)):
        tem = x_preds - other_data_preds[j]
        # use which distance?!! here use L2 norm firstly
        distances[j] = np.linalg.norm(tem)
        # distance_X_tr_target[i, j] = np.sqrt(np.square(tem[FLAGS.target_class]) + np.square(tem[X_label[i]]))

    # sort(from small to large)
    nns_idx = np.argsort(distances)  # argsort every rows
    np.savetxt('similarity_order_X_all_tr_X', nns_idx)
    ordered_nns = other_data[nns_idx]

    return ordered_nns


def save_nns_fig(x, other_data, ckpt_path_final):
    ordered_data = get_nns_of_x(x, other_data, ckpt_path_final)
    for i in ordered_data:
        img_path = '../imgs/'+str(i)+'.png'
        save_fig(i, img_path)
        
    print('all near neighbors imgs saved at'+img_path)
    
    return True
def main(args=None):
    ckpt_path = FLAGS.train_dir + '/' + str(FLAGS.dataset) + '_' + 'train_data.ckpt'
    ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)
    


    x_train, x_test, y_train, y_test = ld_dataset(FLAGS.dataset, relative_path=True)
    x = x_test[0]
    other_data = x_test
    nns_of_x = get_nns_of_x(x, other_data, ckpt_path_final)
    print('nns_of_x.shape:',nns_of_x.shape)     
    
    for i in nns_of_x:
        a = 0
        img_path = '../imgs/'+str(i)+'.png'
        save_fig(i, img_path)
        a = a + 1
    print('all imgs saved at'+img_path)
        
if __name__ == '__main__':
    main()
