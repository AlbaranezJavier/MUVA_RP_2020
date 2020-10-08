import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt

# def checkZero(lines):
#     if (checkLine(lines[14,:]) >= 2 or checkLine(lines[16,:]) >= 2) \
#             and \
#             (checkLine(lines[:,14]) >= 2 or checkLine(lines[:,12]) >= 2 or checkLine(lines[:,17]) >= 2):
#         return 1
#     return 0

def checkZero(lines):
    # Comprueba si tiene un hueco en el centro
    m12 = checkLine(lines[12,:])
    m15 = checkLine(lines[15,:])
    m16 = checkLine(lines[16,:])
    m18 = checkLine(lines[18,:])
    n12 = checkLine(lines[:, 12])
    n14 = checkLine(lines[:, 14])
    n16 = checkLine(lines[:, 16])
    if (m12 and n12) or (m15 and n12) or (m18 and n12) or (m15 and n14) or \
            (m18 and n14) or (m15 and n16) or (m18 and n16):
        if m16:
            return 1
    return 0

# def checkZero(lines):
#     if (checkLine(lines[12,:]) and checkLine(lines[:,12]))\
#             or\
#             (checkLine(lines[15,:]) and checkLine(lines[:,12]))\
#             or\
#             (checkLine(lines[18,:]) and checkLine(lines[:,12])) \
#             or \
#             (checkLine(lines[18, :]) and checkLine(lines[:, 14])) \
#             or \
#             (checkLine(lines[18, :]) and checkLine(lines[:, 16])):
#         if checkLine(lines[16,:]):
#             return 1
#         return 0
#     return 0

# def checkZero(lines):
#     if checkLine(lines[12, :]) or checkLine(lines[14, :]) or checkLine(lines[16, :]):
#         return 1
#     return 0

def checkLine(line):
    # Comprueba el numero de maximos de una linea
    # return True si solo hay un max
    # return False si hay más de un max
    count = 0
    prev = 0
    for x in line:
        if x > 0.2 and prev == 0:
            prev = 1
            count = count + 1
        elif x < 0.2 and prev == 1:
            prev = 0
    return count != 1


#-------------------------------#
# Extracción de características #
#-------------------------------#
def feat_extraction (data, theta=0.1):
    # data: dataframe
    # theta: parameter of the feature extraction
    # features extracted:
    #   'width','W_max1','W_max2','W_max3',
    #   'height','H_max1','H_max2','H_max3',
    #   'area','w_vs_h'
    #
    features = np.zeros([data.shape[0], 13]) #<- allocate memory with zeros
    data = data.values.reshape([data.shape[0],28,28])
    #-> axis 0: id of instance, axis 1: width(cols) , axis 2: height(rows)
    for k in range(data.shape[0]):
        #..current image
        x = data[k,:,:]
        #--width feature
        sum_cols = x.sum(axis=0) #<- axis0 of x, not of data!!
        indc = np.argwhere(sum_cols > theta * sum_cols.max())
        col_3maxs = np.argsort(sum_cols)[-3:]
        features[k,0] = indc[-1] - indc[0]
        features[k,1:4] = col_3maxs
        #meanWMAX
        features[k,10] = np.sum(col_3maxs)/3
        #--tint: cantidad de tinta, suma los valores del grid
        features[k,11] = x.sum()
        #--zero: tiene un hueco en el centro?
        features[k,12] = checkZero(x)
        #--height feature
        sum_rows = x.sum(axis=1) #<- axis1 of x, not of data!!
        indr = np.argwhere(sum_rows > theta * sum_rows.max())
        features[k,4] = indr[-1] - indr[0]
        row_3maxs = np.argsort(sum_rows)[-3:]
        features[k,5:8] = row_3maxs
    #--area
    features[:,8] = features[:,0] * features[:,4]
    #--ratio W/H
    features[:,9] = features[:,0] / features[:,4]

    #features[:,10] = (features[:,1] > features[:,0])*1 + (features[:,2] > features[:,0])*1 + (features[:,3] > features[:,0])*1
    #features[:,10] =  np.sum(features[:,1:3] > [np.concatenate([features[:,0],features[:,0],features[:,0]], axis=1)
    #print(np.count_nonzero(features[:,1:4] > theta, axis=1))
    #print(features[:,11])
    col_names = ['width','W_max1','W_max2','W_max3','height','H_max1','H_max2','H_max3','area','w_vs_h','meanWMAX','tint','zero']
    # scaler = MinMaxScaler()
    # scaler.fit(features)
    # features = scaler.transform(features)
    return pd.DataFrame(features,columns = col_names)

#---------------------#
# Separacion de datos #
#---------------------#
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_set = data.iloc[train_indices]
    test_set  = data.iloc[test_indices]
    return train_set.reset_index(drop=True), test_set.reset_index(drop=True)

#------------------------------#
# Algunas funciones auxiliares #
#------------------------------#

def join_features_labels(X0,X1):
    Y0 = pd.DataFrame(np.zeros(X0.shape[0]),columns=['label'])
    XY0 = pd.concat([X0,Y0],axis=1)
    Y1 = pd.DataFrame(np.ones(X1.shape[0]),columns=['label'])
    XY1 = pd.concat([X1,Y1],axis=1)
    return pd.concat([XY0,XY1],axis=0,ignore_index=True)

def jitter(X,sigma=0.2):
    random_sign = (-1)**np.random.randint(1,3,*X.shape)
    return X + np.random.normal(0,sigma,*X.shape)*random_sign

def extractor(theta):
    np.random.seed(seed=1234) #<- comment this to get randomness
    fraction_Test  = 0.2
    fraction_Valid = 0.2

    # --- Get data -------------------------------------
    FullSet_0 = pd.read_csv('../../Datasets/1000ceros.csv', header=None)
    FullSet_1 = pd.read_csv('../../Datasets/1000unos.csv',  header=None)
    FullSet_0 = FullSet_0 /255. #<- quick rescale to [0,1]
    FullSet_1 = FullSet_1 /255. #<- quick rescale to [0,1]

    # --- Separate Test sets -----------------------------
    TrainSet_0, TestSet_0 = split_train_test(FullSet_0, fraction_Test)
    TrainSet_1, TestSet_1 = split_train_test(FullSet_1, fraction_Test)

    # --- Separate Validation sets -----------------------
    TrainSet_0, ValidSet_0 = split_train_test(TrainSet_0, fraction_Valid)
    TrainSet_1, ValidSet_1 = split_train_test(TrainSet_1, fraction_Valid)

    # --- Ensamble TRAIN SET, VALIDATION SET Y TEST SET --
    #          with the features and the labels
    Feat_train = join_features_labels(feat_extraction(TrainSet_0, theta=theta),
                                      feat_extraction(TrainSet_1, theta=theta))
    Feat_valid = join_features_labels(feat_extraction(ValidSet_0, theta=theta),
                                      feat_extraction(ValidSet_1, theta=theta))
    Feat_test  = join_features_labels(feat_extraction(TestSet_0,  theta=theta),
                                      feat_extraction(TestSet_1,  theta=theta))
    feat_1 = 'tint'
    feat_2 = 'zero'

    #-[2].Fit a LogisticRegression model (a linear classifier) with Feat_train dataframe
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    #clf.fit(Feat_train[[feat_1, feat_2]], Feat_train['label'])
    clf.fit(Feat_train[[feat_1, feat_2]], Feat_train['label'])

    #-[3].Predict the Feat_valid dataframe
    #y_pred = clf.predict( Feat_valid[[feat_1, feat_2]] )
    y_pred = clf.predict(Feat_test[[feat_1, feat_2]])

    #-[4].Compare the predictions with the ground truth
    from sklearn.metrics import confusion_matrix
    # conf_mat = confusion_matrix(y_pred, Feat_valid['label'])
    conf_mat = confusion_matrix(y_pred, Feat_test['label'])
    return conf_mat[0,0]+conf_mat[1,1]

nombres = ['Javier Albaráñez Martínez']
if True:
    #------------------------------#
    # Construcción de los datasets #
    #------------------------------#
    np.random.seed(seed=1234) #<- comment this to get randomness
    fraction_Test  = 0.2
    fraction_Valid = 0.2
    theta = 0.51

    # --- Get data -------------------------------------
    FullSet_0 = pd.read_csv('../../Datasets/1000ceros.csv', header=None)
    FullSet_1 = pd.read_csv('../../Datasets/1000unos.csv',  header=None)
    FullSet_0 = FullSet_0 /255. #<- quick rescale to [0,1]
    FullSet_1 = FullSet_1 /255. #<- quick rescale to [0,1]

    # --- Separate Test sets -----------------------------
    TrainSet_0, TestSet_0 = split_train_test(FullSet_0, fraction_Test)
    TrainSet_1, TestSet_1 = split_train_test(FullSet_1, fraction_Test)

    # --- Separate Validation sets -----------------------
    TrainSet_0, ValidSet_0 = split_train_test(TrainSet_0, fraction_Valid)
    TrainSet_1, ValidSet_1 = split_train_test(TrainSet_1, fraction_Valid)

    # --- Ensamble TRAIN SET, VALIDATION SET Y TEST SET --
    #          with the features and the labels
    Feat_train = join_features_labels(feat_extraction(TrainSet_0, theta=theta),
                                      feat_extraction(TrainSet_1, theta=theta))
    Feat_valid = join_features_labels(feat_extraction(ValidSet_0, theta=theta),
                                      feat_extraction(ValidSet_1, theta=theta))
    Feat_test  = join_features_labels(feat_extraction(TestSet_0,  theta=theta),
                                      feat_extraction(TestSet_1,  theta=theta))

    #----------------------------#
    # Entrenamiento y evaluación #
    #----------------------------#

    #-[1].Select any 2 features from the list:
    #    -features list: 'width','W_max1','W_max2','W_max3','height','H_max1','H_max2','H_max3','area','w_vs_h'
    feat_1 = 'zero'
    feat_2 = 'tint'

    #-[2].Fit a LogisticRegression model (a linear classifier) with Feat_train dataframe
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(Feat_train[[feat_1, feat_2]], Feat_train['label'])
    # clf.fit(Feat_train[[feat_1, feat_2]], Feat_train['label'])

    # -[3].Predict the Feat_valid dataframe
    y_pred = clf.predict( Feat_valid[[feat_1, feat_2]] )
    # y_pred = clf.predict(Feat_test[[feat_1, feat_2]])

    # -[4].Compare the predictions with the ground truth
    from sklearn.metrics import confusion_matrix

    conf_mat = confusion_matrix(y_pred, Feat_valid['label'])
    # conf_mat = confusion_matrix(y_pred, Feat_test['label'])
    print(conf_mat)
    N_success = conf_mat[0,0]+conf_mat[1,1]
    N_fails = conf_mat[0,1]+conf_mat[1,0]
    print (nombres,"\n")
    print("Outcome:")
    strlog = "  :) HIT  = %d, (%0.2f%%)"%(N_success, 100*N_success/(N_success+N_fails))
    print(strlog)
    strlog = "  :( FAIL = %d, (%0.2f%%)"%(N_fails, 100*N_fails/(N_success+N_fails))
    print(strlog)

    #-[5].Show the (linear) model parameters
    print("\nLogistic regression model:")
    print("  clf coef. = ",clf.coef_)
    print("  clf intercept = ",clf.intercept_)

    #-[6].Plot Feat_valid and the model
    ind = Feat_valid['label']==0
    x0, x1 = Feat_valid[ind][feat_1], Feat_valid[~ind][feat_1]
    y0, y1 = Feat_valid[ind][feat_2], Feat_valid[~ind][feat_2]
    plt.plot(jitter(x0),jitter(y0),'yo',jitter(x1),jitter(y1),'bx', alpha=.3)

    w = clf.coef_[0]
    a = -w[0] / w[1]
    xmin=min(x0.min(axis=0),x1.min(axis=0))
    xmax=max(x0.max(axis=0),x1.max(axis=0))
    ymin=min(y0.min(axis=0),y1.min(axis=0))
    ymax=max(y0.max(axis=0),y1.max(axis=0))
    xx = np.linspace(xmin,xmax)
    yy = a * xx - (clf.intercept_[0] / w[1])
    plt.plot(xx,yy,'r')
    strTitle = "w_X = %2.2f, w_Y = %2.2f, w_0 = %2.2f " % (w[0], w[1], clf.intercept_[0])
    plt.axis([xmin-1,xmax+1,ymin-1,ymax+1])
    plt.title(strTitle)
    plt.xlabel(feat_1)
    plt.ylabel(feat_2)
    plt.show()
else:
    max = 0
    best = 0
    for i in range(0,100):
        try :
            hits = extractor(i/100)
            print(hits)
            if hits > max:
                max = hits
                best = i/100
        except ValueError:
            print(f'Fallo en {i}')
    print(f'{best} consiguió {max}')