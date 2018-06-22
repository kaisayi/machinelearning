import xlrd
import os
import pandas as pd 
import numpy as np 

from numpy.linalg import solve
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

def load(sheet=0, colname=None):
    file = os.path.join(os.getcwd(), 'fitting/strain.xlsx')
    if colname is None: 
        return pd.read_excel(file, sheet_name=sheet, header=None)
    else: 
        return pd.read_excel(file, sheet_name=0, names=colname)
    
    

def to_xy(df, targets):

    # check targets is iterable or not
    if isinstance(targets, list):
        depends = [x for x in df.columns if x not in targets]
        
    else:
        depends = df.columns.tolist()
        depends.remove(targets)
        targets = [targets,]
    print(targets)

    # find out the type of the target column.  
    target_type = df[targets].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type

    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if (target_type in (np.int64, np.int32)) and (len(targets)==1):
        # Classification
        dummies = pd.get_dummies(df[targets])
        return df.as_matrix(depends).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        # Regression
        return df.as_matrix(depends).astype(np.float32), df.as_matrix(targets).astype(np.float32)

    

def train_():

    # create sequential model 
    df_ = load(colname=['x', 'y', 'dx', 'dy'])
    df_.drop('dx', axis=1, inplace=True)
    x, y = to_xy(df_, 'dy')


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=32)
    

    model = Sequential()
    model.add(Dense(20, input_dim=x.shape[1], activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(y.shape[1]))
    
    model.compile(loss='mean_squared_error', optimizer='adam')

    weighpath = os.path.join(os.getcwd(), 'best_eps2.hdf5')
    if os.path.exists(weighpath):
        os.remove(weighpath)

    checkpoint = ModelCheckpoint(filepath=weighpath, verbose=0, save_best_only=True)

    model.fit(x, y, validation_data=(x_test, y_test), callbacks=[checkpoint], verbose=2, epochs=1000)

    model.load_weights(weighpath)
    pred_test = model.predict(x_test)
    score_test = np.sqrt(metrics.mean_squared_error(pred_test, y_test))
    print("the score (RMSE) of test_data: ", score_test)

def load_model(path):
    '''
    load model weigh from path
    '''

    model = Sequential()
    model.add(Dense(20, input_dim=2, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.load_weights(path)
    return model 


def calculate_():

    # load model of strain-stress surface
    eps1_path = os.path.join(os.getcwd(), 'best.hdf5')
    eps2_path = os.path.join(os.getcwd(), 'best_eps2.hdf5')

    model_eps1 = load_model(eps1_path)
    model_eps2 = load_model(eps2_path)

    # generate the meshgrids
    df_ = load(colname=['x', 'y', 'dx', 'dy'])
    x = df_['x'].values
    y = df_['y'].values 
    real_z = df_['dy'].values 
    X, Y = np.meshgrid(np.linspace(x.min(), x.max()), np.linspace(y.min(), y.max()))

    # calculate the elestic parameters
    
    def calc_(xc, yc, step = 0.2):
        x_r = np.linspace(xc-step*4, xc+step*4, 5)
        y_r = np.linspace(yc-step*4, yc+step*4, 5)
        Xi, Yi = np.meshgrid(x_r, y_r)
        
        epsx = model_eps1.predict(np.c_[Xi.ravel(), Yi.ravel()])
        epsy = model_eps2.predict(np.c_[Xi.ravel(), Yi.ravel()])
        
        epsx = epsx.reshape(Xi.shape)
        epsy = epsy.reshape(Xi.shape)

        stress_xx = np.square(Xi).sum() 
        stress_yy = np.square(Yi).sum()
        stress_xy = (Xi * Yi).sum()
        epsx_x = (epsx * Xi).sum()
        epsy_y = (epsy * Yi).sum()  
        epsx_y = (epsx * Yi).sum()
        epsy_x = (epsy * Xi).sum()

        A_ = np.array([
            [stress_xx, 0, stress_xy],
            [0, stress_yy, stress_xy],
            [stress_xy, stress_xy, stress_xx+stress_yy]
        ])

        b_ = np.array([
            epsx_x,
            epsy_y,
            epsx_y + epsy_x
        ])
        
        res = solve(A_, b_)

        return res[0], res[1], res[2] 

    df_ = pd.DataFrame(data=np.c_[X.ravel(), Y.ravel()],
                       columns=('x', 'y'))
                  
    df_['E11'], df_['E22'], df_['E12'] = zip(*df_.apply(lambda row: calc_(row['x'], row['y']), axis=1))
                       
    return df_, X, Y 
    
def main():
    df, *_ = calculate_()
    print(df.head())    

if __name__ == '__main__':
    main()


