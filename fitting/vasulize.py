import matplotlib.pyplot as plt 
import numpy as np 

from surf_fitting import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 


def plot():
    eps2_path = os.path.join(os.getcwd(), 'best.hdf5')
    model = load_model(eps2_path)
    df_ = load(colname=['x', 'y', 'dx', 'dy'])
    x = df_['x'].values
    y = df_['y'].values 
    real_z = df_['dx'].values 
    X, Y = np.meshgrid(np.linspace(x.min(), x.max()), np.linspace(y.min(), y.max()))
    print(X)
    print(Y)
    eps1 = model.predict(np.c_[X.ravel(), Y.ravel()])
    eps1 = eps1.reshape(X.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, real_z, marker='o', s=20, c='k', alpha=0.5)

    surf = ax.plot_surface(X, Y, eps1, cmap=cm.rainbow,
                       linewidth=0, antialiased=False)

    plt.show()

def plot_elatic_surf():
    df_, X, Y = calculate_()
    
    e11 = df_['E22'].values
    e11 = e11.reshape(X.shape)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, e11, cmap=cm.rainbow,
                    linewidth=0, antialiased=False)

    plt.show()


if __name__ == '__main__':
    # plot_elatic_surf()
    plot()
    
    