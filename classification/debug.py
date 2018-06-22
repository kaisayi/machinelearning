import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from numpy.linalg import norm
from 
fpath = 'iris.data'

def read_data(file):
    df = pd.read_csv(file, header=None)
    return df.values

# k-NN
# calculate the Euler distance
def distance_(xi, xc):
    xi, xc = np.array(xi), np.array(xc)
    return norm(xi-xc)

def maj_vote(yset):
    return np.bincount(yset).argmax()

# implement KD-search-tree

class Node:
    
    def __init__(self, data=None, left=None, right=None, label=None):
        self.data = data
        self.left = left
        self.right = right
        
    @property
    def is_leaf(self):
        return self.data is None or all(bool(c) for c, _ in self.children)
    
    @property
    def children(self):
        if self.left:
            yield self.left, 0
        if self.right:
            yield self.right, 1
    
    def preorder(self):
        '''
        retrieve the tree by preorder
        '''
        if not self:
            return
        
        yield self
        
        if self.left:
            for x in self.left.preorder():
                yield x
        
        if self.right:
            for x in self.right.preorder():
                yield x
    
    def inorder(self):
        '''
        inorder : left, self, right
        '''     
        if not self:
            return
        
        if self.left:
            for x in self.left.inorder():
                yield x
        
        yield self
        
        if self.right:
            for x in self.right.inorder():
                yield x
    
    def postorder(self):
        '''
        opstorder : left right self
        '''
        if not self:
            return
        
        if self.left:
            for x in self.left.postorder():
                yield x
        
        if self.right:
            for x in self.right.postorder():
                yield x
        
        yield self
        
    def height(self):
        
        min_height = int(bool(self))
        
        return max([min_height] + [c.height()+1 for c, _ in self.children])
    
    def __repr__(self):
        return "<{node}: {data}>".format(node=self.__class__, data=repr(self.data))
    
    def __nonzero__(self):
        return self.data is not None
    
    __bool__ = __nonzero__
    

class KDNode(Node):
    
    def __init__(self, data=None, left=None, right=None, label=None, ax=None, sel_axis=None, dimensions=None):
        '''
        axis : split axis of current node
        sel_axis: split axis of children node
        '''
        super(KDNode, self).__init__(data, left, right, label)
        
        self.ax = ax
        self.sel_axis = sel_axis
        self.dimensions = dimensions
        
    def search_knn(self, point, k, dist=None):
        '''
        return k nearest neighbors of points and distance
        '''
        if k < 1:
            raise 'the k must bigger than 0'
            
        getdist = dist or (lambda nd, p: distance_(nd.data, p))
        
        # k 最近临点结果初始化
        results = []
        # 查找K 最紧邻点的坐标
        self._search_node(point, k, results, getdist, count())
        
        # K 近邻点排序
        return [(node, -d) for node, _, d in sorted(results, reverse=True)]
        
    def _search_node(self, point, k, results, getdist, counter):
        
        if not self:
            return
        
        nodeDist = getdist(self, point)
        # Add current node to the prior queue if it is closer than at least one 
        # one points in queue
        # If the heap is at its capacity, we need to check if the current node 
        # is closer than the current farthest node 
        item = (-nodeDist, next(counter), self)
        if len(results) >= k:
            if -nodeDist > results[0][0]:
                heapq.heapreplace(results, item)
        else:
            heapq.heappush(results, item)
        
        
        # search the children tree
        # get the split plane and the distance of point to plane
        split_plane = self.data[self.ax]
        _dist_plane = point[self.ax] - split_plane
        
        if point[self.ax] <= split_plane:
            if self.left:
                self.left._search_node(point, k, results, getdist, counter)
                
        else:
            if self.right:
                self.right._search_node(point, k, results, getdist, counter)
        
        # if the distance between the point and plane is larger than the largest distance in the results
        # search the next side of the node
        if -_dist_plane**2 > results[0][0] or len(results) < k:
            if point[self.ax] <= split_plane:
                if self.right:
                    self.right._search_node(point, k, results, getdist, counter)
            else:
                if self.left:
                    self.left._search_node(point, k, results, getdist, counter)
                    
                     
    
    def search_nn(self, point, dist=None):
        '''
        find the nearset node of the given point
        '''
        return next(iter(self.search_knn(point, 1, dist)), None)    
        

def create(points, labels=None, dimensions=None, ax=0, sel_axis=None):
    '''
    create kd-tree from the poinst 
    '''
    dimensions = check_dimension(points, dimensions) # self check the dimension of the points
    
    # initiate the rule of generate the next axis
    sel_axis = sel_axis or (lambda prev: (prev + 1)%dimensions)
    
    if points is None or len(points) < 1:
        try:
            return KDNode(ax=ax, sel_axis=sel_axis, dimensions=dimensions)
        except:
            print(ax)
            raise ValueError
    
    if labels is None or len(labels) < 1:
        labels = [None] * len(points)
    
    assert len(points) == len(labels)
    
    # create the kd-tree
    points = np.array(points)
    labels = np.array(labels).reshape(-1, 1)
    
    datas = np.concatenate((points, labels), axis=1)
    
    args_sort = datas[:, ax].argsort()
    
    # 
    median = datas.shape[0] // 2
    
    loc = datas[median, :-1]
    lab = datas[median, -1]
    ldata = datas[args_sort[:median]]
    rdata = datas[args_sort[median+1:]]
    left = create(ldata[:, :-1], ldata[:, -1], dimensions, sel_axis(ax))
    right = create(rdata[:, :-1], rdata[:, -1], dimensions, sel_axis(ax))
    
    return KDNode(loc, left, right, ax=axis, sel_axis=sel_axis, dimensions=dimensions)    


def check_dimension(points, dimensions=None):
    
    dimensions = dimensions or len(points[0])
    
    for i, p in enumerate(points):
        if len(p) != dimensions:
            raise ValueError("the %s-th value of the point has invalid dimension"%(i))
            
    return dimensions

if __name__ == '__main__':
    