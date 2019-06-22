#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# In[4]:


print(np.__version__)
np.show_config()


# In[5]:


Z=np.zeros(10)
print(Z)


# In[6]:


Z=np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))


# In[9]:


np.info(np.add) #前面numpy已经as成np,所以这里就不用numpy而是用np。


# In[12]:


Z=np.zeros(10)
Z[4] = 1
print(Z)


# In[14]:


Z=np.arange(10,50)
print(Z)


# In[15]:


Z=np.arange(50)
Z=Z[::-1]
print(Z)


# In[16]:


Z=np.arange(9).reshape(3,3)
print(Z)


# In[17]:


nz = np.nonzero([1,2,0,0,4,0])
print(nz)


# In[18]:


Z=np.eye(3)
print(Z)


# In[19]:


Z=np.random.random((3,3,3))
print(Z)


# In[20]:


Z=np.random.random((10,10))
Zmin, Zmax, = Z.min(), Z.max()
print(Zmin, Zmax)


# In[24]:


Z=np.random.random(30)
m=Z.mean()
print(m)


# In[26]:


Z=np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)


# In[28]:


Z=np.ones((5,5))
Z=np.pad(Z, pad_width = 1, mode = 'constant', constant_values = 0)
print(Z)


# In[35]:


print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(0.3 != 3 * 0.1)


# In[36]:


Z=np.diag(1+np.arange(4),k=-1)
print(Z)


# In[37]:


Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)


# In[38]:


print(np.unravel_index(100,(6,7,8)))


# In[39]:


Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)


# In[40]:


Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)


# In[41]:


color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])
color


# In[42]:


Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)


# In[43]:


Z=np.arange(11)
Z[(Z>3)&(Z<=8)] *= -1
print(Z)


# In[44]:


print(sum(range(5),-1))


# In[45]:


from numpy import *
print(sum(range(5),-1))


# In[49]:


Z = np.arange(5)
Z ** Z  # legal


# In[55]:


Z = np.arange(5)
2 << Z >> 2  # false


# In[51]:


Z = np.arange(5)
Z <- Z   # legal


# In[52]:


Z = np.arange(5)
1j*Z   # legal


# In[53]:


Z = np.arange(5)
Z/1/1   # legal


# In[54]:


Z = np.arange(5)
Z<Z>Z    # false


# In[56]:


print(np.array(0) / np.array(0))


# In[57]:


print(np.array(0) // np.array(0))


# In[58]:


print(np.array([np.nan]).astype(int).astype(float))


# In[59]:


Z = np.random.uniform(-10,+10,10)
print (np.copysign(np.ceil(np.abs(Z)), Z))


# In[63]:


Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))#找到两个数组中的共同元素


# In[71]:


# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0


# In[74]:


# Back to sanity
_ = np.seterr(**defaults)

#An equivalent way, with a context manager:
    
with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0


# In[75]:


np.sqrt(-1) == np.emath.sqrt(-1)  # False,虚数


# In[76]:


yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print ("Yesterday is " + str(yesterday))
print ("Today is " + str(today))
print ("Tomorrow is "+ str(tomorrow))


# In[77]:


Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)


# In[78]:


A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B) #A+B


# In[79]:


np.divide(A,2,out=A) #A/2


# In[80]:


np.negative(A,out=A)#A取负数


# In[81]:


np.multiply(A,B,out=A)#A*B


# In[83]:


Z = np.random.uniform(0,10,10)#五种方法提取一个随机数组的整数部分

print (Z - Z%1)


# In[84]:


print (np.floor(Z))


# In[85]:


print (np.ceil(Z)-1)


# In[86]:


print (Z.astype(int)) #后面的点去掉了


# In[87]:


print (np.trunc(Z))


# In[88]:


Z = np.zeros((5,5))
Z += np.arange(5)
print (Z)


# In[91]:


def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print (Z)


# In[94]:


Z = np.linspace(0,1,11,endpoint=False)[1:] #创建一个长度为10的随机向量，其值域范围从0到1，但是不包括0和1
print (Z)


# In[95]:


Z=np.random.random(10) #创建一个长度为10的随机向量，并将其排序
Z.sort()
print (Z)


# In[96]:


Z = np.arange(10)
np.add.reduce(Z)


# In[101]:


A = np.random.randint(0,2,5) # 对于两个随机数组A和B，检查它们是否相等
B = np.random.randint(0,2,5) 
#equal = np.allclose(A,B)
equal = np.array_equal(A,B)
print(equal)


# In[102]:


Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1


# In[106]:


Z = np.random.random((10,2)) #44. 将笛卡尔坐标下的一个10x2的矩阵转换为极坐标形式
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)


# In[110]:


Z = np.random.random(10) #45. 创建一个长度为10的向量，并将向量中最大值替换为1
Z[Z.argmax()] = 0
print(Z)


# In[111]:


Z = np.zeros((5,5), [('x',float),('y',float)]) #46. 创建一个结构化数组，并实现 x 和 y 坐标覆盖 [0,1]x[0,1] 区域
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                            np.linspace(0,1,5))
print(Z)


# In[118]:


X = np.arange(8) #??? 47.构造Cauchy矩阵C (Cij =1/(xi - yj)),构建成功了吗
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))


# In[120]:


for dtype in [np.int8, np.int32, np.int64]: #48. 打印每个numpy标量类型的最小值和最大值
    print(np.iinfo(dtype).min) 
    print(np.iinfo(dtype).max)

for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)


# In[131]:


np.set_printoptions(threshold=1) #49为什么threshold must be numeric and non-NAN？threshold换成数字才可以。
Z = np.zeros((16,16))
print (Z)


# In[132]:


Z = np.arange(100) #50. 给定标量时，如何找到数组中最接近标量的值？
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print (Z[index])


# In[133]:


Z = np.zeros(10, [ ('position', [ ('x', float, 1), #51. 创建一个表示位置(x,y)和颜色(r,g,b)的结构化数组
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print (Z)


# In[134]:


Z = np.random.random((10,2)) #52. 对一个表示坐标形状为(100,2)的随机向量，找到点与点的距离
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print (D)


# In[135]:


# # 方法2
# # Much faster with scipy
import scipy
# # Thanks Gavin Heverly-Coulson (#issue 1)
import scipy.spatial
D = scipy.spatial.distance.cdist(Z,Z)
print (D)


# In[137]:


Z = np.arange(10, dtype=np.int32) #将32位的浮点数(float)转换为对应的整数(integer)
Z = Z.astype(np.float32, copy=False)
print (Z)


# In[138]:


Z = np.arange(9).reshape(3,3) #枚举。55、对于numpy数组，enumerate的等价操作
for index, value in np.ndenumerate(Z):
   print (index, value)
for index in np.ndindex(Z.shape):
    print (index, Z[index])


# In[139]:


X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10)) #56. 生成一个通用的二维Gaussian-like数组
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print (G)


# In[144]:


n = 5 #57. 对一个二维数组，如何在其内部随机放置p个元素?
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print (Z)


# In[146]:


X = np.random.rand(5, 10) #58. 减去一个矩阵中的每一行的平均值
# # Recent versions of numpy
Y = X - X.mean(axis=1, keepdims=True)
print(Y)


# In[147]:


# # 方法2
# # Older versions of numpy
Y = X - X.mean(axis=1).reshape(-1, 1)
print (Y)


# In[3]:


import numpy as np #59. 如何通过第n列对一个数组进行排序?
Z = np.random.randint(0,10,(3,3))
print (Z)
print (Z[Z[:,0].argsort()])


# In[4]:


Z = np.random.randint(0,3,(3,10)) #60. 如何检查一个二维数组是否有空列？
print ((~Z.any(axis=0)).any())


# In[5]:


Z = np.random.uniform(0,1,10) #61. 从数组中的给定值中找出最近的值 
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print (m)


# In[6]:


A = np.arange(3).reshape(3,1) #62. 如何用迭代器(iterator)计算两个分别具有形状(1,3)和(3,1)的数组?
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it: 
    z[...] = x + y
print (it.operands[2])


# In[7]:


class NamedArray(np.ndarray): #（再看）63. 创建一个具有name属性的数组类
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)


# In[12]:


Z = np.ones(10) #（有难度！！！）64. 考虑一个给定的向量，如何对由第二个向量索引的每个元素加1(小心重复的索引)?
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)


# In[18]:


# # 方法2
np.add.at(Z, I, 1)
print(Z)


# In[19]:


X = [1,2,3,4,5,6] #65. 根据索引列表(I)，如何将向量(X)的元素累加到数组(F)?
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print (F)


# In[21]:


w,h = 16,16 #66. 考虑一个(dtype=ubyte) 的 (w,h,3)图像，计算其唯一颜色的数量
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte) 
# #Note that we should compute 256*256 first. 
# #Otherwise numpy will only promote F.dtype to 'uint16' and overfolw will occur
F = I[...,0]*(256*256) + I[...,1]*256 +I[...,2]
n = len(np.unique(F))
print (n)


# In[22]:


A = np.random.randint(0,10,(3,4,3,4)) #67. 考虑一个四维数组，如何一次性计算出最后两个轴(axis)的和？
# # solution by passing a tuple of axes (introduced in numpy 1.7.0)
sum = A.sum(axis=(-2,-1))
print (sum)


# In[23]:


# # 方法2
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print (sum)


# In[24]:


D = np.random.uniform(0,1,100) #68. 考虑一个一维向量D，如何使用相同大小的向量S来计算D子集的均值？
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print (D_means)


# In[25]:


# # 方法2
import pandas as pd
print(pd.Series(D).groupby(S).mean())


# In[28]:


A = np.random.uniform(0,1,(5,5)) #69. 如何获得点积 dot prodcut的对角线?
B = np.random.uniform(0,1,(5,5))
# # slow version
np.diag(np.dot(A, B))

## 方法2
# # Fast version
np.sum(A * B.T, axis=1)


# In[29]:


## 方法3
# # Faster version
np.einsum("ij,ji->i", A, B)


# In[30]:


Z = np.array([1,2,3,4,5]) #70. 考虑一个向量[1,2,3,4,5],如何建立一个新的向量，在这个新向量中每个值之间有3个连续的零？
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print (Z0)


# In[31]:


A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print (A * B[:,:,None])


# In[32]:


A = np.arange(25).reshape(5,5) #72. 如何对一个数组中任意两行做交换?
A[[0,1]] = A[[1,0]]
print (A)


# In[33]:


faces = np.random.randint(0,100,(10,3)) #（有难度！！！）73. 考虑一个可以描述10个三角形的triplets，找到可以分割全部三角形的line segment
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print (G)


# In[34]:


C = np.bincount([1,1,2,3,4,4,6]) #74. 给定一个二进制的数组C，如何产生一个数组A满足np.bincount(A)==C(★★★)
A = np.repeat(np.arange(len(C)), C)
print (A)


# In[35]:


def moving_average(a, n=3) : #75. 如何通过滑动窗口计算一个数组的平均数? (★★★)
(提示: np.cumsum)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)

print(moving_average(Z, n=3))


# In[40]:


from numpy.lib import stride_tricks 
#76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)
def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)

print (Z)


# In[42]:


Z = np.random.randint(0,2,100) #77. 如何对布尔值取反，或者原位(in-place)改变浮点数的符号(sign)？(★★★)
np.logical_not(Z, out=Z)


# In[43]:


Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z)


# In[2]:


import numpy as np 
#78. 考虑两组点集P0和P1去描述一组线(二维)和一个点p,如何计算点p到每一条线 i (P0[i],P1[i])的距离？(★★★)
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))

print (distance(P0, P1, p))


# In[3]:


# # based on distance function from previous question
#79.考虑两组点集P0和P1去描述一组线(二维)和一组点集P，如何计算每一个点 j(P[j]) 到每一条线 i (P0[i],P1[i])的距离？(★★★)
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print (np.array([distance(P0,P1,p_i) for p_i in p]))


# In[8]:


#81. 考虑一个数组Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14],如何生成一个数组R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ...,[11,12,13,14]]? (★★★)
Z = np.arange(1,15,dtype=np.uint32) 
#??? NameError: name 'stride_tricks' is not defined.原代码有点问题，需要改成：（加个：np.lib.）np.lib.stride_tricks.as_strided
R = np.lib.stride_tricks.as_strided(Z,(11,4),(4,4))
print (R)


# In[9]:


Z = np.random.uniform(0,1,(10,10)) #82. 计算一个矩阵的秩
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print (rank)


# In[12]:


#83. 如何找到一个数组中出现频率最高的值？
Z = np.random.randint(0,10,50)
print (np.bincount(Z).argmax())


# In[14]:


Z = np.random.randint(0,5,(10,10)) #84. 从一个10x10的矩阵中提取出连续的3x3区块(★★★)
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = np.lib.stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print (C)


# In[17]:


class Symetric(np.ndarray): #85. 创建一个满足 Z[i,j] == Z[j,i]的子类 (★★★) (提示: class 方法)
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print (S)


# In[18]:


p, n = 10, 20 #86. 考虑p个 nxn 矩阵和一组形状为(n,1)的向量，如何直接计算p个矩阵的乘积(n,1)？(★★★)
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print (S)
#It works, because:
#M is (p,n,n)
#V is (p,n,1)
#Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.


# In[19]:


Z = np.ones((16,16)) 
#87. 对于一个16x16的数组，如何得到一个区域(block-sum)的和(区域大小为4x4)? (★★★)
#(提示: np.add.reduceat)
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print (S)


# In[23]:


def iterate(Z): #88. 如何利用numpy数组实现Game of Life? (★★★)
# Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

# Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)
print (Z)


# In[28]:


#89. 如何找到一个数组的第n个最大值? (★★★)
#(提示: np.argsort | np.argpartition)
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

# # 方法1
# # Slow
#print (Z[np.argsort(Z)[-n:]])

# # 方法2
# # Fast
print (Z[np.argpartition(-Z,n)[:n]])


# In[29]:


#90. 给定任意个数向量，创建笛卡尔积(每一个元素的每一种组合)(★★★)
#(提示: np.indices)
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))


# In[30]:


#91. 如何从一个正常数组创建记录数组(record array)? (★★★)
#(提示: np.core.records.fromarrays)
Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T, 
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print (R)


# In[36]:


#92. 考虑一个大向量Z, 用三种不同的方法计算它的立方(★★★)
#(提示: np.power, \*, np.einsum)
x = np.random.rand()
np.power(x,3)

## 方法2
# x*x*x

## 方法3 ???这个方法的代码有点问题！
#np.einsum('i,i,i->i',x,x,x)


# In[37]:


#93. 考虑两个形状分别为(8,3) 和(2,2)的数组A和B. 如何在数组A中找到满足包含B中元素的行？(不考虑B中每行元素顺序)？ (★★★)
#(提示: np.where)
A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print (rows)


# In[39]:


#94. 考虑一个10x3的矩阵，分解出有不全相同值的行 (如 [2,2,3]) (★★★)
Z = np.random.randint(0,5,(10,3))
print (Z)

# # solution for arrays of all dtypes (including string arrays and record arrays)
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print (U)

# # 方法2
# # soluiton for numerical arrays only, will work for any number of columns in Z
# U = Z[Z.max(axis=1) != Z.min(axis=1),:]
# print (U)


# In[40]:


#95. 将一个整数向量转换为matrix binary的表现形式 (★★★)
#(提示: np.unpackbits)
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])

# # 方法2
# print (np.unpackbits(I[:, np.newaxis], axis=1))


# In[41]:


#96. 给定一个二维数组，如何提取出唯一的(unique)行?(★★★)
#(提示: np.ascontiguousarray)

Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print (uZ)


# In[42]:


#97. 考虑两个向量A和B，写出用einsum等式对应的inner, outer, sum, mul函数(★★★)
#(提示: np.einsum)


A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)
print ('sum')
print (np.einsum('i->', A))# np.sum(A)

print ('A * B')
print (np.einsum('i,i->i', A, B)) # A * B

print ('inner')
print (np.einsum('i,i', A, B))    # np.inner(A, B)

print ('outer')
print (np.einsum('i,j->ij', A, B))    # np.outer(A, B)


# In[44]:


#98. 考虑一个由两个向量描述的路径(X,Y)，如何用等距样例(equidistant samples)对其进行采样(sample)? (★★★)
#Considering a path described by two vectors (X,Y), how to sample it using equidistant samples
#(提示: np.cumsum, np.interp)


phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrate path
r_int = np.linspace(0, r.max(), 200) # regular spaced path
x_int = np.interp(r_int, r, x)       # integrate path
y_int = np.interp(r_int, r, y)


# In[45]:


#99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)
#(提示: np.logical_and.reduce, np.mod)

X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print (X[M])


# In[46]:


#100. 对于一个一维数组X，计算它boostrapped之后的95%置信区间的平均值。
#(Compute bootstrapped 95% confidence intervals for the mean of a 1D array X，i.e. resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)
#(提示: np.percentile)

X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print (confint)


# In[ ]:




