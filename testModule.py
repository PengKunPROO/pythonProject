import numpy as np

x=np.array([[0,1],[0,2],[0,3]])#this is a 3 rows 2 columns array
print(x.shape)
X=x[:, 1]#表示取所有的行的第2个数据
print(X)
y=np.array([
            [
            [1,2],[1,3]
             ],
            [
            [1,4],[1,5]
             ]
            ])
print('shape of y is '+str(y.shape))
Y=y[:,:,0]
print(Y.shape)
print(Y)