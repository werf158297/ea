import numpy as np
import math
from matplotlib import pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split  

#欧氏距离计算
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))  
 
#随机选k个初始聚类中心
def randCent(dataSet,k):
    m,n = dataSet.shape
 
    centroids = np.zeros((k,n))
    for i in range(k):
        index = int(np.random.uniform(0,m)) 
        centroids[i,:] = dataSet[index,:]
    return centroids

#凭经验选择
def randCent2(dataSet,k):
 m,n = dataSet.shape
 
 centroids = np.zeros((k,n))
 for j in range(k):
  
  centroids[j,:] = dataSet[j*50+25,:]
 return centroids

#随机分，以重心为初始类中心 
def randCent3(dataSet,k):
 m,n = dataSet.shape
 centroids = np.zeros((k,n))
 k1= np.empty(shape=(m, 1))
 for i in range(m):
  k1[i] = int(np.random.uniform(0,k)) 
 
 for j in range(k):
  pointsInCluster = dataSet[np.nonzero(k1 == j)[0]]  # 获取簇类所有的点
  centroids[j,:] = np.mean(pointsInCluster,axis=0)   # 对矩阵的行求均值
 return centroids

#选择批次距离尽量远的点 
def randCent4(dataSet,k):
 m,n = dataSet.shape
 centroids = np.zeros((k,n))
 index = int(np.random.uniform(0,m)) 
 centroids[0,:] = dataSet[index,:]
 
 for i in range(1,k):
  maxDist = 0.0
  maxIndex = -1
  
#最大最小距离
  for j in range(m):
   minDist = 10000.0
   for p in range(i):
	
    distance = distEclud(centroids[p,:],dataSet[j,:])
    if distance < minDist:
     minDist = distance
     
   if minDist > maxDist: 
     maxDist = minDist
     maxIndex = j
  centroids[i,:] = dataSet[maxIndex,:]
       
 return centroids  

#密度法选择中心    
def randCent5(dataSet,k):
 m,n = dataSet.shape
 centroids = np.zeros((k,n))
 midu=[]
 r=0.5
 dt=25
 z=0
 ind=-1
 
 #找出密度大于阈值的点
 for i in range(m):
  c=0
  for j in range(m):
   distance = distEclud(dataSet[i,:],dataSet[j,:])
  # print(distance)
   if distance < r:
    c=c+1
  if c>dt:
   midu.append(i)
  if c>z:
	  z=c
	  ind=i
	  
 l=len(midu)
 dataset1 = np.zeros((l,n))
 for i in range(l):
	 
  dataset1[i,:] = dataSet[midu[i],:]
 centroids[0,:] = dataset1[ind,:]
 for i in range(1,k):
  maxDist = 0.0
  maxIndex = -1
 
#最大最小距离
  for j in range(l):
   minDist = 10000.0
   for p in range(i):
    distance = distEclud(centroids[p,:],dataset1[j,:])
    if distance < minDist:
     minDist = distance
     
   if minDist > maxDist: 
     maxDist = minDist
     maxIndex = j
  centroids[i,:] = dataset1[maxIndex,:]
       
 return centroids  
      
# k均值
def kmeans_open(dataSet,k):
    m = np.shape(dataSet)[0]  
    # 第一列存样本属于哪一簇，第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m,2)))
    clusterChange = True
 
    # 第1步 初始化centroids
    centroids = randCent(dataSet,k)
   
    while clusterChange:
        clusterChange = False
        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
 
            #第2步 找出最近的类中心
            for j in range(k):
                # 计算该样本到类中心的欧式距离
                distance = distEclud(centroids[j,:],dataSet[i,:])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
           
            # 第3步：更新每一行样本所属的簇
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex,minDist**2
        
        #第4步：更新类中心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]  # 获取簇类所有的点
            centroids[j,:] = np.mean(pointsInCluster,axis=0)   # 对矩阵的行求均值
    
    return clusterAssment.A[:,0], centroids
#测试聚类效果
def tes(x,k_num):
    right = 0
    for k in range(0, k_num):
        checker = [0, 0, 0]
        for i in range(0, 50):
            checker[int(x[i + 50 * k])] += 1
        right += max(checker)
        print("第"+str(k)+"类错误率："+str(1-max(checker)/50))
    print("错误率:"+str(1-right/k_num/50))
    return right	
if __name__=='__main__':
#数据集处理
 data = pd.read_csv('iris.csv')
 data.replace(['setosa','versicolor','virginica'],[0,1,2],inplace=True)
 X=data.values.tolist()

 X=np.delete(X,0,axis=1)#移除第一列
 x, y = np.split(X, [4,], axis=1)# 前四列是特征 
 features = [2,3]
 x = x[:,features]


 #数据集处理，只包含两类
 x1 = x[np.nonzero(y != 2)[0]]
 x2 = x[np.nonzero(y != 1)[0]]
 x3 = x[np.nonzero(y != 0)[0]]

#3类一起分
 label,k=kmeans_open(x,3)

#两两分类
 label12,k12=kmeans_open(x1,2)
 label13,k13=kmeans_open(x2,2)
 label23,k23=kmeans_open(x3,2)

				
			
		
 print(k)		
#c=randCent4(x,3)	
#print(c)	

#测试
 tes(label,3)
 tes(label12,2)
 tes(label13,2)
 tes(label23,2)		
