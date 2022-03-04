import numpy as np
import math
from matplotlib import pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split  

#欧氏距离计算
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))  
 
#随机选k个初始聚类中心
def randCent(dataSet,nc):
    m,n = dataSet.shape
 
    centroids = np.zeros((nc,n))
    for i in range(nc):
        index = int(np.random.uniform(0,m)) 
        centroids[i,:] = dataSet[index,:]
    return centroids

#
def indistance(X,newZ):
    newvalue=np.square(X-newZ)
    Dj=(np.sqrt(newvalue.sum(axis=1)).sum(axis=0))/len(X)
    return Dj
    
def stdvar(X,newZ):
    std=np.square(X-newZ).sum(axis=0)/len(X)
    D=np.sqrt(std)
    return D
    
#参数
def pre():
 
 k=int(input("请输入k:"))#聚类中心数目
 tn=int(input("请输入tn:"))#最少样本数目
 ts=float(input("请输入ts:"))#分布标准差
 tc=float(input("请输入tc:"))#中心最小距离
 L=int(input("请输入一次迭代可合并最大次数L:"))
 I=int(input("请输入迭代运算最大次数I:"))
 return k,tn,ts,tc,L,I

#第一步
def one(dataSet,nc):
 centroids=randCent(dataSet,nc)
 return centroids	

#第二步
def two(dataset,centroids,nc):
 m = np.shape(dataset)[0] 	
 clusterAssment = np.mat(np.zeros((m,2))) 
 for i in range(m):
  minDist = 100000.0
  minIndex = -1
 
            #第2步 找出最近的类中心
  for j in range(nc):
   # 计算该样本到类中心的欧式距离
   distance = distEclud(centroids[j,:],dataset[i,:])
   if distance < minDist:
    minDist = distance
    minIndex = j
           
       
  clusterAssment[i,:] = minIndex,minDist**2
 return clusterAssment

#第三步 
def three(clusterAssment,centroids,nc,tn,dataset):
 for j in range(nc):
  nj=len(np.nonzero(clusterAssment[:,0].A == j)[0])
 
  if nj<tn:
   
  
   nc=nc-1 
   centroids=np.delete(centroids,j, axis=0) 
   break
 return centroids,nc

#第四五六步
def ffs(dataSet,clusterAssment,centroids,nc):
 Dj=np.empty(shape=(nc, 2))
 Db=0
 
 for j in range(nc):
  nj=len(np.nonzero(clusterAssment[:,0].A == j)[0])
  pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]  # 获取簇类所有的点
  centroids[j,:] = np.mean(pointsInCluster,axis=0)   # 对矩阵的行求均值 
 # print(clusterAssment) 
  Dj[j,:]=np.mean(clusterAssment[np.nonzero(clusterAssment[:,0].A == j)[0]],axis=0)
  
  Db+=  Dj[j][1]/nc
 
 return centroids,Dj,Db

#第七步
def seven(runt,nc,I,k,clusterAssment,dataset,centroids,ts):
	if runt>=I:
		tc=0
		
	
	if nc>=2*k:
		clusterAssment,centroids,nc=eleven(centroids,tc,nc)
	else:
		clusterAssment,centroids,nc=eight(nc,clusterAssment,dataset,centroids,ts)	
#第八九十步
def eight(Dj,Db,nc,clusterAssment,dataSet,centroids,ts,tn,k):
 a=0.4
 for j in range(nc):
  nj=len(np.nonzero(clusterAssment[:,0].A == j)[0])
  pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]
  std=np.square(pointsInCluster-centroids[j]).sum(axis=0)/nj
  #print(Dj)
  D=np.sqrt(std)
  list_D = D.tolist()
  max_list =  max(list_D) # 返回最大值
 
  max_index = list_D.index(max(list_D))# 最大值的索引
  if  max_list>ts:
   g=(Dj[j][1]>=Db) and (nj>(2*tn+1))
  # print(Dj[j][1]>=Db)
   if g or nc<=k/2:
    point=centroids[j,:]
    point1=point.copy()
    
    point1[max_index]=point1[max_index]+a*max_list
    
    point2=point.copy()
    point2[max_index]= point2[max_index]-a*max_list
  
    centroids=np.delete(centroids,j, axis=0)
    centroids= np.row_stack((centroids, point1))
    centroids= np.row_stack((centroids, point2))  
   
    nc=nc+1
    
 return centroids,nc

#第十一、十二、十三步		
def eleven(clusterAssment,centroids,tc,nc):
 mind=10000
 a=0
 b=0
 point= np.zeros((1,2))
 for i in range(nc-1):
  for j in range(i+1,nc):
   Dij=distEclud(centroids[i,:],centroids[j,:])
   if Dij<mind:
    mind=Dij
    a=i
    b=j
 if mind<tc:
  na=len(np.nonzero(clusterAssment[:,0].A == a)[0])
  nb=len(np.nonzero(clusterAssment[:,0].A == b)[0])
  point=(na*centroids[a]+nb*centroids[b])/(na+nb)
  print(point)
  centroids=np.delete(centroids, b, axis=0)
  centroids=np.delete(centroids, a, axis=0)
  #np.append(centroids, point[0], axis=0)
  centroids= np.row_stack((centroids, point))
 
  nc=nc-1
 return centroids,nc

# isodata
def isodata(dataset,nc):
 #k,tn,ts,tc,L,I=pre()
 k,tn,ts,tc,L,I=3,35,0.4,1,2,20
 centroids=one(dataset,nc) 
 
 runt=0
 while 1:
  #print(runt)
  if runt>=I:
   break
  else:
   runt+=1
   clusterAssment=two(dataset,centroids,nc)	
   centroids,nc=three(clusterAssment,centroids,nc,tn,dataset) 
   clusterAssment=two(dataset,centroids,nc) 
   centroids,Dj,Db=ffs(dataset,clusterAssment,centroids,nc)
   if runt>=I:
    tc=0
		
	
   if nc>=2*k:
    centroids,nc=eleven(clusterAssment,centroids,tc,nc)
    clusterAssment=two(dataset,centroids,nc)
   else:
   
    centroids,nc=eight(Dj,Db,nc,clusterAssment,dataset,centroids,ts,tn,k)	
    clusterAssment=two(dataset,centroids,nc)
    
    

 return clusterAssment.A[:,0], centroids

#数据集处理
data = pd.read_csv('iris.csv')
data.replace(['setosa','versicolor','virginica'],[0,1,2],inplace=True)
X=data.values.tolist()

X=np.delete(X,0,axis=1)#移除第一列
x, y = np.split(X, [4,], axis=1)# 前四列是特征 
features = [2,3]
x = x[:,features]




#3类一起分
label,k1=isodata(x,2)



#测试聚类效果
def tes(x):
    right = 0
    for k in range(3):
        checker = [0, 0, 0]
        for i in range(0, 50):
            checker[int(x[i + 50 * k])] += 1
        right += max(checker)
    print("错误率:"+str(1-right/150))
    return right					
			
		
print(k1)		


#测试
tes(label)

