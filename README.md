# Machine-Learning
This Repository contains the basic algorithm about machine learning  

Note: All the algorithm use the scikit-learn packages to finish the code

__*PCA*__   
Including *PCA*(linear), *IncrementalPCA*(batch-size data to run once), *KernelPCA*(NonLinear)

1. __数据集__
* 使用的数据集来自scikit-learn自带的鸾尾花数据集  
* 该数据集包含3种类型共150个样本，每一种50个样本，每个样本有4个属性。

2. __PCA__  
scikit-learn中提供了一个PCA类来实现PCA模型，原型为：  
`
class sklearn.decomposition.PCA(n_components=None,copy=True,whiten=False)
`

* 参数：n_components：指定了降维后的维数
* 属性：  
components_:主成分数组  
explained_variance_ratio：数组元素是每个主成分的explained_variance的比例  
mean_：元素是每个特征的统计平均值  
n_components_：整数，只是主成分有多少元素  
* 方法  
fit(X[,y]):训练模型  
transform(X)：执行降维  
fit_transform(X[,y])：训练模型并降维  
inverse_transform(X)：执行升维  
* 结果  
![](https://github.com/wanggongziZeo/Image-folder/blob/master/Images/PCA.png)  

2. __IncrementalPCA__  
原型为：  
`
class sklearn.decomposition.IncrementalPCA(n_components=None,copy=True,whiten=False,batch_size=None)
`

* 结果  
![](https://github.com/wanggongziZeo/Image-folder/blob/master/Images/IncrementalPCA.png)  

3. __KernelPCA__  
原型为：  
`
class sklearn.decomposition.KernelPCA(n_components=None,kernel='linear',gamma=None,degree=3,coef0=1,
kernel_params=None,alpha=1.0,fit_inverse_transform=False,eigen_solver='auto',tot=0,max_iter=None,remove_zero_eig=False)
`  
* 参数  
kernel：指定核函数  
'linear'：线性核  
'poly'：多项式核  
'rbf'：高斯核函数  
'sigmoid'  
* 结果  
![](https://github.com/wanggongziZeo/Image-folder/blob/master/Images/KernelPCA.png)
