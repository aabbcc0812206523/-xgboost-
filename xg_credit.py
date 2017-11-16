# 独热编码后，训练集和测试集中列有丢失，导致不匹配，需要改为xgboost.py
import  scipy as sc
import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from scipy.sparse import hstack,csr_matrix
from pandas import Series
from pandas import DataFrame
import xgboost as xgb
from sklearn import cross_validation, metrics


datafile = "./tb_excel_drjs.csv"
datafile_lsjs = "./tb_excel_lsjs.csv"
#未违约数据
data_z = pd.read_csv(datafile)

data_lsjs = pd.read_csv(datafile_lsjs)
#print(data_lsjs.count())314
#print(data_lsjs.column())

#增加label
#data_lsjs['label']=1
#data_z['label']=0

#data_weiyue = data_lsjs.loc[data_lsjs['jszt']=='买方违约']
#违约数据
data_weiyue = data_lsjs.loc[((data_lsjs['jszt']=='买方违约')&(data_lsjs['wtlx']=='委买'))|
                            ((data_lsjs['jszt']=='卖方违约')&(data_lsjs['wtlx']=='委卖'))]#,['jsbh','jysbh','dbdk','ghsjdk','jsbh','jsrq','jszt','jysbh','lldk', 'qylb', 'sjdk’,’cjsj','jsrq','jsj','jsl','lldk']
#总数据
data = pd.concat((data_z,data_weiyue),ignore_index=True)
#print(len(data))10101
#删除重复、为空、唯一值特征
data=data.drop(['ghsjdk','jszt','jiaosrq','phl','sjghl','pzdl','ghydl','qylb','spdm','spmc','ghdyje','ppdm','yhkje','djdk','dyje','jysmc','sjghl','ghydl','yhkje','djdk','dyje','jsfs','drsj','sjdk','sjsj','byzd1','byzd2','wtlx','dbdk'],axis=1)
#print(data.columns) Index(['cjsj', 'jsbh', 'jsj', 'jsl', 'jsrq', 'jysbh', 'lldk'], dtype='object')

#train_w = data_weiyue[0:70]
#train_z = data_z[0:7000]
#train = pd.concat([train_z,train_w])
##测试集
#test = pd.concat([data_weiyue[71:100],data_z[7001:10000]])#contact 默认的合并方向为axis=0


#对日期进行处理转化为天数
now_time = datetime.now()
clu_date = pd.to_datetime(data['jsrq'].tolist())
#print((now_time-clu_date).days)
date_day = (now_time-clu_date).days
#print(type(date_day))
data['jsrq']=date_day

#print (a.shape)#(101,)
#print(type(a))  <class 'pandas.core.series.Series'>
#print(a.loc[:0].tolist())

#对交易时间处理为5分钟的倍数\
a= data['cjsj'].str.split(':')#pandas 数据框的str列内置的方法 不要忘记str
a.tolist()
li_t =[]
for t in a:
    if int(t[0])<13:
        min5=(int(t[0])-9)*12+int(np.ceil(int(t[1])/5))#以9点半为基准以5分钟为刻度计数
    else:
        min5 = (int(t[0])-13) * 12 +24+ int(np.ceil(int(t[1])/5))
    li_t.append(min5)
data['cjsj']=li_t

#print(max(li_t)) 48
#print(train['pzdl'].value_counts())#查看某一列的属性值以及个数
#print(train.count())
#print(train['lldk'].isnull())
#print(train[train.isnull().values==True])
#增加label

#标准化
min_max_scaler = preprocessing.MinMaxScaler()#feature_range=(0,2)
X_minmax = min_max_scaler.fit_transform(data.loc[:,['cjsj','jsrq','jsj','jsl','lldk']])#,'ghsjdk'结果类型为numpy.ndarray
#独热编码
encoder = preprocessing.OneHotEncoder(categorical_features='all', sparse=True)#改进点：通过此处选择哪一列来编码
X_onehot = encoder.fit_transform(data.loc[:,['jsbh','jysbh']])
data=hstack([csr_matrix(X_minmax),X_onehot])
#print(type(train))<class 'scipy.sparse.coo.coo_matrix'>
#print(train.shape)(7070, 177)
#print(train)

#xgboost
data = data.toarray()
#print(type(data))<class 'numpy.ndarray'>

train_1 = data[0:7000,:]
train_2=data[10001:10071,:]
train = np.concatenate((train_1,train_2),axis=0)
#print(train.shape)(7070, 132)
test_1 = data[7000:10000,:]
test_2=data[10071:10101,:]
test = np.concatenate((test_1,test_2),axis=0)
#print(test.shape)(3030, 132)

label_train = [0 for i in range(7000)]
label_train.extend([1 for i in range(70)])
#print(len(label_train))7070
label_train = np.array(label_train)
label_test = [0 for i in range(3000)]
label_test.extend([1 for i in range(30)])
label_test = np.array(label_test)
#print(len(label_test))3030

dtrain = xgb.DMatrix(train,label=label_train)#参数为ndarray
dtest = xgb.DMatrix(test,label=label_test)
#bst：booster参数   bst:max_depth树的最大深度,默认为6；
# bst:eta和GBM中的 learning rate 参数类似。 通过减少每一步的权重，可以提高模型的鲁棒性。 典型值为0.01-0.2 默认0.3
# silent silent[默认0]，当这个参数值为1时，静默模式开启，不会输出任何信息。 一般这个参数就保持默认的0，因为这样能帮我们更好地理解模型
#objective[默认reg:linear]这个参数定义需要被最小化的损失函数。最常用的值有：
#binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)。 multi:softmax 使用softmax的多分类器，返回预测的类别(不是概率)。
#nthread[默认值为最大可能的线程数]这个参数用来进行多线程控制，应当输入系统的核数。 如果你希望使用CPU全部的核，那就不要输入这个参数，
    # 算法会自动检测它。
#eval_metric[默认值取决于objective参数的取值]对于有效数据的度量方法。 对于回归问题，默认值是rmse，对于分类问题，默认值是error。
    # 典型值有：rmse 均方根误差(∑Ni=1?2N??????√) mae 平均绝对误差(∑Ni=1|?|N) logloss 负对数似然函数值 error 二分类错误率(阈值为0.5)
    #  merror 多分类错误率 mlogloss 多分类logloss损失函数 auc 曲线下面积
#param = {'bst:max_depth': 3, 'bst:eta': 1, 'silent': 1, 'objective': 'binary:logistic'}#2，1，1，
#param['nthread'] = 4
#param['eval_metric'] = 'auc'
#param = {'booster':'gbtree','bst:eta': 1,'objective':'binary:logistic','silent': 1,
#         'bst:max_depth':4,
#         'subsample':1.0,
#         'min_child_weight':5,#当它的值较大时，可以避免避免过拟合，过大会前拟合
#         'colsample_bytree':0.2,
#         'scale_pos_weight':0.1,
#         'eval_metric':'auc',
#         'bst:gamma':0.05,
#         'lambda':300
#}
param = {'booster':'gbtree','bst:eta': 1,'objective':'binary:logistic','silent': 1,'bst:max_depth':3,
         'eval_metric':'auc','subsample':0.9}

evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round =20

bst = xgb.train(param, dtrain, num_round, evallist)
bst.dump_model('dump.raw.txt')
xgb.plot_importance(bst)