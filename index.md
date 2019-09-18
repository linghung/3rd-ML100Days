# 學習筆記：【第三屆】機器學習百日馬拉松

![](https://i.imgur.com/rkvGRsx.png)

學習路程
---
1. 找到問題
2. prototype
    - 為什麼這個問題重要
    - 資料從何⽽來(可靠度)
    - 資料的型態是什麼(轉換/標準化方式)
    - 我們可以回答什麼問題(假設 --> 驗證)
3. 改進
4. 實際使用

相關名詞關係釐清
---
- 人工智慧
    - 機器學習
        - 深度學習
- 機器學習
    - supervised (要標 label)
    - unsupervised
    - reinforcement

ML操作步驟概覽
---
- 資料搜集
    - 資料從何⽽來
    - 多處來源
    - 資料異常的理由與頻率
    - 先了解：
        - 資料量
        - 資料意義
        - 資料關係
        - 是否有空值(填預設-1或999或已被填0)或錯誤資料
- 資料前處理
    - 統⼀資料格式, ⽅便讀寫 (.json, .csv)
    - 錯誤值修正
    - 缺失值填補方法(不要破壞資料分布)
        - 數值型欄位偏態不明顯時填平均值(Mean)
        - 數值型欄位偏態明顯時填中位數(Median)
        - 類別型欄位填眾數(Mode)
        - 依意義補指定值
        - 補預測值(可能導致overfitting)
    - 離群值處理
        - 透過 Range / 五值 / 平均數 / 標準差 / zscore / IQR / scatter / histogram / boxplot 來檢查異常
        - 了解該數值會離群的可能原因, 以免刪除掉重要資訊
        - 解決方法
            - 新增欄位⽤以紀錄異常與否
            - 填補(取代)：以 mean, median, min, max, quantile, na
            - 或整欄資料不使用
    - 特徵縮放(有時不適合)(以合理的⽅式, 平衡特徵間的影響⼒)
        - 影響
            - Regression model：有差
            - Tree-based model：沒有太⼤關係
        - 方法
            - Standard Scaler 標準化適合常態分佈, 不易受到極端值
                - z 轉換法：(x-mean)/std
                - 空間壓縮法
                    - 控制在0~1之間：(x-min)/(max-min)
                    - 控制在-1~1之間：((x-min)/(max-min)-0.5)*2
                    - 或上述的 min, max 改用 q1, q99
            - MinMax Scaler 最⼩最⼤化, 適合均勻分佈, 容易受到極端值影響
- 探索式資料分析(EDA=Exploratory Data Analysis)
    - 相關係數 Correlation Coefficient
        - .00-.19：非常弱相關
        - .20-.39：弱相關
        - .40-.59：中度相關
        - .60-.79：強相關
        - .80-1.0：非常強相關
    - 核密度函數 KDE
        - 某 x 出現機率
        - 線下面積和 = 1
        - 核函數
            - Gaussian(Normal dist)
            - Cosine
        - sns.kdeplot()
    - 離散化(降低異常⼲擾與ovefitting)
        - 等寬法(pd.cut)：受異常值影響較大
        - 等頻法(pd.qcut)：每一份中的個數一樣
        - 聚類法
        - 自訂
    - 數據清理
    - 特徵萃取
    - 視覺化
    - 
    - 統計/量化
        - 計算集中趨勢
            - 平均值 Mean
            - 中位數 Median
            - 眾數 Mode
        - 計算分散程度
            - 最⼩值 Min
            - 最⼤值 Max
            - 範圍 Range
            - 四分位差 Quartiles
            - 變異數 Variance
            - 標準差 Standard deviation
    - 常見於迴歸問題的評估指標
        - 平均絕對誤差 - Mean Absolute Error (MAE)
        - 平均平方誤差(均方差) - Mean Squared Error (MSE)
    - 常見於分類問題的指標
        - Binary Cross Entropy (CE)
    - 調整分析方向
- 定義⽬標與評估準則
    - 回歸問題？分類問題？
    - 要預測的⽬標是啥
    - 用哪些資料預測
    - 資料切分
        - training
        - validation
        - test
- 評估準則
    - 回歸問題 (預測值為實數)
        - RMSE, Root Mean Square Error
        - Mean Absolute Error
        - R-Square
    - 分類問題 (預測值為類別)
        - Accuracy
        - F1-score
        - AUC, Area Under Curve
- 建立模型與調整參數
    - Regression 回歸模型
    - Tree-based model 樹模型
    - Neural network 神經網路
- 驗證模型
- 導入(實際應用的意思？)

資料類型
---
- 現在世界
    - 離散
    - 連續
    - 日期時間
    - 布林
    - 排序
- pandas DataFrame
    - float64：可表⽰離散或連續變數
    - int64：可表⽰離散或連續變數
    - object：包含字串, ⽤於表⽰類別型變數
- 字串/類別轉數值方式
    - Label encoding：
        - 使⽤時機是資料為有序的
        - 把每個類別 mapping 到某個整數, 不會增加新欄位
    - One Hot encoding：
        - 使⽤時機是資料為無序的
        - 為每個類別新增一個欄位, 用 0/1 表示是否
- 數值型轉換方式
    - 函數：y=x*200
    - 條件式：>5給2點, <5給1點

撇步
---
- 有時候原始資料太⼤了, 有些資料的操作很費時, 先在具有同樣結構的資料上測試程式碼是否能夠得到理想中的結果

語法
---
```python=
# 各種 import 寫法
import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings
from collections import defaultdict
from PIL import Image
from scipy.stats import mode
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


# 讓繪圖正常顯示
%matplotlib inline
# 忽略警告訊息
warnings.filterwarnings('ignore')
# help
?pd.read_csv
# 單行註解
"""
多行註解
"""


# 函數
r=sum(arr)
r=abs(x)
r=len(arr)
def mean_absolute_error(y, yp):
    mae = MAE = sum(abs(y - yp)) / len(y)
    return mae
def mean_squared_error(y, yp):
    mse = MSE = sum((y - yp)**2) / len(y)
    return mse
def normalize_value(x):
    x = 2 * (((x-x.min())/(x.max()-x.min())) - 0.5)
    return x

# 讀檔的寫法
import requests
try:
    response = requests.get(target_url)
    data = response.text
    data = data.split('\n')
    arrange_data = []
    for line in data:
        line = line.split('\t')
        arrange_data.append(line)
    df = pd.DataFrame(arrange_data)
    from PIL import Image
    from io import BytesIO
    first_link = df.loc[0][1]
    response = requests.get(first_link)
    img = Image.open(BytesIO(response.content))
    width, height = img.size
    img = img.resize((width*3,height))
    img = np.array(img)
    plt.imshow(img)
    plt.show()
except:
    print("fail")

with open('example.txt', 'r') as f:
    data = f.readlines()

import json
with open('example.json', 'r') as f:
    data = json.load(f)

import scipy.io as sio
data = sio.loadmat('example.mat')

import numpy as np
arr = np.load(example.npy)

import pickle
with open('example.pkl', 'rb') as f:
    arr = pickle.load(f)

import os
file = os.path.join('./data/', 'application_train.csv')
df = pd.read_csv(file)

data = []
with open("data/example.txt", 'r') as f:
    for line in f:
        line = line.replace('\n', '').split(',') # 將每句最後的 /n 取代成空值後，再以逗號斷句
        data.append(line)
df = pd.DataFrame(data[1:])
df.columns = data[0]

import json
df.to_json('data/example01.json')
with open('data/example01.json', 'r') as f:
    j1 = json.load(f)
df.set_index('id', inplace=True)
df.to_json('data/example02.json', orient='index')
with open('data/example02.json', 'r') as f:
    j2 = json.load(f)
    
import numpy as np
# 將 data 的數值部分轉成 numpy array
array = np.array(data[1:])
np.save(arr=array, file='data/example.npy')
array_back = np.load('data/example.npy')

import pickle
with open('data/example.pkl', 'wb') as f:
    pickle.dump(file=f, obj=data)
with open('data/example.pkl', 'rb') as f:
    pkl_data = pickle.load(f)


# image
import skimage.io as skio
img1 = skio.imread('data/examples/example.jpg')
plt.imshow(img1)
plt.show()

from PIL import Image
img2 = Image.open('data/examples/example.jpg') # 這時候還是 PIL object
img2 = np.array(img2)
plt.imshow(img2)
plt.show()

import cv2
img3 = cv2.imread('data/examples/example.jpg')
plt.imshow(img3)
plt.show()

img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
plt.imshow(img3)
plt.show()

%%timeit
im = np.array([skio.imread('data/examples/example.jpg') for _ in range(N_times)])

import scipy.io as sio
sio.savemat(file_name='data/examples/example.mat', mdict={'img': img1})

mat_arr = sio.loadmat('data/examples/example.mat')
print(mat_arr.keys())

mat_arr = mat_arr['img']
print(mat_arr.shape)

plt.imshow(mat_arr)
plt.show()


# 查看資料長相的幾種方式
df.head(3)
df.tail(3)
df.shape
df.shape[0]
df.index
df.size
len(df)
df['a'].values
df['a'].value_counts()
df['a'].unique()
df.dtypes
df.dtypes.value_counts()
df['a'].value_counts().sort_index(ascending = False)
df.columns.tolist()
df.select_dtypes('number').columns
df.select_dtypes(include=["object"]).apply(pd.Series.nunique, axis = 0)

# 資料欄位的類型與數量
dtype_df = df.dtypes.reset_index() 
dtype_df.columns = ["Count", "Column Type"]
dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index()


# 計算
df.describe()
df['a'].describe()
df['a'].min()
df['a'].max()
df['a'].mean()
df['a'].std()
df['a'].nunique()
df.sort_values('a').max()
df.corr()
df.corr()['a']
np.corrcoef(df['a'], df['b'])
np.mean(arr)
np.median(arr)
np.median(df[~df['a'].isnull()]['a'])
np.quantile(arr, q = [0,.5])
np.log1p(df['a'])
np.log(df['a'])
mode(arr) # from scipy.stats import mode
zscore(arr) # from scipy.stats import zscore

# cdf
series = df['a']
li = list(series)
li.sort()
cdf = pd.Series(series.index+1, li)

# 標準化搭配羅吉斯迴歸模型
df_0 = df.fillna(0)
df_temp = StandardScaler().fit_transform(df_0)
train_X = df_temp[:train_num]
r=cross_val_score(LogisticRegression(solver='lbfgs'), train_X, train_Y, cv=5).mean()

train_X = MinMaxScaler().fit_transform(df)
train_X = StandardScaler().fit_transform(df)
cross_val_score(LinearRegression(), train_X, train_Y, cv=5).mean()
cross_val_score(LogisticRegression(solver='lbfgs'), train_X, train_Y, cv=5).mean()

q=[np.percentile(df[~df['a'].isnull()]['a'], q = i) for i in [0, 25, 50, 75, 100]]
q=np.percentile(df[not_null]['a'], q=99)
from scipy.stats import mode
mode(df[~df['A'].isnull()]['A'])
s=sum(app_train['DAYS_EMPLOYED'] == 365243)


# 新變數
cities = ['Austin', 'Dallas', 'Austin', 'Dallas']
weekdays = ['Sun', 'Sun', 'Mon', 'Mon']
visitors = [139, 237, 326, 456]
list_labels = ['city', 'weekday', 'visitor']
list_cols = [cities, weekdays, visitors]
zipped = list(zip(list_labels, list_cols))
visitors_2 = pd.DataFrame(dict(zipped))

df=pd.DataFrame({'B':['B2', 'B3', 'B6', 'B7'],
                   'D':['D2', 'D3', 'D6', 'D7'],
                   'F':['F2', 'F3', 'F6', 'F7']},
                   index=[2, 3, 6, 7])
mode_dict = defaultdict(lambda:0)
sorted(mode_dict.items(), key=lambda kv: kv[1], reverse=True)
np.random.seed(1)
arr=np.random.randint(100,300,size=3)
arr=np.random.randint(0, 50, 1000)
x=np.random.randn() #隨機傳回標準常態分布的取樣值
arr=np.random.randn(101)
arr=np.random.normal(0, 10, 1000)


# 計時
start = time.time()
end = time.time()
dif = end-start
print("Elapsed time: %.3f secs" % (dif))


# df各種操作
df=pd.melt(df)
pd.pivot(columns='var',values='val')
df=pd.concat([df1,df2,df3])
df=pd.concat([df1,df2,df3],axis=1)
df=pd.concat([df1,df2,df3],axis=1,join='inner')
df=pd.merge(df1,df2,on='id',how='outer')
df=pd.merge(df1,df2,on='id',how='inner')
id=df['Id']
df=df.drop(['a', 'b'], axis=1)

# 把只有 2 值 (通常是 0,1) 的欄位去掉
numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])

b=df['age'] != 30
subdf=df[b]
subdf=df[df['age'] != 30]
subdf=df[df.id==20 | df.age!=3]  #邏輯運算子 & | ~ ^
keep_indexs = (df['a']> 1) & (df['a']< 5)
df = df[keep_indexs]
subdf=df.loc[df['a'] > 3, ['b', 'c']]
subdf=df[df.column.isin(value)]
subdf=df[pd.isnull(obj)]
app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY']
subdf = df[df[i].notna()]
subdf=df[pd.notnull(obj)]
df['a'].dropna(inplace =True)
subdf = df[:x]
df[['a','b']]
subdf=df.drop_duplicates()
subdf=df.head(3)
subdf=df.tail(3)
subdf=df.sample(frac=0.5)
subdf=df.sample(n=3)
subdf=df.iloc[n:m]
x=df.loc[0]['TARGET'] #第 0 列資料的 TARGET 欄位的值
arr=df.loc[100] #第 100 列資料
df=df.loc[95:100][['a','b']] #第 95 ~ 100 列的其中兩欄
subdf=df.isnull()
subdf=df['a'].isnull()
df['a'] = df['a'].clip(800, 2500)
#.sum().sort_values(ascending=False).head()

#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)

newdf=df['id','age']
newdf=df.age
newdf=df.filter(regex=...)
df['col'].replace({365243: np.nan}, inplace = True)

subdf=df.groupby(by='a')['b','c']
subdf=df.groupby(['id','age'])
subdf=df.groupby(['a']).apply(lambda x: x / x.mean())
subdf['amount'].mean()
subdf['amount'].apply()
subdf.mean()
df_m1 = df.fillna(-1)


# 繪圖
plt.style.use('default')
plt.style.use('ggplot')
plt.style.use('seaborn')

df['a'].plot.hist(title = 'xxx')
df.groupby(['a'])['b'].hist()
df.groupby(['a'])['b'].hist(bins = 100)
plt.plot(df['a'], np.log10(df['b']), 'b.',label = 'mylabel') #散佈圖
plt.plot(x_lin, y_hat, 'r-', label = 'line')
df['a'].hist()
plt.hist(app_train[~app_train.OWN_CAR_AGE.isnull()]['OWN_CAR_AGE'])
plt.scatter(x, y) #散佈圖
plt.hist(df['age'], edgecolor = 'k', bins = 25)
df.boxplot(column=c, by = b, showfliers = False, figsize=(12,12))
sns.boxplot(x=df[col])
sns.regplot(x = df['GrLivArea'], y=train_Y)
sns.kdeplot(df.loc[df['a'] == 0, 'b'] / 365, label = 'xxx')
sns.kdeplot(df.loc[df['a'] == 1, 'b'] / 365, label = 'xxx', kernel='cos')
sns.distplot(df.loc[df['a'] == 1, 'b'] / 365, label = 'xxx', hist = False)
sns.barplot(px, py)

# 先指定劃在哪格
plt.subplot(321) # 列-欄-位置
# 再呼叫繪製函數
plt.plot([0,1],[0,1], label = 'I am subplot1')

plt.subplot(nrows, ncols, i+1)
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2])
sns.jointplot()
fig, axs = plt.subplots(3, 5)
fig.set_figwidth(15)
fig.set_figheight(9)
fig.subplots_adjust(wspace = 0.2, hspace = 0.3)
p = axs[count//5, count%5]
p.plot(sub_df[i], sub_df['TARGET'], '.')
p.set_title(i)
sub_df.boxplot(column=[i], by=['TARGET'], ax=p)

plt.xlim([cdf.index.min(), cdf.index.max() * 1.05]) # 限制顯示圖片的範圍
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍
plt.legend(loc = 2)
plt.xticks(rotation = 75); plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
plt.suptitle('')
plt.figure(figsize = (10, 8))
plt.show()


# print
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))
print(f'{len(x)} Numeric Features : {x}\n')

# 分組
arr=np.linspace(20, 70, 11) #從20到70，切成10組
pd.cut(df['age'],bins=[10,20,30]) #(0, 10]:'(' 表示不包含, ']' 表示包含
pd.cut(df['age'],4)
pd.cut(df['a'], rule, include_lowest=True)
pd.qcut(df['age'],4)


# 類別轉數值 - label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[col] = le.fit_transform(df[col])
le.fit(df[col])
df[col] = le.transform(df[col])

# 類別轉數值 - one hot encoding
df = pd.get_dummies(df) # 原本A欄位值為x與y的話, 就會做成欄位 A_x 與 A_y


# sort
arr=arr.sort_values()









```


###### tags: `ML` `機器學習`
