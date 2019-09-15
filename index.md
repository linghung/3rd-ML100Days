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
            - 填補(取代)：以中位數, Min, Max, Max+1 或平均數填補(有時會⽤ NA)
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
    - Label encoding：使⽤時機是資料為有序的
    - One Hot encoding：使⽤時機是資料為無序的
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
import os
import numpy as np
import matplotlib.pyplot as plt


# 讓繪圖正常顯示
%matplotlib inline


# 讀檔的寫法
with open(‘example.txt’, ‘r’) as f:
    data = f.readlines()

import json
with open(‘example.json’, ‘r’) as f:
    data = json.load(f)

import scipy.io as sio
data = sio.loadmat(‘example.mat’)

import numpy as np
arr = np.load(example.npy)

import pickle
with open(‘example.pkl’, ‘rb’) as f:
    arr = pickle.load(f)

import os
file = os.path.join('./data/', 'application_train.csv')
df = pd.read_csv(file)


# 查看資料長相的幾種方式
df.head(3)
df.tail(3)
df.shape
df.size()
df.describe()
df['col'].value_counts()
df['col'].unique()


# 統計
np.median(value_array)
np.quantile(value_arrar, q = …)
scipy.stats.mode(value_array)
np.mean(value_array)


⽤ pd.DataFrame 來創建⼀個 dataframe
⽤ np.random.randint 來產⽣隨機數值


# df各種操作
pd.melt(df)
pd.pivot(columns='var',values='val')
pd.concat([df1,df2])
pd.concat([df1,df2],axis=1)
pd.merge(df1,df2,on='id',how='outer')
pd.merge(df1,df2,on='id',how='inner')

subdf=df[df.id==20 | df.age!=3]  #邏輯運算子 & | ~ ^
subdf=df[df.column.isin(value)]
subdf=df[pd.isnull(obj)]
subdf=df[pd.notnull(obj)]
subdf=df.drop_duplicates()
subdf=df.head(3)
subdf=df.tail(3)
subdf=df.sample(frac=0.5)
subdf=df.sample(n=3)
subdf=df.iloc[n:m]

newdf=df['id','age']
newdf=df.age
newdf=df.filter(regex=...)
df['col'].replace({365243: np.nan}, inplace = True)

subdf=df.groupby(['id','age'])
subdf['amount'].mean()
subdf['amount'].apply()
subdf['amount'].hist()
subdf.mean()


# 繪圖
plt.style.use('default')
plt.style.use('ggplot')
plt.style.use('seaborn')

plt.hist(df['age'], edgecolor = 'k', bins = 25)
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1', kernel='cos')
sns.distplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')
sns.barplot(px, py)
plt.legend()

plt.xticks(rotation = 75); plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
plt.figure(figsize = (10, 8))


# 分組
np.linspace(20, 70, 11) #從20到70，切成10組
pd.cut(df['age'],bins=[10,20,30]) #(0, 10]:'(' 表示不包含, ']' 表示包含
pd.cut(df['age'],4)
pd.qcut(df['age'],4)


# 類別轉數值
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
app_train[col] = le.fit_transform(app_train[col])

```

###### tags: `ML` `機器學習`
