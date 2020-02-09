# ML 專案開發流程
* 資料搜集、前處理(EDA)
* 定義目標與評估準則
    * 定義⽬標
        * 回歸問題？分類問題？
        * 要預測的⽬標(Y)是啥
        * 用哪些資料(X)預測
        * 資料切分
            * training set
            * validation set
            * test set
    * 設定評估準則
        * 迴歸問題 (預測值為實數)
            * RMSE, Root Mean Square Error
            * Mean Absolute Error 平均絕對誤差
            * Mean Squared Error 平均平方誤差(均方差)
            * R-Square
        * 分類問題 (預測值為類別)
            * Accuracy
            * F1-score
            * AUC, Area Under Curve
            * AUROC, Area Under the ROC(Receiver Operating Curve)
            * MAP@N
            * CE, Binary Cross Entropy
* 建立模型與調整參數
    * 模型: 
        * Regression 回歸模型
        * Tree-based model 樹模型
        * Neural network 神經網路
* 驗證
* 導入(應用)
