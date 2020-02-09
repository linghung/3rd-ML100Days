# ML 專案開發流程
* 資料搜集、前處理(EDA)
* 定義目標與評估準則
    * 定義⽬標
        * 回歸問題？分類問題？
        * 要預測的⽬標Y是啥
        * 用哪些資料X預測
        * 資料切分
            * training set
            * validation set
            * test set
    * 設定評估準則
        * 回歸問題 (預測值為實數)
            * RMSE, Root Mean Square Error
            * Mean Absolute Error
            * Mean Squared Error
            * R-Square
        * 分類問題 (預測值為類別)
            * Accuracy
            * F1-score
            * AUC, Area Under Curve
            * AUROC, Area Under the ROC(Receiver Operating Curve)
                * 0.5 代表隨機猜測，1 表示預測最好
            * MAP@N
* 建立模型與調整參數
    * 模型: 
        * Regression 回歸模型
        * Tree-based model 樹模型
        * Neural network 神經網路
* 驗證
* 導入(應用)
