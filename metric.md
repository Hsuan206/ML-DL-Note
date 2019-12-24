###### tags: `NLP learning`
# 衡量指標

[TOC]

## Confusion Matrix

> ### 定義
> [color=#C63939] 
> 
    用來評估模型或是演算法的好壞，用一個表格表示，其中一列 / 行代表預測值與實際值
    通常用在分類模型

![](https://i.imgur.com/LDjecVD.png)

> ### 意義
> [color=#C63939] 
> 
    TP 代表真實被標記為「有」的情況，模型預測他為「有」這個類別的個數
    TN 代表真實被標記為「沒有」的情況，模型預測他為「沒有」這個類別的個數
    FP 代表真實被標記為「沒有」的情況，模型預測他為「有」這個類別的個數
    FN 代表真實被標記為「有」的情況，模型預測他為「沒有」這個類別的個數

| 名稱 | 真實情況 | 模型預測 |
| -------- | -------- | -------- |
| True Positive (TP) (1,1)| 有 | 有 |
| Negative (TN) (0,0)| 沒有 | 沒有 |
| Positive (FP) (0,1)| 沒有 | 有 |
| Negative (FN) (1,0)| 有 | 沒有 | 

![](https://i.imgur.com/ab3ILVx.png)

* 依照這個是否懷孕的二元分類例子，結果會是

| 名稱 | 真實情況 | 模型預測 |
| -------- | -------- | -------- |
| True Positive (TP) (1,1)| 有懷孕 | 有懷孕 |
| Negative (TN) (0,0)| 沒有懷孕 | 沒有懷孕 |
| Positive (FP) (0,1)| 沒有懷孕 | 有懷孕 |
| Negative (FN) (1,0)| 有懷孕 | 沒有懷孕 |

**Example**
```python=
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from matplotlib.font_manager import FontProperties
import seaborn as sns
import matplotlib.pyplot as plt    
import numpy as np

# 假設真實與預測類別 (可以改成自己的資料)
y_true = ["不是貓","貓","貓","貓","不是貓","貓","不是貓","貓",
          "貓","貓","不是貓","貓","不是貓","貓","貓","貓"]
y_pred = ["貓","貓","貓","不是貓","不是貓","不是貓","貓","不是貓",
          "貓","不是貓","不是貓","貓","不是貓","貓","不是貓","貓"]
class_names = ["貓","不是貓"]

# 計算 confusion_matrix 的元素
element_count = confusion_matrix(y_true, y_pred)

# confusion_matrix
def plot_confusion_matrix(y_true, y_pred, classes, count, cmap=plt.cm.Blues):
    
    ax = plt.subplot()
    # heatmap 常⽤於呈現變數間的相關性，或用於呈現不同條件下，數量的高低關係
    sns.heatmap(count, annot = True, ax = ax, cmap = cmap);    # matplotlib 也有 heatmap

    # 用微軟正黑體
    myfont = FontProperties(fname=r'msjh.ttc')

    # 設定各種標籤、標題
    ax.set_xlabel('True labels',fontsize=12)
    ax.set_ylabel('Predicted labels',fontsize=12) 
    ax.set_title('Confusion Matrix',fontsize=20) 
    ax.xaxis.set_ticklabels(classes,fontproperties=myfont,fontsize=15)  # fontproperties 解決中文無法秀出問題
    ax.yaxis.set_ticklabels(classes,fontproperties=myfont,fontsize=15)

# 畫圖    
plot_confusion_matrix(y_true, y_pred, classes=class_names, count=element_count)
```
![](https://i.imgur.com/rjMzGus.png)


>### 參考資料
>
>[sklearn - Confusion matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)
[python matplotlib 中 axes 與 axis 的區別 ?](https://www.zhihu.com/question/51745620)
>
>[name=可愛的瑄]

---
### Accuracy
> ### 意義
> [color=#C63939] 
> 
    正確率，實際值與預測值一致時即為預測正確
:sparkles: 依照confusion matrix，TP 和 TN 加總即為正確個數
* 公式可表示為
![](https://i.imgur.com/hdHazTU.png)

```python=
'''
同上述例子 
'''
# 降維度的感覺
TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel() # array([3, 2, 5, 6])
accuracy = (TP + TN) / (TP + FN + FP + TN)
print(accuracy)
```
output :
    
    0.5625

:sparkles: confusion_matrix 這個函式與我們常見之 matrix 會有相反的狀況
### Precision
> ### 意義
> [color=#C63939] 
> 
    精確率，被模型預測為「有」的情況下，有多少是真正為「有」的
:sparkles: 例如：辨識垃圾郵件，如果 Precision 太低，代表實際非垃圾郵件中被模型預測為垃圾郵件相對較高(FP)，使用者因此會失去許多重要訊息
    
* 公式可表示為
![](https://i.imgur.com/po51L37.png)

```python=
'''
同上述例子 
'''
TN, FP, FN, TP= confusion_matrix(y_true, y_pred).ravel()
precision = TP / (TP + FP)
print(precision)
```
output :
    
    0.75

### Recall
> ### 意義
> [color=#C63939] 
> 
    召回率，被實際值為「有」的情況下，有多少是被模型預測出結果為「有」的
:sparkles: 例如：辨識垃圾郵件，如果 recall 太低，代表真實為垃圾郵件，被模型預測為非垃圾郵件的比例相對較高(FN)，使用者則會收到許多垃圾郵件

* 公式可表示為
![](https://i.imgur.com/DyHPSfD.png)

```python=
'''
同上述例子 
'''
TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
recall = TP / (TP + FN)
print(recall)
```
output :
    
    0.5454545454545454

![](https://i.imgur.com/BOjp4gD.png)
圈圈內的代表被模型預測為「有」的
實心點代表實際為「有」的

> ### 小注意
> [color=#398bc6] 
> 
    recall 和 precision 通常一個高另外一個就會低
:sparkles: 低 recall，高 precision：這代表我們錯過了是 positive 的資料（高 FN）但我們預測為 positive 的資料確實大部分都是 positive 的（低 FP）
:sparkles: 高 recall，低 precision：大多數 positive 都能被正確識別（低 FN），但存在大量誤報（高 FP）。

### ROC

> ### 定義
> [color=#C63939] 
> 
    用來呈現 sensitivity 及 1-specificity 的圖形

> ### 意義
> [color=#C63939] 
> 
    若是曲線愈往圖形左上方移動表示 sensitivity 愈高，FP 愈低
![](https://i.imgur.com/V53NKyj.png)
![](https://i.imgur.com/wz4Q1ox.png)
![](https://i.imgur.com/1U4dpIA.png)
![](https://i.imgur.com/l3TqqGY.png)

```python=
'''
同上述例子 
'''
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_true_binarized, y_pred_binarized)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='ROC')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

```
output :
    
![](https://i.imgur.com/SIv88LD.png)


看起來非常呆

### AUC

> ### 定義
> [color=#C63939] 
> 
    曲線下的面積 (Area under curve)，用來判別 ROC 曲線的鑑別力

> ### 意義
> [color=#C63939] 
> 
    範圍從為 0~1之間，數值愈大愈好
![](https://i.imgur.com/PsYV6qr.png)

```python=
'''
同上述例子 
'''
from sklearn.metrics import auc

auc(fpr, tpr)
```
output：
    
    0.5727272727272728
### F1
> ### 意義
> [color=#C63939] 
>
    想要在 Precision 和 Recall 之間尋求平衡時使用
    
* 公式可表示為
![](https://i.imgur.com/BJAMP10.png)

```python=
'''
同上述例子 
'''
TP, FN, FP, TN = confusion_matrix(y_true, y_pred).ravel()
F1_score = 2 * ((precision * recall) / (precision + recall))
print(F1_score)
```
output :
    
    0.5

> ### 直接使用 sklearn.metrics 的方法
> [color=#398bc6] 
> 
    
```python=
'''
同上述例子
使用 sklearn.metrics 的方法，計算時需填入每個樣本已分類的結果
字串沒辦法處理，所以轉成 1,0
'''

# 字串沒辦法處理，所以轉成 1,0
y_true_binarized = list(map(lambda x: 0 if x in ["不是貓"] else 1, y_true))
y_pred_binarized = list(map(lambda x: 0 if x in ["不是貓"] else 1, y_pred))

accuracy = metrics.accuracy_score(y_true_binarized, y_pred_binarized)
precision = metrics.precision_score(y_true_binarized, y_pred_binarized) # 使用 Precision 評估
recall  = metrics.recall_score(y_true_binarized, y_pred_binarized) # 使用 Recall 評估
f1 = metrics.f1_score(y_true_binarized, y_pred_binarized) # 使用 F1-Score 評估

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1) 
```
output :
    
    Accuracy:  0.5625
    Precision:  0.75
    Recall:  0.5454545454545454
    F1-Score:  0.631578947368421


### F1-micro
> ### 意義
> [color=#C63939] 
>
    同常用在多類別的分類，計算所有的類別的 TP、FN、FP
    
    
* 公式可表示為    
![](https://i.imgur.com/nUOfJiq.png)
![](https://i.imgur.com/PPjUG4k.png)
![](https://i.imgur.com/XTsdl2G.png)

```python=
'''
同上述例子
'''
precision = metrics.precision_score(y_true_binarized, y_pred_binarized, average='micro')
recall  = metrics.recall_score(y_true_binarized, y_pred_binarized, average='micro') 
f1_micro = metrics.f1_score(y_true_binarized, y_pred_binarized, average='micro')
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1_micro) 
```
output :
    
    Precision:  0.5625
    Recall:  0.5625
    F1-Score:  0.5625

### F1-macro
> ### 意義
> [color=#C63939] 
>
    同常用在多類別的分類，計算每個類別的 precision、recall，並計算平均值
    但是未考慮類別不平衡
    
* 公式可表示為
![](https://i.imgur.com/OV8Mq9K.png)
![](https://i.imgur.com/pl0o1Lj.png)
![](https://i.imgur.com/fs6bzkh.png)
![](https://i.imgur.com/Y2jtE98.png)
![](https://i.imgur.com/04dTmAp.png)

```python=
'''
同上述例子
'''
precision = metrics.precision_score(y_true_binarized, y_pred_binarized, average='macro') 
recall  = metrics.recall_score(y_true_binarized, y_pred_binarized, average='macro') 
f1_macro = metrics.f1_score(y_true_binarized, y_pred_binarized, average='macro') 
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1_macro) 
```

output :
    
    Precision:  0.5625
    Recall:  0.5727272727272728
    F1-Score:  0.5465587044534412

>### 參考資料
>
>[F1-micro 和 F1-macro 差別](http://sofasofa.io/forum_main_post.php?postid=1001112)
>
>[name=可愛的瑄]
>

> ### 小補充
> [color=#398bc6] 
> 
    classification_report 可以看到各類別的 precision、recall 等其他 F1 計算結果
```python=
'''
同上述例子
'''
print('分類資訊: \n', metrics.classification_report(y_true, y_pred))
```

output :
    
    分類資訊: 
               precision    recall  f1-score   support

         不是貓       0.38      0.60      0.46         5
           貓       0.75      0.55      0.63        11

       micro avg       0.56      0.56      0.56        16
       macro avg       0.56      0.57      0.55        16
    weighted avg       0.63      0.56      0.58        16
