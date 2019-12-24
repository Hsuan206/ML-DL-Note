###### tags: `NLP learning`
[TOC]
# NN with Pytorch 

以下範例在 colab 進行
## 回顧

* 以 [SMS spam collection dataset](https://www.kaggle.com/ishansoni/sms-spam-collection-dataset) 為範例的資料經過一連串的[文字前處理](https://hackmd.io/M3dsMTNuQsORAs8Q3PvI5w?view)後，將文字轉成文字特徵向量，並使用 pytorch 進行下一步的模型預測
```python=
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = data_rawSMS['content']

'''
轉換成 array 放到一個欄位
'''
tfidf = TfidfVectorizer()
data_rawSMS['textsVect'] = list(tfidf.fit_transform(corpus).toarray())

data_rawSMS.head()
```

output：
![](https://i.imgur.com/0UoFblr.png)
## Import Module
* import 一些可能需要用到的 module

```python=
from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
```

## Train / Test Split
* 分成訓練資料與驗證資料

**Example**
```python=
'''
train_test_split(data(feature),target(label),shuffle,test_size,random_state)
'''
train_feature, val_feature, train_label, val_label = train_test_split([tfidf for tfidf in data_rawSMS['textsVect']],
                                                                     data_rawSMS["spam"], 
                                                                     shuffle=True, 
                                                                     test_size=0.2, 
                                                                     random_state=2019)
```    

## Data Loading and Preprocessing

* 當資料量龐大時，為了能夠更好的提升訓練速度、模型訓練的結果，我們可以將資料分批的進行訓練，可以實現偉大目標的容器叫做 `dataloader`

### Design a Dataset Class for Constructing a Dataset Object

* 要把資料放在 `dataloader` 上前需要把資料包成一個 `Dataset` 物件
    * **`torch.utils.data.Dataset`**：此方法可建立自訂的`Dataset` 類別，例如：把一張圖片旋轉
    * **`torch.utils.data.TensorDataset`**
     
**Example**
```python=
'''
自行定義的 dataset，也可直接使用 TensorDataset
'''
class spamDataset(Dataset):
    def __init__(self, feature, label):
        
        # 接受 input 進來的 feature 和 label
        # 將資料轉成array 形式再轉成 Tensor
        self.feature = torch.Tensor(np.asarray(feature)).float() # input
        self.label = torch.Tensor(np.asarray(label)).long() # label

    def __len__(self):
        
        # 取的長度
        return len(self.feature)
        
    def __getitem__(self, index):
      
        # 回傳指定 index 的資料
        feature = self.feature[index]
        label = self.label[index]
        
        return feature, label
```
* 把資料都放進 dataset 裡面

```python=
'''
spamDataset(data_tensor, target_tensor)
'''
train_dataset = spamDataset(train_feature, train_label)
val_dataset = spamDataset(val_feature, val_label)

# 可以查看一下 dataset 裡面的資料
print('訓練集資料長度：', train_dataset.__len__())
print(train_dataset.__getitem__(0))
print('驗證集資料長度：', val_dataset.__len__())
print(val_dataset.__getitem__(0))
```    
output：

    訓練集資料長度： 4457
    (tensor([0., 0., 0.,  ..., 0., 0., 0.]), tensor(1))
    驗證集資料長度： 1115
    (tensor([0., 0., 0.,  ..., 0., 0., 0.]), tensor(1))

:sparkles: 都是 tensor 的形式ㄛ!
### Use the `DataLoader` Object to Load Data

* **`torch.utils.data.DataLoader`**：用來打包我們的 `Dataset`
    * 較常使用的幾個參數：
        * **dataset** (Dataset)：載入你的資料
        * **batch_size** (int, optional)：一次要載入多少資料 (預設值：1)
        * **shuffle** (bool, optional)：資料是否要在每一個 epoch 後重新打亂 (預設值：False)
        * **num_workers** (int, optional)：你要使用多少個 CPU 去載入資料 0 means that the data will be loaded in the main process. (預設值：0)
        * **pin_memory** (bool, optional)：在回傳資料前是否要複製 tensors 到 CUDA 的 pinned memory (預設值：False)
        * **drop_last** (bool, optional)： 最後的資料無法滿足 batch 的大小的時候是否要丟棄，否則最後一個 batch 會較小 (預設值：False)。例如：資料量 1234 筆，batch_size = 30，最後的 4 筆會被丟棄

**Example**
```python=
'''
使用 DataLoader 將 dataset 包起來
'''
BATCH_SIZE = 10

train_loader = DataLoader(train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True, 
                          num_workers=4, 
                          pin_memory=True)

val_loader = DataLoader(val_dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True, 
                        num_workers=4, 
                        pin_memory=True)
```

## Neural Network

### Building NN Model

**Example**
```python=
'''
先設一些超參數
'''
INPUT = len(train_feature[0]) # 資料維度
NUM_CLASSES = 2 # label 數量
EPOCH = 50 # 所有資料跑幾次
LR = 0.01 # learning rate
```
```python=
class spamModel(nn.Module):
    def __init__(self, INPUT, NUM_CLASSES):
        super(spamModel, self).__init__()
        
        # nn.Sequential() 快速搭建神經網路的方法，簡化程式的感覺
        self.classifier = nn.Sequential(  
            nn.Dropout(0.2), # 隨機 20% 的神經元被捨棄
            nn.Linear(INPUT, 1024),
            nn.Dropout(0.2),
            nn.ReLU(True), # 激勵函數
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), # 批次正規化，讓每層的值能夠在有效的範圍內傳遞下去
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, NUM_CLASSES)
         )
        
    def forward(self, x):
        output = self.classifier(x)
        return output         

```
> ### 小補充
> [color=#398bc6] 
> 
    Dropout 和 Batch Normalization 的作用都是避免 overfitting
    Batch Normalization 通常加在全連接層與激勵函數中間
    
>### 參考資料
>
> [Dropout](https://morvanzhou.github.io/tutorials/machine-learning/torch/5-03-dropout/)
> [Batch Normalization](https://morvanzhou.github.io/tutorials/machine-learning/torch/5-04-A-batch-normalization/)
>
>[name=可愛的瑄]
>

### Training and Validation
* 在訓練模型前需要設 Optimizer 與 Cost Function：
    * **Optimizer**：
        * `SGD`：隨機梯度下降法這種方法是將資料分成⼀小批⼀小批的進⾏訓練。但是速度比較慢。
        * `Adagrad`：改變 learning rate 的⽅式
        * `RMSporp`：跟 Adagrad 有點像，⾃適應調節 learning rate。
        * `Adam`：Adagrad + RMSporp 的結合
    * **Cost Function**：衡量模型**預測值**與**實際值**之間差異的函數
        * `BCELoss` (Binary Cross Entropy)：只有兩類的話可以使用此方法
        * `CrossEntropyLoss`：多類別的分類時使用
        * `MSELoss` (mean squared error)：均⽅誤差，預測值與實際值的差距之平均值

**Example**

```python=
'''
設 Optimizer 與 Cost Function
'''
# 把模型生出來丟到 GPU，如果不想用 GPU 就不要加 cuda
spam_model = spamModel(INPUT, NUM_CLASSES).cuda()
# 使用 Adam
optimizer = torch.optim.Adam(spam_model.parameters(), lr=LR)
# 使用 Cross Entropy 方法去計算 loss
criterion = nn.CrossEntropyLoss()
```

* 訓練與驗證模型

**Example**
```python=
train_loss = list()
val_loss = list()
'''
訓練模型
'''
print("-----------------------訓練資料--------------------------------")
for epoch in range(EPOCH):

    # 訓練模式
    spam_model.train()
    for index, (feature, label) in enumerate(train_loader): 
        
        # 有 Variable 才能做 Back Propagation
        # 也要記得把資料都丟進 GPU (如果模型使用 GPU)
        feature = Variable(feature).cuda()
        label = Variable(label).cuda()

        # 下一次訓練的時候先把前次的梯度清空
        optimizer.zero_grad()
        # 預測值
        outputs = spam_model(feature)
        # 計算誤差
        loss = criterion(outputs, label)
        # Back Propagation
        loss.backward()
        # 更新 gradients
        optimizer.step()

        if (index+1) % 100 == 0:
            # 預測正確的有幾個
            correct = torch.sum(torch.argmax(outputs,dim=1)==label)
            print ('Epoch: [%d/%d], Batch: [%d/%d], Loss: %.10f, Accuracy: %.6f'
                   % (epoch+1, EPOCH, index+1, len(train_dataset)/BATCH_SIZE, loss.item(), correct.item()/BATCH_SIZE))
    train_loss.append(loss)
'''
驗證模型
'''
print("-----------------------驗證資料--------------------------------")   
# 切換到評估模式
spam_model.eval()      
# 不需要進行 backward 所以也就不用計算梯度值
with torch.no_grad():
    for epoch in range(EPOCH):
        # 驗證資料
        for index, (feature, label) in enumerate(val_loader): 

            feature = Variable(feature).cuda()
            label = Variable(label).cuda()

            val_outputs = spam_model(feature)
            loss = criterion(val_outputs, label)
            
            if (index+1) % 100 == 0:
                val_correct = torch.sum(torch.argmax(val_outputs,dim=1)==label)
                print ('Epoch: [%d/%d], Batch: [%d/%d], Loss: %.10f, Accuracy: %.6f'
                       % (epoch+1, EPOCH, index+1, len(val_dataset)/BATCH_SIZE, loss.item(), val_correct.item()/BATCH_SIZE))
        val_loss.append(loss)
```
output：

    -----------------------訓練資料--------------------------------
    Epoch: [1/50], Batch: [100/445], Loss: 0.1673426926, Accuracy: 0.900000
    Epoch: [1/50], Batch: [200/445], Loss: 0.0977606103, Accuracy: 1.000000
    Epoch: [1/50], Batch: [300/445], Loss: 0.2467552125, Accuracy: 0.900000
    Epoch: [1/50], Batch: [400/445], Loss: 0.1291934401, Accuracy: 1.000000
    Epoch: [2/50], Batch: [100/445], Loss: 0.0328372903, Accuracy: 1.000000
    Epoch: [2/50], Batch: [200/445], Loss: 0.0641695485, Accuracy: 1.000000
    Epoch: [2/50], Batch: [300/445], Loss: 0.1754610986, Accuracy: 0.900000
    Epoch: [2/50], Batch: [400/445], Loss: 0.4433955252, Accuracy: 0.900000
    Epoch: [3/50], Batch: [100/445], Loss: 0.0418901481, Accuracy: 1.000000
    Epoch: [3/50], Batch: [200/445], Loss: 0.0220499989, Accuracy: 1.000000
    Epoch: [3/50], Batch: [300/445], Loss: 0.0886580124, Accuracy: 0.900000
    Epoch: [3/50], Batch: [400/445], Loss: 0.0150807742, Accuracy: 1.000000
    Epoch: [4/50], Batch: [100/445], Loss: 0.0478897467, Accuracy: 1.000000
    Epoch: [4/50], Batch: [200/445], Loss: 0.0192247629, Accuracy: 1.000000
    Epoch: [4/50], Batch: [300/445], Loss: 0.0089771152, Accuracy: 1.000000
    Epoch: [4/50], Batch: [400/445], Loss: 0.2824593186, Accuracy: 0.900000
    Epoch: [5/50], Batch: [100/445], Loss: 0.0919226184, Accuracy: 0.900000
    .
    .
    .
    -----------------------驗證資料--------------------------------
    Epoch: [1/50], Batch: [100/111], Loss: 0.1565511823, Accuracy: 0.900000
    Epoch: [2/50], Batch: [100/111], Loss: 0.0056394697, Accuracy: 1.000000
    Epoch: [3/50], Batch: [100/111], Loss: 0.0011286974, Accuracy: 1.000000
    Epoch: [4/50], Batch: [100/111], Loss: 0.0211444013, Accuracy: 1.000000
    Epoch: [5/50], Batch: [100/111], Loss: 0.8300634623, Accuracy: 0.900000
    Epoch: [6/50], Batch: [100/111], Loss: 0.0029574155, Accuracy: 1.000000
    Epoch: [7/50], Batch: [100/111], Loss: 0.0023370266, Accuracy: 1.000000
    Epoch: [8/50], Batch: [100/111], Loss: 0.0006031751, Accuracy: 1.000000
    Epoch: [9/50], Batch: [100/111], Loss: 0.0114302989, Accuracy: 1.000000
    Epoch: [10/50], Batch: [100/111], Loss: 0.0121690985, Accuracy: 1.000000
    Epoch: [11/50], Batch: [100/111], Loss: 0.0070082424, Accuracy: 1.000000
    Epoch: [12/50], Batch: [100/111], Loss: 0.0168771390, Accuracy: 1.00000




> ### 小補充
> [color=#398bc6] 
> 
    .train() 與 .eval() 模式有什麼差 ? 有時候可以不需要用 ?
    
    如果神經網路有 dropout 或是 batch Normalization，在訓練期間他們防止 overfitting 的功能
    
    設置 .eval() 是要把 dropout 和 batch Normalization 鎖住，要讓我們使用 .train() 
    下的模型來預測，因為 dropout 在訓練時會丟掉一些神經元，有些值為 0，若沒有設置 .eval() 
    可能會使模型預測結果不一樣(例如：圖片失真)


>### 參考資料
>
> [train( ) and eval( )-1](https://jamesmccaffrey.wordpress.com/2019/01/23/pytorch-train-vs-eval-mode/)
> [train( ) and eval( )-2](https://zhuanlan.zhihu.com/p/54986509)
> 
>[name=可愛的瑄]


* 可以畫圖來看是否 overfitting

**Example**
```python=
'''
過度擬合 (overfitting) 代表
1. 訓練集的 loss 下降的遠比驗證集的 loss 還來的快
2. 驗證集的 loss 隨訓練時間增長，反⽽上升
'''
# 以視覺化方式檢視訓練過程
import matplotlib.pyplot as plt

plt.plot(range(len(train_loss)), train_loss, label="train loss")
plt.plot(range(len(val_loss)), val_loss, label="val loss")
plt.legend()
plt.title("Loss")
plt.show()
```

output：
![](https://i.imgur.com/hVkjTY2.png)


:sparkles: 更明顯看出 overfitting 的例子
```python=
self.classifier = nn.Sequential(  
    nn.Dropout(0.2), # 隨機 20% 的神經元被捨棄
    nn.Linear(INPUT, 128),
    nn.ReLU(True), # 激勵函數
    nn.Linear(128, NUM_CLASSES)
)
```
![](https://i.imgur.com/nUy0d2W.png)


### Prediction
* 此例直接使用驗證資料做測試預測

```python=
pred = spam_model(val_feature.cuda())
pred = torch.max(pred,1)[1].data.cpu().numpy()

print('預測值：', pred)

```
output：

    預測值： [1 1 0 ... 0 0 0]

### Evaluation

* 對預測的資料進行評估，[點我回顧衡量指標](https://hackmd.io/qBqUxoOBRq225G5mxWBNfw)
    * Accuracy
    * AUC
    * Confusion Matrix

**Example**
```python=
'''
承上預測資料範例
'''
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn import metrics
from sklearn.metrics import confusion_matrix

test = np.asarray(val_label).astype(int)
print('實際值：', test)

'''
正確率
'''
accuracy = accuracy_score(test, pred)
print('Accuracy：', accuracy)

'''
AUC
'''
fpr, tpr, _ = metrics.roc_curve(pred, test)
auc = auc(fpr, tpr)
print('AUC：', auc)

'''
Confusion Matrix 與 各實際值與預測值的關係
'''
cm = confusion_matrix(test, pred)
print('Confusion Matrix： \n', cm)

TN, FP, FN, TP = confusion_matrix(test, pred).ravel()
print(TN, FP, FN, TP)
```
output：

    實際值： [1 1 0 ... 0 0 0]
    Accuracy： 0.9829596412556054
    AUC： 0.9699921290830382
    Confusion Matrix： 
     [[956   7]
     [ 12 140]]
    956 7 12 140
     
> ### 小補充
> [color=#398bc6] 
> 
    想要 import 之前在衡量指標寫好，畫 confusion matrix 圖的 function
```python=
'''
本例是上傳至 colab 後，import .ipynb 檔案
也可自行在其他地方 import .py 
'''
import self_confusionMatrix

classes = ["ham","spam"]
self_confusionMatrix.plot_confusion_matrix(test, pred, classes, cm)

```
output：
![](https://i.imgur.com/pBXSeNY.png)

## CNN

* 切分資料
```python=
features = data_SMS.content.values
labels = data_SMS.spam.values

train_feature, val_feature, train_label, val_label= train_test_split(features,
                                                                     labels, 
                                                                     shuffle = True, 
                                                                     test_size=0.2, 
                                                                     random_state=2019)
```
* 把 text (content) 長度變成一致
```python=
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_fatures = 6000

# Tokenize
tokenizer = Tokenizer(num_words=max_fatures)
tokenizer.fit_on_texts(train_feature)
word_index = tokenizer.word_index
print('找到 %s 個不重複的單字' % len(word_index))
print('計數 \n', tokenizer.word_counts)

# Sequences
train_feature = tokenizer.texts_to_sequences(train_feature)
val_feature = tokenizer.texts_to_sequences(val_feature)
# 一句的每個 word 的 index
print(train_feature[0])

# Padding 至一樣長度
train_feature = pad_sequences(train_feature, maxlen=55)
val_feature = pad_sequences(val_feature, maxlen=55)
print(len(train_feature[0]))
```
output：

    Using TensorFlow backend.
    找到 4375 個不重複的單字
    計數 
     OrderedDict([('urgent', 55), ('trying', 30), ('contact', 55), ('last', 62), ('weekend', 31), ('draw', 34), ('show', 47), ('prize', 79) ...
    [91, 205, 92, 80, 194, 177, 121, 56, 133, 1, 48, 249, 294, 124]
    55
> ### 小補充
> [color=#398bc6] 
> 
    有時候 train 和 val 資料 padding 後的長度會不一樣，所以最好是設一下 maxlen
* 資料整理後丟進 dataset
```python=
train_feature = torch.Tensor(train_feature)
train_label = torch.Tensor(train_label).long()

val_feature = torch.Tensor(val_feature)
val_label = torch.Tensor(val_label).long()

train_dataset = TensorDataset(train_feature, train_label)
val_dataset = TensorDataset(val_feature, val_label)
```
* 設定所需參數
```python=
'''
先設一些超參數
'''
textCNN_param = {
    'OUTPUT': 2, # label 數量
    'EPOCH': 10, # 所有資料跑幾次
    'LR': 0.001, # learning rate
    'BATCH_SIZE': 32,
    'INPUT': (len(word_index)+1), # vocab_size
    'EMBEDDING_LENGTH': 100 
}
```


* Dataset
```python=
'''
這裡直接用 TensorDataset
'''
train_dataset = TensorDataset(train_feature, train_label)
val_dataset = TensorDataset(val_feature, val_label)
```
### Building CNN Model

* 如何知道經過每一層 size 會變成怎樣 ?
**output_size = ceil(( input_size - kernel_size + pad * 2  ) / stride) + 1**
:sparkles: 經過第一層後產生 20 個 32 x 54 大小的 feature map
:sparkles: 經過第二層產生 40 個 32 x 53 大小的 feature map
* 如果要把 map 補到原來大小，要加多少 padding ?
**(kernel_size - 1) / 2**

* batch_size x embedding_size x vocab_size 形式丟入 CNN

**Example**
```python=
class spamModel(nn.Module):
    def __init__(self, args):

        super(spamModel, self).__init__()
        vocab_size = args['INPUT']
        batch_size = args['BATCH_SIZE']
        num_class = args['OUTPUT']
        embedding_length=args['EMBEDDING_LENGTH']

        self.embedding = nn.Embedding(vocab_size, embedding_length)
        self.conv1 = nn.Sequential(         
            nn.Conv1d(  
                in_channels=embedding_length,              
                out_channels=20,    
                kernel_size=2,
                stride=1,                   
                padding=0,                 
            ),             
            nn.BatchNorm1d(20),
            nn.ReLU(True),        # 32 x 20 x 54              
        )
        self.conv2 = nn.Sequential(        
            nn.Conv1d(20, 40, 2, 1, 0),
            nn.BatchNorm1d(40),
            nn.ReLU(True),        # 32 x 20 x 53                    
        )

        self.classifier = nn.Sequential(  
            nn.Linear(40 * 53, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, num_class)

         )

    def forward(self, x):
        x = self.embedding(x) # torch.Size([32, 55, 100])
        x = x.permute(0,2,1) # torch.Size([32, 100, 55])
        conv1 = self.conv1(x) # torch.Size([32, 20, 54])
        conv2 = self.conv2(conv1) #torch.Size([32, 40, 53])

        x = conv2.view(conv2.size(0), -1) 
        output = self.classifier(x)
        return  output
```
* 訓練及驗證模型
```python=
'''
設 Optimizer 與 Cost Function
'''
spam_model = spamModel(textCNN_param).cuda()
# 使用 Adam
optimizer = torch.optim.Adam(spam_model.parameters(), lr=textCNN_param['LR'])
# 使用 Cross Entropy 方法去計算 loss
criterion = nn.CrossEntropyLoss()

'''
訓練模型
'''

##### code here

'''
驗證模型
'''
##### code here
```
>### 參考資料
>
> [textCNN](https://blog.csdn.net/yellow_red_people/article/details/80406552)
> [con1 詳解](https://www.cnblogs.com/pythonClub/p/10421799.html)
>[name=可愛的瑄]
>
## LSTM
* 切分資料
```python=
# array 形式
features = data_SMS.content.values
labels = data_SMS.spam.values

train_feature, val_feature, train_label, val_label= train_test_split(features,
                                                                     labels, 
                                                                     shuffle = True, 
                                                                     test_size=0.2, 
                                                                     random_state=2019)
```
* 把 text (content) 長度變成一致
```python=
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_fatures = 6000

# Tokenize
tokenizer = Tokenizer(num_words=max_fatures)
tokenizer.fit_on_texts(train_feature)
word_index = tokenizer.word_index
print('找到 %s 個不重複的單字' % len(word_index))
print('計數 \n', tokenizer.word_counts)

# Sequences
train_feature = tokenizer.texts_to_sequences(train_feature)
val_feature = tokenizer.texts_to_sequences(val_feature)
# 一句的每個 word 的 index
print(train_feature[0])

# Padding 至一樣長度
train_feature = pad_sequences(train_feature)
val_feature = pad_sequences(val_feature)
print(len(train_feature[0]))
```
output：

    Using TensorFlow backend.
    找到 4375 個不重複的單字
    計數 
     OrderedDict([('urgent', 55), ('trying', 30), ('contact', 55), ('last', 62), ('weekend', 31), ('draw', 34), ('show', 47), ('prize', 79) ...
    [91, 205, 92, 80, 194, 177, 121, 56, 133, 1, 48, 249, 294, 124]
    57

* 資料整理後丟進 dataset
```python=
train_feature = torch.Tensor(train_feature)
train_label = torch.Tensor(train_label).long()

val_feature = torch.Tensor(val_feature)
val_label = torch.Tensor(val_label).long()

train_dataset = TensorDataset(train_feature, train_label)
val_dataset = TensorDataset(val_feature, val_label)
```
* 設定所需參數
```python=
'''
先設一些超參數
'''
OUTPUT = 2 # label 數量
# 資料維度
INPUT = (len(word_index)+1) # vocab_size
HIDDEN_SIZE = 256
EMBEDDING_LENGTH = 100 
EPOCH = 10 # 所有資料跑幾次
LR = 0.01 # learning rate
```

* DataLoader：drop_last 設 True 避免報錯
```python=
'''
使用 DataLoader 將 dataset 包起來
'''
BATCH_SIZE = 16

# drop_last 設 True 
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers= 4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers= 4, pin_memory=True , drop_last=True)
```

### Building LSTM Model

> ### 參數說明
> [color=#398bc6] 
>  
    batch_size : 批次處理的大小
    output_size : 2 = (spam, ham)
    hidden_size : LSTM hidden state 的大小
    vocab_size : 包含唯一單詞的詞彙表的大小
    embedding_length : 經 word embeddings 之 Embedding 後維度) 多少維度表達一個字

**Example**
```python=
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class LSTM(nn.Module):

    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length):
       
        # Initializer
        super(LSTM, self).__init__()
        
        # model construction
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        
        # 丟幾個字進去 embedding
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.lstm = nn.LSTM(embedding_length, hidden_size, num_layers=2 , bidirectional=True)
        self.label = nn.Linear(4*hidden_size, output_size)

        self.is_first = True
  
    def forward(self , x , batch_size=None):

        x = self.word_embeddings(x)

        if self.is_first:
            print('input shape:' , x.shape)

        x = x.permute(1, 0, 2)

        if self.is_first:
            print('input permute shape:' , x.shape)
 
        if batch_size is None:
            h_0 = Variable(torch.zeros(4, self.batch_size, self.hidden_size).cuda()) # 4 = num_layers*num_directions
        else:
            h_0 =  Variable(torch.zeros(4, batch_size, self.hidden_size).cuda())    
   
        if batch_size is None:
            c_0 = Variable(torch.zeros(4, self.batch_size, self.hidden_size).cuda()) # 4 = num_layers*num_directions
        else:
            c_0 =  Variable(torch.zeros(4, batch_size, self.hidden_size).cuda()) 
        if self.is_first:
            print('h_0 shape:' , h_0.shape)
            
            
        # 這邊的 h_n 是存 LSTM 跑完後最後一層(t)的 hidden state，我們只要取出來接上一層 output layer 算 output 即可
        output, (h_n , c_n) = self.lstm(x, (h_0,c_0))

        if self.is_first:
            print('h_n shape:' , h_n.shape)    
        
        h_n = h_n.contiguous().view(h_n.size()[1], h_n.size()[0]*h_n.size()[2])
        if self.is_first:
            print('h_n contiguous shape:' , h_n.shape)
            
        logits = self.label(h_n)
        if self.is_first:
            print('logits shape:' , logits.shape)
            self.is_first = False
            
            
        return logits   
```

* 訓練及驗證模型
```python=
'''
設 Optimizer 與 Cost Function
'''
spam_model = LSTM(BATCH_SIZE, OUTPUT, HIDDEN_SIZE, INPUT, EMBEDDING_LENGTH).cuda()

##### code here

'''
訓練模型
'''

##### code here

'''
驗證模型
'''
##### code here
```
