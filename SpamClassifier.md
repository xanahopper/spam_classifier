# 邮件分类

## 问题描述

给定邮件组的历史邮件，将其中的邮件识别分类，验证分类识别的程度。

## 问题分析

邮件数据本身并没有做标注，且数量巨大，所以不可能进行人工标注分类，所以要解决问题首先需要对训练样本做分类。在这里简单认为在一个文件夹中的文件均属于一个类型。

其次，根据样本所训练出来的模型，对需要预测的样本做预测，解决好这两个问题，问题就解决了。

## 数据预处理

- 使用 `python` 库 `email` 对原始邮件组文件进行读取，获取到其内容
- 通过常用英文分割方法将文本分割
- 过滤 停用词、无用词
- 提取所有文件的词典
- 提取所有文件对应的单词频率，成为文件对应的特征

## 数据分类、预测与验证

这里采用交叉验证的方法，将原始数据随机的分为 3:1 的训练与测试数据，进行训练与预测，预测的数据与原分类进行比对，获得以下结果：

```text
                          precision    recall  f1-score   support

   talk.politics.mideast       0.04      0.03      0.03       239
               rec.autos       0.01      0.02      0.01       242
   comp.sys.mac.hardware       0.00      0.00      0.00       241
             alt.atheism       0.00      0.00      0.00       263
      rec.sport.baseball       0.01      0.01      0.01       245
 comp.os.ms-windows.misc       0.44      0.02      0.03       265
        rec.sport.hockey       0.03      0.04      0.03       247
               sci.crypt       0.03      0.03      0.03       262
                 sci.med       0.03      0.02      0.02       258
      talk.politics.misc       0.12      0.16      0.14       239
         rec.motorcycles       0.01      0.02      0.01       242
          comp.windows.x       0.03      0.03      0.03       265
           comp.graphics       0.03      0.03      0.03       247
comp.sys.ibm.pc.hardware       0.02      0.04      0.02       249
         sci.electronics       0.05      0.05      0.05       260
      talk.politics.guns       0.03      0.02      0.03       250
               sci.space       0.01      0.01      0.01       235
  soc.religion.christian       0.03      0.03      0.03       241
            misc.forsale       0.01      0.01      0.01       258
      talk.religion.misc       0.03      0.02      0.03       252

             avg / total       0.05      0.03      0.03      5000
```

预测度很低（捂脸）