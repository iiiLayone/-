# transformer (Attention is all you need)
残差连接网络 N编码器重复几次 复制模块 d_model 表示每一层的输出的维度
![image](https://github.com/iiiLayone/base-of-mgbert/blob/main/images/transformer.png)



## layer norm和batch norm  
seq:sequence序列的长度n。 feat :feature 特征 


layer norm：对每个样本标准化把每一行标准化 拿样本自己来求更稳定  把layer norm的数据转置放到batch norm中处理，再转置回去就是想要的结果。

batch norm： 每次取一个特征把均值变为0方差变为1 二维中把每一列标准化 变长的数据中不使用batch norm 
