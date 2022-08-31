# transformer (Attention is all you need)

## 网络模型
残差连接网络 N编码器重复几次 复制模块 d_model 表示每一层的输出的维度

![image](https://github.com/iiiLayone/base-of-mgbert/blob/main/images/transformer.png)



### layer norm和batch norm  
seq:sequence序列的长度n。 feat :feature 特征 


layer norm：对每个样本标准化把每一行标准化 拿样本自己来求更稳定  把layer norm的数据转置放到batch norm中处理，再转置回去就是想要的结果。

batch norm： 每次取一个特征把均值变为0方差变为1 二维中把每一列标准化 变长的数据中不使用batch norm 

![image](https://github.com/iiiLayone/base-of-mgbert/blob/main/images/layernorm%2Cbatchnorm.png)

### attention 注意力层

![image](https://github.com/iiiLayone/base-of-mgbert/blob/main/images/attention11.png)

注意力函数是一个将quary和一些key-value对映射成一个输出的函数，output是value的加权和，value和权重等价于key和quary的相似度。相似度通过内积来判断
softmax是对每一行做softmax。
Softmax函数就可以将多分类的输出值转换为范围在[0, 1]和为1的概率分布。

![image](https://github.com/iiiLayone/base-of-mgbert/blob/main/images/attention.png)

#### Scale Dot-Product Attention

图解：Q和K是两个矩阵，接下来除以根号d_k.

mask是为了避免t时刻看到t时刻以后的东西。具体来说t时刻的Quary即q_t，只能看到k_1,k_2,...,k_t-1，不应该看到之后的东西。
但是注意力机制中可以看到全部的k_1,k_2,...,k_t-1,k_t,...,k_n。所以使用mask将t时刻之后的值全部换成非常大的负数，softmax之后权重就变成0了。

#### Muti-Head Attention
linear 线性层用来把矩阵投影到比较低的维度







