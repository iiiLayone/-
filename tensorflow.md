### 张量（Tensor）
tf使用张量作为数据的基本单位，张量在概念上等同于多维数组
```
A = tf.constant([1,2],[3,4])  #定义一个2*2常量矩阵
```
张量的重要属性是其形状、类型和值可以通过张量的**shape**,**dtype**属性和**numpy()** 方法得到。
```
print(A.shape)
print(A.dtype)
print(A.numpy())  #张量的 numpy() 方法是将张量的值转换为一个 NumPy 数组。
```
TensorFlow 的大多数 API 函数会根据输入的值自动推断张量中元素的类型（一般默认为 tf.float32 ）。不过你也可以通过加入 dtype 参数来自行指定类型

### 自动求导机制
使用tf.GradientTape()自动求导记录器，实现自动求导
```
x = tf.Variable(initial_value=3.)
# x初始化为3的变量，通过tf.Variable()声明 变量与张量一样同样具有形状类型值三个属性
# 与张量的区别在于变量能被tf的自动求导机制求导，常用于定义机器学习模型的参数

with tf.GradientTape() as tape:     
# 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
# 只要进入了该语句的上下文环境，该环境中的计算步骤都会被自动记录
  y = tf.square(x)
# 此步骤被记录
y_grad = tape.gradient(y, x)        
# 计算y关于x的导数
# 离开上下文环境，记录停止，但是tape依然可用
print(y, y_grad)
```

















### 线性回归 f(x) = ax + b
预测值f(x)与真实值误差y越小越好：使用**均方差**作为成本函数:

优化的目标：找到合适的a和b，使得(f(x)-y)^2最小

顺序模型：
```
model = tf.keras.Sequential()  //空的

model.add(tf.keras.layers.Dense(1,input_shape=(1,))) 
//输出维度为1，输入维度为1

model.summary()               //返回模型整体的形状
//output shape 第一个维度代表样本的个数不需要考虑 第二个维度代表输出维度
//Param 代表参数

model.compile(optimizer = 'adam'，loss='mse')  
//adam最常用优化方法梯度下降算法 
//损失函数使用均方差函数mse

history = model.fit(x,y,epochs = 5000)
//fit 进行训练  epochs 对所有数据训练多少次 5000次

model.predict(x)
model.predict(pd.Series[20]) 
//使用训练好的模型进行预测  输入数据的格式要与x相同
```

### 梯度下降算法
optimizer 致力于找到函数极值点

tf会随机初始化很多值，计算梯度寻找函数变化最快的方向，沿着这个方向去改变参数值

每次移动的值叫做学习速率，不可过大也不可过小，需要人为固定的超参数

### 多层感知器











