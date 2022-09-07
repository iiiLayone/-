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
![image](https://github.com/iyi111111/BaseOfMGBERT/blob/main/images/wb%E7%9A%84%E5%81%8F%E5%AF%BC%E6%95%B0.png)
```
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
    # tf.square() 操作代表对输入张量的每一个元素求平方，不改变张量形状
    # tf.reduce_sum() 操作代表对输入张量的所有元素求和，输出一个形状为空的纯量张量
w_grad, b_grad = tape.gradient(L, [w, b])        
# 计算L(w, b)关于w, b的偏导数
print(L, w_grad, b_grad)
```
### 线性回归
- tape.gradient(ys, xs) 自动计算梯度 
- optimizer.apply_gradients(grads_and_vars) 自动更新模型参数
```
import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
# 首先进行归一化操作

X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
# 声明了一个梯度下降 优化器 （Optimizer），其学习率为 5e-4。
# 优化器可以根据计算出的求导结果更新模型参数，从而最小化某个特定的损失函数
# 具体使用方式是调用其 apply_gradients() 方法

for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
    
    # 更新模型参数的方法optimizer.apply_gradients() 需要提供参数 grads_and_vars，
    # 即待更新的变量（如上述代码中的 variables ）及损失函数关于这些变量的偏导数（如上述代码中的 grads ）
    # 这里需要传入一个 Python 列表（List），列表中的每个元素是一个 （变量的偏导数，变量） 对。
    # 比如上例中需要传入的参数是 [(grad_a, a), (grad_b, b)] 。
    # 通过 grads = tape.gradient(loss, variables) 求出 tape 中记录的 loss 关于 variables = [a, b] 中每个变量的偏导数，也就是 grads = [grad_a, grad_b]，
    # 再使用 Python 的 zip() 函数将 grads = [grad_a, grad_b] 和 variables = [a, b] 拼装在一起，就可以组合出所需的参数了。
    
    # 如果 a = [1, 3, 5]， b = [2, 4, 6]，那么 zip(a, b) = [(1, 2), (3, 4), ..., (5, 6)] 
    # zip() 函数返回的是一个 zip 对象，本质上是一个生成器，需要调用 list() 来将生成器转换成列表
```
### 线性回归 f(x) = ax + b
预测值f(x)与真实值误差y越小越好：使用**均方差**作为成本函数:

优化的目标：找到合适的a和b，使得(f(x)-y)^2最小

顺序模型：
```
model = tf.keras.Sequential()  # 空的

model.add(tf.keras.layers.Dense(1,input_shape=(1,))) 
# 输出维度为1，输入维度为1

model.summary()                # 返回模型整体的形状
# output shape 第一个维度代表样本的个数不需要考虑 第二个维度代表输出维度
# Param 代表参数

model.compile(optimizer = 'adam'，loss='mse')  
# adam最常用优化方法梯度下降算法 
# 损失函数使用均方差函数mse

history = model.fit(x,y,epochs = 5000)
# fit 进行训练  epochs 对所有数据训练多少次 5000次

model.predict(x)
model.predict(pd.Series[20]) 
# 使用训练好的模型进行预测  输入数据的格式要与x相同
```
### python 复习
#### 类的初始化
类有一个名为 __init__() 的特殊方法（构造方法），该方法在类实例化时会自动调用
```
def __init__(self):
    self.data = []
```
__init__() 方法可以有参数，参数通过 __init__() 传递到类的实例化操作上
```
class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart
x = Complex(3.0, -4.5)
```
类的方法与普通的函数只有一个特别的区别——它们必须有一个额外的第一个参数名称, 按照惯例它的名称是**self**

self 代表的是类的实例，代表当前对象的地址，而 self.class 则指向类。
#### 类的方法
类的内部，使用 def 关键字来定义一个方法，与一般函数定义不同，类方法必须包含参数 self, 且为第一个参数，self 代表的是类的实例。

#### 类的继承
- 私有属性： __ private_attrs：两个下划线开头，声明该属性为私有，不能在类的外部被使用或直接访问。在类内部的方法中使用时 self.__ private_attrs。
- 类的方法：在类的内部，使用 def 关键字来定义一个方法，与一般函数定义不同，类方法必须包含参数 self，且为第一个参数，self 代表的是类的实例。self 的名字并不是规定死的，也可以使用 this，但是最好还是按照约定使用 self。
- 类的私有方法：__ private_method：两个下划线开头，声明该方法为私有方法，只能在类的内部调用 ，不能在类的外部调用。self.__ private_methods。
- 子类（派生类 DerivedClassName）会继承父类（基类 BaseClassName）的属性和方法。
- 类的私有属性 和 私有方法，都不会被子类继承，子类也无法访问； 私有属性 和 私有方法 往往用来处理类的内部事情，不通过对象处理，起到安全作用。
- 私有方法和私有属性，子类通过调用通过实例化方法调用私有属性和方法，不能直接调用。

#### 类的专有方法
```
__init__ : 构造函数，在生成对象时调用
__del__ : 析构函数，释放对象时使用
__repr__ : 打印，转换
__setitem__ : 按照索引赋值
__getitem__: 按照索引获取值
__len__: 获得长度
__cmp__: 比较运算
__call__: 函数调用
__add__: 加运算
__sub__: 减运算
__mul__: 乘运算
__truediv__: 除运算
__mod__: 求余运算
__pow__: 乘方
```
#### super()函数
super() 函数是用于调用父类(超类)的一个方法。单继承的类层级结构中，super 引用父类而不必显式地指定它们的名称，从而令代码更易维护。（官方文档描述）
也就是说，在子类中不再用父类名调用父类方法，而是用一个代理对象调用父类方法，这样当父类名改变或者继承关系发生变化时，不用对每个调用处都进行修改。

super() 是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。

MRO 就是类的方法解析顺序表, 其实也就是继承父类方法时的顺序表。
```
# super().xxx

class FooParent(object):
    def __init__(self):
        self.parent = 'I\'m the parent.'
        print ('Parent')
    
    def bar(self,message):
        print ("%s from Parent" % message)
 
class FooChild(FooParent):
    def __init__(self):
        # super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），然后把类 FooChild 的对象转换为类 FooParent 的对象
        super(FooChild,self).__init__()    
        print ('Child')
        
    def bar(self,message):
        super(FooChild, self).bar(message)
        print ('Child bar fuction')
        print (self.parent)
 
if __name__ == '__main__':
    fooChild = FooChild()
    fooChild.bar('HelloWorld')
```
执行结果
```
Parent
Child
HelloWorld from Parent
Child bar fuction
I'm the parent.
```
#### __call__()
任何类，只需要定义一个__call__()方法，就可以直接对实例进行调用

### model与layer

Keras 有两个重要的概念： 模型（Model） 和 层（Layer） 。层将各种计算流程和变量进行了封装（例如基本的全连接层，CNN 的卷积层、池化层等），而模型则将各种层进行组织和连接，并封装成一个整体，描述了如何将输入数据通过各种层以及运算而得到输出。
在需要模型调用的时候，使用 y_pred = model(X) 的形式即可。Keras 在 tf.keras.layers 下内置了深度学习中大量常用的的预定义层，同时也允许我们自定义层。

Keras 模型以类的形式呈现，我们可以通过**继承** tf.keras.Model 这个 Python 类来定义自己的模型。
在继承类中，我们需要**重写** __init__() （构造函数，初始化）和 call(input) （模型调用）两个方法，同时也可以根据需要增加自定义的方法。



从 DataLoader 中随机取一批训练数据；

将这批数据送入模型，计算出模型的预测值；

将模型预测值与真实值进行比较，计算损失函数（loss）。这里使用 tf.keras.losses 中的交叉熵函数作为损失函数；

计算损失函数关于模型变量的导数；

将求出的导数值传入优化器，使用优化器的 apply_gradients 方法更新模型参数以最小化损失函数

TensorFlow 的图像表示为 [图像数目，长，宽，色彩通道数] 的四维张量

池化层（Pooling Layer）的理解则简单得多，其可以理解为对图像进行降采样的过程，对于每一次滑动窗口中的所有值，输出其中的最大值（MaxPooling）、均值或其他方法产生的值。



















