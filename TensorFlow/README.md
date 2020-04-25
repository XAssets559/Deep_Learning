# TensorFlow笔记

[toc]

## 基本

### 图和Operation

TensorFlow中每一个Tensor定义都是在图中完成的,但是并没有进行实际操作，还有Operation以及Viartion也是在图中定义的，如果直接去输出他们的值返回的是一个对象，并非是值。如果要输出值，需要在会话中开启Session.run()才能显示值，也可以在会话中用**实例.eval**()显示值。



构建图：定义数据和操作

执行图：开启会话，调用各方资源，用run方法运行起来

#### OP

![1](https://github.com/XAssets559/Deep-Learning/blob/master/TensorFlow/Picture/1.jpg)

操作函数									&												操作对象

tf.constant(Tensor对象)					  输入Tensor对象-Const	-输出Tensor对象

tf.add(Tensor对象1，Tensor对象2)	Tensor对象1，Tensor对象2-Add-Tensor对象3



在图中打印语句打印的是对象

------

### 图与TensorBord

> 图结构：数据（Tensor）+操作（Op）

#### 分类：默认图、自定义图

> 一张图一个命名空间

##### 查看默认图的方法

> TF会自动帮我们创建一张默认图

1）调用方法：tf.get_defult_graph()

2）查看属性：.graph

##### 创建图：

- 可以通过tf.Graph()自定义创建图

- 如果要在这张图中创建OP，典型用法是使用上下文管理器：with xxx.as_defult():
- 开启自己图的会话：with tf.Session(graph = XXX) as sess:

#### TensorBord

图在TensorFlow中是有一个TensorBoard（可视化学习）可以显示的，

​	打开TensorBoard的方法：

​		1.将图序列化到本地：tf.summary.FileWriter('path',graph = [会话中的图])

​				执行完以后，将在指定目录中生成一个event文件，其名称格式如下：events.out.tfevents.{timestamp}.{hostname}

​		2.启动TensorBoard：tensorboard	--logdir = "Path"

​		3.在浏览器中打开TensorBoard的图页面127.0.0.1:6006。

------

### 会话

> 会话包含两种开启方式：
>
> - tf.Session:用于完整的程序当中
> - tf.InteractiveSession:用于交互式上下文中的TensorFlow，例如Shell

#### 初始化

1）**会话掌握资源，用完要回收**

1.上下文管理器With

2.sess = tf.Session()以及sess.close()

2）**初始化对象时的参数**

1.graph = None：图

2.target = ' '：如果将此参数留空（默认值），会话将仅使用本地计算机中的设备。可以指定grpc://网址，以便制定TensorFlow服务器的地址，这使得会话可以访问该服务器控制的计算机上的所有设备。

3.config = None：此参数允许您制定一个tf.ConfigProto以便控制会话的行为，例如：ConfigProto协议用于打印设备使用信息。

3）**会话的Run()**

> run(fetches,feed_dict = None,options = None,run_metadata = None)

1.fetchrs：单一的Operation，或者列表、元组（其它不属于tensorflow的类型不行）

2.feed_dict：参数允许调用者覆盖图中张量的值，运行时赋值

​		与tf.placeholdre搭配使用，则会检查值得形状是否与占位符兼容

4）**feed操作**

- placeholder提供占位符，run时候通过feed_dict指定参数

------

### 张量Tensor

TensorFlow的张量就是一个n维数组，类型为tf.Tensor。Tensor具有以下两个重要的属性

type：数据类型

shape：形状（阶）

#### 张量的类型

![2](https://github.com/XAssets559/Deep-Learning/blob/master/TensorFlow/Picture/2.jpg)

如果不指定类型，默认	tf.float32

#### 创建随机张量

一般我们经常使用的随机函数Math.random()产生的是服从均匀分布的随机数，能够模拟等概率出现的情况，例如扔一个骰子，1到6点的概率应该相等，但现实生活中更多的随机现象是符合正态分布的，例如20岁成年人的体重分布等。

```python
tf.truncated_normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None,
)
```

```python
tf.random_normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None,
)
```

从正态分布中输出随机值，由随机正态分布的数字组成的矩阵

- 其他特殊的创建张量的OP
  - **tf.Variable**
  - tf.placeholder

#### 张量的变换

##### 类型改变

提供了如下一些改变张量中数值类型的函数

- ```python
  tf.string_to_number(string_tensor, out_type=tf.float32, name=None)
  ```

- ```python
  tf.to_float(x, name='ToFloat')
  ```

- ```python
  tf.to_double(x, name='ToDouble')
  ```

- ```python
  tf.to_int32(x, name='ToInt32')
  ```

- ```python
  tf.to_int32(x, name='ToInt64')
  ```

- ```python
  tf.cast(x, dtype, name=None)
  ```

##### 形状改变

TensorFlow的张量具有两种形状变换，动态形状和静态形状

- tf.reshape
- tf.set_shape

关于动态形状和静态形状必须符合以下规则

- 静态形状
  - 转换静态形状的时候，1-D到1-D,2-D到2-D不能跨阶数改变形状
  - 对于已经固定静态形状的张量，不能再次设置静态形状
  - 只有在形状没有完全固定下来（tf.placeholder）的情况才可以改变静态形状：tf.set_shape(shape)
- 动态形状
  - tf.reshape(x,shape)动态创建新张量时，张量的元素个数必须匹配

------

### 变量OP

#### 创建变量

- ```python
  tf.Variable(
      initial_value=None,
      trainable=True,
      collections=None,
      validate_shape=True,
      caching_device=None,
      name=None,
      variable_def=None,
      dtype=None,
      expected_shape=None,
      import_scope=None,
      constraint=None,
  )
  ```

  - Initial_value：初始化变量
  - trainable：是否被训练
  - Colloctions：新变量将添加到列出的图的集合中collections，默认为[GraphKeys.GLOBAL_VARIABLES]，如果trainable时True变量也被添加到图形集合GraphKeys.GLOBAL_VARIABLES

- 变量需要显式初始化，才能运行值

  - init = tf.global_variables_initializer()
  - Sess.run(init)

#### 修改变量命名空间

```
tf.variable_scope(
    name_or_scope,
    default_name=None,
    values=None,
    initializer=None,
    regularizer=None,
    caching_device=None,
    partitioner=None,
    custom_getter=None,
    reuse=None,
    dtype=None,
    use_resource=None,
    constraint=None,
    auxiliary_name_scope=True,
)
```

- name_or_scope：“命名空间”

### 基础API和改机API

#### 基础API

**tf.app**

这个模块相当于为TensorFlow 进行的脚本提供一个 main 函数入口，可以定义脚本运行的 flags。

**tf.image**

Tensorflow 的图像处理操作。主要是一些颜色变换、变形和图像的编码和解码。

**tf.gfile**

这个模块提供了一组文件操作函数

**tf.summary**

用来生成TensorBoard可用的统计目标，目前 Summary 主要提供了4种类型：audio、image、histogram、scalar

**tf.python_io**

用来读写 TFRecords 文件

**tf.train**

这个模块提供了一些训练器，与 tf.nn 组合起来，实现一些网络的优化计算。

**tf.nn**

这个模块提供了一些构建神经网络的底层函数。TensorFlow 构建网络的核心模块。

#### 高级API

**tf.keras**

Keras 本来是一个独立的深度学习库。tensorflow 将其学习过来，增加这部分模块在于快速构建模型。

**tf.layers**

以更高级的概念层来定义一个模型。类似tf.keras。

**tf.contrib**

tf.contrib.layers提供够计算机图中的 网络层、正则化、摘要操作，是构建计算机图的高级操作，但是tf.contrib 包含不稳定和实验代码，可能以后API会改变。

**tf.estimator**

一个Estimator 相当于Model + Training + Evaluate 的合体。在模块中，已经实现了几种简单的分类器和回归器，包括：Baseline、Learning 和 DNN。这里的DNN 的网络，只是全连接网络，没有提供卷积之类的。

------

## 实战

### 自实现线性回归

### 文件读取通用流程

#### 第一阶段：创建文件名队列

- tf.train.string_input_producer(string_tensor,shuffle = True)
  - string_tensor:含有文件名+路径的一阶张量
  - num_epochs:过几遍数据，默认无限过
  - return 文件名队列

#### 第二阶段：读取与解码

#####     读取：

- tf.TextLineReader:
  - 阅读文本文件逗号分隔值(CSV)格式，默认按行读取
  - return 读取器实例
- tf.WholeFileReader :用于读取图片文件
  - return 读取器实例
- tf.FixedLengthRecordReader(record_bytes):二进制文件
  - 要读取每个记录数是固定数量字节的二进制文件
  - record_bytes:整形，指定每次读取（一个样本）的字节数
  - return 读取器实例
- tf.TFRecordReader:读取TFRecords文件
  - return 读取器实例

> 1.他们有共同的读取方法：read(file_queue),并且都会返回一个Tensor元组（key文件名字，value默认内容（一个样本））
>
> 2.由于默认只会读取一个样本，所以如果想要进行批处理，需要使用tf.train.batch或tf.train.shuffle_batch进行批处理，便于之后指定每批次多个样本的训练

##### 解码：

- tf.decode_csv:解码文本文件内容
- tf.image.decode_jpeg(contents)
  - 将JPEG编码的图像解码成uint8张量
  - <font color = "red">如果图片大小不一样，要调成一样的形状</font>
  - return：uint8张量，3-D形状[heitght，width，cannels]
- tf.image.decode_png(contents)
  - 将PNG编码的图像解码成uint8张量
  - return：张良类型，3-D形状[height,width,cannels]
- tf.decode_raw:解码二进制文件内容
  - 与tf.FixedLengthRecoderReader搭配使用，二进制读取为uint8张量

> 解码阶段，默认所有的内容都解码成tf.uint8类型,如果之后需要其他类型，可使用tf.cast()进行相应转换

#### 第三阶段：批处理

- tf.train.batch(tensors,batch_size,num_treads = 1,capacity = 32,name = None)
  - 读取指定大小(个数)的张量
  - tensors：可以使包含张量的列表，批处理内容放到列表当
  - batch_size：从列表中读取的批处理大小
  - num_treads：进入队列线程数
  - capacity：证书，队列中元素的最大数量
  - return ：tensors
  - tf.train.shuffle_batch

##### 线程操作

> 以上用到的队列都是tf.train.QueueRunner对象。
>
> 每个对象负责一个阶段，tf.train.start_queue_runners函数会要求图中的每个QueueRunner启动它的运行队列操作的线程。（需要在会话中开启）

- tf.train.start_queuerunners(sess=None,coord = None)
  - 收集图中所有的队列线程，默认同时启动线程
  - sess:所在的会话
  - coord:线程协调器
- tf.train.Coordinator()
  - 线程协调员，对县城进行管理和控制
  - request_stop():请求停止
  - should_stop():询问是否结束
  - join(threads=None,stop_grace_period_secs=120):回收线程
    - return 协调员实例

