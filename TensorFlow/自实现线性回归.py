# coding=utf-8
import tensorflow as tf
import os
# x = tf.constant([[0], [0.5], [1], [1.5], [2], [2.5]])
# y_true = tf.constant([[0.0031], [0.2023], [0.4014], [0.6006], [0.8000], [0.9995]])
def regression():
    '''
    自实现线性回归
    :return:
    '''
    x = tf.random_normal(shape=[100, 1])
    y_true = tf.matmul(x,[[0.8]]) + 0.7
    # 模型定义
    with tf.variable_scope("model_define") :
        weight = tf.Variable(tf.random_normal(shape=[1, 1]))
        bias = tf.Variable(tf.random_normal(shape=[1, 1]))
        y_predict = tf.matmul(x, weight) + bias
    # 最小二乘
    with tf.variable_scope("error") :
        error = tf.reduce_mean(tf.square(y_predict-y_true))
    # 迭代器
    with tf.variable_scope("optimizer") :
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
    # 初始化变量
    with tf.variable_scope("initlizer_variable") :
        init = tf.global_variables_initializer()
    # 收集变量
    tf.summary.scalar("error",error)
    tf.summary.histogram("weight",weight)
    tf.summary.histogram("bias",bias)
    # 一键合并变量
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        # 生成文件读写器
        file_writer = tf.summary.FileWriter("./summary",graph = sess.graph)
        # 模型保存~~
        saver = tf.train.Saver()
        print ("训练前weight：%.3f,bias:%.3f"%(weight.eval(),bias.eval()))
        for i in range(1000):
            sess.run(optimizer)
            # run合并操作
            summary = sess.run(merged)
            # 在文件读写器中加入summary
            file_writer.add_summary(summary,i)
            if i%10 == 0:
                saver.save(sess,"./model/model.ckpt")
        # if os.path.exists("./model/checkpoint"):
        #     saver.restore(sess,"./model/model.ckpt")
        print("拟合曲线为：y = %.3f x + %.3f" % (weight.eval(), bias.eval()))
    return None
if __name__ == "__main__":
    regression()