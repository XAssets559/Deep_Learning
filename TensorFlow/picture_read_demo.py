# coding=utf-8
import tensorflow as tf
import os
# 图片读取案例


#命令行参数设置
# tf.app.flags.DEFINE_integer("max_step",0,"最大训练步数")
# tf.app.flags.DEFINE_string("model_save_path","123456","模型保存位置")
# FLAGS = tf.app.flags.FLAGS
# def command_demo():
#     print ("max_step:\n",FLAGS.max_step)
#     prinnt ("model_save_path:\n",FLAGS.model_save_path)
#     return None

# def main(argv):
#     print(argv)

def picture_demo(file_list):
    # 构造文件名队列
    file_queue = tf.train.string_input_producer(file_list)
    # 读取与解码
    reader = tf.WholeFileReader()
    #读取
    key,value = reader.read(file_queue)
    # 解码
    image = tf.image.decode_jpeg(value)
    print("image:\n",image)
    # 图像形状调整、类型修改
    image_resized = tf.image.resize_images(image,[227,227])
    print(image_resized)
    # 静态形状改变
    image_resized.set_shape([227,227,3])
    print (image_resized)

    # 批处理
    tf.train.batch([image_resized],batch_size=100,num_threads=1,capacity=100)
    with tf.Session() as sess:
        # 开启线程
        # 线程协调员
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        # key_new,value_new = sess.run([key,value])
        # print ("key:\n",key_new)
        # print ("value:\n",value_new)
        coord.request_stop()
        coord.join(threads)
if __name__ == "__main__":
    # command_demo()
    # tf.app.run()
    # 构造文件名+列表
    filename = os.listdir("./dogs/alasijia_train")
    file_list = [os.path.join("./dogs/alasijia_train",file)for file in filename]
    # print(file_list)
    picture_demo(file_list)