# coding=utf-8
import tensorflow as tf
import os
class CIFAR(object):
    def __init__(self):
        self.height = 32
        self.width = 32
        self.channels = 3

        self.image_bytes = self.height*self.width*self.channels
        self.label_bytes = 1
        self.all_bytes = self.image_bytes+self.label_bytes
    def read_and_decode(self,file_list):
        # 构建文件名队列
        file_queue = tf.train.string_input_producer(file_list)
        # 读取器
        reader = tf.FixedLengthRecordReader(self.all_bytes)
        # 读取
        key,value = reader.read(file_queue)
        # 解码
        decoded = tf.decode_raw(value,tf.uint8)
        print(decoded)
        # 对Tensor对象进行切片
        label = tf.slice(decoded,[0],[self.label_bytes])
        image = tf.slice(decoded,[self.label_bytes],[self.image_bytes])
        print(image)
        image_reshaped = tf.reshape(image,shape=[self.channels,self.height,self.width])
        print(image_reshaped)
        # 图片转置height,width,channels
        image_transposed = tf.transpose(image_reshaped,[1,2,0])
        # 调整类型
        image_cast = tf.cast(image_transposed,tf.float32)
        # 批处理
        label_batch,image_batch = tf.train.batch([label,image_cast],batch_size=100,capacity=100)
        print(image_cast)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord= coord)
            # print("image_reshaped:\n",sess.run(image_reshaped))
            # print("image_transposed:\n",sess.run((image_transposed)))
            coord.request_stop()
            coord.join(threads)
        return None
if __name__ == "__main__":
    # 文件名+路径列表
    filename = os.listdir("./cifar-10-batches-py")
    file_list = [os.path.join("./cifar-10-batches-py/",file)for file in filename if file[-3:] == "bin"]
    cifar = CIFAR()
    cifar.read_and_decode(file_list)