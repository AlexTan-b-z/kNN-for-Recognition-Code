# encoding=utf8
import operator
import pdb
from os import listdir
from numpy import *
from PIL import Image


class MyKnn(object):

    def encode_img(self, im):
        #把图片转化成一维数组
        width = im.size[0]
        height = im.size[1]
        img_encoding = []
        for i in range(0, width):
            for j in range(0, height):
                cl = im.getpixel((i, j))
                clall = cl[0] + cl[1] + cl[2]
                if(clall == 0):  # 黑色
                    img_encoding.append(1)
                else:
                    img_encoding.append(0)
        array_img = array(img_encoding)
        return array_img

    def traindata(self, datadir):
        labels = []
        #labels代表种类的意思，一共有10类，即数字从0到9
        trainfile = listdir(datadir)
        num = len(trainfile)
        trainarr = zeros((num, 200))
        #trainarr为初始化为0的高为num，宽为200的矩阵（2维数组）
        for i in range(num):
            thisfname = trainfile[i]
            thislabel = thisfname.split('_')[0]
            labels.append(thislabel)
            # pdb.set_trace()
            trainarr[i, :] = loadtxt(datadir + '/' + thisfname)
        return trainarr, labels

    def knn(self, k, testdata, traindata, labels):
        # testdata:一维数组
        # traindata：二维数组
        # labels：一维列表，跟traindata一一对应
        # 以下shape取的是训练数据的第一维，即其行数，也就是训练数据的个数
        traindatasize = traindata.shape[0]
        dif = tile(testdata, (traindatasize, 1)) - traindata
        #tile()的意思是给一维的测试数据转为与训练数据一样的行和列的格式
        sqdif = dif ** 2
        sumsqdif = sqdif.sum(axis=1)
        #axis=1-----》横向相加的意思
        #sumsqdif在此时已经成为1维的了
        distance = sumsqdif ** 0.5
        sortdistance = distance.argsort()
        #sortdistance为测试数据到各个训练数据的距离按近到远排序之后的结果
        count = {}
        for i in range(k):
            vote = labels[sortdistance[i]]
            #sortdistance[i]测试数据最近的K个训练数据的下标
            #vote测试数据最近的K个训练数据的类别
            count[vote] = count.get(vote, 0) + 1
        sortcount = sorted(
            count.items(), key=operator.itemgetter(1), reverse=True)
        return sortcount[0][0]

    def recognize_code(self, img_path):
        img = Image.open(img_path)
        img = img.convert('RGB')
        code = []
        for i in range(5):
            #把图片切分成五部分，每一部分都调用一次knn算法
            region = (i * 10 - 3.2, 0, 10 + i * 10 - 3.2, 20)
            #3.2是测试出来的
            cropImg = img.crop(region)
            img_array = self.encode_img(cropImg)
            trainarr, labels = self.traindata('train/data')
            number = self.knn(6, img_array, trainarr, labels)
            code.append(number)
        return "".join(code)

if __name__ == '__main__':
    knn = MyKnn()
    code = knn.recognize_code('test/test3.jpg')
    print(code)