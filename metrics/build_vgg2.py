'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''


import os
import random
import shutil
import tqdm

def cpfile_rand(img, outfile, num):
    list_ = os.listdir(img)
    if num > len(list_):
        print('输出数量必须小于：', len(list_))
        exit()
    numlist = random.sample(range(0,len(list_)),num) # 生成随机数列表a
    cnt = 0
    for n in numlist:
        filename = list_[n]
        oldpath = os.path.join(img, filename)
        newpath = os.path.join(outfile, filename)
        shutil.copy(oldpath, newpath)
#         print('剩余文件：', num-cnt)
        cnt = cnt + 1
#     print('==========task OK!==========')
def check_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

root_dir='/media/Storage/facedata/vgg_mtcnnpy_160/'
save_dir='/media/Storage/facedata/vgg_mtcnnpy_160_shuffled/'
list_root = os.listdir(root_dir)
for person in tqdm.tqdm(list_root):
    dir_person = root_dir + person
    dest_dir = save_dir + person
    check_folder(dest_dir)
    cpfile_rand(dir_person, dest_dir, 6)