
# from PIL import Image
# ###转为灰度图  公式Y = 0.299 R + 0.587 G + 0.114 B
#
# im = Image.open(r'C:\Users\yinghe\pyworkspace\467benz\new\1dln.jpg')
# im1 = im.convert('L')
# im1.save(r'C:\Users\yinghe\pyworkspace\467benz\new\1dln.png')

from os.path import splitext
import glob
from PIL import Image


def get_all_file(filename):                                               #glob模块是用来查找匹配的文件的
    files = glob.glob(filename)
    return files


def to_ather_file(files, type):
    for jpg in files:
        im = Image.open(jpg).convert('L')                                 #转为灰度图
        im = im.resize((60, 160))
        png = splitext(jpg)[0] + "." + type
        im.save(png)


if __name__ == "__main__":
    filename = "C:/Users/yinghe/pyworkspace/467benz/0915/*.[Jj][Pp][Gg]"    #指定图片文件路径
    files = get_all_file(filename)
    to_ather_file(files, "png")
