# 简单爬取中国裁判文书网验证码图片
import urllib
import requests
import time

for i in range(1, 1500):
    time.sleep(1)
    urllib.request.urlretrieve('http://wenshu.court.gov.cn/ValiCode/CreateCode/?guid=0eef5144-8306-590d9a25-c4f6c794dd8f',
                       r'C:\Users\yinghe\pyworkspace\467benz\0912\%s.jpg' % i)
