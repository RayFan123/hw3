# 数字图像处理实验报告——project3 histogram


## 姓名：范睿霖   班级：自动化66  学号：2160504145



## 摘要：直方图图像增强技术，包括图像直方图的输出，均衡，匹配，局部增强以及图像分割的内容


-------



###section1，输出图像的直方图：
lena1
![Figure_1](media/15528716586498/Figure_1.png)
lena2
![Figure_2](media/15528716586498/Figure_2.png)
lena4
![Figure_3](media/15528716586498/Figure_3.png)
elain1
![Figure_4](media/15528716586498/Figure_4.png)
elain2
![Figure_5](media/15528716586498/Figure_5.png)
elain3
![Figure_6](media/15528716586498/Figure_6.png)
citywall1
![Figure_7](media/15528716586498/Figure_7.png)
citywall2
![Figure_8](media/15528716586498/Figure_8.png)
woman1
![Figure_9](media/15528716586498/Figure_9.png)
woman2
![Figure_10](media/15528716586498/Figure_10.png)

-------

###section2,直方图均衡
lena1
![Figure_11](media/15528716586498/Figure_11.png)
lena2
![Figure_12](media/15528716586498/Figure_12.png)
lena4
![Figure_13](media/15528716586498/Figure_13.png)
elain1
![Figure_14](media/15528716586498/Figure_14.png)
elain2
![Figure_15](media/15528716586498/Figure_15.png)
elain3
![Figure_16](media/15528716586498/Figure_16.png)
citywall1
![Figure_17](media/15528716586498/Figure_17.png)
citywall2
![Figure_18](media/15528716586498/Figure_18.png)
woman1
![Figure_19](media/15528716586498/Figure_19.png)
woman2
![Figure_20](media/15528716586498/Figure_20.png)

分析：直方图均衡后，大部分图像的对比度都得到了改善，根据对直方图的分析发现，本身具有比较均匀分布像素的图像改善效果相对更好，而本身像素值大量重叠的图像改善效果相对较差。

-------
###section3,直方图匹配
lena2
![Figure_21](media/15528716586498/Figure_21.png)
lena4
![Figure_22](media/15528716586498/Figure_22.png)
elain1

![Figure_23](media/15528716586498/Figure_23.png)
elain2
![Figure_24](media/15528716586498/Figure_24.png)
elain3
![Figure_25](media/15528716586498/Figure_25.png)
citywall1
![Figure_26](media/15528716586498/Figure_26.png)
citywall2
![Figure_27](media/15528716586498/Figure_27.png)
woman1
![Figure_28](media/15528716586498/Figure_28.png)
woman2
![Figure_29](media/15528716586498/Figure_29.png)

-------

### section4，直方图局部增强
参数设置为：


| k0 | k1 | k2 | m |
| --- | --- | --- | --- |
| 0.9 | 0.3 | 1.8 | 2 |

 
lena:
![Figure_30](media/15528716586498/Figure_30.png)
elain:
![Figure_31](media/15528716586498/Figure_31.png)


-------
###section5,图像分割  
elain:
![Figure_32](media/15528716586498/Figure_32.png)
lena:
![Figure_33](media/15528716586498/Figure_33.png)



-------

##附录

代码：
直方图输出


```
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('woman2.BMP').convert('L')
a = np.array(img)
print (np.shape(a))
#hist = np.array(img.histogram())/(np.shape(a)[0]*np.shape(a)[1])
hist = np.zeros(256)

for i in range(np.shape(a)[0]):
    for j in range(np.shape(a)[1]):
        hist[a[i,j]] = hist[a[i,j]]+1

x = np.arange(256)

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax1 = fig.add_subplot(1,2,2)
ax1.imshow(a,cmap = 'gray')
ax.bar(x,hist)
plt.show()

```

直方图均衡

```
rom PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('woman2.bmp')

img = img.convert('L')

x = np.array(img)
x_1 = x.copy()
a = np.array(img.histogram())
a1 = a.copy()
mn =np.shape(x)[0]*np.shape(x)[1]
for i in range(np.shape(a)[0]-1,-1,-1):
    for j in range (i):
        a[i] = a[i] + a[j]
    a[i] = round(a[i] *255/mn)
'''add = 0
l = []
for i in range(len(a)):
    add += a[i]
    l.append(add)'''

for i in range (np.shape(x)[0]):
    for j in range (np.shape(x)[1]):
        x [i,j] = a[x[i,j]]


t = np.arange(256)

img = Image.fromarray(x)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(x_1,cmap='gray')
ax3.imshow(x,cmap='gray')
ax2.bar(t,a1/mn)
ax4.bar(t,np.array(img.histogram())/mn)
plt.show()

```

直方图匹配

```
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('elain3.bmp').convert('L')
img_1 = Image.open('elain.bmp').convert('L')

r = np.array(img.histogram())
r_1 = np.array(img_1.histogram())
x = np.array(img)
x_1 = np.array(img_1)
t = np.arange(np.shape(r)[0])
'''
fig_1 = plt.figure()
axa = fig_1.add_subplot(1,2,1)
axb = fig_1.add_subplot(1,2,2)
axa.bar(t,r)
axb.bar(t,r_1)
plt.show()
'''
x_m = x.copy()
mn = np.shape(x)[0] * np.shape(x)[1]
#r = r / mn
#r_1 = r_1 / mn
s = np.zeros((np.shape(r)[0],1))
s_1 = np.zeros((np.shape(r_1)[0],1))

for i in range(np.shape(r)[0]-1,-1,-1):   #分别计算两幅图像的均衡变换
    for j in range(i):
        s[i] = s[i] + r[j]
        s_1[i] = s_1[i] + r_1[j]
s = s * (np.shape(r)[0]-1) / mn
s_1 = s_1 * (np.shape(r_1)[0]-1) / mn
s = np.floor(s)
s_1 = np.floor(s_1)

j = 0
for i in range(np.shape(s)[0]):
    while j < len(s_1):
        if s[i] == s_1[j] or s[i] < s_1[j]:
            s[i] = j
            break
        j += 1

for i in range(np.shape(x)[0]):           #执行映射
    for j in range(np.shape(x)[1]):
        x_m[i, j] = s[x[i, j]]

img_m = Image.fromarray(x_m)
r_m = np.array(img_m.histogram())

fig = plt.figure()
ax1 = fig.add_subplot(2,3,1)
ax2 = fig.add_subplot(2,3,2)
ax3 = fig.add_subplot(2,3,3)
ax4 = fig.add_subplot(2,3,4)
ax5 = fig.add_subplot(2,3,5)
ax6 = fig.add_subplot(2,3,6)
ax1.imshow(x,cmap='gray')
ax2.imshow(x_1,cmap='gray')
ax3.imshow(x_m,cmap='gray')
ax4.bar(t,r)
ax5.bar(t,r_1)
ax6.bar(t,r_m)
plt.show()

```

直方图局部增强

```
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def compute_m (array):
    m = 0
    for i in range(np.shape(array)[0]):
        for j in range(np.shape(array)[1]):
            m = m + array[i,j]
    m = m / np.shape(array)[0] / np.shape(array)[1]
    return (m)

k0 = 0.9
k1 = 0.3
k2 = 1.8
e = 2
img = Image.open('elain.bmp').convert('L')
hist = np.array(img.histogram())
x = np.array(img)
x_o = x.copy()
m_g = np.mean(x)
sigma_g = np.var(x)
index = np.array([[1,1]])
for i in range(3,np.shape(x)[0]-3):
    for j in range(3,np.shape(x)[1]-3):
        y = x [i - 3 : i + 4, j - 3 : j + 4]
        m_l = np.mean(y)
        sigma_l = np.var(y)
        if m_l < k0 * m_g and sigma_l < k2 * sigma_g and sigma_l > k1 * sigma_g:
            index = np.row_stack((index,[i,j]))
index = np.delete (index,0,0)

for i in range(np.shape(index)[0]):
    x [index[i,0],index[i,1]] = x [index[i,0],index[i,1]] * e

t = np.arange(np.shape(hist)[0])
img_1 = Image.fromarray(x)
hist_1 = np.array(img_1.histogram())
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(x_o,cmap = 'gray')
ax2.bar(t,hist)
ax3.imshow(x,cmap = 'gray')
ax4.bar(t,hist_1)
plt.show()

```
直方图图像分割

```
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('lena.bmp').convert('L')
x = np.array(img)
x_n = x.copy()
m = np.array([[np.mean(x)]])

for i in range(10):
    x_1 = x[x > m[i]]
    x_2 = x[x <= m[i]]
    m_1 = np.mean(x_1)
    m_2 = np.mean(x_2)
    m = np.row_stack((m,(m_1 + m_2) / 2))
    if (m[i] - m[i + 1])**2 <= 1:
        break

for i in range(np.shape(x)[0]):
    for j in range(np.shape(x)[1]):
        if x_n[i,j] > m[-1]:
            x_n[i,j] = 255
        else:
            x_n[i,j] = 0

img_1 = Image.fromarray(x_n)
hist = np.array(img.histogram())
hist_1 = np.array(img_1.histogram())
t = np.arange(np.shape(hist)[0])

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.imshow(x,cmap = 'gray')
ax2.bar(t,hist)
ax3.imshow(x_n,cmap = 'gray')
ax4.bar(t,hist_1)
plt.show()

```


