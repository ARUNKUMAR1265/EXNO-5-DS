# EXNO-5-DS-DATA VISUALIZATION USING MATPLOT LIBRARY

# Aim:
  To Perform Data Visualization using matplot python library for the given datas.

# EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# Algorithm:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

# Coding and Output:
from google.colab import drive

drive.mount('/content/drive')

import matplotlib.pyplot as plt

x_values=[0, 1, 2, 3, 4, 5]

y_values=[0, 1, 4, 9, 16, 25]

plt.plot(x_values, y_values)

plt.show()

![Screenshot 2024-12-15 090622](https://github.com/user-attachments/assets/c94e8a4b-44ad-420c-b49d-df2883dff450)

import matplotlib.pyplot as plt

x=[1,2,3]

y=[2,4,1]

plt.plot(x,y)

plt.xlabel('x-axis')

plt.ylabel('y-axis')

plt.title('My first graph!')

plt.show()

![Screenshot 2024-12-15 090634](https://github.com/user-attachments/assets/1ea1d9d6-203e-4bbb-b14f-b59613d9b400)

x1=[1,2,3]

y1=[2,4,1]

plt.plot(x1,y1,label="line 1")

x2=[1,2,3]

y2=[4,1,3]

plt.plot(x2,y2,label="line 2")

plt.xlabel('x-axis')

plt.ylabel('y-axis')

plt.title('Two lines on same graph!')

plt.legend()

plt.show()

![Screenshot 2024-12-15 090644](https://github.com/user-attachments/assets/0b901f6b-5692-4fe7-b2b6-e01f3b62179a)

x=[1,2,3,4,5,6]

y=[2,4,1,5,2,6]

plt.plot(x,y,color='green',linestyle='dashed',linewidth=3,marker='o',markerfacecolor='blue',markersize=12)

plt.ylim(1,8)

plt.xlim(1,8)

plt.xlabel('x-axis')

plt.ylabel('y-axis')

plt.title('Some cool customizations!')

plt.show()

![Screenshot 2024-12-15 090652](https://github.com/user-attachments/assets/566fedb9-070a-4352-8467-b61361858f32)

yield_apples=[0.895,0.91,0.919,0.926,0.929,0.931]

plt.plot(yield_apples)

![Screenshot 2024-12-15 090706](https://github.com/user-attachments/assets/0ab23c36-33c6-49aa-a0c5-f9c1a951b505)


years=[2010,2011,2012,2013,2014,2015]

yields=[0.895,0.91,0.919,0.926,0.929,0.931]

plt.plot(years, yields)

![Screenshot 2024-12-15 090714](https://github.com/user-attachments/assets/a175a65e-49fb-4177-9d92-60ea7cf7e663)

years=range(2000,2012)

apples=[0.895,0.91,0.919,0.926,0.929,0.931,0.934,0.936,0.937,0.9375,0.932,0.939]

oranges=[0.962,0.941,0.930,0.923,0.918,0.908,0.907,0.904,0.901,0.898,0.9,0.896]

plt.plot(years,apples)

plt.plot(years,oranges)

plt.xlabel('years')

plt.ylabel('Yield (tons per hectare)');

![Screenshot 2024-12-15 090730](https://github.com/user-attachments/assets/dff63db9-ab37-44f3-ab30-aaf551d7fdf3)

plt.plot(years,apples)

plt.plot(years,oranges)

plt.xlabel('years')

plt.ylabel('yield (tons per hectare)')

plt.title("crop yields in kanto")

plt.legend(['apples', 'oranges']);

![Screenshot 2024-12-15 090807](https://github.com/user-attachments/assets/9e19855c-b970-4749-b455-b482d13ed327)

years=[2010,2011,2012,2013,2014,2015]

yields=[0.895,0.91,0.919,0.926,0.929,0.931]

plt.plot(years,yields)

plt.xlabel('years')

plt.ylabel('yield (tons per hectare)');

![Screenshot 2024-12-15 090819](https://github.com/user-attachments/assets/efc76ea4-83f8-4994-a666-3dc7a1972b85)

years=range(2000,2012)

oranges=[0.962,0.941,0.930,0.923,0.918,0.908,0.907,0.904,0.901,0.898,0.9,0.896]

plt.figure(figsize=(12,6))

plt.plot(years,oranges,marker='o')

plt.xlabel('years')

plt.ylabel('yield (tons per hectare)')

plt.title("Yields of Oranges (tons per hectare)");

![Screenshot 2024-12-15 090831](https://github.com/user-attachments/assets/ac2a0141-b6a7-4a56-948c-456235a18fc5)

years=range(2000,2012)

apples=[0.895,0.91,0.919,0.926,0.929,0.931,0.934,0.936,0.937,0.9375,0.932,0.939]

oranges=[0.962,0.941,0.930,0.923,0.918,0.908,0.907,0.904,0.901,0.898,0.9,0.896]

plt.plot(years,apples,marker='o')

plt.plot(years,oranges,marker='x')

plt.xlabel('years')

plt.ylabel('yield (tons per hectare)')

plt.title("crops yields in kanto");

plt.legend(['apples','oranges'])

![Screenshot 2024-12-15 090841](https://github.com/user-attachments/assets/5de9a323-859c-4f19-8400-5b8a62b80904)

x_values=[0,1,2,3,4,5]

y_values=[0,1,4,9,16,25]

plt.scatter(x_values,y_values,s=30,color="blue")

plt.show()

![Screenshot 2024-12-15 090852](https://github.com/user-attachments/assets/e6eedc3a-91c5-4acc-af33-4041ce61af33)

x=[1,2,3,4,5,6,7,8,9,10]

y=[2,4,5,7,6,8,9,11,12,12]

plt.scatter(x,y,label="stars",color="green",marker="*",s=30)

plt.xlabel('x-axis')

plt.ylabel('y-axis')

plt.title('My Scatter plot!')

plt.legend()

plt.show()

![Screenshot 2024-12-15 090902](https://github.com/user-attachments/assets/9fc3f216-c974-4de8-beb7-5cd0f32956e5)

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

x=np.arange(0,10)

y=np.arange(11,21)



x

![Screenshot 2024-12-15 090911](https://github.com/user-attachments/assets/0aa6ccf3-8054-4283-8e31-e4b45089a576)

y

![Screenshot 2024-12-15 090919](https://github.com/user-attachments/assets/e64dfe68-ee3e-48f5-b2b6-f04c0e3f03ea)

plt.scatter(x,y,c='r')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Graph in 2D')

plt.savefig('Test.png')

![Screenshot 2024-12-15 090927](https://github.com/user-attachments/assets/e6e2783c-4098-4976-a8d2-f3ccb0f9847a)

y=x*x

y

![Screenshot 2024-12-15 090935](https://github.com/user-attachments/assets/1c399b8b-0941-4b76-ae45-60b07b72a8da)

plt.plot(x,y,'g*',linestyle='dashed',linewidth=2,markersize=12)

plt.title('2D diagram')

plt.ylabel('Y axis')

plt.xlabel('X axis')

![Screenshot 2024-12-15 090943](https://github.com/user-attachments/assets/cb66890a-da2b-4dfb-973e-c1fce3a91495)

plt.subplot(2,2,1)

plt.plot(x,y,'r--')

plt.subplot(2,2,2)

plt.plot(x,y,'g*-')

plt.subplot(2,2,3)

plt.plot(x,y,'bo')

plt.subplot(2,2,4)

plt.plot(x,y,'go')

![Screenshot 2024-12-15 090952](https://github.com/user-attachments/assets/bf1d9e65-5134-4a7a-a66e-fa2d4a6ad286)

np.pi

![Screenshot 2024-12-15 091004](https://github.com/user-attachments/assets/b70f67a2-f0a9-4296-8765-e3a110c39938)

x=np.arange(0,4*np.pi,0.1)

y=np.sin(x)

plt.title("sine wave form")

plt.plot(x,y)

plt.show()

![Screenshot 2024-12-15 091012](https://github.com/user-attachments/assets/c62cc53a-2dfa-4a6b-a358-569a04b347c7)

import matplotlib.pyplot as plt

import numpy as np

x=[1,2,3,4,5]

y1=[10,12,14,16,18]

y2=[5,7,9,11,13]

y3=[2,4,6,8,10]

plt.fill_between(x,y1,color='blue')

plt.fill_between(x,y2,color='green')

plt.plot(x,y1,color='red')

plt.plot(x,y2,color='black')

plt.legend(['y1','y2'])

plt.show()

![Screenshot 2024-12-15 091022](https://github.com/user-attachments/assets/f88cff06-f0a0-48ff-891d-15ce99386cb5)

plt.stackplot(x,y1,y2,y3,labels=['Line 1','Line 2','Line 3'])

plt.legend(loc='upper left')

plt.title('Stacked Line Chart')

plt.xlabel('X-axis')

plt.ylabel('Y-axis')

plt.show()

![Screenshot 2024-12-15 091033](https://github.com/user-attachments/assets/f5dd5276-8b9d-470d-b876-8dd6c7887ce9)

import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline

x=np.array([1,2,3,4,5,6,7,8,9,10])

y=np.array([2,4,5,7,8,8,9,10,11,12])

spl=make_interp_spline(x,y)

x_smooth = np.linspace(x.min(),x.max(),100)

y_smooth=spl(x_smooth)

plt.plot(x,y,'o',label='data')

plt.plot(x_smooth,y_smooth,'-',label='spline')

plt.legend()

plt.show()

![Screenshot 2024-12-15 091042](https://github.com/user-attachments/assets/b787f4fc-317a-466a-9f80-eb6c8ada0674)

import matplotlib.pyplot as plt

values=[5,6,3,7,2]

names=["A","B","C","D","E"]

plt.bar(names,values,color='green')

plt.show()

![Screenshot 2024-12-15 091050](https://github.com/user-attachments/assets/6f1751df-cd21-4735-bb13-8119c434b097)

plt.barh(names,values,color='yellowgreen')

plt.show()

![Screenshot 2024-12-15 091058](https://github.com/user-attachments/assets/859b3e5d-e3ba-4718-a8df-474f7d93ca2a)

height=[10,24,36,40,5]

names=['one','two','three','four','five']

c1=['red','green']

c2=['b','g']

plt.bar(names,height,width=0.8,color=c1)

plt.xlabel('x-axis')

plt.ylabel('y-label')

plt.title('My barchart!')

plt.show()

![Screenshot 2024-12-15 091107](https://github.com/user-attachments/assets/18beef1d-c70c-48ca-9e06-4f1427787561)

x=[2,8,10]

y=[11,16,9]

x2=[3,9,11]

y2=[6,15,7]

plt.bar(x,y,color='r')

plt.bar(x2,y2,color='g')

plt.title('Bar graph')

plt.ylabel('y axis')

plt.xlabel('x axis')

plt.show()

![Screenshot 2024-12-15 091117](https://github.com/user-attachments/assets/35038df0-4519-490c-bd51-b0b7e826b83a)

ages=[2,5,70,40,30,45,50,45,43,44,60,7,13,57,18,90,77,32,21,20,40]

range=(0,100)

bins=10

plt.hist(ages,bins,range,color='green',histtype='bar',rwidth=0.8)

plt.xlabel('age')

plt.ylabel('no. of people')

plt.title('My histogram')

plt.show()

![Screenshot 2024-12-15 091125](https://github.com/user-attachments/assets/7fa177b7-59e3-4270-95b0-a8963ba079fe)

x=[2,1,6,4,2,4,8,9,4,2,4,10,6,4,5,7,7,3,2,7,5,3,5,9,2,1]

plt.hist(x,bins=10,color='blue',alpha=0.5)

plt.show()

![Screenshot 2024-12-15 091138](https://github.com/user-attachments/assets/fe91337e-d607-477a-8bfe-776c5c773c19)

import matplotlib.pyplot as plt

import numpy as np

np.random.normal(loc=0,scale=1,size=100)

data=np.random.normal(loc=0,scale=1,size=100)

data

![Screenshot 2024-12-15 091150](https://github.com/user-attachments/assets/26a04523-f903-42b4-bbbd-75cef67f2d82)

fig,ax=plt.subplots()

ax.boxplot(data)

ax.set_xlabel('Data')

ax.set_ylabel('Value')

ax.set_title('Box Plot')

plt.show()

![Screenshot 2024-12-15 091201](https://github.com/user-attachments/assets/24cb4c47-bfd0-47d7-8fe7-8c09d6753d76)

activities=['eat','sleep','work','play']

slices=[3,7,8,6]

colors=['r','y','g','b']

plt.pie(slices,labels=activities,colors=colors,startangle=90,shadow=True,explode=(0,0,0.1,0),radius=1.2,autopct='%1.1f%%')

plt.legend()

plt.show()

![Screenshot 2024-12-15 091210](https://github.com/user-attachments/assets/c2580dd2-caf9-4425-9663-3c3683936f23)

labels='python','c++','c','java'

sizes=[215,130,245,210]

colors=['gold','yellowgreen','lightcoral','lightskyblue']

explode=(0,0.4,0,0.5)

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)

plt.axis('equal')

plt.show()

![Screenshot 2024-12-15 091221](https://github.com/user-attachments/assets/f5ea279b-9238-4ae6-865c-c13fb0e62276)

activities=['eat','sleep','work','play']

slices=[3,7,8,6]

colors=['r','y','g','b']

plt.pie(slices,labels=activities,colors=colors,startangle=90,shadow=True,explode=(0,0,0.1,0),radius=1.2,autopct='%1.1f%%')

plt.legend()

![Screenshot 2024-12-15 091227](https://github.com/user-attachments/assets/ac581798-820d-44ee-8ec9-99d9a939a058)

# Result:
 The code is run succesfully.
