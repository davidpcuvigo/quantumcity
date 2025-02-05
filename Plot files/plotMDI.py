import matplotlib.pyplot as plt
import numpy as np

"""This scripts produces a plot of the number of successfull MDI-QKD rounds per second for a ballon link and a fiber link
 using the txt outputs from MDI.py"""

file1 = open("MDIfree.txt","r")
L = file1.read().splitlines()
key1 =[]
for i in L:
    key1.append(float(i))
file1.close()

file2 = open("MDIfiber.txt","r")
L = file2.read().splitlines()
key2 =[]
for i in L:
    key2.append(float(i))
file2.close()



dist_cities = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
plt.figure(figsize=(15,9)) 
plt.semilogy(dist_cities,key1[0:15],label="With balloon ",marker = '*', markersize=15, color='m')
plt.semilogy(dist_cities,key2[0:15],label="With fiber links",marker = '+', markersize=15, color='y')

plt.xlabel('$d_{\\rm cities}$ (km) ',size=30)
plt.ylabel('Successfull MDI round per second',size=30)
plt.legend(loc='best',prop={'size':24})
plt.tick_params(axis='both', labelsize=25)
plt.savefig("MDItest.pdf", format = 'pdf')
plt.show()