#from lib2to3.pgen2.token import LSQB
#from tkinter import Label
from distutils.command.config import config
from torchvision.prototype import datasets
#from torchvision.prototype.datasets._builtin.stanford_cars import StanfordCars
dataset= datasets.load("stanford_cars")
ninja2 =0 
#print(type(dataset))
print(dataset, "jojo")
#print(label)
#print(dat
#print(len(dataset[0]))
for sample in dataset:
    #print(len(sample)
    print(sample,"is a saple00")
    #print(ninja,"hehe")
    #print(c,d,e)
    ninja2+=1
    if ninja2>10:
        break
#    print(sample)  
#    print(label)# this is the content of an item in datapipe returned by _make_datapipe()
#    if ninja>29:
#        break 
#    ninja+=2