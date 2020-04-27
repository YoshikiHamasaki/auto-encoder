import numpy as np
import matplotlib.pyplot as plt

class Dog:

    def bark(self):
        print("woof")
 
sizzles = Dog()
sizzles.bark()

class Dog2:
    def __init__(self,petname,temp):
        self.name = petname;
        self.temperature = temp;

    def status(self):
        print("dog name is",self.name)
        print("dog temprature is",self.temperature)

    def setTemperature(self,iu):
        self.temperature = iu;

    def bark(self):
        print("woof")

#lassie = Dog2("michael",23)
#lassie.status()
#lassie.setTemperature(45)
#lassie.status()
#lassie.bark()

print("a","b","c")
a= 50
if a <= 10:
    print("10<")
elif 10 < a <=25:
    print("10<a<=25")
else:
    print("else")

b = 8%3
print("8/3of%={}".format(b))
