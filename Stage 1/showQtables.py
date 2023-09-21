import json
import numpy as np

with open("Qtable.json","r") as f:
    Qtable=json.load(f)
    
    
def printJson(Qtable):
    
    for i in Qtable:
        print("'"+str(i)+"' : \n"+ str(np.array(Qtable[i]))+"\n")
    
printJson(Qtable)
    
    