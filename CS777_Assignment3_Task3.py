import sys
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext

# spark = SparkSession.builder.master("yarn").getOrCreate()
sc = SparkContext.getOrCreate()
# sqlContext = SQLContext(sc)

lines = sc.textFile(sys.argv[1])

taxilines = lines.map(lambda x:x.split(','))

def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

def correct_rows(x):
    if len(x) == 17:
        if is_float(x[5]) and is_float(x[11]):
            amount = float(x[11])
            if float(x[5]) != 0 and amount >= 1 and amount <= 600:
                return x

taxilinesCorrected = taxilines.filter(correct_rows)


#taxi_points = taxilinesCorrected.map(lambda x:(float(x[4]), float(x[5]), float(x[11]), float(x[12])))
taxi_points = taxilinesCorrected.map(lambda x:(np.array([float(x[4]),float(x[5]),float(x[11]),float(x[15]),1]),float(x[16])))

taxi_points.persist()



learningRate = 0.001
num_iteration = 50
theta = np.zeros(5)


lastTimeCost = taxi_points.map(lambda x:x[1]-np.dot(x[0],theta)).map(lambda x:x**2).treeAggregate(0,lambda x,y:np.add(x,y),lambda x,y:np.add(x,y),3)



#beta = [0.1, 0.1]

size = taxi_points.count()

for i in range(num_iteration):
    gradientCost = taxi_points.map(lambda x:(x[0],x[1]-np.dot(x[0],theta))).map(lambda x:np.array([x[0]*x[1],x[1]**2])).treeAggregate(np.zeros(2),lambda x,y:np.add(x,y),lambda x,y:np.add(x,y),3)
    cost = gradientCost[1] / size
    gradient = -2 * gradientCost[0] / size
    if cost < lastTimeCost:
        learningRate *= 1.05
    else:
        learningRate *= 0.5
    print("step {} : sample cost={}, m0={}, m1={}, m2={}, m3={}, b={}, newLearningRate={}".format(i,cost,theta[0],theta[1],theta[2],theta[3],theta[4],learningRate))
    lastTimeCost = cost
    theta = theta - learningRate * gradient
    taxi_points.unpersist()


sc.stop()

