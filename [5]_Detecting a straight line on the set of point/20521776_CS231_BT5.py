import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from tqdm import tqdm


n = int(input("Enter the number of random points: "))

point = []
X_arrayPoints, Y_arrayPoints = [], []
x = randint(n, size = int(n*0.7)) # 70%
y = randint(100)*x + randint(100)

x_rand, y_rand = randint(0, n, size = n - int(n*0.7)), randint(0, n, size = n - int(n*0.7)) # 30%

for i, j in zip(x, y):
    point.append((i,j))
for i, j in zip(x_rand, y_rand):
    point.append((i,j))

for p in point:
    X_arrayPoints.append(p[0])
    Y_arrayPoints.append(p[1])
X_arrayPoints = np.array(X_arrayPoints)
Y_arrayPoints = np.array(Y_arrayPoints)

# print(point)
    

a, b= 0, 0
while True:
    while True:
        p1, p2 = point[randint(len(point))], point[randint(len(point))]
        if p1[0] != p2[0] and p1[1] != p2[1]:
            break
    # print(p1, p2)
    
    ## y = a*x + b
    ### y1 = a*x1 + b, y2 = a*x2 + b
    ### a = (y1-y2)/(x1-x2), b = y2 - a*x2 
    a = (p1[1] - p2[1])/(p1[0] - p2[0])
    b = p2[1] - (a*p2[0])

    print(f"Linear equation: y = {round(a,4)}*x + {round(b,4)} ")
    flag = 0
    for i in tqdm(range(len(point))):
        if a * point[i][0] + b == point[i][1]:
            flag+=1
            
    if flag/ len(point) >= 0.7:
        plt.scatter(X_arrayPoints, Y_arrayPoints)
        plt.plot(X_arrayPoints, a*X_arrayPoints+b, marker = 'o', ms = 7, mec = 'b', color = '#4CAF50')
        plt.xlabel("X value")
        plt.ylabel(f"Y = {a}X + {b}")
        plt.show()
        
        print(f"a value = {a}, b value = {b}")
        break
    else:
        print(f"Through {flag} points!")
    





