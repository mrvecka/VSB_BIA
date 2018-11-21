import numpy as np
import Functions as f
import math as m
import random

def FindMinimum(field,w):    
    for i in range(len(field)):
            if field[i].z < w.z:
                w = field[i]

    return w

def FindMinimum2(field):
    init = field[0]
    for i in range(len(field)):
        if field[i].z < init.z:
            init = field[i]

    return init

def GetWalkerOnCoordinates(field,x,y):
    for i in range(len(field)):
        for j in range(len(field[i])):
            if field[i][j].coordinates[0] == x and field[i][j].coordinates[1] == y:
                return field[i][j]

def GenerateField(w_x,w_y):
    field = []
    mean = [w_x, w_y]
    cov = [[1, 0], [0, 100]]
    x = np.random.multivariate_normal(mean, cov, 10)
    for i in range(len(x)):
        field.append(f.Walker((x[i][0],x[i][1]),0))
    return field

def GenerateRandomUniformField(min,max,size):
    x = np.random.uniform(min,max,size)
    y = np.random.uniform(min,max,size)
    field = []
    for i in range(len(x)):
        field.append(f.Walker((x[i],y[i]),0))
    return field


def GenerateRandomWalker(w_x,w_y):
    mean = [w_x, w_y]
    cov = [[1, 0], [0, 100]]
    x = np.random.multivariate_normal(mean, cov, 1)
    w = f.Walker((x[0][0],x[0][1]),0)    
    return w

def SelectNRandomWalkers(field,count,actualIndex):
    tmp_field = field.copy()
    tmp_field.pop(actualIndex)
    indexes = random.sample(tmp_field,count)
    
    return indexes

def GetListOfCities():
    cities = []

    cities.append(f.City((60,200), 'A',0))
    cities.append(f.City((80,200),'B',1))
    cities.append(f.City((80,180),'C',2))
    cities.append(f.City((140,180),'D',3))
    cities.append(f.City((20,160),'E',4))
    cities.append(f.City((100,160),'F',5))
    cities.append(f.City((200,160),'G',6))
    cities.append(f.City((140,140),'H',7))
    cities.append(f.City((40,120),'I',8))
    cities.append(f.City((100,120),'J',9))
    cities.append(f.City((180,100), 'K',10))
    cities.append(f.City((60,80), 'L',11))
    cities.append(f.City((120,80), 'M',12))
    cities.append(f.City((180,60),'N',13))
    cities.append(f.City((20,40),'O',14))
    cities.append(f.City((100,40),'P',15))
    cities.append(f.City((200,40),'Q',16))
    cities.append(f.City((20,20),'R',17))
    cities.append(f.City((60,20),'S',18))
    cities.append(f.City((160,20),'T',19))

    return cities

def CalculateEuclideanDistance(cities):
    matrix = []
    for i in range(len(cities)):
        arr = []
        for j in range(len(cities)):
            if cities[i].Id == cities[j].Id:
                arr.append(0)
            else:
                sum = 0.0
                for c in range(len(cities[j].Coordinates)):
                    sum += pow(abs(cities[i].Coordinates[c] - cities[j].Coordinates[c]),2)
                sum = m.sqrt(sum)
                arr.append(sum)

        matrix.append(arr)

    return matrix 

def GenerateRandomUnsortedPopulation(cities,count,lengthMatrix):
    population = []
    for i in range(count):
        pop = random.sample(cities, len(cities))
        traveler = f.Traveler(pop,lengthMatrix,i,0)
        population.append(traveler)
    
    return population

def FindBestWay(population):
    best = population[0]
    for i in range(1,len(population)):
        if population[i].Length < best.Length:
            best = population[i]

    return best

def Crossover(chromosome1, chromosome2):
    end = random.randint(0, len(chromosome1.ListOfCities))
    start = random.randint(0, end)
    section = chromosome1.ListOfCities[start:end]
    offspring_genes = list(
        gene if gene not in section else None for gene in chromosome2.ListOfCities)
    g = (x for x in section)
    for i, x in enumerate(offspring_genes):
        if x is None:
            offspring_genes[i] = next(g)
    
    return offspring_genes

def GetPerturbationVector(dim,ptr_value):
    x = np.random.uniform(0,1,dim)
    for i in range(len(x)):
        if (x[i] < ptr_value):
            x[i] = 1
        else:
            x[i] = 0

    return x