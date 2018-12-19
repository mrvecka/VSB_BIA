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

def GetRandomBestWalkerForDE(field,actualIndex):
    indexes = SelectNRandomWalkers(field,5,actualIndex)
    best = FindMinimum2(indexes)

    return best


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

def GetCitiesIndexes(cities):
    indexes = []
    for i in range(len(cities)):
        indexes.append(cities[i].Id)

    return indexes


def CalculateEuclideanDistance(cities):
    matrix = []
    for i in range(len(cities)):
        arr = []
        for j in range(len(cities)):
            if cities[i].Id == cities[j].Id:
                arr.append(np.inf)
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

def Crossover(chromosome1, chromosome2, lengthMatrix,indexes):
    # end = random.randint(0, len(chromosome1.ListOfCities))
    # start = random.randint(0, end)
    # section = chromosome1.ListOfCities[start:end]
    # offspring_genes = list(
    #     gene if gene not in section else None for gene in chromosome2.ListOfCities)
    # g = (x for x in section)
    # for i, x in enumerate(offspring_genes):
    #     if x is None:
    #         offspring_genes[i] = next(g)

    x1,x2 = random.sample(indexes,2)
    tempLeft = chromosome2.ListOfCities[:x1]
    tempRight = chromosome2.ListOfCities[x2:]
    tempFinal = tempLeft + tempRight
    for i in range(len(chromosome1.ListOfCities)):
        if not chromosome1.ListOfCities[i] in tempFinal:
            tempLeft.append(chromosome1.ListOfCities[i])
    
    temp = tempLeft + tempRight
    newTraveler = f.Traveler(temp,lengthMatrix,chromosome2.Id)
    
    return newTraveler

def GetPerturbationVector(dim,ptr_value):
    x = np.random.uniform(0,1,dim)
    for i in range(len(x)):
        if (x[i] < ptr_value):
            x[i] = 1
        else:
            x[i] = 0

    return x

def generate_paths(start,count_of_ants,distances,pheromone):
    all_paths = []
    for i in range(count_of_ants):
        path = gen_path(start,distances,pheromone)
        all_paths.append((path, gen_path_dist(path,distances)))
    return all_paths

def gen_path_dist(path,distances):
    total_dist = 0
    for ele in path:
        total_dist += distances[ele]
    return total_dist

def gen_path(start,distances,pheromone):
    path = []
    visited = set()
    visited.add(start)
    prev = start
    for i in range(len(distances) - 1):
        move = pick_move(pheromone[prev], distances[prev], visited, distances)
        path.append((prev, move))
        prev = move
        visited.add(move)
    path.append((prev, start)) # going back to where we started    
    return path

def pick_move(pheromone, dist, visited, distances):
    pheromone = np.copy(pheromone)
    pheromone[list(visited)] = 0

    row = pheromone ** 1* (( 1.0 / dist) ** 1)

    norm_row = row / row.sum()
    move = np.random.choice(len(distances), 1, p=norm_row)[0]
    return move

def spread_pheromone(paths, n_best, pheromone, distances):
    path = paths[0]
    total_sum = 0
    for i in path:
        total_sum += pheromone[i]
    for move in path:
        pheromone[move] += 1.0 / distances[move]
        #pheromone[move] += (1 - 0.6) * pheromone[move] + total_sum


def GetXYOfAntPath(path,cities):
    x = []
    y = []

    for i in range(len(path[0])):
        start_city_index = path[0][i][0]
        end_city_index = path[0][i][1]
        start_city = cities[start_city_index]
        end_city = cities[end_city_index]

        x.append(start_city.Coordinates[0])
        y.append(start_city.Coordinates[1])
        x.append(end_city.Coordinates[0])
        y.append(end_city.Coordinates[1])

    return x,y

def CalcuateEuclideanDistance(vec1,vec2):    
    sum =0
    for i in range(len(vec1)):
        sum += pow(abs(vec2[i] - vec1[i]),2)
    
    sum = m.sqrt(sum)
    return sum

def GetMultiObjectOptimalizationPopulation(min, max, size):
    population = np.random.randint(min,max,size)
    pop = []
    # for i in range(size):
    #     x = -population[i] **2
    #     y =  -(population[i]-2)**2
    #     pop.append((x,y))

    pop.append((-4,-16))
    pop.append((-1,-9))
    pop.append((0,-4))
    pop.append((-4,0))
    pop.append((-16,-4))
    pop.append((-1,-1))

    return pop

def Contains(pop,item):
    contains = False
    for i in range(len(pop)):
        if isinstance(pop[i],list):
            for j in range(len(pop[i])):
                if pop[i][j] == item:
                    contains = True
        else:
            if pop[i] == item:
                contains = True
                
    return contains

def CreateMultiObjectiveRanks(pop,size):
    np_res = np.zeros(size)  
    sp_temp = []
    sp_res = []
    for j in range(len(pop)):
        for i in range(len(pop)):
            if j != i:
                if pop[i][0] >= pop[j][0] and pop[i][1] >= pop[j][1]:
                    np_res[j] = np_res[j] +1 #np
                if pop[i][0] <= pop[j][0] and pop[i][1] <= pop[j][1]:
                    sp_temp.append(i) #sp
        sp_res.append(sp_temp)
        sp_temp = []

    q_res = []
    for m in range(len(np_res)):
        q = []
        for i in range(len(np_res)):
            if np_res[i] == 0 and not Contains(q_res,i):
                q.append(i)
        if len(q) == 0:
            break
        q_res.append(q)
        for i in range(len(q)):
            tmp = sp_res[q[i]]
            for j in range(len(tmp)):
                np_res[tmp[j]] = np_res[tmp[j]] -1 

    return q_res 

def CreateNewMultiObjectivePopulation(old_pop,delta):

    new_pop = []
    for i in range(len(old_pop)):
        parents = random.sample(old_pop,2)
        x = (parents[0][0] + parents[1][0]) / 2 
        y = (parents[0][1] + parents[1][1]) / 2

        x += delta
        y += delta
        new_pop.append((x,y))
    
    return new_pop

def ReduceFullPopulationMO(pop,size):
    sorted_pop = []
    sorted_pop.append(pop[0])
    for i in range(1,len(pop)):
        actual = pop[i]

        for j in range(len(sorted_pop)):
            sort = sorted_pop[j]

            if actual[0] >= sort[0] and actual[1] >= sort[1]:
                sorted_pop.insert(j,actual)
                break
            elif j == len(sorted_pop) -1:
                sorted_pop.insert(j+1,actual)
                break
    
    return sorted_pop[:size]


     

    

