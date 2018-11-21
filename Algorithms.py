import numpy as np
import math as m
import random
import Functions as f
import time
import PlotFunctions as myPlot
import Helper as h


#       hillclimb(prieaveragemer)       annealing(average)
# f1    0.0011081368048272478           0.016460038186587873
# f2    0.28496469037391975             1.7615023764806759
# f3    0.012414146683253805            5.276187574210139
# f4    1.9785660576933395              16.713162055571043
# f5    809.9377074209818               691.8634345551397
def HillClimbAndAnnealingComparison(func):
    """
    30 times run function 'func' for HillClimb and Annealing algorithm with counter for fitness calculation ( count of fitness calculation should be same )
    Allways take the best value and at make average from these 30 best values
    
    Methode print average for HillClimb and Annealing algorithm

    @param: func Function for calculating fitness 
    """
    res = 0
    res2 = 0
    iterations = 0
    for i in range(30):
        tmp, iterations = AnnealingAlgorithm(-2,1,func)
        #print(iterations)
        res2 += tmp
        res += HillClimb(iterations,-2,1,func)

    print("HillClimb algorithm:",res/30)
    print("Annealing algorithm:",res2/30)


def HillClimb(iterations,x,y,func):

    field = h.GenerateField(x,y)
    func.CalculateField(field)
    init_w = f.Walker((x,y),0)
    init_w.z = func.CalculateVector(init_w.coordinates)
    # ax = myPlot.PlotShow(func)
    # myPlot.AddScatter(ax,init_w.coordinates[0],init_w.coordinates[1],init_w.z,"k")

    for i in range(iterations):
        w = h.FindMinimum(field,init_w)
        # myPlot.AddScatter(ax,init_w.coordinates[0],init_w.coordinates[1],init_w.z,"r")
        field = h.GenerateField(w.coordinates[0],w.coordinates[1])
        func.CalculateField(field)
        init_w = w

    return init_w.z
    # myPlot.Show()

def AnnealingAlgorithm(x,y,func):

    temperature = 5000
    final_temperature = 0.00001
    alpha = 0.99
    init_w = f.Walker((x,y),0)
    init_w.z = func.CalculateVector(init_w.coordinates)

    # ax = myPlot.PlotShow(func)
    # myPlot.AddScatter(ax,init_w.coordinates[0],init_w.coordinates[1],init_w.z,"b")

    iter = 0
    while temperature > final_temperature:
        for i in range(1):
            iter +=1
            neighbour_w = h.GenerateRandomWalker(init_w.coordinates[0],init_w.coordinates[1])
            neighbour_w.z = func.CalculateVector(neighbour_w.coordinates)
            delta = neighbour_w.z - init_w.z
            if delta < 0:
                init_w = neighbour_w
        # myPlot.AddScatter(ax,init_w.coordinates[0],init_w.coordinates[1],init_w.z,"b")
            else:
                r = np.random.uniform(0,1)
                if r < m.exp(-delta/temperature):
                    init_w = neighbour_w
                    # myPlot.AddScatter(ax,init_w.coordinates[0],init_w.coordinates[1],init_w.z,"b")


        temperature = alpha * temperature
    # myPlot.AddScatter(ax,init_w.coordinates[0],init_w.coordinates[1],init_w.z,"r")
    # myPlot.Show()
    return init_w.z, iter

def BlindAlgorithm(iterations,range_x,range_y,func):

    ax = myPlot.PlotShow(func)

    fitness = None
    last_w = None
    x, y = 0, 0
    for i in range(iterations):
        x = random.uniform(range_x[0], range_x[1])
        y = random.uniform(range_y[0], range_y[1])  
        fit = func.CalculateVector((x,y))
        if fitness == None or fit < fitness:
            fitness = fit
            last_w = f.Walker((x,y),fit)
        myPlot.AddScatter(ax,x,y,fit,"r")

    myPlot.AddScatter(ax,last_w.coordinates[0],last_w.coordinates[1],last_w.z,"r")
    myPlot.Show()


def SomaAlgorithm(func):
    pathLength = 3
    step = 0.11
    ptrValue = 0.1

    ax = myPlot.PlotShow(func)

    field = h.GenerateRandomUniformField(func.Min,func.Max,20)
    func.CalculateField(field)
    leader = h.FindMinimum2(field)

    myPlot.AddScatters(ax,field,"r")
    myPlot.AddScatter(ax,leader.coordinates[0],leader.coordinates[1],leader.z,"yellow")

    for s in range(10):
        for k in range(len(field)):
            
            jumps = []
            for i in range(0,int(pathLength/step),1):
                ptr = h.GetPerturbationVector(2,ptrValue)
                w_x = field[k].coordinates[0] + (leader.coordinates[0] - field[k].coordinates[0]) * (i*  step)*ptr[0]
                w_y = field[k].coordinates[1] + (leader.coordinates[1] - field[k].coordinates[1]) * (i*step)*ptr[1]
                w = f.Walker((w_x,w_y),0)
                w.z = func.CalculateVector(w.coordinates)
                jumps.append(w)
            
            new_pos = h.FindMinimum2(jumps)
            field[k] = new_pos
        
        ax = myPlot.PlotShowAnimated(ax,func)
        myPlot.AddScatters(ax,field,"r")

        leader = h.FindMinimum2(field)

        myPlot.AddScatter(ax,leader.coordinates[0],leader.coordinates[1],leader.z,"yellow")
        myPlot.PlotPause(1.0)
    myPlot.Show()


def ParticalSwarnAlgorithm(func):

    iterations = 20
    particles = 10;
    c1 = 2
    c2 = 2
    weightStart = 0.9
    wightEnd = 0.4

    ax = myPlot.PlotShow(func)

    field = h.GenerateRandomUniformField(func.Min,func.Max,particles)    
    func.CalculateField(field)
    gBest = h.FindMinimum2(field)

    myPlot.AddScatters(ax,field,"black")

    initialVelocity = (abs(func.Min) + abs(func.Max)) / 20
    for i in range(particles):
        field[i].rand1 = np.random.uniform(0,1,1)
        field[i].rand2 = np.random.uniform(0,1,1)
        field[i].velocity = np.tile(initialVelocity,len(field[0].coordinates))

    for i in range(iterations):
        for j in range(particles):
            newVelocityVector = []
            newCoordinatesVector = []
            weight = weightStart - ((weightStart - wightEnd)*i)/iterations
            for k in range(len(field[j].coordinates)):
                vel = 0.0
                if (field[j].pBest == None):
                    field[j].pBest = field[j]

                vel = weight * field[j].velocity[k] + c1 * field[j].rand1 * (field[j].pBest.coordinates[k] - field[j].coordinates[k]) + c2 * field[j].rand2 * (gBest.coordinates[k] - field[j].coordinates[k]) 
                
                if (vel > initialVelocity):
                    vel = initialVelocity
                newVelocityVector.append(vel)
                pos = field[j].coordinates[k] + vel
                newCoordinatesVector.append(pos)
                           
            field[j].coordinates = newCoordinatesVector
            field[j].velocity = newVelocityVector

            field[j].z = func.CalculateVector(field[j].coordinates)
            if (field[j].z < field[j].pBest.z):                
                field[j].pBest = field[j]
            if (field[j].pBest.z < gBest.z):
                gBest = field[j].pBest

        myPlot.PlotPause(0.5)
        ax = myPlot.PlotShowAnimated(ax,func)

        myPlot.AddScatters(ax,field,"r")
        myPlot.AddScatter(ax,gBest.coordinates[0],gBest.coordinates[1],gBest.z,"yellow")
    myPlot.Show()

def DifferentialEvolutionAlgorithm(func):

    generations = 20
    threshold = 0.7
    mutation = 0.8
    dim = 2;
    field = h.GenerateRandomUniformField(func.Min,func.Max,10*dim)    
    func.CalculateField(field)

    ax = myPlot.PlotShow(func)
    myPlot.AddScatters(ax,field,"black")

    for i in range(generations):
        newPopulation = []
        for item in range(len(field)):

            rand_walkers = h.SelectNRandomWalkers(field,3,item)
            final_vector = []
            for coor in range(len(rand_walkers[0].coordinates)):
                differential_value = (rand_walkers[0].coordinates[coor] - rand_walkers[1].coordinates[coor]) * mutation
                noisy_val = differential_value + rand_walkers[2].coordinates[coor]

                cr_param = np.random.uniform(0,1,1)
                if cr_param < threshold:
                    final_vector.append(field[item].coordinates[coor])
                else:
                    final_vector.append(noisy_val)                    
            
            z = func.CalculateVector(final_vector)
            if z < field[item].z:
                newPopulation.append(f.Walker(final_vector,z))
            else:
                newPopulation.append(f.Walker(field[item].coordinates,field[item].z))

        field = newPopulation
        myPlot.PlotPause(0.5)
        ax = myPlot.PlotShowAnimated(ax,func)

        myPlot.AddScatters(ax,field,"r")
    myPlot.Show()


# + current to pBest strategy


# jeden jedinec je postupnost miest a fitness je dlzka vsetkych vzdialenosti
# x jedincov
# k jedincovi vyberiem nahodne rodica s ktorym sa bude krizit
# 2 bodove krizenie (nahodne)
# okraje z prveho potomka a stred doskladam z druheho tak aby som zachoval postupnost 
# (idem od zaciatku a do vyslednej postupnosti pridam mesto ak ho vysledna postupnost este neobsahuje )
# mutaciu robim tak ze nahodne vyberiem dve mesta a prehodim ich 
# skusit kolko krat to treba spustit
def TravelerSalesManGA():

    indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    cities = h.GetListOfCities();
    lengthMatrix = h.CalculateEuclideanDistance(cities)
    population = h.GenerateRandomUnsortedPopulation(cities,20,lengthMatrix)
    
    bestWay = h.FindBestWay(population)
    C_param = 0.9
    M_param = 0.08
    ax = myPlot.PlotTravelerPoints(bestWay)

    for iter in range(500):
        newPopulation = []
        for pop in range(len(population)):
            random_crossover = random.uniform(0,1)
            random_mutation = random.uniform(0,1)
            newTraveler = None
            actualTraveler = population[pop]
            if random_crossover < C_param:
                parent =random.sample(population,1)[0] 
                while actualTraveler.Id == parent.Id:
                    parent =random.sample(population,1)[0]         
                # x1,x2 = random.sample(indexes,2)
                # tempLeft = actualTraveler.ListOfCities[:x1]
                # tempRight = actualTraveler.ListOfCities[x2:]
                # tempFinal = tempLeft + tempRight
                # for i in range(len(parent.ListOfCities)):
                #     if not parent.ListOfCities[i] in tempFinal:
                #         tempLeft.append(parent.ListOfCities[i])
                
                # temp = tempLeft + tempRight
                # newTraveler = f.Traveler(temp,lengthMatrix,actualTraveler.Id)
                newTraveler = f.Traveler(h.Crossover(parent,actualTraveler),lengthMatrix,actualTraveler.Id)
            else:
                newTraveler = actualTraveler

            if random_mutation < M_param:
                x1,x2 = random.sample(indexes,2)
                newTraveler.ListOfCities[x1],newTraveler.ListOfCities[x2] = newTraveler.ListOfCities[x2],newTraveler.ListOfCities[x1] 

            newTraveler.Length=newTraveler.CalculateLength(lengthMatrix)
            actualTraveler.Length = actualTraveler.CalculateLength(lengthMatrix)
            if newTraveler.Length < actualTraveler.Length:
                newPopulation.append(newTraveler)
            else:
                newPopulation.append(actualTraveler)
            

        newBest = h.FindBestWay(newPopulation)
        newBest.Length = newBest.CalculateLength(lengthMatrix)
        bestWay.Length = bestWay.CalculateLength(lengthMatrix)
        if newBest.Length < bestWay.Length:
            bestWay = f.Traveler(newBest.ListOfCities,lengthMatrix,newBest.Id)
            bestWay.Length = bestWay.CalculateLength(lengthMatrix)
        population = []
        population = sorted(
            newPopulation, key=lambda x: x.Length, reverse=False)
        myPlot.PlotPause(.01)
        myPlot.PlotTravelerPointsAnimated(ax,bestWay,iter)

    myPlot.Show()

# HillClimb(-2,1,f.SphereFunction(-2,2,0.1))
# BlindAlgorithm(20,(-2,2),(-2,2),f.SphereFunction(-2,2,0.1))
# AnnealingAlgorithm(-2,1,f.SchwefelFunction(-500,500,1))
# SomaAlgorithm(f.SphereFunction(-2,2,0.1))
# ParticalSwarnAlgorithm(f.RosenbrockFunction(-2,3,0.1))
# DifferentialEvolutionAlgorithm(f.SphereFunction(-2,2,0.1))

TravelerSalesManGA()












    

    

            

                
            

