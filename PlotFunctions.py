import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def PlotShow(func):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x = y = np.arange(func.Min, func.Max, func.Density)
    X, Y = np.meshgrid(x, y)
    zs = np.array([func.CalculateVector((x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])

    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    return ax

def PlotShowAnimated(ax,func):
    ax.clear()
    x = y = np.arange(func.Min, func.Max, func.Density)
    X, Y = np.meshgrid(x, y)
    zs = np.array([func.CalculateVector((x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])  
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    return ax

def PlotTravelerPoints(bestWay):
    x,y = bestWay.GetXYVectors()
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    last_x = [] 
    last_x.append(x[:1])
    last_x.append(x[len(x)-1:])

    last_y = []
    last_y.append(y[:1])
    last_y.append(y[len(y)-1:])

    for i, txt in enumerate(bestWay.ListOfCities):
        ax.annotate(txt.Mark, (x[i], y[i]))
    ax.annotate(bestWay.Length,(250,250))
    plt.plot(x, y, '-ok')
    plt.plot(last_x,last_y,'-ok')
    return ax

def PlotTravelerPointsAnimated(ax,bestWay,iter):
    ax.clear()
    x,y = bestWay.GetXYVectors()
    ax.scatter(x, y)
    last_x = [] 
    last_x.append(x[:1])
    last_x.append(x[len(x)-1:])

    last_y = []
    last_y.append(y[:1])
    last_y.append(y[len(y)-1:])

    for i, txt in enumerate(bestWay.ListOfCities):
        ax.annotate(txt.Mark, (x[i], y[i]))

    info = iter 
    ax.annotate(str(iter) + ': ' + str(bestWay.Length),(170,200))
    plt.plot(x, y, '-ok')
    plt.plot(last_x,last_y,'-ok')
    return ax

def PlotAntPathStart(cities):

    x = []
    y = []
    for i in range(len(cities)):
        x.append(cities[i].Coordinates[0])
        y.append(cities[i].Coordinates[1])

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    
    ax.annotate('just scatters',(150,250))

    return ax

def PlotAntPath(ax,x,y):
    ax.clear()
    ax.scatter(x, y)
    ax.scatter(x[0],y[0],color='yellow')
    
    ax.annotate('desc',(250,250))
    plt.plot(x, y, '-ok')
    return ax

def AddScatter(ax,x,y,z,col,sDef=20):
    ax.scatter(x,y,z,color=col,s=sDef)

def AddScatters(ax,field,col,sDef=20):
    for u in range(len(field)):
        ax.scatter(field[u].coordinates[0],field[u].coordinates[1],field[u].z-0.2,color=col,s=sDef)

def Show():
    plt.show()

def PlotClear():
    plt.clf()

def PlotPause(time):
    plt.pause(time)

