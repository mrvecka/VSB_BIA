def Permute(cities, start, results):

    if start >= len(cities):
        results.append(cities[:])
    else:
        for i in range(start, len(cities)):
            cities[i], cities[start] = cities[start], cities[i]
            Permute(cities, start +1, results)
            cities[start], cities[i] = cities[i], cities[start]