import csv

import numpy

def convert_pace(PaceStr):
    m, s = [int(i) for i in PaceStr.split(':')]
    return 60*m + s

def convert_time(TimeStr):
    h, m, s = [int(i) for i in TimeStr.split(':')]
    return 3600*h + 60*m + s

def categorize_age(Age):
    return Age - (Age%5)

def run(Filename):
    ifile  = open(Filename, "rb")
    rows = csv.reader(ifile)
    data = {} 
    
    next(rows)
    for row in rows:

        age = categorize_age(int(row[2]))
        rank = int(row[4])
        timing = convert_time(row[5])
        pace = convert_pace(row[6])

        Id = int(row[0])
        if row[3] == 'M':
            gender = 1
        else:
            gender = 0

        d = (age, rank, timing, pace)
        
        year = int(row[7])

        if Id in data:
            (gender, _, races) = data[Id]
            races[year] = d
            data[Id] = (gender, age, races)
        else:
            races = {}
            races[year] = d
            data[Id] = (gender, age, races)
      
    for attribute, (g, a, val) in data.items():
        data[attribute] = (g, a, len(val), val)

    #x = numpy.zeros(shape=(len(data), 45))
    x = []
    for attribute, value in data.items():
        features = [0 for col in range(45)]
        features[0] = value[0]
        features[1] = value[1]
        features[2] = value[2]

        for year, t in value[3].items():
            if year >= 2003:
                index = (year - 2003) * 3 + 3
                features[index] = t[1]
                index += 1
                features[index] = t[2]
                index += 1
                features[index] = t[3]
        
        x.append(features)

    s = numpy.array(x)
    return s
