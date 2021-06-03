with open("velocity_sac.csv", "a+") as f:
    for line in f:
        split = line.split(',')
        f.write(split[0] + ',' + split[2])