import os
for dirname, _, filenames in os.walk('C:/Users/tam/Documents/data/Data Warehouse'):
    for filename in filenames:
        print(filename)

for dirname, _, filenames in os.walk('C:/Users/tam/Downloads/PlantVillage'):
    print(dirname)