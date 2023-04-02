import bpy
import csv

positions = []

with open("/home/mbowditch/Documents/MA932/blender_scripts/positions.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    positions.append([float(x) for x in row])

col = bpy.data.collections[1].objects

for x in col:
    f = 1
    for p in positions:
        x.location = p
        x.keyframe_insert(data_path = "location", frame=f)
        f += 1

# For each 'bird' create an object in collection

# Loop through objects in collection and set new locations
# Done
