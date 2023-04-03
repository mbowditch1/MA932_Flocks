import bpy
import csv

curr_positions = []
all_positions = []
with open("/home/mbowditch/Documents/MA932/blender_scripts/positions.csv", 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        if row[0] != "n":
            curr_positions.append([float(x) for x in row])
        else:
            all_positions.append(curr_positions)
            curr_positions = []


for pos in all_positions:
    f = 1
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.5, location=pos[0])

col = bpy.data.collections[1].objects
for i,x in enumerate(col):
    f = 1
    for p in all_positions[i]:
        x.location = p
        x.keyframe_insert(data_path = "location", frame=f)
        f += 5

# For each 'bird' create an object in collection

# Loop through objects in collection and set new locations
# Done
