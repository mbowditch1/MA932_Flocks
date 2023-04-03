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
    bpy.ops.mesh.primitive_cone_add(radius1=0.5, location=pos[0])

col = bpy.data.collections[1].objects
for i,x in enumerate(col):
    f = 1
    for j,p in enumerate(all_positions[i]):
        if j >= 1:
            x.rotation_euler.x = p[0] - all_positions[i][j-1][0]
            x.rotation_euler.y = p[1] - all_positions[i][j-1][1]
            x.rotation_euler.z = p[2] - all_positions[i][j-1][2]
            
        x.location = p
        x.keyframe_insert(data_path = "location", frame=f)
        x.keyframe_insert(data_path = "rotation_euler", frame=f)
        f += 5

# For each 'bird' create an object in collection

# Loop through objects in collection and set new locations
# Done
