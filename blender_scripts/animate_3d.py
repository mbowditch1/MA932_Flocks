import bpy

col = bpy.data.collections[1].objects

for x in col:
    x.location = [0,0,0]
    x.keyframe_insert(data_path = "location", frame=1)
    x.location = [10,10,10]
    x.keyframe_insert(data_path = "location", frame=50)
    
# For each 'bird' create an object in collection

# Loop through objects in collection and set new locations
# Done