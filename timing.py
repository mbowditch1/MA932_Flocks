import FlockModel
import time

# Time running model 
st = time.time()
model = FlockModel.Model(dt = 0.1, maxtime = 100, noise = 0.1, phenotype = [0,1,0], density=1)
model.run()
et = time.time()

elapsed = et - st
print("Execution time:", elapsed, "seconds")
