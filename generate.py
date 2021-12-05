import model
import main
import multi_training
import cPickle as pickle
import theano
import time
import sys

input_dir = sys.argv[1]
weights_file = sys.argv[2]
comp_name = sys.argv[3]

theano.config.experimental.unpickle_gpu_on_cpu=True

print("Using " + input_dir + " and " + weights_file)

timeA = time.time()
m = model.Model([300,300],[100,50],dropout=0.5)
timeB = time.time()
print("Time to construct model: " + str(timeB - timeA) + "s")

timeA = time.time()
pcs = multi_training.loadPieces(input_dir)
timeB = time.time()
print("Time to load pieces: " + str(timeB - timeA) + "s")

timeA = time.time()
m.learned_config = pickle.load(open(weights_file,"rb"))
for i in range(50):
    main.gen_adaptive(m,pcs,1,name=comp_name+str(i))
timeB = time.time()
print("Time to generate 50 new pieces: " + str(timeB - timeA) + "s")
