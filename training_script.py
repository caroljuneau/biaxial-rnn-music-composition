# Carol Juneau

import sys 
import model
import main
import multi_training
import cPickle as pickle
import theano
import time

input_dir = sys.argv[1]
output_dir = sys.argv[2]
print ("input dir: " + input_dir)
print("output dir: " + output_dir)

startTime = time.time()
print("using device: " + theano.config.device)

pcs = multi_training.loadPieces(input_dir)
print("finished loading pieces")

m = model.Model([300,300],[100,50],dropout=0.5)
print("constructed m")

multi_training.trainPiece(m,pcs,10001)
print("trained m")

pickle.dump(m.learned_config, open((output_dir+'/params_final.p'), 'wb'))
print("saved model weights")

# main.gen_adaptive(m, pcs, 1, name="composition") # TODO this??
# print("composed piece")

executionTime = (time.time() - startTime)
print("Total training time: " + time.strftime("%Hh %Mm %Ss", time.gmtime(executionTime)))
