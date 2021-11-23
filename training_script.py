# Carol Juneau

import sys 
input_dir = sys.argv[0]
output_dir = sys.argv[1]

print ("input dir: " + input_dir)
print("output dir: " + output_dir)




# import model
# import main
# import multi_training
# import cPickle as pickle
# import theano
# import time

# startTime = time.time()
# print("using device: " + theano.config.device)

# pcs = multi_training.loadPieces("music/gulfcoastcountry") #TODO variable for dataset name
# print("finished loading pieces")

# m = model.Model([300,300],[100,50],dropout=0.5)
# print("constructed m")

# multi_training.trainPiece(m,pcs,10001)
# print("trained m")

# pickle.dump(m.learned_config, open('output/params_final.p', 'wb')) # TODO path
# print("saved model weights")

# # main.gen_adaptive(m, pcs, 1, name="composition") # TODO this??
# # print("composed piece")

# executionTime = (time.time() - startTime)
# print("Total training time: " + time.strftime("%Hh %Mm %Ss", time.gmtime(executionTime)))