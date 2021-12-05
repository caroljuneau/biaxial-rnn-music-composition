# Biaxial Recurrent Neural Network for Music Composition

Project for UofSC CSCE 585: Machine Learning Systems

Carol Juneau, Joseph Cammarata, Brody Norton

The code for this project has largely been adapted from [Daniel Johnson's project](https://github.com/danieldjohnson/biaxial-rnn-music-composition).

We have updated the instructions from the original project.

## Generating Instructions
This requires that you have conda and GCC. Unlike training it does not require a GPU.

First, clone the github repository to your machine.

Then create and activate a conda environment that uses python=2.7. There is a yml file included in the repo with all the requirements.

There are midi files of different genres located in the "music" directory. Each zipped directory has midi files for a particular genre. Start by unzipping "classical.zip".

You will also need to go into the "output" directory and unzip a weights file. Start by unzipping "classical1.zip". You will notice that it contains several other files, but the one to notice is "params_final.p" -- this file contains the trained weights that will be used in the model.

In a terminal, make sure you're in the "biaxial-rnn-music-composition" directory and run python. Then enter the following.
```python
import model
import main
import multi_training
import cPickle as pickle
```
Load in your pieces:
```python
pcs = multi_training.loadPieces("music/classical")
```
Construct the model; this will take some time:
```python
m = model.Model([300,300],[100,50],dropout=0.5)
```
Load in the weights:
```python
m.learned_config = pickle.load("open("output/classical1/params_final.p","rb"))
```
If you are using a CPU rather than a GPU, run this line: 
```python
theano.config.experimental.unpickle_gpu_on_cpu = True
```
Generate a composition:
```python
main.gen_adaptive(m,pcs,1,name="composition")
```
Alternatively, you can use a for-loop to generate several pieces at a time:
```python
for i in range(10):
    main.gen_adaptive(m,pcs,1,name="composition"+str(i))
```


## Training Instructions

This requires that you have conda and GCC and that you are using a GPU with CUDA 9.0.

Note that this process will take a long time. For me it takes around 7.5 hours in total. 

Activate the conda environment described in the previous section. 

Again, you will need some midi files. You can use the same ones in "music/classical" as before, or another genre. 

In a terminal, cd into the "biaxial-rnn-music-composition" directory. Use the following flags to run python from your terminal:
```
THEANO_FLAGS="device=cuda,floatX=float32" python
```
```python
import model
import main
import multi_training
import cPickle as pickle
```
Load in your pieces:
```python
pcs = multi_training.loadPieces("music/classical")
```
Construct the model; this will take some time:
```python
m = model.Model([300,300],[100,50],dropout=0.5)
```
Train the model; this step takes the most time:
```python
multi_training.trainPiece(m,pcs,10000,"output")
```
Finally save the final trained weights:
```python
pickle.dump(m.learned_config, open("output/params_final.p","wb"))
```
You can also generate pieces as in the previous section:
```python
main.gen_adaptive(m,pcs,1,name="composition")
```

## Notes

I used the university's RCI computer to run the training for this project.

Those scripts are included in this repo -- training.sh and training_script.py. I cannot guarantee that these will work for your machine. 

There is also a script called using_gpu_test.py which is taken from the Theano documentation. This tests whether Theano is correctly running on the GPU or if it is running on the CPU. 
