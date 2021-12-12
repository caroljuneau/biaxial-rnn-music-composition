# Biaxial Recurrent Neural Network for Music Composition

Project for UofSC CSCE 585: Machine Learning Systems

Carol Juneau, Joseph Cammarata, Brody Norton

The code for this project has largely been adapted from [Daniel Johnson's project](https://github.com/danieldjohnson/biaxial-rnn-music-composition).

We have updated the instructions from the original project.

Video Presentation Link: https://youtu.be/lCkIf_xvrh4

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

## Guide to Files in this Repository

Original code from Daniel Johnson's project:
- LICENSE.txt
- data.py
- main.py
- midi_to_statematrix.py
- model.py
- multi_training.py (slightly modified)
- out_to_in_op.py
- visualize.py

Our scripts, which were run on the university's HPC research computers:
- generate.py
- generating.sh
- training_script.py
- training.sh

"music" directory:
- zip files containing midi files for each genre, used for training and generating

"output" directory:
- note: versions 1,2,3 of each genre were trained separately but using the same datasets
- composition0.mid through composition 49.mid, generated using the final weights
- gen#.err, gen#.out, job#.err, job#.out, logs
- params_final.p, the final training weights
- sample0.mid through sample10000.mid, samples generated every 500 iterations during training
- note: johnson_weights does not have logs or samples since we only had the weights provided by Johnson

"johnson" directory:
- zip files containing the compositions generated using Johnson's training weights for each genre

Miscellaneous:
- readme.md
- .gitignore
- midi-env.yml, the conda environment to correctly set up everything
- visualize.py, taken from Theano documentation to help determine if Theano is correctly using the GPU or if it is defaulting to the CPU
- 585 Project Presentation.pdf, the presentation slides
- audio_samples, a guide and youtube link to our samples used in the surveys
- metadata_spreadsheet.xlsx, a spreadsheet with metadata from the logs created during training
- results_spreadhseet.xlsx, a spreadsheet with the results from our surveys
