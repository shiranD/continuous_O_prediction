<p>This project is aimed at continuous representation learning<br>
With python version 3.6.4 install the following pkgs<br><p>

This projects has the code for training and inference of continuous models, including simple regression with LSTM, and a GAN based version, and a categorical version copared with in the paper. At inference time it has a focus on long-tail terms and measures the successfulness of correctly prediction infrequenct types, and tokens.

To use this repo:

`pip install -r requirements.txt`

The input to training.sh is assuming the follwoing tree for training data:


```
data
  ├── pos_pb_big (dataset name)
        └── set_0
             ├── train (a sentence per line)
             ├── test
             ├── valid
             ├── train_pos (optional, each line corresponding POSs)
             ├── test_pos
             └── valid_pos
```
if you go to [pubmed_dataset_prep](https://github.com/shiranD/pubmed_dataset_prep) you can build an appropriate training set for pubmed, or apply a few of the commands for a different dataset. 
```
embeddings
  ├── embedding method_dimension_dataset name ('w2v_50_pos_nyt_big')
      (a line from an embedding file: 'the' <float> <float> ...)
      (don't forget to normalize vectors)
```
To generate the embeddings of your like, 


  
 
  
