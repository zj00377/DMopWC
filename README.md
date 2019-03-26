# MopWC

## Running Code

We implemented the model as described in the paper based on Tensorflow library, [Tensorflow](https://www.tensorflow.org/).



### Run examples

In this code, you can run our model on test dataset. If you want to use our surface based morphometry features (mTBM), please cnotact me to get the access of the features. 

If you execute dmopwc.py, you can reproduce our model.  

```
python dmopwc.py --max_iter 1200 --text_iter 200 --label MMSE --feature sparse --mode lifelong
```
