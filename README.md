# Deep Multi-order Preserving Weight Consolidation (DMopWC)

Alzheimer's disease (AD) is the most common type of dementia. Identifying biomarkers that can track AD at early stages is crucial for therapy to be successful. Many researchers have developed models to predict cognitive impairments by employing valuable longitudinal imaging information along the progression of the disease. However, previous methods model the problem either in the isolated single-task mode or multi-task batch mode, which ignores the fact that the longitudinal data always arrive in a continuous time sequence and, in reality, there are rich types of longitudinal data to apply our learned model to. To this end, we continually model the AD progression in time sequence via a proposed novel Deep Multi-order Preserving Weight Consolidation (DMopWC) to simultaneously 1) discover the inter and inner relations among different cognitive measures at different time points and utilize such relations to enhance the learning of associations between imaging features and clinical scores; 2) continually learn new longitudinal patients' images to overcome forgetting the previously learned knowledge without access to the old data. Moreover, inspired by recent breakthroughs of Recurrent Neural Network, we consider time-order knowledge to further reinforce the statistical power of DMopWC and ensure features at a particular time will be temporally ahead of the features at its subsequential times. Empirical studies on the longitudinal brain image dataset demonstrate that DMopWC achieves superior performance over other AD prognosis algorithms.


## Running Code

We implemented the model as described in the paper based on Tensorflow library, [Tensorflow](https://www.tensorflow.org/).



### Run examples

In this code, you can run our model on test dataset. If you want to use our surface based morphometry features (mTBM), please cnotact me to get the access of the features. 

If you execute dmopwc.py, you can reproduce our model.  

```
python dmopwc.py --max_iter 1200 --text_iter 200 --label MMSE --feature sparse --mode lifelong

```

### Contacts and Paper

Our work has been accepted by MICCAI 2019. If you use this code as part of any published research, please refer the following paper.

```
@misc{zhang2019deep,
  title={Continually Modeling Alzheimer's Disease Progression via Deep Multi-Order Preserving Weight Consolidation},
  author={Zhang, Jie and Wang, Yalin},
  year={2019},
  publisher={MICCAI}
}
```

If you have any questions feel free to contact me:

Jie (Joena) Zhang           JieZhang.Joena@asu.edu
Yalin Wang                  ylwang@asu.edu
