# Does Long-Term Series Forecasting Need Complex Attention and Extra Long Inputs?


## Introduction
 Transformer-based application on Long-term Time series Forecasting (LTSF) tasks still has two major issues that need to be further investigated: 1) Whether the sparse attention mechanism designed by these methods actually reduce the running time on real devices; 2) Whether these models need extra long input sequences to guarantee their performance? The answers given in this paper are negative. Therefore, to better copy with these two issues, we design a lightweight Period-Attention mechanism (Periodformer). Furthermore, to take full advantage of GPUs for fast hyperparameter optimization (e.g., finding the suitable input length), a Multi-GPU Asynchronous parallel algorithm based on Bayesian Optimization (MABO) is presented. Compared with the state-of-the-art methods, the prediction error of Periodformer reduced by 13% and 26% for multivariate and univariate forecasting, respectively. In addition, MABO reduces the average search time by 46% while finding better hyperparameters. As a conclusion, this paper indicates that LTSF may not need complex attention and extra long input sequences.

<img src="https://github.com/liangdaojun/Periodformer/blob/main/Images/performance.jpg">


## Contributions

 - It is found that although the computational complexity of those traditional Transformer-based LTSF methods is theoretically reduced, their running time on practical devices remains unchanged. Meanwhile, it is found that both the input length of the series and the kernel size of the MA have impacts on the final forecast.
 - A novel Period-Attention mechanism (Periodformer) is proposed, which renovates the aggregation of long-term subseries via explicit periodicity and short-term subseries via built-in proximity. In addition, a gate mechanism is built into Period-Attention to adjust the influence of the attention score to its output, which guarantees higher prediction performance and shorter running time on real devices.
 - A multi-GPU asynchronous parallel search algorithm based on Bayesian optimization (MABO) is presented. MABO allocates a process to each GPU via a queue mechanism, and then creates multiple trials at a time for asynchronous parallel search, which greatly accelerates the search speed.
 - Periodformer reduces the prediction error of state-of-the-art (SOTA) methods by around 14.8\% and 22.6\% for multivariate and univariate forecasting, respectively. Besides, MABO reduces the average search time by around 46\% while finding out better hyperparameters.

<img src="https://github.com/liangdaojun/Periodformer/blob/main/Images/period_att.jpg">


## Training Periodformer with MABO
Clone the code repository
```git
git clone git@github.com:Anoise/MHE.git
```

### Training on ETTm2 Dataset
Go to the directory "Periodformer/", and run
```python
python eval_async_mGPUs_PF.py
--num_gpus 8
--num_trails 32
--model Periodformer
--data ETTm2
--root_path ../data/ETT-small/
--data_path ETTm2.csv
--pred_len 192
--enc_in 7
--enc_in 7
--c_out 7
```

### Training on Electricity Dataset
Go to the directory "Periodformer/", and run
```python
python eval_async_mGPUs_PF.py
--num_gpus 8
--num_trails 32
--model Periodformer
--data custom
--root_path ../data/electricity/
--data_path electricity.csv
--pred_len 192
--enc_in 321
--dec_in 321
--c_out 321
```

### Training on Exchange Dataset
Go to the directory "Periodformer/", and run
```python
python eval_async_mGPUs_PF_v2.py 
--num_gpus 8 
--num_trails 32 
--model Periodformer 
--data custom 
--root_path ../data/exchange_rate/ 
--data_path exchange_rate.csv  
--pred_len 192 
--enc_in 8 
--dec_in 8 
--c_out 8
```

### Training on Traffic Dataset
Go to the directory "Periodformer/", and run
```python
python eval_async_mGPUs_PF_v2.py
--num_gpus 8 
--num_trails 32 
--model Periodformer 
--data custom 
--root_path ../data/traffic/ 
--data_path traffic.csv  
--pred_len 192 
--enc_in 862 
--dec_in 862 
--c_out 862
```

### Training on Weather Dataset
Go to the directory "Periodformer/", and run
```python
python eval_async_mGPUs_PF_v2.py 
--num_gpus 8 
--num_trails 32 
--model Periodformer 
--data custom 
--root_path ../data/weather/ 
--data_path weather.csv  
--pred_len 192 
--enc_in 21 
--dec_in 21 
--c_out 21
```

### Training on ILI Dataset
Go to the directory "Periodformer/", and run
```python
python eval_async_mGPUs_PF_v2.py 
--num_gpus 8 
--num_trails 32 
--model Periodformer 
--data custom 
--root_path ../data/illness/ 
--data_path national_illness.csv  
--pred_len 36 
--enc_in 7 
--dec_in 7 
--c_out 7
```

## Testing, taking ETTm2 dataset as an example,
```python
python run.py
--is_training 0
--model Periodformer
--data ETTm2
--root_path ../data/ETT-small/
--data_path ETTm2.csv
--pred_len 192
--enc_in 7
--enc_in 7
--c_out 7
```

Note that:
- Model was trained with Python 3.7 with CUDA 10.X.
- Model should work as expected with pytorch >= 1.7 support was recently included.

## Performace on Multivariate setting

<img src="https://github.com/liangdaojun/Periodformer/blob/main/Images/tb_per_1.jpg">

## Performace on Univariate setting

<img src="https://github.com/liangdaojun/Periodformer/blob/main/Images/tb_per_2.jpg">

## Effectiveness of MABO

<img src="https://github.com/liangdaojun/Periodformer/blob/main/Images/mabo_per.jpg">

## Citations

Daojun Liang, Haixia Zhang, Dongfeng Yuan, Xiaoyan Ma, Dongyang Li and Minggao Zhang, Does Long-Term Serie Forecasting Need Complex Attention and Extra Long Inputs? arXiv preprint arXiv: 2306.05035 (2023).

