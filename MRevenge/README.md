# Hy-Q in Montezuma's Revenge
This repository contains our DQN-style algorithm for Montezuma's Revenge.

## Collect offline data
(easy/medium/hard) dataset: 
```bash
random-network-distillation-pytorch/get_offline_[easy/med/hard].py
```

## Run our code

To reproduce our result, please run:
```bash
python train_atari.py --dueling True --seperate_buffer True --ratio_ann True --offline_buffer_path offline_data/[easy/medium/hard] --seed [seed]
```

Please use seed [1,12,123,1234,12345] to reproduce our results. 

## Credit
RND code adapted from: https://github.com/jcwleo/random-network-distillation-pytorch
Priortized Experience Replay code adapted from: https://github.com/felix-kerkhoff/DQfD