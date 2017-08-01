# Frozen Lake

This repository has two programs that take different approaches at trying to solve the [Frozen Lake](https://gym.openai.com/envs/FrozenLake-v0) challenge.

`frozen_lake.py` uses a simple table to track Q, because the number of states and actions is small.

`frozen_lake_net.py` uses a tensorflow neural network to estimate Q. In this case it doesn't work as well as using a table.