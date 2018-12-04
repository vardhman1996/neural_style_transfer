#!/bin/bash

set -x

python main.py --name run3 --content_weight 1e5 --style_weight 1e12

python main.py --name run6 --content_weight 1e5 --style_weight 1e13

python main.py --name run4 --content_weight 1e5 --style_weight 5e11

python main.py --name run5 --content_weight 1e5 --style_weight 1e14

python main.py --name run7 --content_weight 1e3 --style_weight 1e10

python main.py --name run8 --content_weight 1e5 --style_weight 1e15
