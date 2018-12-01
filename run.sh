#!/bin/bash

set -x
python main.py --name run1 --content_weight 1e5 --style_weight 1e10

python main.py --name run2 --content_weight 1e5 --style_weight 1e11

python main.py --name run3 --content_weight 1e5 --style_weight 1e12

python main.py --name run4 --content_weight 1e5 --style_weight 5e11

python main.py --name run5 --content_weight 1e4 --style_weight 1e10

python main.py --name run6 --content_weight 1e5 --style_weight 1e13
