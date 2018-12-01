#!/bin/bash

set -x

python3 main.py --name g_cloud_run2 --content_weight 1e5 --style_weight 1e11
python3 main.py --name g_cloud_run3 --content_weight 1e5 --style_weight 1e12
python3 main.py --name g_cloud_run1 --content_weight 1e5 --style_weight 1e10
python3 main.py --name g_cloud_run4 --content_weight 1e4 --style_weight 1e10
python3 main.py --name g_cloud_run5 --content_weight 1e4 --style_weight 1e11
python3 main.py --name g_cloud_run6 --content_weight 1e4 --style_weight 5e10
python3 main.py --name g_cloud_run7 --content_weight 1e3 --style_weight 1e10