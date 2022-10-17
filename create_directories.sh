#!/bin/sh

# Create directories if they do not exist
mkdir -p object_reconstruction/data
mkdir -p object_reconstruction/data/checkpoints
mkdir -p object_reconstruction/data/obj_pointcloud
mkdir -p object_reconstruction/data/objects
mkdir -p object_reconstruction/data/touch_charts
mkdir -p object_reconstruction/data/checkpoints/touch_model
mkdir -p object_reconstruction/data/checkpoints/deformation_model


# Add files __init__.py
touch object_reconstruction/data/__init__.py
touch object_reconstruction/data/checkpoints/__init__.py
touch object_reconstruction/data/obj_pointcloud/__init__.py
touch object_reconstruction/data/objects/__init__.py
touch object_reconstruction/data/touch_charts/__init__.py

# Download data
cd object_reconstruction/data/touch_charts
wget "https://uob-my.sharepoint.com/:u:/g/personal/ri21540_bristol_ac_uk/ET1hJIrHIFRLt4o3nl6U8OIBnjdgA7X2abtqzacBl3Vzhw?e=OLfKLU&download=1" -O touch_chart.obj
wget "https://uob-my.sharepoint.com/:u:/g/personal/ri21540_bristol_ac_uk/EeIsekJK_FlHiGyoqob0J_UBY6YG2_9_XsRZmDb5cl15vQ?e=8wlrH7&download=1" -O vision_charts.obj
wget "https://uob-my.sharepoint.com/:u:/g/personal/ri21540_bristol_ac_uk/EVGbUV38uZ5LoL52NakmLNAB8aLW8RtqKERBY6rXLK6frQ?e=wiD1W7&download=1" -O adj_info.npy