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

