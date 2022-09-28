#!/bin/sh

# Create directories if they do not exist
mkdir -p tactile_gym_sim2real/data_collection/sim/active_reconstruction/data
mkdir -p tactile_gym_sim2real/data_collection/sim/active_reconstruction/data/checkpoints
mkdir -p tactile_gym_sim2real/data_collection/sim/active_reconstruction/data/obj_pointcloud
mkdir -p tactile_gym_sim2real/data_collection/sim/active_reconstruction/data/objects
mkdir -p tactile_gym_sim2real/data_collection/sim/active_reconstruction/data/touch_charts
mkdir -p tactile_gym_sim2real/data_collection/sim/active_reconstruction/data/checkpoints/touch_model
mkdir -p tactile_gym_sim2real/data_collection/sim/active_reconstruction/data/checkpoints/deformation_model


# Add files __init__.py
touch tactile_gym_sim2real/data_collection/sim/active_reconstruction/data/__init__.py
touch tactile_gym_sim2real/data_collection/sim/active_reconstruction/data/checkpoints/__init__.py
touch tactile_gym_sim2real/data_collection/sim/active_reconstruction/data/obj_pointcloud/__init__.py
touch tactile_gym_sim2real/data_collection/sim/active_reconstruction/data/objects/__init__.py
touch tactile_gym_sim2real/data_collection/sim/active_reconstruction/data/touch_charts/__init__.py

