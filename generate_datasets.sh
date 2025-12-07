#!/bin/bash

# This script automates the process of generating RLBench datasets for a list of tasks.
# It changes into the rlbench/rlbench directory and then runs dataset_generator.py for each task.

# List of tasks to process (snake_case format)
TASKS=(
    "open_box"
    "open_door"
    "open_drawer"
    "open_grill"
    "open_microwave"
    "open_washing_machine"
    "open_window"
    "phone_on_base"
    "pick_up_cup"
    "play_jenga"
    "press_switch"
    "push_button"
    "put_books_on_bookshelf"
    "put_knife_on_chopping_board"
    "put_rubbish_in_bin"
    "scoop_with_spatula"
    "slide_block_to_target"
    "take_plate_off_colored_dish_rack"
    "toilet_seat_up"
    "lamp_on"
    "turn_tap"
)

# Common parameters for the dataset_generator.py script
SAVE_PATH="../../data/train_data" # Relative to rlbench/rlbench
IMAGE_SIZE="256 256"
RENDERER="opengl"
EPISODES_PER_TASK=25
VARIATIONS=1
PROCESSES=1
ARM_MAX_VELOCITY=2.0
ARM_MAX_ACCELERATION=8.0

echo "Starting dataset generation for all tasks..."
echo "--------------------------------------------------"

# Change to the rlbench/rlbench directory
# The script will be executed from the project root, so the path is relative to that.
cd rlbench/rlbench || { echo "Failed to change directory to rlbench/rlbench. Exiting."; exit 1; }

# Loop through each task and run the dataset generator
for task_name in "${TASKS[@]}"; do
    echo "--------------------------------------------------"
    echo "Generating dataset for task: $task_name"
    echo "--------------------------------------------------"

    python dataset_generator.py \
        --save_path="$SAVE_PATH" \
        --tasks="$task_name" \
        --image_size $IMAGE_SIZE \
        --renderer="$RENDERER" \
        --episodes_per_task=$EPISODES_PER_TASK \
        --variations=$VARIATIONS \
        --processes=$PROCESSES \
        --arm_max_velocity=$ARM_MAX_VELOCITY \
        --arm_max_acceleration=$ARM_MAX_ACCELERATION

    # Check the exit code of the last command
    if [ $? -ne 0 ]; then
        echo "Error generating dataset for task: $task_name. Aborting."
        exit 1
    fi
done

echo "--------------------------------------------------"
echo "All datasets generated successfully."
echo "--------------------------------------------------"

# Go back to the original directory
cd ../..
