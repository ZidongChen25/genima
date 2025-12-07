#!/bin/bash

# This script automates the process of generating rendered datasets for a list of tasks.
# It iterates through a predefined list of task names and runs the rendering script for each.

# List of tasks to process
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

# Common parameters for the rendering script
DATASET_ROOT="/home/zc1525/Desktop/genima/data/train_data"
ACTION_HORIZON=20
NUM_PROCESSES=5
EPISODES=25 # You can change this value if needed

# Loop through each task and run the rendering script
for task_name in "${TASKS[@]}"; do
    echo "--------------------------------------------------"
    echo "Processing task: $task_name"
    echo "--------------------------------------------------"

    python render/render_data_no_tex.py \
        episodes=$EPISODES \
        dataset_root=$DATASET_ROOT \
        action_horizon=$ACTION_HORIZON \
        num_processes=$NUM_PROCESSES \
        draw.rnd_bg=False \
        task="$task_name"

    # Check the exit code of the last command
    if [ $? -ne 0 ]; then
        echo "Error processing task: $task_name. Aborting."
        exit 1
    fi
done

echo "--------------------------------------------------"
echo "All tasks processed successfully."
echo "--------------------------------------------------"
