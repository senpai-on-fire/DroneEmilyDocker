#!/bin/bash

# Loop through a range of integers (e.g., 1 to 5)
for i in {1..10}; do
    # Run the Python script with the loop variable as the argument
    python3 OnyiParamEstReplayGPU_updated.py --model ltc --id $i
done

