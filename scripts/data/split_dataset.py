#!/usr/bin/env python3
"""Split dataset into train/val/test."""
import json, sys
input_file = sys.argv[1]
lines = open(input_file).readlines()
train = lines[:int(len(lines)*0.8)]
val = lines[int(len(lines)*0.8):int(len(lines)*0.9)]
test = lines[int(len(lines)*0.9):]
for name, data in [('train', train), ('val', val), ('test', test)]:
    with open(f'{name}.jsonl', 'w') as f:
        f.writelines(data)
