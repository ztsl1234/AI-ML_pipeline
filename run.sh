#!/bin/sh
echo "Starting to run Pipeline..."
echo "First arg: $1"

python src/MLPipeline.py $1 &> MLPipeline$1.log

echo "Pipeline completed running"
