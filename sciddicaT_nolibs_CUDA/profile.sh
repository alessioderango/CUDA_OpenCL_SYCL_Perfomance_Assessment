#!/bin/bash

srun nvprof ./sciddicaTcuda ../data/tessina_header.txt ../data/tessina_dem.txt ../data/tessina_source.txt ./tessina_output_cuda 4000 &&  md5sum ./tessina_output_cuda && cat ../data/tessina_header.txt ./tessina_output_cuda > ./tessina_output_cuda.qgis && rm ./tessina_output_cuda

