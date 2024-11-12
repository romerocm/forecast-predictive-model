#!/bin/bash

case "$1" in
  "train")
    docker-compose up --build ml_model
    ;;
  "analyze")
    docker-compose up --build analyze
    ;;
  "predict")
    docker-compose up --build predict
    ;;
  "jupyter")
    docker-compose up --build jupyter
    ;;
  *)
    echo "Usage: ./run.sh [train|analyze|predict|jupyter]"
    exit 1
    ;;
esac
