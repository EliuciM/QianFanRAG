#!/bin/bash

stamp=$(date +"%Y%m%d%H%M%S")

PYTHON_BIN=/usr/bin/python3
#UPLOAD_PORT=${UPLOAD_PORT}
#QUERY_PORT=${QUERY_PORT}

# nohup $PYTHON_BIN -u ./embedding_api.py > ./embedding_api_${stamp}.log 2>&1 &

nohup $PYTHON_BIN -u ./query_api.py > ./query_api_${stamp}.log 2>&1 &

nohup $PYTHON_BIN -u ./upload_api.py > ./upload_api_${stamp}.log 2>&1 &
