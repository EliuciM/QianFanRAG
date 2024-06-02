#!/bin/bash

# Kill embedding_api.py process
pkill -f "embedding_api.py"

# Kill query_api.py process
pkill -f "query_api.py"

pkill -f "upload_api.py"