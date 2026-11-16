#!/bin/sh
for p in 8000 8001 8002 8003; do
  llama-server -m bge-small-en-v1.5-q8_0.gguf --embedding -np 1 -t 1 --port $p &
done
