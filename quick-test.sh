#!/bin/bash
# Quick single test

echo "Testing proxy with simple request..."

curl -s http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b-q2",
    "messages": [
      {"role": "user", "content": "Say hello"}
    ],
    "temperature": 0.7,
    "max_tokens": 20
  }' | jq '.'
