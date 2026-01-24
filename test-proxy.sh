#!/bin/bash
# Simple test for compression proxy

echo "Testing compression proxy..."
echo ""

# Test 1: Simple completion (no tools)
echo "=== Test 1: Simple completion ==="
curl -s http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b-q2",
    "messages": [
      {"role": "user", "content": "Say hello in one word"}
    ],
    "temperature": 0.7,
    "max_tokens": 10
  }' | jq -r '.choices[0].message.content'

echo ""
echo ""

# Test 2: With tools
echo "=== Test 2: Tool calling ==="
curl -s http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b-q2",
    "messages": [
      {"role": "user", "content": "List files in the current directory"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "Shell",
          "description": "Execute a shell command",
          "parameters": {
            "type": "object",
            "properties": {
              "command": {
                "type": "string",
                "description": "The shell command to execute"
              }
            },
            "required": ["command"]
          }
        }
      }
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }' | jq '.'

echo ""
echo ""
echo "✓ Tests complete"
