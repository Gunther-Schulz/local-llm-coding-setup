#!/bin/bash
# Test the vLLM API server

echo "🧪 Testing vLLM API server..."
echo ""

# Test 1: Check if server is running
echo "1. Checking server status..."
curl -s http://localhost:8000/v1/models | python -m json.tool || echo "❌ Server not responding"

echo ""
echo "2. Testing with a simple coding query..."
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "messages": [{"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}],
        "max_tokens": 200
    }' | python -m json.tool

echo ""
echo "✅ Test complete!"

