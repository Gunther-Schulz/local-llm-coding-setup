#!/usr/bin/env python3
"""
Benchmark script for vLLM endpoints
Tests inference speed, latency, and throughput
"""

import time
import requests
import json
import statistics
from typing import Dict, List

def benchmark(endpoint_url: str, num_requests: int = 10, model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
    """Benchmark the vLLM endpoint"""
    
    prompt = "Write a Python function to sort a list in ascending order and explain how it works."
    
    print(f"🚀 Benchmarking {endpoint_url}")
    print(f"Model: {model_name}")
    print(f"Requests: {num_requests}\n")
    
    times = []
    tokens_per_second = []
    time_to_first_token = []
    successful_requests = 0
    
    for i in range(num_requests):
        print(f"Request {i+1}/{num_requests}...", end=" ", flush=True)
        start = time.time()
        
        try:
            response = requests.post(
                f"{endpoint_url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.7
                },
                timeout=120
            )
            
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                
                if completion_tokens > 0:
                    tps = completion_tokens / elapsed
                    tokens_per_second.append(tps)
                    times.append(elapsed)
                    successful_requests += 1
                    print(f"✅ {elapsed:.2f}s ({completion_tokens} tokens, {tps:.1f} tok/s)")
                else:
                    print(f"⚠️  {elapsed:.2f}s (no tokens)")
            else:
                print(f"❌ Error {response.status_code}: {response.text[:100]}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)[:100]}")
    
    # Calculate statistics
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS")
    print("="*60)
    
    if times:
        print(f"\n✅ Successful Requests: {successful_requests}/{num_requests}")
        print(f"⏱️  Average Latency: {statistics.mean(times):.2f}s")
        print(f"⏱️  Median Latency: {statistics.median(times):.2f}s")
        print(f"⏱️  Min Latency: {min(times):.2f}s")
        print(f"⏱️  Max Latency: {max(times):.2f}s")
        
        if len(times) > 1:
            print(f"📈 Std Deviation: {statistics.stdev(times):.2f}s")
    
    if tokens_per_second:
        print(f"\n🚀 Average Tokens/Second: {statistics.mean(tokens_per_second):.1f}")
        print(f"🚀 Median Tokens/Second: {statistics.median(tokens_per_second):.1f}")
        print(f"🚀 Min Tokens/Second: {min(tokens_per_second):.1f}")
        print(f"🚀 Max Tokens/Second: {max(tokens_per_second):.1f}")
        
        if len(tokens_per_second) > 1:
            print(f"📈 Std Deviation: {statistics.stdev(tokens_per_second):.1f}")
    
    return {
        "endpoint": endpoint_url,
        "model": model_name,
        "successful_requests": successful_requests,
        "total_requests": num_requests,
        "avg_latency": statistics.mean(times) if times else 0,
        "avg_tokens_per_second": statistics.mean(tokens_per_second) if tokens_per_second else 0
    }

if __name__ == "__main__":
    import sys
    
    # Default to localhost, but can override
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-Coder-32B-Instruct"
    num_reqs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    results = benchmark(endpoint, num_reqs, model)
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to benchmark_results.json")

