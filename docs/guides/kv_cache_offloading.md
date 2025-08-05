<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV Cache Offloading

## Overview

KV cache offloading stores KV cache data across multiple memory hierarchies (GPU, CPU, Disk) to overcome GPU memory limitations. The KV cache manager automatically moves blocks between tiers based on access patterns and memory pressure.

## Architecture

### Memory Hierarchy

Dynamo's KV cache manager implements a multi-tiered offloading system:

| Memory Tier | Capacity | Access Speed | Use Case |
|-------------|----------|--------------|----------|
| **GPU Memory (Device)** | Limited (24-80GB) | Fastest | Active inference, hot cache |
| **CPU Memory (Host)** | Large (256GB-2TB) | Fast | Warm cache, recent contexts |
| **Disk/NVMe (Storage)** | Largest (Multi-TB) | Moderate | Cold storage, archival |

### Components

- **Offload Manager**: Manages block transfers between memory tiers
- **Block Pools**: Allocate memory within each tier
- **Transfer Context**: Handles NIXL-based data movement
- **Event Manager**: Publishes KV cache events for routing

### Workflow

1. New KV blocks allocated in GPU memory
2. Block access patterns tracked via events
3. Offload decisions based on memory pressure and access frequency
4. Non-blocking transfers using NIXL or UCX
5. Blocks moved back to GPU when needed
6. State changes broadcast to router

## Configuration

### Block Manager Parameters

| Parameter | Type | Description | Required | Default |
|-----------|------|-------------|----------|---------|
| `worker_id` | int | Unique identifier for the worker | Yes | - |
| `num_layer` | int | Number of transformer layers | Yes | - |
| `outer_dim` | int | Number of outer dimensions (1 for MLA, 2 for K/V) | Yes | - |
| `page_size` | int | Tokens per block (16, 32, or 64) | Yes | - |
| `inner_dim` | int | Hidden dimension size | Yes | - |
| `dtype` | str | Data type ("fp16", "fp32", "bf16") | No | "fp16" |
| `device_num_blocks` | int | GPU memory blocks to allocate | No | None |
| `host_num_blocks` | int | CPU memory blocks to allocate | No | None |
| `device_id` | int | GPU device ID | No | 0 |

### Memory Tier Configuration

```python
from dynamo._core import BlockManager

# Basic configuration with host offloading
block_manager = BlockManager(
    worker_id=1,
    num_layer=32,          # Number of transformer layers
    outer_dim=2,           # 2 for key and value
    page_size=16,          # Tokens per block (commonly 16 or 64)
    inner_dim=4096,        # Hidden dimension
    dtype="fp16",
    device_num_blocks=1024,    # GPU memory blocks
    host_num_blocks=4096,      # CPU memory blocks (4x GPU capacity)
    device_id=0
)

# Advanced configuration with larger pools
block_manager = BlockManager(
    worker_id=1,
    num_layer=40,          # Larger model
    outer_dim=2,
    page_size=32,          # Larger page size
    inner_dim=8192,        # Larger hidden dimension
    dtype="bf16",          # Brain float 16
    device_num_blocks=2048,    # More GPU blocks
    host_num_blocks=16384,     # Much larger host pool (8x GPU)
    device_id=0
)
```

### Memory Sizing Guidelines

| Model Size | GPU Blocks | Host Blocks | Ratio | Use Case |
|------------|------------|-------------|-------|----------|
| 7B params | 1024 | 4096 | 4:1 | Basic offloading |
| 13B params | 1536 | 9216 | 6:1 | Moderate contexts |
| 34B params | 2048 | 16384 | 8:1 | Long contexts |
| 70B+ params | 3072 | 24576 | 8:1 | Extended contexts |

### Backend Configuration

#### vLLM Backend

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `--enable-prefix-caching` | Enable KV cache reuse and events | Yes | False |
| `--enforce-eager` | Disable CUDA graphs for compatibility | Recommended | False |
| `--gpu-memory-utilization` | Fraction of GPU memory to use | No | 0.9 |
| `--dynamo-port-min` | Minimum port for Dynamo services | No | 20000 |
| `--dynamo-port-max` | Maximum port for Dynamo services | No | 30000 |

```bash
# Basic vLLM configuration
python -m dynamo.vllm \
    --model microsoft/DialoGPT-medium \
    --enable-prefix-caching \
    --enforce-eager

# Production configuration  
python -m dynamo.vllm \
    --model meta-llama/Llama-2-70b-chat-hf \
    --enable-prefix-caching \
    --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 4
```

#### TensorRT-LLM Backend

| Transfer Method | Description | Stability | Use Case |
|----------------|-------------|-----------|----------|
| UCX | Default high-performance communication | Stable | Production |
| NIXL | Experimental transfer library | Experimental | Testing |

```bash
# Default UCX method
python -m dynamo.trtllm \
    --model-path /models/llama-2-70b \
    --engine-config ./configs/ucx_engine.yaml \
    --tensor-parallel-size 4

# NIXL method (experimental)
python -m dynamo.trtllm \
    --model-path /models/llama-2-70b \
    --engine-config ./configs/nixl_engine.yaml \
    --tensor-parallel-size 4
```

**Note**: NIXL support requires building with `--trtllm-use-nixl-kvcache-experimental`.

#### SGLang Backend

| Parameter | Description | Required | Values |
|-----------|-------------|----------|--------|
| `--disaggregation-mode` | Worker specialization | Yes | prefill, decode |
| `--disaggregation-transfer-backend` | Transfer method | Yes | nixl, ucx |
| `--page-size` | KV cache block size | No | 16, 32, 64 |

```bash
# Prefill worker
python -m dynamo.sglang.worker \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --page-size 16 \
    --tp 2

# Decode worker
python -m dynamo.sglang.decode_worker \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --page-size 16 \
    --tp 2
```

### Engine Configuration Files

For TensorRT-LLM, configure KV cache settings via YAML:

```yaml
# engine_config.yaml (UCX backend - default)
tensor_parallel_size: 1
max_num_tokens: 8192
kv_cache_config:
  free_gpu_memory_fraction: 0.95
cache_transceiver_config:
  backend: default

# engine_nixl.yaml (NIXL backend - experimental)
tensor_parallel_size: 1
max_num_tokens: 8192
kv_cache_config:
  free_gpu_memory_fraction: 0.95
cache_transceiver_config:
  backend: nixl
```

## Deployment

### Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| ETCD | 3.5+ | Service discovery and configuration |
| NATS | 2.8+ | Message queue with JetStream |
| NVIDIA Driver | 535+ | GPU support |
| CUDA | 12.0+ | GPU computation |

### Step 1: Infrastructure Setup

```bash
# Start ETCD
docker run -d --name etcd -p 2379:2379 quay.io/coreos/etcd:v3.5.0 \
    etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379

# Start NATS with JetStream
docker run -d --name nats -p 4222:4222 nats:2.8.4 nats-server -js

# Verify services
curl http://localhost:2379/health
curl http://localhost:4222/varz
```

### Step 2: Enable KV-Aware Routing

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--router-mode` | Routing strategy | round-robin | kv |
| `--kv-overlap-score-weight` | Weight for cache overlap | 1.0 | 1.0-2.0 |
| `--router-temperature` | Sampling randomness | 0.0 | 0.0-0.1 |
| `--http-port` | Frontend HTTP port | 8080 | 8080 |

```bash
# Basic KV routing
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8080 \
    --kv-overlap-score-weight 1.0

# Production configuration
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8080 \
    --kv-overlap-score-weight 1.5 \
    --router-temperature 0.05
```

### Step 3: Deploy Workers

#### Single Node Deployment

```bash
# Terminal 1: Frontend
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8080

# Terminal 2: vLLM Worker
python -m dynamo.vllm \
    --model microsoft/DialoGPT-medium \
    --enable-prefix-caching \
    --enforce-eager \
    --gpu-memory-utilization 0.8
```

#### Multi-Node Deployment

**Node 1 (Frontend):**
```bash
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8080
```

**Node 2 (Worker):**
```bash
# Export ETCD/NATS endpoints
export ETCD_ENDPOINTS=http://node1:2379
export NATS_URL=nats://node1:4222

python -m dynamo.vllm \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --enforce-eager \
    --tensor-parallel-size 2
```

#### Disaggregated Deployment

**Prefill Worker:**
```bash
python -m dynamo.sglang.worker \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --tensor-parallel-size 2
```

**Decode Worker:**
```bash
CUDA_VISIBLE_DEVICES=2,3 python -m dynamo.sglang.decode_worker \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --tensor-parallel-size 2
```

### Step 4: Verify KV Events

KV events are published on dynamically allocated ports. Check worker logs for port assignment.

#### Check Event Port Allocation

```bash
# Method 1: Check worker logs
grep "Allocated ZMQ KV events port" /tmp/worker.log

# Method 2: Check worker output
grep "ZMQ KV events port" /var/log/dynamo/worker.log

# Method 3: Monitor process output
ps aux | grep dynamo.vllm
```

#### Monitor KV Events

```bash
# Basic monitoring script
cat > monitor_kv_events.py << EOF
import zmq
import json
import sys

if len(sys.argv) != 2:
    print("Usage: python monitor_kv_events.py <port>")
    sys.exit(1)

port = sys.argv[1]
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(f'tcp://localhost:{port}')
socket.setsockopt(zmq.SUBSCRIBE, b'')

print(f"Monitoring KV events on port {port}...")
while True:
    try:
        message = socket.recv_json(zmq.NOBLOCK)
        print(f"KV Event: {json.dumps(message, indent=2)}")
    except zmq.Again:
        continue
    except KeyboardInterrupt:
        break

socket.close()
context.term()
EOF

# Run monitoring (replace PORT with actual port from logs)
python monitor_kv_events.py <PORT>
```

#### Event Types

| Event Type | Description | Fields |
|------------|-------------|---------|
| `block_created` | New KV block allocated | block_id, sequence_hash, worker_id |
| `block_deleted` | KV block evicted | block_id, sequence_hash, worker_id |
| `block_accessed` | KV block cache hit | block_id, sequence_hash, access_count |

### Step 5: Test KV Cache Offloading

#### Basic Functional Test

```python
import requests

base_url = "http://localhost:8080/v1"
model_name = "your-model"

def send_request(messages):
    return requests.post(f"{base_url}/chat/completions", json={
        "model": model_name, "messages": messages, "max_tokens": 100
    }).json()

# Test 1: Cold cache
messages1 = [{"role": "user", "content": "Explain quantum computing"}]
response1 = send_request(messages1)

# Test 2: Warm cache - shared prefix
messages2 = [
    {"role": "user", "content": "Explain quantum computing"},
    {"role": "assistant", "content": response1['choices'][0]['message']['content']},
    {"role": "user", "content": "How does it compare to classical computing?"}
]
response2 = send_request(messages2)

# Test 3: Cache miss - different context
messages3 = [{"role": "user", "content": "What is the weather like today?"}]
response3 = send_request(messages3)
```

#### Load Testing for Offloading

```python
import requests
import threading
import random

def conversation_thread(thread_id, num_requests=10):
    topics = ["artificial intelligence", "machine learning", "quantum computing"]
    topic = random.choice(topics)
    messages = [{"role": "user", "content": f"Tell me about {topic}"}]
    
    for i in range(num_requests):
        response = requests.post("http://localhost:8080/v1/chat/completions", json={
            "model": "your-model", "messages": messages, "max_tokens": 50
        })
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            messages.extend([
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Can you elaborate? (request {i+1})"}
            ])

# Run 5 concurrent conversations
threads = [threading.Thread(target=conversation_thread, args=(i, 8)) for i in range(5)]
for t in threads: t.start()
for t in threads: t.join()
```

## Performance Testing

### Benchmark Methodology

#### Test Environment Setup

| Component | Specification | Purpose |
|-----------|---------------|---------|
| GPU | H100 80GB or A100 80GB | Primary compute and cache |
| CPU | 64+ cores | Host memory management |
| RAM | 512GB+ | Host cache storage |
| Storage | NVMe SSD 2TB+ | Disk cache tier |
| Network | 200Gbps+ | Multi-node transfers |

#### Routing Mode Comparison

```bash
# Test 1: Random routing (baseline)
python -m dynamo.frontend \
    --router-mode random \
    --http-port 8080 &
FRONTEND_PID=$!

# Run benchmark
python benchmark_script.py --mode random --duration 300 --concurrency 10

# Stop frontend
kill $FRONTEND_PID

# Test 2: KV-aware routing
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8080 \
    --kv-overlap-score-weight 1.0 &
FRONTEND_PID=$!

# Run same benchmark
python benchmark_script.py --mode kv --duration 300 --concurrency 10

kill $FRONTEND_PID
```

#### Memory Monitoring Script

```bash
#!/bin/bash
# monitor_memory.sh - Comprehensive memory monitoring

LOG_FILE="/tmp/memory_monitor.log"
INTERVAL=5

echo "Timestamp,GPU_Used_MB,GPU_Total_MB,CPU_Used_MB,CPU_Total_MB,Disk_Used_MB" > $LOG_FILE

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # GPU memory (first GPU)
    GPU_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1)
    GPU_USED=$(echo $GPU_INFO | cut -d',' -f1 | tr -d ' ')
    GPU_TOTAL=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')
    
    # CPU memory
    MEM_INFO=$(free -m | grep '^Mem:')
    CPU_TOTAL=$(echo $MEM_INFO | awk '{print $2}')
    CPU_USED=$(echo $MEM_INFO | awk '{print $3}')
    
    # Disk usage for KV cache
    DISK_USED=$(du -sm /tmp/kv_cache 2>/dev/null | cut -f1 || echo "0")
    
    echo "$TIMESTAMP,$GPU_USED,$GPU_TOTAL,$CPU_USED,$CPU_TOTAL,$DISK_USED" >> $LOG_FILE
    sleep $INTERVAL
done
```

### Performance Metrics

#### Primary Metrics

| Metric | Description | Target | Measurement |
|--------|-------------|--------|-------------|
| TTFT | Time to First Token | <2s | Client-side timing |
| ITL | Inter-Token Latency | <50ms | Token generation rate |
| Throughput | Requests per second | >10 RPS | Total completed requests |
| Cache Hit Rate | Percentage of cache hits | >60% | KV event analysis |

#### Memory Utilization Metrics

| Tier | Utilization Target | Monitoring Command |
|------|-------------------|-------------------|
| GPU | 80-95% | `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` |
| CPU | 60-80% | `free -h` |
| Disk | <50% | `df -h /tmp` |

#### Advanced Monitoring

```python
# performance_monitor.py - Comprehensive performance monitoring
import requests
import time
import statistics
import json
from datetime import datetime

class PerformanceMonitor:
    def __init__(self, base_url="http://localhost:8080", model_name="your-model"):
        self.base_url = base_url
        self.model_name = model_name
        self.metrics = {
            'ttft_times': [],
            'request_times': [],
            'token_counts': [],
            'error_count': 0,
            'total_requests': 0
        }
    
    def send_request(self, messages, max_tokens=100):
        start_time = time.time()
        first_token_time = None
        
        try:
            response = requests.post(f"{self.base_url}/v1/chat/completions", 
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "stream": True
                }, stream=True)
            
            token_count = 0
            for line in response.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.time()
                    token_count += 1
            
            end_time = time.time()
            
            ttft = first_token_time - start_time if first_token_time else None
            total_time = end_time - start_time
            
            self.metrics['ttft_times'].append(ttft)
            self.metrics['request_times'].append(total_time)
            self.metrics['token_counts'].append(token_count)
            self.metrics['total_requests'] += 1
            
            return True
            
        except Exception as e:
            print(f"Request failed: {e}")
            self.metrics['error_count'] += 1
            self.metrics['total_requests'] += 1
            return False
    
    def run_benchmark(self, duration_seconds=300, concurrency=5):
        import threading
        
        def worker():
            end_time = time.time() + duration_seconds
            request_id = 0
            
            while time.time() < end_time:
                messages = [{"role": "user", "content": f"Test request {request_id}"}]
                self.send_request(messages)
                request_id += 1
                time.sleep(1)  # 1 RPS per thread
        
        threads = []
        for i in range(concurrency):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
    
    def print_results(self):
        if self.metrics['ttft_times']:
            ttft_filtered = [t for t in self.metrics['ttft_times'] if t is not None]
            print(f"TTFT - Mean: {statistics.mean(ttft_filtered):.3f}s, "
                  f"P95: {statistics.quantiles(ttft_filtered, n=20)[18]:.3f}s")
        
        if self.metrics['request_times']:
            print(f"Total Time - Mean: {statistics.mean(self.metrics['request_times']):.3f}s")
        
        print(f"Success Rate: {(self.metrics['total_requests'] - self.metrics['error_count']) / self.metrics['total_requests'] * 100:.1f}%")
        print(f"Total Requests: {self.metrics['total_requests']}")

# Usage
if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.run_benchmark(duration_seconds=300, concurrency=5)
    monitor.print_results()
```

## Troubleshooting

### Common Issues

| Symptom | Possible Cause | Solution | Debug Command |
|---------|---------------|----------|---------------|
| KV Events Not Publishing | Missing `--enable-prefix-caching` | Add flag to vLLM worker | `grep "KV.*event" /tmp/worker.log` |
| High Memory Usage | Insufficient host allocation | Increase `host_num_blocks` | `free -h && nvidia-smi` |
| NIXL Transfer Failures | Missing infrastructure | Start ETCD/NATS services | `curl http://localhost:2379/health` |
| Router Not Finding Workers | Service discovery issues | Check worker registration | `curl http://localhost:8080/v1/models` |
| Poor Cache Hit Rate | Suboptimal routing | Tune `--kv-overlap-score-weight` | Monitor KV events |
| OOM on GPU | Excessive GPU allocation | Reduce `--gpu-memory-utilization` | `nvidia-smi` |
| Slow Block Transfers | Network bottleneck | Check NIXL configuration | `iftop -i eth0` |
| Worker Registration Failed | Port conflicts | Check port allocation | `netstat -tulpn \| grep :8080` |

### Diagnostic Scripts

#### System Health Check

```bash
#!/bin/bash
# health_check.sh

echo "=== Dynamo KV Cache Health Check ==="

# Check infrastructure
curl -s http://localhost:2379/health > /dev/null && echo "ETCD: Running" || echo "ETCD: Not accessible"
curl -s http://localhost:4222/varz > /dev/null && echo "NATS: Running" || echo "NATS: Not accessible"

# Check GPU and memory
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
free -h | grep Mem:

# Check Dynamo services
curl -s http://localhost:8080/v1/models > /dev/null && echo "Frontend: Running" || echo "Frontend: Not accessible"

# Check KV event ports
grep "Allocated ZMQ KV events port" /tmp/worker.log 2>/dev/null | tail -3
```

#### KV Cache Analysis

```python
# kv_analysis.py - Analyze KV cache performance
import zmq
import json
import time
from collections import defaultdict, deque
from datetime import datetime

class KVCacheAnalyzer:
    def __init__(self, port):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f'tcp://localhost:{port}')
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        
        self.stats = {
            'total_events': 0,
            'block_created': 0,
            'block_deleted': 0,
            'block_accessed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'active_blocks': set(),
            'recent_events': deque(maxlen=100)
        }
        
        self.block_access_count = defaultdict(int)
        self.sequence_blocks = defaultdict(set)
    
    def monitor(self, duration_seconds=300):
        print(f"Monitoring KV events on port {self.port} for {duration_seconds} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            try:
                message = self.socket.recv_json(zmq.NOBLOCK)
                self.process_event(message)
                
            except zmq.Again:
                time.sleep(0.1)
                continue
            except KeyboardInterrupt:
                break
        
        self.print_analysis()
    
    def process_event(self, event):
        self.stats['total_events'] += 1
        self.stats['recent_events'].append({
            'timestamp': datetime.now().isoformat(),
            'event': event
        })
        
        event_type = event.get('type', 'unknown')
        block_id = event.get('block_id')
        sequence_hash = event.get('sequence_hash')
        
        if event_type == 'block_created':
            self.stats['block_created'] += 1
            self.stats['active_blocks'].add(block_id)
            if sequence_hash:
                self.sequence_blocks[sequence_hash].add(block_id)
                
        elif event_type == 'block_deleted':
            self.stats['block_deleted'] += 1
            self.stats['active_blocks'].discard(block_id)
            
        elif event_type == 'block_accessed':
            self.stats['block_accessed'] += 1
            self.block_access_count[block_id] += 1
            
            # Determine if this was a cache hit or miss
            if self.block_access_count[block_id] > 1:
                self.stats['cache_hits'] += 1
            else:
                self.stats['cache_misses'] += 1
    
    def print_analysis(self):
        print("\n=== KV Cache Analysis Results ===")
        print(f"Total Events: {self.stats['total_events']}")
        print(f"Blocks Created: {self.stats['block_created']}")
        print(f"Blocks Deleted: {self.stats['block_deleted']}")
        print(f"Block Accesses: {self.stats['block_accessed']}")
        print(f"Active Blocks: {len(self.stats['active_blocks'])}")
        
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
            print(f"Cache Hit Rate: {hit_rate:.2%}")
        
        # Top accessed blocks
        if self.block_access_count:
            top_blocks = sorted(self.block_access_count.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\nTop Accessed Blocks:")
            for block_id, count in top_blocks:
                print(f"  Block {block_id}: {count} accesses")
        
        # Recent events
        print(f"\nRecent Events (last {min(5, len(self.stats['recent_events']))}):")
        for event_data in list(self.stats['recent_events'])[-5:]:
            print(f"  {event_data['timestamp']}: {event_data['event']}")

# Usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python kv_analysis.py <kv_event_port>")
        sys.exit(1)
    
    port = sys.argv[1]
    analyzer = KVCacheAnalyzer(port)
    analyzer.monitor(duration_seconds=300)
```

### Debug Commands

#### Service Status

```bash
# Check all Dynamo services
systemctl status dynamo-* 2>/dev/null || echo "No systemd services found"

# Check Docker containers
docker ps | grep -E "(etcd|nats|dynamo)"

# Check listening ports
netstat -tulpn | grep -E ":(2379|4222|8080|20000|30000)"
```

#### Log Analysis

```bash
# Find all Dynamo log files
find /tmp /var/log -name "*dynamo*" -o -name "*worker*" 2>/dev/null

# Extract KV event information
grep -r "KV.*event\|Allocated.*port\|ZMQ" /tmp/ 2>/dev/null | head -10

# Monitor real-time logs
tail -f /tmp/worker.log | grep -E "KV|event|port|error"
```

#### Network Diagnostics

```bash
# Check NIXL network configuration
env | grep -E "NIXL|UCX" | sort

# Test network connectivity between nodes
ping -c 3 <remote_node_ip>
nc -zv <remote_node_ip> 2379  # ETCD
nc -zv <remote_node_ip> 4222  # NATS
```

## Advanced Configuration

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `NIXL_UCX_TLS` | UCX transport layers | `all` | `self,tcp,cuda_copy,cuda_ipc` |
| `NIXL_UCX_NET_DEVICES` | Network devices for UCX | Auto-detect | `mlx5_0:1,mlx5_1:1` |
| `ETCD_ENDPOINTS` | ETCD server addresses | `http://localhost:2379` | `http://node1:2379,http://node2:2379` |
| `NATS_URL` | NATS server URL | `nats://localhost:4222` | `nats://cluster.example.com:4222` |
| `DYNAMO_LOG_LEVEL` | Logging verbosity | `info` | `debug`, `trace` |

### Custom Memory Configurations

#### Large Model Configuration (70B+)

```python
from dynamo._core import BlockManager

block_manager = BlockManager(
    worker_id=1,
    num_layer=80,
    outer_dim=2,
    page_size=32,
    inner_dim=8192,
    dtype="bf16",
    device_num_blocks=4096,
    host_num_blocks=32768,
    device_id=0
)
```

#### Multi-GPU Configuration

```python
# GPU 0
block_manager_gpu0 = BlockManager(
    worker_id=1, num_layer=40, outer_dim=2, page_size=16,
    inner_dim=4096, dtype="fp16", device_num_blocks=2048,
    host_num_blocks=8192, device_id=0
)

# GPU 1
block_manager_gpu1 = BlockManager(
    worker_id=2, num_layer=40, outer_dim=2, page_size=16,
    inner_dim=4096, dtype="fp16", device_num_blocks=2048,
    host_num_blocks=8192, device_id=1
)
```

### NIXL Optimization

#### Network Configuration

```bash
# NIXL configuration
export NIXL_UCX_TLS=self,tcp,cuda_copy,cuda_ipc,rc
export NIXL_UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1
export NIXL_UCX_RNDV_THRESH=8192
export NIXL_UCX_ZCOPY_THRESH=1024

# Multi-node
export NIXL_UCX_NET_DEVICES=$(ibdev2netdev | awk '{print $5":1"}' | paste -sd,)

python -m dynamo.vllm \
    --model meta-llama/Llama-2-70b-chat-hf \
    --enable-prefix-caching \
    --tensor-parallel-size 4
```

#### Memory Pool Tuning

```bash
# Configure memory pools for optimal performance
export NIXL_MEMORY_POOL_SIZE=8GB
export NIXL_CUDA_MEMORY_POOL_SIZE=2GB
export NIXL_PINNED_MEMORY_POOL_SIZE=4GB

# Enable memory pool recycling
export NIXL_ENABLE_MEMORY_RECYCLING=true
export NIXL_MEMORY_RECYCLING_THRESHOLD=0.8
```

### Production Deployment

#### Resource Limits

```yaml
# kubernetes_deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamo-vllm-worker
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm-worker
        image: dynamo/vllm:latest
        resources:
          requests:
            memory: "128Gi"
            nvidia.com/gpu: "2"
            cpu: "16"
          limits:
            memory: "256Gi"
            nvidia.com/gpu: "2"
            cpu: "32"
        env:
        - name: NIXL_UCX_TLS
          value: "self,tcp,cuda_copy,cuda_ipc"
        - name: DYNAMO_LOG_LEVEL
          value: "info"
        command:
        - python
        - -m
        - dynamo.vllm
        args:
        - --model
        - meta-llama/Llama-2-70b-chat-hf
        - --enable-prefix-caching
        - --enforce-eager
        - --tensor-parallel-size
        - "2"
        - --gpu-memory-utilization
        - "0.85"
```

#### Monitoring Configuration

```bash
# prometheus_config.yml
global:
  scrape_interval: 15s

scrape_configs:
- job_name: 'dynamo-metrics'
  static_configs:
  - targets: ['localhost:8000']  # Metrics endpoint
  metrics_path: /metrics
  scrape_interval: 5s

- job_name: 'nvidia-gpu'
  static_configs:
  - targets: ['localhost:9400']  # nvidia-exporter
  scrape_interval: 10s
```

### Development Configuration

#### Debug Mode

```bash
# Enable comprehensive debugging
export DYNAMO_LOG_LEVEL=trace
export NIXL_LOG_LEVEL=debug
export RUST_LOG=debug

# Launch with debug flags
python -m dynamo.vllm \
    --model microsoft/DialoGPT-medium \
    --enable-prefix-caching \
    --enforce-eager \
    --gpu-memory-utilization 0.5  # Reduced for debugging
```

#### Profiling Setup

```python
# profiling_config.py
import cProfile
import pstats
import io
from dynamo._core import BlockManager

def profile_block_manager():
    pr = cProfile.Profile()
    pr.enable()
    
    # Your block manager operations here
    block_manager = BlockManager(
        worker_id=1, num_layer=32, outer_dim=2,
        page_size=16, inner_dim=4096, dtype="fp16",
        device_num_blocks=1024, host_num_blocks=4096
    )
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())

if __name__ == "__main__":
    profile_block_manager()
```

## Monitoring and Observability

### Metrics Collection

| Metric Category | Metrics | Collection Method |
|----------------|---------|-------------------|
| **Memory Usage** | GPU/CPU/Disk utilization | System monitoring |
| **Cache Performance** | Hit rate, miss rate, eviction rate | KV event analysis |
| **Request Latency** | TTFT, ITL, total request time | Client instrumentation |
| **Throughput** | Requests/second, tokens/second | Load balancer metrics |
| **Transfer Performance** | NIXL bandwidth, transfer latency | Network monitoring |

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Dynamo KV Cache Offloading",
    "panels": [
      {
        "title": "Memory Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_ml_py_memory_used_bytes / nvidia_ml_py_memory_total_bytes * 100",
            "legendFormat": "GPU Memory %"
          },
          {
            "expr": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100",
            "legendFormat": "CPU Memory %"
          }
        ]
      },
      {
        "title": "KV Cache Hit Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(kv_cache_hits_total[5m]) / (rate(kv_cache_hits_total[5m]) + rate(kv_cache_misses_total[5m])) * 100",
            "legendFormat": "Hit Rate %"
          }
        ]
      }
    ]
  }
}
```

## References

### Documentation

- [Dynamo Architecture Documentation](../architecture/architecture.md)
- [KV Cache Routing](../architecture/kv_cache_routing.md) 
- [Disaggregated Serving](../architecture/disagg_serving.md)
- [Request Migration](../architecture/request_migration.md)
- [NIXL Documentation](https://github.com/ai-dynamo/nixl)

### Research Papers

- [PagedAttention: Efficient Memory Management for Dynamic Neural Networks](https://arxiv.org/abs/2309.06180)
- [RadixAttention: Efficient Reuse of KV-Cache via Prefix Sharing](https://arxiv.org/abs/2312.07104)
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670)

### External Tools

- [vLLM Documentation](https://docs.vllm.ai/)
- [TensorRT-LLM Repository](https://github.com/NVIDIA/TensorRT-LLM)
- [SGLang Documentation](https://github.com/sgl-project/sglang)
- [UCX Documentation](https://openucx.readthedocs.io/)

## Performance Considerations

KV cache offloading provides:

- **Memory Efficiency**: Larger effective capacity using CPU memory pools
- **Extended Context Support**: Handle longer sequences without OOM  
- **Cache Hit Acceleration**: Route requests to workers with relevant cache
- **Non-blocking Transfers**: Overlap computation with data movement