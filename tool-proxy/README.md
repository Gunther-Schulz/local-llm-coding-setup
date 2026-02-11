# Tool Proxy Server

HTTP proxy that sits between Cursor IDE and Llama.cpp backend, intercepting tool calls to inject reminders and prevent loops.

## Features

- **Tool Reminders**: Inject helpful reminders when tools are used (e.g., "Use Grep/Glob first")
- **Read Deduplication**: Track same file/range reads per turn and warn when excessive
- **Overlapping Range Detection**: Detect redundant overlapping reads and suggest consolidation
- **Turn-Based State**: Track reads per interaction turn, reset automatically
- **Fully Configurable**: YAML config for all rules, limits, and messages

## Directory Structure

```
tool-proxy/
├── server.py          # HTTP proxy entry point
├── interceptor.py     # Tool call interceptor with loop prevention
├── config/
│   ├── __init__.py
│   └── default_rules.yaml  # Default configuration
├── requirements.txt
└── README.md
```

## Installation

```bash
cd tool-proxy
pip install -r requirements.txt
```

## Usage

### Basic

```bash
python server.py --backend-url http://localhost:8080
```

### With Custom Config

```bash
python server.py \
    --port 8080 \
    --backend-url http://localhost:8080 \
    --config config/default_rules.yaml
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--port`, `-p` | `8080` | Port to listen on |
| `--backend-url`, `-b` | *required* | URL of Llama.cpp backend |
| `--config`, `-c` | `config/default_rules.yaml` | Path to config file |
| `--verbose`, `-v` | `false` | Enable debug logging |

### Debug logging

- **CLI:** `python server.py ... --verbose` (or `-v`)
- **Env:** `DEBUG=1 ./start-proxy.sh` (or set `DEBUG=1` before running)
- **Config:** In `config/default_rules.yaml`, set `logging.debug: true` or `logging.level: DEBUG`

When debug is on, the proxy logs: request path, Content-Length, turn_id, request keys, truncated body, tool call count and params, reminders, backend response keys and errors.

## Configuration

Edit `config/default_rules.yaml` to customize:

### Tool Reminders

```yaml
tools:
  Read:
    enabled: true
    message: |
      REMINDER: Use Grep/Glob first for discovery...
  Grep:
    enabled: true
    message: |
      REMINDER: Grep is for finding patterns...
```

### Read Coalescing (Loop Prevention)

```yaml
read_coalescing:
  enabled: true
  max_reads_per_turn: 3
  reminder_message: |
    WARNING: This file/range has been read {count} times...
```

### Overlapping Range Detection

```yaml
overlapping_ranges:
  enabled: true
  max_overlapping_reads: 2
  reminder_message: |
    WARNING: Multiple overlapping reads detected...
```

### Turn Tracking

```yaml
turn_tracking:
  enabled: true
  max_turns_in_memory: 100
  auto_reset_turn: true
```

## How It Works

### Request Flow

```
Cursor IDE
    ↓ (POST /tool-calls)
Tool Proxy (port 8080)
    ↓ Intercepts tool calls, injects reminders
    ↓ Tracks reads per turn (loop prevention)
    ↓ Forwards to backend
Llama.cpp Backend
    ↓ Response
Tool Proxy
    ↓ Returns to Cursor IDE
```

### Loop Prevention

1. On each `Read` tool call, track the file path and range
2. If same file/range read ≥ `max_reads_per_turn` times → inject warning
3. If overlapping ranges detected ≥ `max_overlapping_reads` times → inject warning
4. Turn state resets when new `turn_id` is received

### Turn Detection

Turns are detected via:
- Query param: `?turn_id=abc123`
- Header: `X-Turn-ID: abc123`
- Request body: `{"turn_id": "abc123"}`

## Example Config

See `config/default_rules.yaml` for complete example with all options.

## Integration with Cursor IDE

Configure Cursor IDE to use the proxy as your tool backend:

```json
{
  "agentBackend": "http://localhost:8080"
}
```

## Troubleshooting

### Proxy not receiving requests

- Check Cursor IDE is pointing to proxy port (default 8080)
- Verify backend URL is correct (`--backend-url`)

### Reminders not appearing

- Check `enabled: true` in config for the tool
- Verify config path is correct (`--config`)

### Read deduplication not working

- Ensure `turn_id` is being sent with requests
- Check `read_coalescing.enabled: true` in config

### Backend unreachable

- Verify Llama.cpp is running and accessible
- Check backend URL includes protocol (`http://` or `https://`)

## License

MIT