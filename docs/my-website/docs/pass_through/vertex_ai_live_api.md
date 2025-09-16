# Vertex AI Live API Passthrough

This guide shows how to use LiteLLM's passthrough functionality to connect to Google Vertex AI's Realtime (Live) API for Gemini Live models.

## Overview

The Vertex AI Live API (Realtime API) enables real-time, streaming interactions with Gemini models. LiteLLM's passthrough support allows you to use these capabilities without waiting for full unified spec implementation.

## Supported Models

- `gemini-2.0-flash-live-preview`
- `gemini-live-2.5-flash` (Private GA)
- `gemini-live-2.5-flash-preview-native-audio` (Public preview)
- `gemini-2.0-flash-live-001`

## Prerequisites

1. **Google Cloud Project** with Vertex AI API enabled
2. **Service Account** with Vertex AI permissions
3. **LiteLLM** installed with Vertex AI support

## Authentication Setup

### Option 1: Service Account Key File

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
export VERTEXAI_PROJECT="your-project-id"
export VERTEXAI_LOCATION="us-central1"  # or "global" for Live API
```

### Option 2: Application Default Credentials

```bash
gcloud auth application-default login
export VERTEXAI_PROJECT="your-project-id"
export VERTEXAI_LOCATION="us-central1"
```

## Basic Usage

### SDK Passthrough

```python
import litellm
from litellm.passthrough.main import llm_passthrough_route

# Basic passthrough request
response = llm_passthrough_route(
    method="POST",
    endpoint="live/stream",
    model="gemini-2.0-flash-live-preview",
    custom_llm_provider="vertex_ai",
    json={
        "model": "gemini-2.0-flash-live-preview",
        "messages": [
            {"role": "user", "content": "Hello! Can you help me with a question?"}
        ],
        "stream": True,
        "max_tokens": 100
    }
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
```

### Async Passthrough

```python
import asyncio
from litellm.passthrough.main import allm_passthrough_route

async def async_example():
    response = await allm_passthrough_route(
        method="POST",
        endpoint="realtime/connect",
        model="gemini-live-2.5-flash",
        custom_llm_provider="vertex_ai",
        json={
            "model": "gemini-live-2.5-flash",
            "messages": [
                {"role": "user", "content": "What's the weather like today?"}
            ],
            "stream": True,
            "temperature": 0.7
        }
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

# Run the async function
asyncio.run(async_example())
```

### Router Passthrough

```python
from litellm import Router

# Create router with Vertex AI configuration
router = Router(
    model_list=[
        {
            "model_name": "gemini-live-preview",
            "litellm_params": {
                "model": "vertex_ai/gemini-2.0-flash-live-preview",
                "vertex_project": "your-project-id",
                "vertex_location": "global",  # Use global for Live API
            }
        }
    ]
)

# Use router for passthrough
response = router.allm_passthrough_route(
    method="POST",
    endpoint="live/stream",
    model="gemini-live-preview",
    custom_llm_provider="vertex_ai",
    json={
        "model": "gemini-2.0-flash-live-preview",
        "messages": [
            {"role": "user", "content": "Tell me a short story about AI"}
        ],
        "stream": True
    }
)
```

## Proxy Usage

### 1. Setup config.yaml

```yaml
model_list:
  - model_name: gemini-live-preview
    litellm_params:
      model: vertex_ai/gemini-2.0-flash-live-preview
      vertex_project: your-project-id
      vertex_location: global
```

### 2. Start the Proxy

```bash
litellm proxy --config config.yaml
# RUNNING on http://localhost:4000
```

### 3. Use the Proxy

```bash
curl -X POST http://localhost:4000/vertex_ai/live/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-litellm-api-key>" \
  -d '{
    "model": "gemini-2.0-flash-live-preview",
    "messages": [
      {"role": "user", "content": "Hello from Vertex AI Live API!"}
    ],
    "stream": true
  }'
```

## WebSocket Support

For real-time streaming, you can use WebSocket connections:

```python
import websocket
import json

# WebSocket URL construction
project_id = "your-project-id"
location = "global"  # Use global for Live API
model = "gemini-2.0-flash-live-preview"

websocket_url = f"wss://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:streamRawPredict"

# WebSocket connection (conceptual example)
def on_message(ws, message):
    print(f"Received: {message}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("Connection opened")
    # Send your request here
    ws.send(json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True
    }))

# Create WebSocket connection
ws = websocket.WebSocketApp(
    websocket_url,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)

# Run the WebSocket
ws.run_forever()
```

## Supported Endpoints

The Vertex AI Live API passthrough supports various endpoints:

- `live/stream` - Main Live API streaming endpoint
- `realtime/connect` - Real-time connection endpoint
- `streamRawPredict` - Raw prediction streaming
- Custom endpoints with proper URL construction

## Error Handling

```python
import litellm
from litellm.passthrough.main import llm_passthrough_route

try:
    response = llm_passthrough_route(
        method="POST",
        endpoint="live/stream",
        model="gemini-2.0-flash-live-preview",
        custom_llm_provider="vertex_ai",
        json={
            "model": "gemini-2.0-flash-live-preview",
            "messages": [{"role": "user", "content": "Test"}],
            "stream": True
        }
    )
    
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")
        
except Exception as e:
    print(f"Exception: {str(e)}")
```

## Common Issues and Solutions

### 1. Authentication Errors

**Problem**: `401 Unauthorized` or credential errors

**Solution**: Ensure your service account has proper permissions and credentials are correctly set:

```bash
# Check if credentials are set
echo $GOOGLE_APPLICATION_CREDENTIALS
echo $VERTEXAI_PROJECT

# Test authentication
gcloud auth application-default print-access-token
```

### 2. Project ID Not Found

**Problem**: `Vertex AI project ID is required for Live API`

**Solution**: Set the project ID in environment variables or litellm_params:

```python
# In your code
litellm_params = {
    "vertex_project": "your-project-id",
    "vertex_location": "global"
}
```

### 3. Model Not Supported

**Problem**: Model not found or not supported

**Solution**: Use one of the supported Live API models:

- `gemini-2.0-flash-live-preview`
- `gemini-live-2.5-flash`
- `gemini-live-2.5-flash-preview-native-audio`

### 4. Region Issues

**Problem**: Wrong region for Live API

**Solution**: Use `global` region for Live API:

```python
litellm_params = {
    "vertex_location": "global"  # Use global for Live API
}
```

## Performance Considerations

1. **Latency**: Live API is optimized for low-latency interactions
2. **Streaming**: Use streaming for real-time responses
3. **Connection Pooling**: Reuse connections when possible
4. **Error Handling**: Implement proper retry logic

## Monitoring and Logging

Enable LiteLLM logging to monitor passthrough requests:

```python
import litellm

# Enable verbose logging
litellm.set_verbose = True

# Your passthrough requests will be logged
response = llm_passthrough_route(...)
```

## Best Practices

1. **Use Global Region**: Live API works best with global region
2. **Handle Streaming**: Implement proper streaming response handling
3. **Error Recovery**: Implement retry logic for failed requests
4. **Resource Management**: Close connections properly
5. **Security**: Keep credentials secure and rotate regularly

## Example Integration

See the complete example in `examples/vertex_ai_live_api_passthrough.py` for a full working implementation.

## Support

For issues and questions:

1. Check the [LiteLLM documentation](https://docs.litellm.ai/)
2. Review [Vertex AI Live API documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/live-api)
3. Open an issue on the [LiteLLM GitHub repository](https://github.com/BerriAI/litellm/issues)
