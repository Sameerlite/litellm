"""
Vertex AI Live API Passthrough Example

This example demonstrates how to use LiteLLM's passthrough functionality
to connect to Google Vertex AI's Realtime (Live) API for Gemini Live models.

Prerequisites:
1. Google Cloud Project with Vertex AI API enabled
2. Service account credentials with Vertex AI permissions
3. LiteLLM installed with Vertex AI support

Environment Setup:
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
export VERTEXAI_PROJECT="your-project-id"
export VERTEXAI_LOCATION="us-central1"  # or "global" for Live API
"""

import asyncio
import json
import os
from typing import Dict, Any

import litellm
from litellm import Router
from litellm.passthrough.main import llm_passthrough_route, allm_passthrough_route


def setup_environment():
    """Setup environment variables for Vertex AI"""
    # Set your Google Cloud credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/service-account-key.json"
    os.environ["VERTEXAI_PROJECT"] = "your-project-id"
    os.environ["VERTEXAI_LOCATION"] = "us-central1"  # Use "global" for Live API


def example_basic_passthrough():
    """Example: Basic passthrough to Vertex AI Live API"""
    print("=== Basic Passthrough Example ===")
    
    # Configure LiteLLM logging
    litellm.set_verbose = True
    
    try:
        # Make a passthrough request to Vertex AI Live API
        response = llm_passthrough_route(
            method="POST",
            endpoint="live/stream",
            model="gemini-2.0-flash-live-preview",
            custom_llm_provider="vertex_ai",
            api_base=None,  # Will use default Vertex AI endpoint
            api_key=None,   # Will use service account credentials
            json={
                "model": "gemini-2.0-flash-live-preview",
                "messages": [
                    {"role": "user", "content": "Hello! Can you help me with a quick question?"}
                ],
                "stream": True,
                "max_tokens": 100
            }
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Response Data: {json.dumps(response_data, indent=2)}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error in basic passthrough: {str(e)}")


async def example_async_passthrough():
    """Example: Async passthrough to Vertex AI Live API"""
    print("\n=== Async Passthrough Example ===")
    
    try:
        # Make an async passthrough request
        response = await allm_passthrough_route(
            method="POST",
            endpoint="realtime/connect",
            model="gemini-live-2.5-flash",
            custom_llm_provider="vertex_ai",
            api_base=None,
            api_key=None,
            json={
                "model": "gemini-live-2.5-flash",
                "messages": [
                    {"role": "user", "content": "What's the weather like today?"}
                ],
                "stream": True,
                "temperature": 0.7
            }
        )
        
        print(f"Async Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Async Response Data: {json.dumps(response_data, indent=2)}")
        else:
            print(f"Async Error: {response.text}")
            
    except Exception as e:
        print(f"Error in async passthrough: {str(e)}")


def example_router_passthrough():
    """Example: Using Router for passthrough requests"""
    print("\n=== Router Passthrough Example ===")
    
    # Create a router with Vertex AI configuration
    router = Router(
        model_list=[
            {
                "model_name": "gemini-live-preview",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.0-flash-live-preview",
                    "vertex_project": os.getenv("VERTEXAI_PROJECT"),
                    "vertex_location": os.getenv("VERTEXAI_LOCATION", "global"),
                }
            }
        ]
    )
    
    try:
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
        
        print(f"Router Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Router Response Data: {json.dumps(response_data, indent=2)}")
        else:
            print(f"Router Error: {response.text}")
            
    except Exception as e:
        print(f"Error in router passthrough: {str(e)}")


def example_different_live_models():
    """Example: Testing different Gemini Live models"""
    print("\n=== Different Live Models Example ===")
    
    # Supported Live API models
    live_models = [
        "gemini-2.0-flash-live-preview",
        "gemini-live-2.5-flash",
        "gemini-live-2.5-flash-preview-native-audio",
        "gemini-2.0-flash-live-001"
    ]
    
    for model in live_models:
        print(f"\nTesting model: {model}")
        
        try:
            response = llm_passthrough_route(
                method="POST",
                endpoint="live/stream",
                model=model,
                custom_llm_provider="vertex_ai",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": f"Hello from {model}!"}
                    ],
                    "stream": True
                }
            )
            
            print(f"  Status: {response.status_code}")
            if response.status_code != 200:
                print(f"  Error: {response.text}")
                
        except Exception as e:
            print(f"  Error: {str(e)}")


def example_websocket_connection():
    """Example: WebSocket connection for real-time streaming"""
    print("\n=== WebSocket Connection Example ===")
    
    # Note: This is a conceptual example. Actual WebSocket implementation
    # would require additional setup for real-time streaming.
    
    print("For WebSocket connections to Vertex AI Live API:")
    print("1. Use the streamRawPredict endpoint")
    print("2. Set up WebSocket connection with proper authentication")
    print("3. Handle real-time message streaming")
    print("4. Implement proper error handling and reconnection logic")
    
    # Example WebSocket URL construction
    project_id = os.getenv("VERTEXAI_PROJECT", "your-project-id")
    location = os.getenv("VERTEXAI_LOCATION", "global")
    model = "gemini-2.0-flash-live-preview"
    
    websocket_url = f"wss://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:streamRawPredict"
    
    print(f"WebSocket URL: {websocket_url}")
    print("Use this URL with a WebSocket client for real-time streaming")


def example_error_handling():
    """Example: Error handling for passthrough requests"""
    print("\n=== Error Handling Example ===")
    
    try:
        # Test with invalid model
        response = llm_passthrough_route(
            method="POST",
            endpoint="live/stream",
            model="invalid-model",
            custom_llm_provider="vertex_ai",
            json={
                "model": "invalid-model",
                "messages": [{"role": "user", "content": "Test"}],
                "stream": True
            }
        )
        
        print(f"Invalid model response: {response.status_code}")
        
    except Exception as e:
        print(f"Expected error for invalid model: {str(e)}")
    
    try:
        # Test with missing credentials
        original_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        
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
        
        print(f"Missing credentials response: {response.status_code}")
        
    except Exception as e:
        print(f"Expected error for missing credentials: {str(e)}")
    finally:
        # Restore credentials
        if original_creds:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = original_creds


def main():
    """Main function to run all examples"""
    print("Vertex AI Live API Passthrough Examples")
    print("======================================")
    
    # Setup environment
    setup_environment()
    
    # Run examples
    example_basic_passthrough()
    
    # Run async example
    asyncio.run(example_async_passthrough())
    
    example_router_passthrough()
    example_different_live_models()
    example_websocket_connection()
    example_error_handling()
    
    print("\n=== Examples Complete ===")
    print("For more information, see the LiteLLM documentation:")
    print("https://docs.litellm.ai/docs/pass_through/vertex_ai")


if __name__ == "__main__":
    main()
