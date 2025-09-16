from typing import TYPE_CHECKING, List, Optional, Tuple

from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

if TYPE_CHECKING:
    from httpx import URL


class VertexAIPassthroughConfig(VertexBase, BasePassthroughConfig):
    """
    Passthrough configuration for Vertex AI Live API (Realtime API)
    
    This configuration enables passthrough support for Google Vertex AI's 
    Realtime (Live) API, allowing teams to connect to Gemini Live via 
    Vertex AI endpoints without requiring a full unified spec implementation.
    """
    
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        """
        Check if the request is a streaming request for Vertex AI Live API
        
        Vertex AI Live API uses WebSocket connections for real-time streaming,
        so we check for streaming indicators in the endpoint or request data.
        """
        # Check for streaming indicators in endpoint
        if any(keyword in endpoint.lower() for keyword in ["stream", "live", "realtime"]):
            return True
            
        # Check for streaming indicators in request data
        if request_data.get("stream", False):
            return True
            
        # Check for realtime/live specific parameters
        if any(keyword in request_data for keyword in ["realtime", "live"]):
            return True
            
        return False

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        endpoint: str,
        request_query_params: Optional[dict],
        litellm_params: dict,
    ) -> Tuple["URL", str]:
        """
        Get the complete URL for Vertex AI Live API requests
        
        For Vertex AI Live API, we need to:
        1. Determine the correct region (global for Live API)
        2. Build the proper endpoint URL
        3. Handle WebSocket vs HTTP endpoints
        """
        # Extract Vertex AI specific parameters
        vertex_project = self.get_vertex_ai_project(litellm_params)
        vertex_location = self.get_vertex_ai_location(litellm_params)
        vertex_credentials = self.get_vertex_ai_credentials(litellm_params)
        
        # For Live API, we typically use global region
        if vertex_location is None:
            vertex_location = "global"
            
        # Get the base API URL
        base_target_url = self.get_api_base(
            api_base=api_base, 
            vertex_location=vertex_location
        )
        
        if base_target_url is None:
            raise Exception("Vertex AI api base not found")
            
        # For Live API, we need to construct the proper endpoint
        # The Live API uses a different URL structure than regular Vertex AI
        if "live" in endpoint.lower() or "realtime" in endpoint.lower():
            # Live API endpoint structure
            if vertex_project is None:
                raise Exception("Vertex AI project ID is required for Live API")
                
            # Construct Live API endpoint
            live_endpoint = f"v1beta1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:streamRawPredict"
            if not endpoint.startswith("/"):
                endpoint = "/" + endpoint
            if not endpoint.startswith("/v1beta1"):
                endpoint = "/" + live_endpoint
        else:
            # Regular Vertex AI endpoint
            if not endpoint.startswith("/"):
                endpoint = "/" + endpoint
                
        return (
            self.format_url(endpoint, base_target_url, request_query_params),
            base_target_url,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str],
        api_base: str,
    ) -> dict:
        """
        Validate and set up authentication for Vertex AI Live API
        
        This method handles authentication for both regular Vertex AI
        and Live API endpoints using the same credential system.
        """
        # Get Vertex AI credentials and project
        vertex_credentials = self.get_vertex_ai_credentials(litellm_params)
        vertex_project = self.get_vertex_ai_project(litellm_params)
        
        # Get access token using the base class method
        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        
        # Set up headers for Vertex AI
        headers.update({
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        })
        
        # Add any additional headers for Live API
        if "live" in model.lower() or "realtime" in str(optional_params).lower():
            headers.update({
                "X-Goog-User-Project": project_id,
            })
        
        return headers

    def get_api_key(self, api_key: Optional[str]) -> Optional[str]:
        """
        Get API key for Vertex AI
        
        For Vertex AI, we don't use traditional API keys but rather
        OAuth2 access tokens obtained through the credential system.
        """
        # Vertex AI uses OAuth2 tokens, not API keys
        # The actual authentication is handled in validate_environment
        return None

    def get_models(self, api_key: Optional[str] = None, api_base: Optional[str] = None) -> List[str]:
        """
        Returns a list of models supported by Vertex AI Live API
        """
        return [
            "gemini-2.0-flash-live-preview",
            "gemini-live-2.5-flash",
            "gemini-live-2.5-flash-preview-native-audio",
            "gemini-2.0-flash-live-001"
        ]

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        """
        Returns the base model name from the given model name
        """
        # For Live API models, return the model name as is
        if "live" in model.lower() or "realtime" in model.lower():
            return model
        return None
