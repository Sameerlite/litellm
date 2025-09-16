"""
Integration tests for Vertex AI Live API passthrough with LiteLLM proxy
"""
import json
import pytest
from unittest.mock import MagicMock, patch, Mock
from httpx import Response, Headers

from litellm.passthrough.main import llm_passthrough_route, allm_passthrough_route
from litellm.types.utils import LlmProviders


class TestVertexAIPassthroughIntegration:
    """Integration tests for Vertex AI Live API passthrough"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_project_id = "test-project-123"
        self.mock_location = "us-central1"
        self.mock_credentials = {"type": "service_account", "project_id": self.mock_project_id}

    @patch('litellm.passthrough.main.litellm.module_level_client')
    @patch('litellm.utils.ProviderConfigManager.get_provider_passthrough_config')
    @patch('litellm.litellm_core_utils.get_litellm_params.get_litellm_params')
    @patch('litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider')
    def test_llm_passthrough_route_vertex_ai_live(self, mock_get_llm_provider, mock_get_litellm_params, 
                                                  mock_get_config, mock_client):
        """Test llm_passthrough_route with Vertex AI Live API"""
        # Setup mocks
        mock_get_llm_provider.return_value = ("gemini-2.0-flash-live-preview", "vertex_ai", None, None)
        mock_get_litellm_params.return_value = {
            "vertex_project": self.mock_project_id,
            "vertex_location": self.mock_location,
            "vertex_credentials": self.mock_credentials
        }
        
        # Mock the Vertex AI passthrough config
        mock_config = MagicMock()
        mock_config.get_complete_url.return_value = (
            "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project-123/locations/us-central1/publishers/google/models/gemini-2.0-flash-live-preview:streamRawPredict",
            "https://us-central1-aiplatform.googleapis.com"
        )
        mock_config.get_api_key.return_value = None
        mock_config.validate_environment.return_value = {
            "Authorization": "Bearer mock_token",
            "Content-Type": "application/json",
            "X-Goog-User-Project": self.mock_project_id
        }
        mock_config.sign_request.return_value = ({}, None)
        mock_config.is_streaming_request.return_value = True
        mock_get_config.return_value = mock_config
        
        # Mock HTTP client response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = Headers({"content-type": "application/json"})
        mock_response.json.return_value = {"response": "test response"}
        mock_client.send.return_value = mock_response
        
        # Mock logging object
        mock_logging_obj = MagicMock()
        mock_logging_obj.update_environment_variables = MagicMock()
        
        # Test the passthrough route
        response = llm_passthrough_route(
            method="POST",
            endpoint="live/stream",
            model="gemini-2.0-flash-live-preview",
            custom_llm_provider="vertex_ai",
            api_base=None,
            api_key=None,
            json={
                "model": "gemini-2.0-flash-live-preview",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            },
            litellm_logging_obj=mock_logging_obj
        )
        
        # Verify the response
        assert response.status_code == 200
        mock_config.get_complete_url.assert_called_once()
        mock_config.validate_environment.assert_called_once()
        mock_config.sign_request.assert_called_once()

    @patch('litellm.passthrough.main.litellm.module_level_aclient')
    @patch('litellm.utils.ProviderConfigManager.get_provider_passthrough_config')
    @patch('litellm.litellm_core_utils.get_litellm_params.get_litellm_params')
    @patch('litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider')
    @pytest.mark.asyncio
    async def test_allm_passthrough_route_vertex_ai_live(self, mock_get_llm_provider, mock_get_litellm_params, 
                                                        mock_get_config, mock_aclient):
        """Test allm_passthrough_route with Vertex AI Live API"""
        # Setup mocks
        mock_get_llm_provider.return_value = ("gemini-2.0-flash-live-preview", "vertex_ai", None, None)
        mock_get_litellm_params.return_value = {
            "vertex_project": self.mock_project_id,
            "vertex_location": self.mock_location,
            "vertex_credentials": self.mock_credentials
        }
        
        # Mock the Vertex AI passthrough config
        mock_config = MagicMock()
        mock_config.get_complete_url.return_value = (
            "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project-123/locations/us-central1/publishers/google/models/gemini-2.0-flash-live-preview:streamRawPredict",
            "https://us-central1-aiplatform.googleapis.com"
        )
        mock_config.get_api_key.return_value = None
        mock_config.validate_environment.return_value = {
            "Authorization": "Bearer mock_token",
            "Content-Type": "application/json",
            "X-Goog-User-Project": self.mock_project_id
        }
        mock_config.sign_request.return_value = ({}, None)
        mock_config.is_streaming_request.return_value = True
        mock_get_config.return_value = mock_config
        
        # Mock async HTTP client response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = Headers({"content-type": "application/json"})
        mock_response.json.return_value = {"response": "test response"}
        mock_response.raise_for_status.return_value = None
        mock_aclient.send.return_value = mock_response
        
        # Mock logging object
        mock_logging_obj = MagicMock()
        mock_logging_obj.update_environment_variables = MagicMock()
        
        # Test the async passthrough route
        response = await allm_passthrough_route(
            method="POST",
            endpoint="live/stream",
            model="gemini-2.0-flash-live-preview",
            custom_llm_provider="vertex_ai",
            api_base=None,
            api_key=None,
            json={
                "model": "gemini-2.0-flash-live-preview",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            },
            litellm_logging_obj=mock_logging_obj
        )
        
        # Verify the response
        assert response.status_code == 200
        mock_config.get_complete_url.assert_called_once()
        mock_config.validate_environment.assert_called_once()
        mock_config.sign_request.assert_called_once()

    def test_vertex_ai_live_api_endpoints(self):
        """Test various Vertex AI Live API endpoints"""
        from litellm.llms.vertex_ai.passthrough.transformation import VertexAIPassthroughConfig
        
        config = VertexAIPassthroughConfig()
        
        # Test different Live API endpoints
        live_endpoints = [
            "live/stream",
            "realtime/connect", 
            "streamRawPredict",
            "v1beta1/projects/{project}/locations/{location}/publishers/google/models/{model}:streamRawPredict"
        ]
        
        for endpoint in live_endpoints:
            # Test streaming detection
            assert config.is_streaming_request(endpoint, {}) == True
            
            # Test with request data
            assert config.is_streaming_request(endpoint, {"stream": True}) == True
            assert config.is_streaming_request(endpoint, {"realtime": True}) == True

    def test_vertex_ai_authentication_flow(self):
        """Test Vertex AI authentication flow for Live API"""
        from litellm.llms.vertex_ai.passthrough.transformation import VertexAIPassthroughConfig
        
        config = VertexAIPassthroughConfig()
        
        # Mock the authentication methods
        with patch.object(config, '_ensure_access_token', return_value=("mock_token", self.mock_project_id)):
            with patch.object(config, 'get_vertex_ai_credentials', return_value=self.mock_credentials):
                with patch.object(config, 'get_vertex_ai_project', return_value=self.mock_project_id):
                    
                    headers = {}
                    litellm_params = {
                        "vertex_project": self.mock_project_id,
                        "vertex_credentials": self.mock_credentials
                    }
                    
                    result_headers = config.validate_environment(
                        headers=headers,
                        model="gemini-2.0-flash-live-preview",
                        messages=[],
                        optional_params={"realtime": True},
                        litellm_params=litellm_params,
                        api_key=None,
                        api_base="https://us-central1-aiplatform.googleapis.com"
                    )
                    
                    # Verify authentication headers
                    assert "Authorization" in result_headers
                    assert result_headers["Authorization"] == "Bearer mock_token"
                    assert "X-Goog-User-Project" in result_headers
                    assert result_headers["X-Goog-User-Project"] == self.mock_project_id

    def test_vertex_ai_model_support(self):
        """Test support for different Vertex AI Live models"""
        from litellm.llms.vertex_ai.passthrough.transformation import VertexAIPassthroughConfig
        
        config = VertexAIPassthroughConfig()
        
        # Test supported Live API models
        live_models = [
            "gemini-2.0-flash-live-preview",
            "gemini-live-2.5-flash",
            "gemini-live-2.5-flash-preview-native-audio",
            "gemini-2.0-flash-live-001"
        ]
        
        for model in live_models:
            # Test URL construction
            with patch.object(config, 'get_vertex_ai_project', return_value=self.mock_project_id):
                with patch.object(config, 'get_vertex_ai_location', return_value=self.mock_location):
                    with patch.object(config, 'get_api_base', return_value="https://us-central1-aiplatform.googleapis.com"):
                        
                        url, base_url = config.get_complete_url(
                            api_base=None,
                            api_key=None,
                            model=model,
                            endpoint="live/stream",
                            request_query_params=None,
                            litellm_params={"vertex_project": self.mock_project_id}
                        )
                        
                        # Verify the URL contains the model name
                        assert model in str(url)
                        assert "streamRawPredict" in str(url)
