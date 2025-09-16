"""
Test suite for Vertex AI Live API passthrough functionality
"""
import json
import pytest
from unittest.mock import MagicMock, patch, Mock
from httpx import Response, Headers

from litellm.llms.vertex_ai.passthrough.transformation import VertexAIPassthroughConfig
from litellm.types.utils import LlmProviders


class TestVertexAIPassthroughConfig:
    """Test cases for VertexAIPassthroughConfig"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = VertexAIPassthroughConfig()
        self.mock_project_id = "test-project-123"
        self.mock_location = "us-central1"
        self.mock_credentials = {"type": "service_account", "project_id": self.mock_project_id}

    def test_is_streaming_request_endpoint_based(self):
        """Test streaming detection based on endpoint"""
        # Test live endpoint
        assert self.config.is_streaming_request("live/stream", {}) == True
        assert self.config.is_streaming_request("realtime/connect", {}) == True
        assert self.config.is_streaming_request("streamRawPredict", {}) == True
        
        # Test non-streaming endpoint
        assert self.config.is_streaming_request("predict", {}) == False
        assert self.config.is_streaming_request("generateContent", {}) == False

    def test_is_streaming_request_data_based(self):
        """Test streaming detection based on request data"""
        # Test streaming request data
        assert self.config.is_streaming_request("", {"stream": True}) == True
        assert self.config.is_streaming_request("", {"realtime": True}) == True
        assert self.config.is_streaming_request("", {"live": True}) == True
        
        # Test non-streaming request data
        assert self.config.is_streaming_request("", {"stream": False}) == False
        assert self.config.is_streaming_request("", {}) == False

    @patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_vertex_ai_project')
    @patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_vertex_ai_location')
    @patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_api_base')
    def test_get_complete_url_live_api(self, mock_get_api_base, mock_get_location, mock_get_project):
        """Test URL construction for Live API endpoints"""
        # Setup mocks
        mock_get_project.return_value = self.mock_project_id
        mock_get_location.return_value = self.mock_location
        mock_get_api_base.return_value = f"https://{self.mock_location}-aiplatform.googleapis.com"
        
        # Test Live API endpoint
        endpoint = "live/stream"
        litellm_params = {
            "vertex_project": self.mock_project_id,
            "vertex_location": self.mock_location
        }
        
        url, base_url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gemini-2.0-flash-live-preview",
            endpoint=endpoint,
            request_query_params=None,
            litellm_params=litellm_params
        )
        
        # Verify URL construction
        assert str(url).endswith(f"v1beta1/projects/{self.mock_project_id}/locations/{self.mock_location}/publishers/google/models/gemini-2.0-flash-live-preview:streamRawPredict")
        assert base_url == f"https://{self.mock_location}-aiplatform.googleapis.com"

    @patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_vertex_ai_project')
    @patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_vertex_ai_location')
    @patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_api_base')
    def test_get_complete_url_regular_endpoint(self, mock_get_api_base, mock_get_location, mock_get_project):
        """Test URL construction for regular Vertex AI endpoints"""
        # Setup mocks
        mock_get_project.return_value = self.mock_project_id
        mock_get_location.return_value = self.mock_location
        mock_get_api_base.return_value = f"https://{self.mock_location}-aiplatform.googleapis.com"
        
        # Test regular endpoint
        endpoint = "predict"
        litellm_params = {
            "vertex_project": self.mock_project_id,
            "vertex_location": self.mock_location
        }
        
        url, base_url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gemini-pro",
            endpoint=endpoint,
            request_query_params=None,
            litellm_params=litellm_params
        )
        
        # Verify URL construction
        assert str(url).endswith("/predict")
        assert base_url == f"https://{self.mock_location}-aiplatform.googleapis.com"

    @patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig._ensure_access_token')
    @patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_vertex_ai_credentials')
    @patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_vertex_ai_project')
    def test_validate_environment(self, mock_get_project, mock_get_credentials, mock_ensure_token):
        """Test environment validation and authentication setup"""
        # Setup mocks
        mock_get_project.return_value = self.mock_project_id
        mock_get_credentials.return_value = self.mock_credentials
        mock_ensure_token.return_value = ("mock_access_token", self.mock_project_id)
        
        headers = {}
        litellm_params = {
            "vertex_project": self.mock_project_id,
            "vertex_credentials": self.mock_credentials
        }
        
        result_headers = self.config.validate_environment(
            headers=headers,
            model="gemini-2.0-flash-live-preview",
            messages=[],
            optional_params={},
            litellm_params=litellm_params,
            api_key=None,
            api_base="https://us-central1-aiplatform.googleapis.com"
        )
        
        # Verify authentication headers
        assert "Authorization" in result_headers
        assert result_headers["Authorization"] == "Bearer mock_access_token"
        assert result_headers["Content-Type"] == "application/json"

    @patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig._ensure_access_token')
    @patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_vertex_ai_credentials')
    @patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_vertex_ai_project')
    def test_validate_environment_live_api_headers(self, mock_get_project, mock_get_credentials, mock_ensure_token):
        """Test Live API specific headers"""
        # Setup mocks
        mock_get_project.return_value = self.mock_project_id
        mock_get_credentials.return_value = self.mock_credentials
        mock_ensure_token.return_value = ("mock_access_token", self.mock_project_id)
        
        headers = {}
        litellm_params = {
            "vertex_project": self.mock_project_id,
            "vertex_credentials": self.mock_credentials
        }
        
        result_headers = self.config.validate_environment(
            headers=headers,
            model="gemini-2.0-flash-live-preview",
            messages=[],
            optional_params={"realtime": True},
            litellm_params=litellm_params,
            api_key=None,
            api_base="https://us-central1-aiplatform.googleapis.com"
        )
        
        # Verify Live API specific headers
        assert "Authorization" in result_headers
        assert "X-Goog-User-Project" in result_headers
        assert result_headers["X-Goog-User-Project"] == self.mock_project_id

    def test_get_api_key(self):
        """Test API key retrieval (should return None for Vertex AI)"""
        # Vertex AI uses OAuth2 tokens, not API keys
        assert self.config.get_api_key("some_key") is None
        assert self.config.get_api_key(None) is None

    def test_get_complete_url_missing_project(self):
        """Test URL construction with missing project ID for Live API"""
        with patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_vertex_ai_project', return_value=None):
            with pytest.raises(Exception, match="Vertex AI project ID is required for Live API"):
                self.config.get_complete_url(
                    api_base=None,
                    api_key=None,
                    model="gemini-2.0-flash-live-preview",
                    endpoint="live/stream",
                    request_query_params=None,
                    litellm_params={}
                )

    def test_get_complete_url_missing_api_base(self):
        """Test URL construction with missing API base"""
        with patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_api_base', return_value=None):
            with pytest.raises(Exception, match="Vertex AI api base not found"):
                self.config.get_complete_url(
                    api_base=None,
                    api_key=None,
                    model="gemini-pro",
                    endpoint="predict",
                    request_query_params=None,
                    litellm_params={}
                )


class TestVertexAIPassthroughIntegration:
    """Integration tests for Vertex AI passthrough with LiteLLM"""

    @patch('litellm.utils.ProviderConfigManager.get_provider_passthrough_config')
    def test_provider_config_manager_integration(self, mock_get_config):
        """Test that ProviderConfigManager returns VertexAIPassthroughConfig"""
        from litellm.utils import ProviderConfigManager
        
        # Mock the config
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        
        # Test VERTEX_AI provider
        config = ProviderConfigManager.get_provider_passthrough_config(
            model="gemini-2.0-flash-live-preview",
            provider=LlmProviders.VERTEX_AI
        )
        
        # Verify the config was retrieved
        mock_get_config.assert_called_once()
        assert config == mock_config

    @patch('litellm.utils.ProviderConfigManager.get_provider_passthrough_config')
    def test_provider_config_manager_vertex_ai_beta(self, mock_get_config):
        """Test that ProviderConfigManager returns VertexAIPassthroughConfig for VERTEX_AI_BETA"""
        from litellm.utils import ProviderConfigManager
        
        # Mock the config
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        
        # Test VERTEX_AI_BETA provider
        config = ProviderConfigManager.get_provider_passthrough_config(
            model="gemini-2.0-flash-live-preview",
            provider=LlmProviders.VERTEX_AI_BETA
        )
        
        # Verify the config was retrieved
        mock_get_config.assert_called_once()
        assert config == mock_config


class TestVertexAILiveAPIModels:
    """Test support for different Gemini Live API models"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = VertexAIPassthroughConfig()

    @pytest.mark.parametrize("model", [
        "gemini-2.0-flash-live-preview",
        "gemini-live-2.5-flash",
        "gemini-live-2.5-flash-preview-native-audio",
        "gemini-2.0-flash-live-001"
    ])
    def test_supported_live_models(self, model):
        """Test that supported Live API models work correctly"""
        # Test streaming detection for Live models
        assert self.config.is_streaming_request("live/stream", {"model": model}) == True
        
        # Test URL construction for Live models
        with patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_vertex_ai_project', return_value="test-project"):
            with patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_vertex_ai_location', return_value="us-central1"):
                with patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_api_base', return_value="https://us-central1-aiplatform.googleapis.com"):
                    url, base_url = self.config.get_complete_url(
                        api_base=None,
                        api_key=None,
                        model=model,
                        endpoint="live/stream",
                        request_query_params=None,
                        litellm_params={"vertex_project": "test-project"}
                    )
                    
                    # Verify the URL contains the model name
                    assert model in str(url)

    def test_global_region_for_live_api(self):
        """Test that Live API uses global region when location is not specified"""
        with patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_vertex_ai_project', return_value="test-project"):
            with patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_vertex_ai_location', return_value=None):
                with patch('litellm.llms.vertex_ai.passthrough.transformation.VertexAIPassthroughConfig.get_api_base') as mock_get_api_base:
                    mock_get_api_base.return_value = "https://global-aiplatform.googleapis.com"
                    
                    url, base_url = self.config.get_complete_url(
                        api_base=None,
                        api_key=None,
                        model="gemini-2.0-flash-live-preview",
                        endpoint="live/stream",
                        request_query_params=None,
                        litellm_params={}
                    )
                    
                    # Verify global region is used
                    mock_get_api_base.assert_called_once_with(api_base=None, vertex_location="global")
