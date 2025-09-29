"""Model wrapper for automatic token tracking."""

import time
from typing import Any, Dict, List, Optional, Generator
from smolagents.models import Model, ChatMessage

from .token_tracker import TokenTracker

# Import LiteLLM for accurate tokenization
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class TrackedModel:
    """Wrapper around any model to automatically track token usage."""
    
    def __init__(self, 
                 base_model: Model, 
                 token_tracker: TokenTracker,
                 agent_name: str = "unknown"):
        """Initialize tracked model.
        
        Args:
            base_model: The base model to wrap
            token_tracker: Token tracker instance
            agent_name: Name of the agent using this model
        """
        self.base_model = base_model
        self.token_tracker = token_tracker
        self.agent_name = agent_name
        
        # Forward all attributes to base model
        for attr in dir(base_model):
            if not attr.startswith('_') and attr not in ['generate', 'generate_stream', '__call__']:
                setattr(self, attr, getattr(base_model, attr))
    
    def _extract_content_from_messages(self, messages: List[ChatMessage]) -> str:
        """Extract text content from messages for logging."""
        contents = []
        for msg in messages:
            if hasattr(msg, 'content'):
                if isinstance(msg.content, str):
                    contents.append(msg.content)
                elif isinstance(msg.content, list):
                    for content_item in msg.content:
                        if isinstance(content_item, dict) and content_item.get('type') == 'text':
                            contents.append(content_item.get('text', ''))
                        else:
                            contents.append(str(content_item))
                else:
                    contents.append(str(msg.content))
        return '\n'.join(contents)
    
    def _extract_content_from_response(self, response: ChatMessage) -> str:
        """Extract text content from response for logging."""
        if hasattr(response, 'content'):
            if isinstance(response.content, str):
                return response.content
            elif isinstance(response.content, list):
                contents = []
                for content_item in response.content:
                    if isinstance(content_item, dict) and content_item.get('type') == 'text':
                        contents.append(content_item.get('text', ''))
                    else:
                        contents.append(str(content_item))
                return '\n'.join(contents)
        return str(response)
    
    def _get_tokenizer_model(self, model_id: str) -> str:
        """Map model ID to appropriate tokenizer model for LiteLLM."""
        model_lower = model_id.lower()
        
        # GPT OSS models -> use gpt-4 tokenizer
        if "gpt-oss" in model_lower:
            return "gpt-4"
        
        # Qwen models -> use qwen tokenizer
        if "qwen" in model_lower:
            return "qwen3"
        
        # DeepSeek models -> use deepseek tokenizer
        if "deepseek" in model_lower:
            return "deepseek-chat"
        
        # Claude models -> use claude tokenizer
        if "claude" in model_lower:
            return "claude-3-sonnet-20240229"
        
        # Default fallback to gpt-4 for unknown models
        return "gpt-4"
    
    def generate(self, messages: List[ChatMessage], **kwargs) -> ChatMessage:
        """Generate response and track token usage."""
        start_time = time.time()
        
        # Extract input content for logging
        input_content = self._extract_content_from_messages(messages)
        
        # Call the base model
        response = self.base_model.generate(messages, **kwargs)
        
        end_time = time.time()
        call_duration = end_time - start_time
        
        # Extract output content for logging
        output_content = self._extract_content_from_response(response)
        
        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        model_id = getattr(self.base_model, 'model_id', str(type(self.base_model).__name__))
        
        # Helper function to estimate tokens from content (including reasoning)
        def estimate_tokens_from_content():
            # Check for reasoning content in GPT OSS models
            total_output_content = output_content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'reasoning_content'):
                    reasoning_content = getattr(choice.message, 'reasoning_content', '')
                    if reasoning_content:
                        total_output_content += f" {reasoning_content}"
            
            if LITELLM_AVAILABLE:
                try:
                    # Map model IDs to appropriate tokenizers
                    tokenizer_model = self._get_tokenizer_model(model_id)
                    
                    # Use LiteLLM's tokenizer for accurate counting
                    estimated_input = max(1, litellm.token_counter(model=tokenizer_model, text=input_content))
                    estimated_output = max(1, litellm.token_counter(model=tokenizer_model, text=total_output_content))
                    return estimated_input, estimated_output
                except Exception as e:
                    # Fallback to character-based estimation if LiteLLM fails
                    print(f"🔍 DEBUG: LiteLLM tokenizer failed for {model_id}: {e}")
            
            # Fallback: character-based estimation (original method)
            estimated_input = max(1, len(input_content) // 4)
            estimated_output = max(1, len(total_output_content) // 4)
            return estimated_input, estimated_output
        
        # Try to get token usage from response
        if hasattr(response, 'token_usage') and response.token_usage:
            input_tokens = response.token_usage.input_tokens
            output_tokens = response.token_usage.output_tokens
            
            # If tokens are 0, check raw response or estimate
            if input_tokens == 0 and output_tokens == 0:
                if hasattr(response, 'raw') and hasattr(response.raw, 'usage'):
                    raw_usage = response.raw.usage
                    if raw_usage.prompt_tokens > 0 or raw_usage.completion_tokens > 0:
                        input_tokens = raw_usage.prompt_tokens
                        output_tokens = raw_usage.completion_tokens
                    else:
                        input_tokens, output_tokens = estimate_tokens_from_content()
                        # [Bug]: Litellm does not work with AWS Bedrock Application Inference profiles #8911
                        tokenizer_method = "LiteLLM tokenizer" if LITELLM_AVAILABLE else "char-based estimation"
                        print(f"🔍 DEBUG: {model_id} tokens estimated using {tokenizer_method} (API returned 0) | Litellm bug #8911")
                else:
                    input_tokens, output_tokens = estimate_tokens_from_content()
                    tokenizer_method = "LiteLLM tokenizer" if LITELLM_AVAILABLE else "char-based estimation"
                    print(f"🔍 DEBUG: {model_id} tokens estimated using {tokenizer_method} (no raw usage)")
        else:
            # Fallback to model's internal tracking
            if hasattr(self.base_model, '_last_input_token_count'):
                input_tokens = getattr(self.base_model, '_last_input_token_count', 0)
            if hasattr(self.base_model, '_last_output_token_count'):
                output_tokens = getattr(self.base_model, '_last_output_token_count', 0)
            
            # If still 0, estimate from content
            if input_tokens == 0 and output_tokens == 0:
                input_tokens, output_tokens = estimate_tokens_from_content()
                tokenizer_method = "LiteLLM tokenizer" if LITELLM_AVAILABLE else "char-based estimation"
                print(f"🔍 DEBUG: {model_id} tokens estimated using {tokenizer_method} (no token_usage)")
        
        # Track the call
        model_id = getattr(self.base_model, 'model_id', str(type(self.base_model).__name__))
        
        self.token_tracker.track_model_call(
            model_id=model_id,
            agent_name=self.agent_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_content=input_content,
            output_content=output_content,
            call_duration=call_duration,
            metadata={
                'method': 'generate',
                'kwargs': str(kwargs)
            }
        )
        
        return response
    
    def generate_stream(self, messages: List[ChatMessage], **kwargs) -> Generator:
        """Generate streaming response and track token usage."""
        start_time = time.time()
        
        # Extract input content for logging
        input_content = self._extract_content_from_messages(messages)
        
        # Call the base model
        stream = self.base_model.generate_stream(messages, **kwargs)
        
        # Collect streaming output
        output_content = ""
        input_tokens = 0
        output_tokens = 0
        
        try:
            for chunk in stream:
                if hasattr(chunk, 'content') and chunk.content:
                    output_content += chunk.content
                
                # Try to get token usage from chunk if available
                if hasattr(chunk, 'token_usage') and chunk.token_usage:
                    input_tokens = chunk.token_usage.input_tokens
                    output_tokens = chunk.token_usage.output_tokens
                
                yield chunk
                
        finally:
            end_time = time.time()
            call_duration = end_time - start_time
            
            # Fallback to model's internal tracking if we didn't get tokens from stream
            if input_tokens == 0 and hasattr(self.base_model, '_last_input_token_count'):
                input_tokens = getattr(self.base_model, '_last_input_token_count', 0)
            if output_tokens == 0 and hasattr(self.base_model, '_last_output_token_count'):
                output_tokens = getattr(self.base_model, '_last_output_token_count', 0)
            
            # Track the call
            model_id = getattr(self.base_model, 'model_id', str(type(self.base_model).__name__))
            
            self.token_tracker.track_model_call(
                model_id=model_id,
                agent_name=self.agent_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_content=input_content,
                output_content=output_content,
                call_duration=call_duration,
                metadata={
                    'method': 'generate_stream',
                    'kwargs': str(kwargs)
                }
            )
    
    def __call__(self, messages: List[ChatMessage], **kwargs) -> ChatMessage:
        """Make the model callable and track token usage."""
        return self.generate(messages, **kwargs)
    
    def __getattr__(self, name):
        """Forward any other attribute access to the base model."""
        return getattr(self.base_model, name)
