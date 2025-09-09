# ==============================================================================
# MODULE: CustomGroqLLM.py - CORRECTED VERSION
# ==============================================================================
import re
from typing import Any, List, Optional, Union

from groq import AsyncGroq, Groq, GroqError

from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse, LLMMessage
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.message_history import MessageHistory

class CustomGroqLLM(LLMInterface):
    """
    A fully compliant adapter for neo4j-graphrag that allows the pipeline to use the Groq API.
    It inherits from LLMInterface and implements the required invoke/ainvoke methods.
    """
    def __init__(
        self,
        api_key: str,
        model: str = "llama3-8b-8192",
        model_params: Optional[dict[str, Any]] = None,
    ):
        if not api_key:
            raise ValueError("Groq API key must be provided.")
        self.sync_client = Groq(api_key=api_key)
        self.async_client = AsyncGroq(api_key=api_key)
        self.model_name = model  # Align with LLMInterface's model_name
        self.model_params = model_params or {}
        print(f"  - [CustomGroqLLM Initialized] Model: {self.model_name}")

    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """
        Synchronously sends the input to the Groq API and returns the response.
        """
        messages = [{"role": "user", "content": input}]  # Basic handling; extend for history/system if needed
        
        try:
            completion = self.sync_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
                stream=False,
            )
            
            full_response = completion.choices[0].message.content
            cleaned_response = re.sub(r'^```json\s*|\s*```$', '', full_response, flags=re.MULTILINE | re.DOTALL).strip()
            
            return LLMResponse(content=cleaned_response)
        
        except GroqError as e:
            raise LLMGenerationError(f"A Groq API error occurred: {e}")
        except Exception as e:
            raise LLMGenerationError(f"An unexpected error occurred: {e}")

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """
        Asynchronously sends the input to the Groq API and returns the response.
        """
        messages = [{"role": "user", "content": input}]  # Basic handling; extend for history/system if needed
        
        try:
            completion_stream = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
                stream=True,
            )
            
            full_response = ""
            async for chunk in completion_stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
            
            cleaned_response = re.sub(r'^```json\s*|\s*```$', '', full_response, flags=re.MULTILINE | re.DOTALL).strip()
            
            return LLMResponse(content=cleaned_response)
        
        except GroqError as e:
            raise LLMGenerationError(f"A Groq API error occurred: {e}")
        except Exception as e:
            raise LLMGenerationError(f"An unexpected error occurred: {e}")