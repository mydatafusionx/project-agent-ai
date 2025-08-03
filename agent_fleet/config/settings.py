from pydantic_settings import BaseSettings
from typing import Dict, List, Optional
from enum import Enum

class ModelType(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

class Settings(BaseSettings):
    # Configurações de Autenticação
    OPENAI_API_KEY: str = ""
    HUGGINGFACEHUB_API_TOKEN: str = ""
    
    # Configurações do Sistema
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Configurações do Banco de Dados Vetorial
    VECTOR_STORE_PATH: str = "./data/vector_store"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Configurações dos Modelos
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIM: int = 768
    
    # Modelos disponíveis
    AVAILABLE_MODELS: Dict[str, Dict] = {
        "gpt-4": {
            "type": ModelType.OPENAI,
            "name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000
        },
        "llama2-7b": {
            "type": ModelType.HUGGINGFACE,
            "name": "meta-llama/Llama-2-7b-chat-hf",
            "temperature": 0.7,
            "max_length": 2000
        },
        "flan-t5-xxl": {
            "type": ModelType.HUGGINGFACE,
            "name": "google/flan-t5-xxl",
            "temperature": 0.7,
            "max_length": 2000
        }
    }
    
    # Configurações da Interface Web
    STREAMLIT_PORT: int = 8501
    STREAMLIT_THEME: str = "light"
    
    # Configurações dos Agentes
    AGENT_TIMEOUT: int = 300  # segundos
    MAX_ITERATIONS: int = 10
    
    # Configurações de Memória
    SHORT_TERM_MEMORY_LIMIT: int = 10  # itens
    LONG_TERM_MEMORY: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = "ignore"

# Instância de configuração
settings = Settings()
