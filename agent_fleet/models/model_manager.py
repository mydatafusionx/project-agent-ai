from typing import Dict, Any, Optional
from langchain_community.llms import HuggingFaceHub, OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.llms.base import BaseLLM
from config.settings import settings, ModelType
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Gerenciador centralizado para modelos de linguagem e embeddings."""
    
    _instance = None
    _models: Dict[str, Any] = {}
    _embeddings: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._initialize_models()
    
    def _initialize_models(self):
        """Inicializa os modelos configurados."""
        for model_id, config in settings.AVAILABLE_MODELS.items():
            try:
                self.get_model(model_id)
                logger.info(f"Modelo {model_id} inicializado com sucesso.")
            except Exception as e:
                logger.error(f"Erro ao inicializar modelo {model_id}: {str(e)}")
    
    def get_model(self, model_id: str) -> BaseLLM:
        """Obtém uma instância do modelo pelo ID."""
        if model_id in self._models:
            return self._models[model_id]
        
        if model_id not in settings.AVAILABLE_MODELS:
            raise ValueError(f"Modelo {model_id} não encontrado nas configurações.")
        
        model_config = settings.AVAILABLE_MODELS[model_id]
        model_type = model_config.get("type")
        
        try:
            if model_type == ModelType.OPENAI:
                self._models[model_id] = OpenAI(
                    model_name=model_config["name"],
                    temperature=model_config.get("temperature", 0.7),
                    max_tokens=model_config.get("max_tokens", 2000),
                    openai_api_key=settings.OPENAI_API_KEY
                )
            elif model_type == ModelType.HUGGINGFACE:
                self._models[model_id] = HuggingFaceHub(
                    repo_id=model_config["name"],
                    model_kwargs={
                        "temperature": model_config.get("temperature", 0.7),
                        "max_length": model_config.get("max_length", 512)
                    },
                    huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
                )
            else:
                raise ValueError(f"Tipo de modelo não suportado: {model_type}")
            
            return self._models[model_id]
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_id}: {str(e)}")
            raise
    
    def get_embeddings(self, model_name: str = None) -> Any:
        """Obtém um modelo de embeddings."""
        model_name = model_name or settings.EMBEDDING_MODEL
        
        if model_name in self._embeddings:
            return self._embeddings[model_name]
        
        try:
            if model_name.startswith("text-embedding"):
                # Modelo da OpenAI
                self._embeddings[model_name] = OpenAIEmbeddings(
                    model=model_name,
                    openai_api_key=settings.OPENAI_API_KEY
                )
            else:
                # Modelo do HuggingFace
                self._embeddings[model_name] = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"}
                )
            
            return self._embeddings[model_name]
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de embeddings {model_name}: {str(e)}")
            raise
    
    def list_available_models(self) -> Dict[str, Dict]:
        """Lista todos os modelos disponíveis."""
        return settings.AVAILABLE_MODELS.copy()

# Instância global
def get_model_manager() -> ModelManager:
    return ModelManager()
