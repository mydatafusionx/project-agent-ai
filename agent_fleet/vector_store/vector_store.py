import os
import logging
from typing import List, Dict, Any, Optional
from langchain.vectorstores import FAISS, Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from agent_fleet.config.settings import settings
from agent_fleet.models.model_manager import get_model_manager

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Gerenciador de armazenamento vetorial para RAG."""
    
    def __init__(self):
        self.model_manager = get_model_manager()
        self.embeddings = self.model_manager.get_embeddings()
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Inicializa o armazenamento vetorial."""
        try:
            if os.path.exists(settings.VECTOR_STORE_PATH):
                self.vector_store = FAISS.load_local(
                    settings.VECTOR_STORE_PATH, 
                    self.embeddings
                )
                logger.info("Banco de dados vetorial carregado com sucesso.")
            else:
                # Cria um banco de dados vazio
                self.vector_store = FAISS.from_texts(
                    ["Bem-vindo ao sistema de agentes autônomos."],
                    self.embeddings
                )
                self._save_vector_store()
                logger.info("Novo banco de dados vetorial criado.")
        except Exception as e:
            logger.error(f"Erro ao inicializar o banco de dados vetorial: {str(e)}")
            raise
    
    def _save_vector_store(self):
        """Salva o banco de dados vetorial no disco."""
        try:
            os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
            self.vector_store.save_local(settings.VECTOR_STORE_PATH)
        except Exception as e:
            logger.error(f"Erro ao salvar o banco de dados vetorial: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]):
        """Adiciona documentos ao banco de dados vetorial."""
        try:
            if not documents:
                return
                
            # Cria um novo FAISS com os documentos
            new_store = FAISS.from_documents(documents, self.embeddings)
            
            # Se já existir um banco de dados, mescla com o novo
            if self.vector_store is not None:
                self.vector_store.merge_from(new_store)
            else:
                self.vector_store = new_store
            
            # Salva as alterações
            self._save_vector_store()
            logger.info(f"Adicionados {len(documents)} documentos ao banco de dados vetorial.")
            
        except Exception as e:
            logger.error(f"Erro ao adicionar documentos ao banco de dados vetorial: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Realiza uma busca por similaridade no banco de dados vetorial."""
        try:
            if self.vector_store is None:
                logger.warning("Banco de dados vetorial não inicializado.")
                return []
                
            return self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
        except Exception as e:
            logger.error(f"Erro na busca por similaridade: {str(e)}")
            return []
    
    def as_retriever(self, **kwargs):
        """Retorna o banco de dados como um retriever."""
        if self.vector_store is None:
            raise ValueError("Banco de dados vetorial não inicializado.")
        return self.vector_store.as_retriever(**kwargs)

# Instância global
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()
