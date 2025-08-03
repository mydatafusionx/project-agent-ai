"""
Frota de Agentes Autônomos

Este pacote implementa um sistema de agentes autônomos que colaboram para resolver tarefas complexas,
combinando RAG (Retrieval-Augmented Generation), Agentic Systems, CrewAI, HuggingFace e LangFlow.
"""

__version__ = "0.1.0"

# Importações principais
from .crew.crew_manager import get_crew_manager
from .vector_store.vector_store import get_vector_store
from .models.model_manager import get_model_manager

# Inicializa os gerenciadores principais
def init():
    """Inicializa os componentes principais do sistema."""
    try:
        # Inicializa o gerenciador de modelos
        model_manager = get_model_manager()
        
        # Inicializa o banco de dados vetorial
        vector_store = get_vector_store()
        
        # Inicializa o gerenciador de equipes
        crew_manager = get_crew_manager()
        
        return {
            'model_manager': model_manager,
            'vector_store': vector_store,
            'crew_manager': crew_manager
        }
    except Exception as e:
        import logging
        logging.error(f"Erro ao inicializar o sistema: {str(e)}")
        raise

# Inicializa o sistema quando o pacote é importado
init()
