import streamlit as st
import os
import logging
from datetime import datetime
from typing import Dict, Any, List

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Título da aplicação
st.set_page_config(
    page_title="Frota de Agentes Autônomos",
    page_icon="🤖",
    layout="wide"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .agent-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f5f5;
        margin-bottom: 15px;
        border-left: 5px solid #1E88E5;
    }
    .message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #E3F2FD;
        margin-left: auto;
        margin-right: 0;
    }
    .agent-message {
        background-color: #f5f5f5;
        margin-right: auto;
        margin-left: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Classe para gerenciar o estado da sessão
class SessionState:
    def __init__(self):
        self.messages = []
        self.agents = {}
        self.active_agent = None
        self.conversation_history = []
        self.vector_store_initialized = False

# Inicializa o estado da sessão
if 'state' not in st.session_state:
    st.session_state.state = SessionState()

def initialize_agents():
    """Inicializa os agentes e ferramentas."""
    try:
        from agent_fleet.crew.crew_manager import get_crew_manager
        from agent_fleet.vector_store.vector_store import get_vector_store
        
        # Inicializa o gerenciador de equipes
        crew_manager = get_crew_manager()
        
        # Adiciona ferramentas ao agente de pesquisa
        research_tools = [
            # Adicione aqui as ferramentas de pesquisa
        ]
        
        # Atualiza os agentes no estado da sessão
        st.session_state.state.agents = {
            'researcher': {
                'name': 'Pesquisador',
                'description': 'Especialista em encontrar e analisar informações',
                'tools': research_tools
            },
            'analyst': {
                'name': 'Analista',
                'description': 'Especialista em análise de dados e geração de insights',
                'tools': []
            },
            'executor': {
                'name': 'Executor',
                'description': 'Especialista em executar tarefas e implementar soluções',
                'tools': []
            }
        }
        
        st.session_state.state.vector_store_initialized = True
        st.success("Agentes inicializados com sucesso!")
        
    except Exception as e:
        st.error(f"Erro ao inicializar agentes: {str(e)}")
        logger.error(f"Erro ao inicializar agentes: {str(e)}", exc_info=True)

def display_chat():
    """Exibe o histórico de mensagens do chat."""
    st.markdown("<div class='main-title'>🤖 Frota de Agentes Autônomos</div>", unsafe_allow_html=True)
    
    # Barra lateral para seleção de agente
    with st.sidebar:
        st.header("🔧 Configurações")
        
        # Seletor de agente
        agent_options = {info['name']: agent_id for agent_id, info in st.session_state.state.agents.items()}
        selected_agent_name = st.selectbox(
            "Selecione um agente:",
            options=list(agent_options.keys()),
            index=0
        )
        st.session_state.state.active_agent = agent_options[selected_agent_name]
        
        # Exibe informações do agente selecionado
        if st.session_state.state.active_agent:
            agent_info = st.session_state.state.agents[st.session_state.state.active_agent]
            st.markdown(f"### {agent_info['name']}")
            st.markdown(f"**Descrição:** {agent_info['description']}")
            
            # Exibe ferramentas disponíveis
            if agent_info.get('tools'):
                st.markdown("**Ferramentas:**")
                for tool in agent_info['tools']:
                    st.markdown(f"- {tool.name}")
        
        # Botão para limpar histórico
        if st.button("Limpar Conversa"):
            st.session_state.state.conversation_history = []
            st.experimental_rerun()
    
    # Exibe o histórico da conversa
    st.markdown("### 💬 Conversa")
    
    for message in st.session_state.state.conversation_history:
        if message['role'] == 'user':
            st.markdown(f"<div class='message user-message'><strong>Você:</strong> {message['content']}</div>", 
                       unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='message agent-message'><strong>{message['agent']}:</strong> {message['content']}</div>", 
                       unsafe_allow_html=True)
    
    # Entrada do usuário
    user_input = st.text_area("Digite sua mensagem:", key="user_input", height=100)
    
    if st.button("Enviar") and user_input:
        # Adiciona a mensagem do usuário ao histórico
        st.session_state.state.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Processa a mensagem com o agente selecionado
        process_agent_response(user_input)
        
        # Limpa a entrada do usuário
        st.session_state.user_input = ""
        st.experimental_rerun()

def process_agent_response(user_input: str):
    """Processa a entrada do usuário e obtém a resposta do agente."""
    try:
        from crew.crew_manager import get_crew_manager
        
        agent_id = st.session_state.state.active_agent
        if not agent_id:
            st.error("Nenhum agente selecionado.")
            return
        
        # Obtém o gerenciador de equipes
        crew_manager = get_crew_manager()
        
        # Cria uma tarefa para o agente
        task_id = f"task_{len(st.session_state.state.conversation_history)}"
        crew_manager.add_task(
            task_id=task_id,
            description=user_input,
            agent_id=agent_id,
            expected_output="Resposta detalhada e útil para o usuário."
        )
        
        # Cria uma equipe com um único agente para esta tarefa
        crew_id = f"crew_{task_id}"
        crew_manager.create_crew(
            crew_id=crew_id,
            task_ids=[task_id]
        )
        
        # Executa a tarefa
        response = crew_manager.run_crew(crew_id=crew_id)
        
        # Adiciona a resposta ao histórico
        st.session_state.state.conversation_history.append({
            'role': 'assistant',
            'agent': st.session_state.state.agents[agent_id]['name'],
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = f"Erro ao processar a solicitação: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg, exc_info=True)
        
        # Adiciona a mensagem de erro ao histórico
        st.session_state.state.conversation_history.append({
            'role': 'assistant',
            'agent': 'Sistema',
            'content': f"❌ {error_msg}",
            'timestamp': datetime.now().isoformat()
        })

def main():
    # Verifica se os agentes foram inicializados
    if not st.session_state.state.vector_store_initialized:
        with st.spinner('Inicializando agentes...'):
            initialize_agents()
    
    # Exibe a interface do chat
    display_chat()

if __name__ == "__main__":
    main()
