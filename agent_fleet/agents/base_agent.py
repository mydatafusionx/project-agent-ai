from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.agents import Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from config.settings import settings

class BaseAgent(ABC):
    """Classe base para todos os agentes da frota."""
    
    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str,
        model_name: str = "gpt-4",
        tools: Optional[List[Tool]] = None,
        verbose: bool = False
    ):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.model_name = model_name
        self.tools = tools or []
        self.verbose = verbose
        self.memory = self._initialize_memory()
        self.agent = self._create_agent()
        self.agent_executor = self._create_agent_executor()
    
    def _initialize_memory(self) -> ConversationBufferMemory:
        """Inicializa a memória do agente."""
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
    
    @abstractmethod
    def _create_agent(self):
        """Método abstrato para criar o agente específico."""
        pass
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Cria o executor do agente."""
        return AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            max_iterations=settings.MAX_ITERATIONS,
            handle_parsing_errors=True
        )
    
    def run(self, input_data: str, **kwargs) -> str:
        """Executa o agente com a entrada fornecida."""
        try:
            result = self.agent_executor.invoke({"input": input_data, **kwargs})
            return result.get("output", "Nenhuma saída gerada.")
        except Exception as e:
            return f"Erro ao executar o agente {self.name}: {str(e)}"
    
    def add_tool(self, tool: Tool):
        """Adiciona uma ferramenta ao agente."""
        self.tools.append(tool)
        self.agent_executor = self._create_agent_executor()
    
    def get_memory(self) -> List[BaseMessage]:
        """Obtém o histórico de mensagens da memória."""
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        """Limpa a memória do agente."""
        self.memory.clear()
        self.memory = self._initialize_memory()
        self.agent_executor = self._create_agent_executor()


class ResearchAgent(BaseAgent):
    """Agente especializado em pesquisa e coleta de informações."""
    
    def _create_agent(self):
        from langchain.agents import initialize_agent
        from langchain.agents import AgentType
        from models.model_manager import get_model_manager
        
        llm = get_model_manager().get_model(self.model_name)
        
        return initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
            agent_kwargs={
                "prefix": f"""Você é {self.name}, {self.role}.
                Seu objetivo: {self.goal}
                
                Histórico:
                {self.backstory}
                
                Use as ferramentas fornecidas para realizar sua tarefa.
                """,
                "suffix": """
                Use as ferramentas para realizar a pesquisa. Seja minucioso e detalhado.
                
                Pergunta: {input}
                {agent_scratchpad}"""
            }
        )


class AnalysisAgent(BaseAgent):
    """Agente especializado em análise de dados e geração de insights."""
    
    def _create_agent(self):
        from langchain.agents import initialize_agent
        from langchain.agents import AgentType
        from models.model_manager import get_model_manager
        
        llm = get_model_manager().get_model(self.model_name)
        
        return initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
            agent_kwargs={
                "prefix": f"""Você é {self.name}, {self.role}.
                Seu objetivo: {self.goal}
                
                Histórico:
                {self.backstory}
                
                Analise os dados fornecidos e gere insights valiosos.""",
                "suffix": """
                Use as ferramentas para analisar os dados. Seja analítico e preciso.
                
                Dados para análise: {input}
                {agent_scratchpad}"""
            }
        )


class ExecutionAgent(BaseAgent):
    """Agente especializado em executar ações com base em instruções."""
    
    def _create_agent(self):
        from langchain.agents import initialize_agent
        from langchain.agents import AgentType
        from models.model_manager import get_model_manager
        
        llm = get_model_manager().get_model(self.model_name)
        
        return initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
            agent_kwargs={
                "prefix": f"""Você é {self.name}, {self.role}.
                Seu objetivo: {self.goal}
                
                Histórico:
                {self.backstory}
                
                Execute as tarefas de forma eficiente e precisa.""",
                "suffix": """
                Use as ferramentas para executar a tarefa. Seja direto e eficiente.
                
                Tarefa: {input}
                {agent_scratchpad}"""
            }
        )
