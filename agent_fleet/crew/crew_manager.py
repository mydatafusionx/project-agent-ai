from typing import Dict, List, Optional, Any
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from crewai import Agent, Task, Crew
from agent_fleet.models.model_manager import get_model_manager
from agent_fleet.vector_store.vector_store import get_vector_store
import logging

logger = logging.getLogger(__name__)

class CrewManager:
    """Gerenciador da frota de agentes e suas interações."""
    
    def __init__(self):
        self.model_manager = get_model_manager()
        self.vector_store = get_vector_store()
        self.agents: Dict[str, Any] = {}
        self.tasks: Dict[str, Any] = {}
        self.crews: Dict[str, Any] = {}
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Inicializa os agentes padrão da frota."""
        # Agente de Pesquisa
        self.add_agent(
            agent_id="researcher",
            role="Pesquisador Sênior",
            goal="Encontrar e analisar informações relevantes de forma precisa",
            backstory="""Você é um especialista em pesquisa que usa ferramentas avançadas para 
            encontrar informações precisas e relevantes em diversas fontes.""",
            model_name="gpt-4"
        )
        
        # Agente de Análise
        self.add_agent(
            agent_id="analyst",
            role="Analista de Dados Sênior",
            goal="Analisar dados e gerar insights valiosos",
            backstory="""Você é um analista especializado em transformar dados complexos em 
            insights acionáveis e compreensíveis.""",
            model_name="gpt-4"
        )
        
        # Agente Executor
        self.add_agent(
            agent_id="executor",
            role="Especialista em Implementação",
            goal="Executar tarefas com base nas informações fornecidas",
            backstory="""Você é um executor eficiente que transforma planos e instruções em 
            ações concretas e resultados mensuráveis.""",
            model_name="gpt-4"
        )
    
    def add_agent(self, agent_id: str, role: str, goal: str, backstory: str, 
                 model_name: str = "gpt-4", tools: Optional[List[BaseTool]] = None,
                 verbose: bool = True) -> Agent:
        """Adiciona um novo agente à frota."""
        if agent_id in self.agents:
            logger.warning(f"Agente com ID '{agent_id}' já existe. Atualizando...")
        
        llm = self.model_manager.get_model(model_name)
        
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=verbose,
            llm=llm,
            allow_delegation=True,
            tools=tools or []
        )
        
        self.agents[agent_id] = agent
        return agent
    
    def add_task(self, task_id: str, description: str, agent_id: str, 
                expected_output: str = "", async_execution: bool = False) -> Task:
        """Adiciona uma nova tarefa à frota."""
        if agent_id not in self.agents:
            raise ValueError(f"Agente com ID '{agent_id}' não encontrado.")
        
        task = Task(
            description=description,
            agent=self.agents[agent_id],
            expected_output=expected_output or description,
            async_execution=async_execution
        )
        
        self.tasks[task_id] = task
        return task
    
    def create_crew(self, crew_id: str, task_ids: List[str], 
                   verbose: int = 2, process: str = "sequential") -> Crew:
        """Cria uma nova equipe de agentes para executar tarefas."""
        tasks = [self.tasks[task_id] for task_id in task_ids 
                if task_id in self.tasks]
        
        if not tasks:
            raise ValueError("Nenhuma tarefa válida fornecida.")
        
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            verbose=verbose,
            process=process
        )
        
        self.crews[crew_id] = crew
        return crew
    
    def run_crew(self, crew_id: str, inputs: Optional[Dict] = None) -> str:
        """Executa uma equipe de agentes."""
        if crew_id not in self.crews:
            raise ValueError(f"Equipe com ID '{crew_id}' não encontrada.")
        
        try:
            result = self.crews[crew_id].kickoff(inputs=inputs)
            return result
        except Exception as e:
            logger.error(f"Erro ao executar a equipe {crew_id}: {str(e)}")
            raise
    
    def add_tool_to_agent(self, agent_id: str, tool: BaseTool):
        """Adiciona uma ferramenta a um agente existente."""
        if agent_id not in self.agents:
            raise ValueError(f"Agente com ID '{agent_id}' não encontrado.")
        
        self.agents[agent_id].tools.append(tool)
    
    def get_agent(self, agent_id: str) -> Agent:
        """Obtém um agente pelo ID."""
        if agent_id not in self.agents:
            raise ValueError(f"Agente com ID '{agent_id}' não encontrado.")
        return self.agents[agent_id]
    
    def get_task(self, task_id: str) -> Task:
        """Obtém uma tarefa pelo ID."""
        if task_id not in self.tasks:
            raise ValueError(f"Tarefa com ID '{task_id}' não encontrada.")
        return self.tasks[task_id]
    
    def get_crew(self, crew_id: str) -> Crew:
        """Obtém uma equipe pelo ID."""
        if crew_id not in self.crews:
            raise ValueError(f"Equipe com ID '{crew_id}' não encontrada.")
        return self.crews[crew_id]

# Instância global
def get_crew_manager() -> CrewManager:
    return CrewManager()
