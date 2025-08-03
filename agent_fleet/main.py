from crewai import Agent, Task, Crew, Process
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent, AgentType
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from config.settings import settings

# Carrega variáveis de ambiente
load_dotenv()

class AgentFleet:
    def __init__(self):
        # Inicializa o modelo de linguagem
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )
        
        # Inicializa embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        
        # Inicializa agentes
        self.agents = self._initialize_agents()
        
    def _initialize_agents(self) -> Dict[str, Agent]:
        """Inicializa os agentes da frota"""
        # Agente de Pesquisa
        researcher = Agent(
            role='Pesquisador',
            goal='Encontrar e analisar informações relevantes',
            backstory="""Você é um especialista em pesquisa que usa ferramentas avançadas para 
            encontrar informações precisas e relevantes.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=True
        )
        
        # Agente de Análise
        analyst = Agent(
            role='Analista',
            goal='Analisar dados e gerar insights',
            backstory="""Você é um analista especializado em transformar dados em 
            insights acionáveis.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=True
        )
        
        # Agente Executor
        executor = Agent(
            role='Executor',
            goal='Executar tarefas com base nas informações fornecidas',
            backstory="""Você é um executor eficiente que transforma planos em ações concretas.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        return {
            'researcher': researcher,
            'analyst': analyst,
            'executor': executor
        }
    
    def create_crew(self, task_description: str):
        """Cria uma equipe de agentes para uma tarefa específica"""
        # Define as tarefas
        research_task = Task(
            description=f"""Realize uma pesquisa aprofundada sobre: {task_description}
            Certifique-se de coletar informações relevantes e de fontes confiáveis.""",
            agent=self.agents['researcher']
        )
        
        analysis_task = Task(
            description="""Analise as informações coletadas e gere insights valiosos.
            Identifique padrões, tendências e oportunidades.""",
            agent=self.agents['analyst']
        )
        
        execution_task = Task(
            description="""Com base na análise, execute as ações necessárias.
            Forneça um plano detalhado de implementação.""",
            agent=self.agents['executor']
        )
        
        # Cria a equipe
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=[research_task, analysis_task, execution_task],
            verbose=2,
            process=Process.sequential
        )
        
        return crew

if __name__ == "__main__":
    # Exemplo de uso
    fleet = AgentFleet()
    crew = fleet.create_crew("Tecnologias de IA para automação de negócios")
    result = crew.kickoff()
    print("\nResultado final:")
    print(result)
