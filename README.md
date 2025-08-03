# Frota de Agentes Autônomos de IA

Este projeto implementa uma frota de agentes de IA autônomos que colaboram para resolver tarefas complexas, combinando RAG (Retrieval-Augmented Generation), Agentic Systems, CrewAI, HuggingFace e LangFlow.

## 🚀 Funcionalidades

- **Agentes Especializados**: Pesquisador, Analista e Executor trabalhando em equipe
- **RAG (Retrieval-Augmented Generation)**: Integração com bancos de dados vetoriais para respostas baseadas em conhecimento
- **Sistema Agêntico**: Tomada de decisão autônoma e colaboração entre agentes
- **Integração com Modelos do HuggingFace**: Suporte a diversos modelos de linguagem
- **Orquestração com CrewAI**: Gerenciamento eficiente de tarefas entre agentes

## 🛠️ Instalação

1. Clone o repositório:
   ```bash
   git clone [URL_DO_REPOSITÓRIO]
   cd project-agent-ai
   ```

2. Crie e ative o ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   .\venv\Scripts\activate  # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Crie um arquivo `.env` na raiz do projeto com suas chaves de API:
   ```
   OPENAI_API_KEY=sua_chave_aqui
   HUGGINGFACEHUB_API_TOKEN=seu_token_aqui
   ```

## 🚦 Como Usar

1. Execute o script principal:
   ```bash
   python -m agent_fleet.main
   ```

2. Ou use o LangFlow para uma interface visual:
   ```bash
   langflow
   ```
   Acesse `http://localhost:7860` no seu navegador.

## 🏗️ Estrutura do Projeto

```
agent_fleet/
├── agents/           # Definições dos agentes
├── tasks/            # Tarefas que os agentes podem executar
├── tools/            # Ferramentas personalizadas para os agentes
├── config/           # Configurações do projeto
├── data/             # Dados e bancos de dados vetoriais
├── models/           # Modelos treinados
└── main.py           # Script principal
```

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e enviar pull requests.

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.