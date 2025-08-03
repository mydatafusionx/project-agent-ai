#!/usr/bin/env python3
"""
Ponto de entrada principal para a Frota de Agentes Autônomos.

Este script inicia a interface web da aplicação.
"""
import os
import sys
import logging
from pathlib import Path

# Configura o caminho para o diretório raiz do projeto
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(ROOT_DIR))

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT_DIR / 'agent_fleet.log')
    ]
)

def main():
    """Função principal para iniciar a aplicação."""
    try:
        # Verifica se o arquivo .env existe
        if not (ROOT_DIR / '.env').exists():
            logging.error("Arquivo .env não encontrado. Por favor, crie um arquivo .env com as configurações necessárias.")
            sys.exit(1)
        
        # Importa o Streamlit e configura para rodar o app
        import subprocess
        import webbrowser
        import time
        
        # Define o caminho para o arquivo app.py
        app_path = str(ROOT_DIR / 'agent_fleet' / 'web' / 'app.py')
        
        # Inicia o servidor Streamlit em uma porta específica
        port = 8501
        url = f"http://localhost:{port}"
        
        # Tenta abrir o navegador após um pequeno delay
        def open_browser():
            time.sleep(2)
            webbrowser.open(url)
        
        # Inicia o Streamlit em um processo separado
        import threading
        threading.Thread(target=open_browser).start()
        
        # Executa o comando Streamlit
        cmd = ["streamlit", "run", app_path, "--server.port", str(port), "--server.headless", "false", "--browser.gatherUsageStats", "false"]
        subprocess.run(cmd)
        
    except ImportError as e:
        logging.error(f"Erro ao importar módulos necessários: {str(e)}")
        logging.info("Certifique-se de que todas as dependências estão instaladas corretamente.")
        logging.info("Execute: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Erro ao iniciar a aplicação: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
