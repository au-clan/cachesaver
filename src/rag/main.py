import yaml
import asyncio
from pathlib import Path
from src.rag.eval.experiment_helper import experiment_loop

async def main():
    config_path = 'C:/root/code_repositories/Uni_AU/Semester5/RAG_Project/cachesaver/src/rag/experiments/config.yml'
    config = yaml.safe_load(Path(config_path).read_text())
    await experiment_loop(config=config, verbose=True)
  

if __name__ == '__main__':
    asyncio.run(main())