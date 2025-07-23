from vllm.entrypoints.cli.main import main as vllm_main
from vllm.logger import init_logger
logger = init_logger(__name__)

def main():
    logger.warning('vllm.scripts.main() is deprecated. Please re-install vllm or use vllm.entrypoints.cli.main.main() instead.')
    vllm_main()