from pathlib import Path
from typing import Literal
import os

def on_startup(command: Literal['build', 'gh-deploy', 'serve'], dirty: bool):
    if os.getenv('READTHEDOCS_VERSION_TYPE') == 'tag':
        mkdocs_dir = Path(__file__).parent.parent
        announcement_path = mkdocs_dir / 'overrides/main.html'
        if announcement_path.exists():
            os.remove(announcement_path)