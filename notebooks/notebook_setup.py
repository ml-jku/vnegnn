import os

import rootutils
from dotenv import load_dotenv


def setup_notebook():
    """Setup notebook environment with proper
    path handling and environment variables.
    """
    notebook_dir = os.getcwd()

    root_path = rootutils.setup_root(
        notebook_dir, indicator=".project-root", pythonpath=True
    )
    os.chdir(root_path)

    env_path = os.path.join(root_path, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)

    return root_path
