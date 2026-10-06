import subprocess
import requests
import time

QDRANT_URL = "http://localhost:6333"
DOCKER_IMAGE = "qdrant/qdrant:v1.16.3"
DOCKER_VOLUME = "qdrant_storage"
DOCKER_NAME = "qdrant_auto"

def ensure_qdrant_running():
    try:
        resp = requests.get(QDRANT_URL, timeout=2)
        if resp.status_code == 200:
            return
        else:
            print("Qdrant is NOT running")
    except requests.RequestException:
        print("Qdrant not running, starting Docker container...")

    # Remove old container if exists
    subprocess.run(["docker", "rm", "-f", DOCKER_NAME], stderr=subprocess.DEVNULL)

    # Run detached
    subprocess.run([
        "docker", "run", "-d", "-p", "6333:6333",
        "-v", f"{DOCKER_VOLUME}:/qdrant/storage",
        "--name", DOCKER_NAME,
        DOCKER_IMAGE
    ], check=True)

    # Wait for Qdrant to be ready
    for _ in range(20):
        try:
            resp = requests.get(QDRANT_URL, timeout=2)
            if resp.status_code == 200:
                print("Qdrant is ready")
                return
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError("Failed to start Qdrant after waiting 20 seconds")
