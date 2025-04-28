import subprocess
import sys

def image_exists(image_name):
    """Check if a Docker image already exists."""
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return bool(result.stdout.strip())

def build_image(image_name="webshop", data_size="small"):
    """Build the Docker image."""
    print(f"Building Docker image '{image_name}' with data size '{data_size}'...")
    subprocess.run([
        "docker", "build",
        "--build-arg", f"DATA_SIZE={data_size}",
        "-t", image_name,
        "."
    ], check=True)

def run_container(image_name="webshop"):
    """Run the Docker container."""
    print(f"Running Docker container from image '{image_name}'...")
    subprocess.run([
        "docker", "run", image_name
    ], check=True)

if __name__ == "__main__":
    IMAGE_NAME = "webshop"
    DATA_SIZE = "small"  # for the small dataset pr the docs

    if image_exists(IMAGE_NAME):
        print(f"Image '{IMAGE_NAME}' already exists. Skipping build.")
    else:
        build_image(IMAGE_NAME, DATA_SIZE)

    run_container(IMAGE_NAME)