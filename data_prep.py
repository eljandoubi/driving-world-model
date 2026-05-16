from datasets import load_dataset
from dotenv import load_dotenv

print("Loading environment variables...", load_dotenv())

ds = load_dataset("immanuelpeter/carla-autopilot-multimodal-dataset", split="test")

ds = ds.sort(["run_id","frame"])

print(ds)


