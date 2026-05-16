from datasets import load_dataset
from dotenv import load_dotenv

# from tqdm import tqdm

print("Loading environment variables...", load_dotenv())

ds = load_dataset("immanuelpeter/carla-autopilot-multimodal-dataset", split="validation", streaming=True)

# ds = ds.sort(["run_id","frame"])
# runids = []
# for d in tqdm(ds):
#     if d["run_id"] not in runids:
#         i=-1
#         print("\nnew run",d["run_id"])
#         runids.append(d["run_id"])

#     assert i<d["frame"], "data must be sorted"
#     i = d["frame"]

# print(runids)
frames = {}
for i,d in enumerate(ds):
    if i==0:
        frames["frame_t"]=d["image_front"]
        frames["action"]=[d["throttle"],d["steer"],d["brake"]]
    elif i==1:
        frames["frame_tp1"]=d["image_front"]
    else:
        break
print(frames)