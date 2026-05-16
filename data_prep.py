import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torchvision import transforms
from tqdm import tqdm

print("Loading environment variables...", load_dotenv())

ds = load_dataset("immanuelpeter/carla-autopilot-multimodal-dataset", split="train", streaming=True)

if __name__ == "__main__":
    j=0
    a_sum = torch.zeros(3)
    a_min = torch.zeros(3)
    a_max = torch.zeros(3)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),])
    im_sum = 0
    im_2_sum = 0
    runids = []
    for d in tqdm(ds):
        if d["run_id"] not in runids:
            i=-1
            print("\nnew run",d["run_id"])
            runids.append(d["run_id"])

        assert i<d["frame"], "data must be sorted"
        i = d["frame"]
        a_t = torch.tensor([d["throttle"],d["steer"],d["brake"]])
        a_sum += a_t
        a_max = torch.maximum(a_max,a_t)
        a_min = torch.minimum(a_max,a_t)
        j +=1

        im = transform(d["image_front"])
        im_sum += im.mean([-2,-1])
        im_2_sum += im.pow(2).mean([-2,-1])

    print(runids,end="\n*************\n")
    print("action",a_min,a_sum/j,a_max,sep="\n",end="\n*************\n")
    im_mean = im_sum/j
    print("frame",im_mean, ((im_2_sum/j-im_mean.pow(2))).pow(0.5), sep="\n*************\n")

else:
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