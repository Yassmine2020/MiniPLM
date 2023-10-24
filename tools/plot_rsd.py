import matplotlib.pyplot as plt
import re
import os
import pickle

# path = "/home/aiscuser/sps/results/fairseq/train/pt_rsd/fairseq_125M/t500K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/rsd1.0/"
path = "/home/aiscuser/sps/results/fairseq/train/pt_rsd/fairseq_125M/t500K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/rsd0.0/"
# path = "/home/aiscuser/sps/results/fairseq/train/pt_rsd/fairseq_125M/t500K-bs4-lr0.0003cosine3e-05-G4-N16-NN2-scr/rsd2.0/"


with open(os.path.join(path, "log.txt")) as f:
    lines = f.readlines()

r = r"train.*global_steps (\d+)/.* \| base_lm_loss: (.*) \| total_lm_loss: (.*) \| loss: (.*) \| elasped_time: .*"

steps = []

d = {
    "base_lm_losses": [],
    "total_lm_losses": [],
    "losses": []
}

for line in lines:
    m = re.match(r, line)
    if m is not None:
        steps.append(int(m.group(1)))
        d["base_lm_losses"].append(float(m.group(2)))
        d["total_lm_losses"].append(float(m.group(3)))
        d["losses"].append(float(m.group(4)))

plot_save_path = os.path.join(path, "plot")
os.makedirs(plot_save_path, exist_ok=True)

print(plot_save_path)

for k in d:
    plt.plot(steps, d[k])
    plt.savefig(os.path.join(plot_save_path, f"{k}.png"))
    plt.close()

with open(os.path.join(plot_save_path, "data.pkl"), "wb") as f:
    pickle.dump((steps, d), f)