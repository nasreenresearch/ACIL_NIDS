import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_per_task(csv_paths, labels, outfile):
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        acc = df[df['metric']=='Top1_Acc_Stream/eval_phase/test_stream']['value']
        plt.plot(range(1, len(acc)+1), acc, marker='o', label=label)
    plt.xlabel("Task")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile)

plot_accuracy_per_task(
    ["results/replay_cicids2017.csv", 
     "results/gem_cicids2017.csv",
     "results/cbrs_cicids2017.csv",
     "results/srr_cicids2017.csv"],
    ["Replay", "GEM", "CBRS", "SRR (ours)"],
    "figures/accuracy_per_task.pdf"
)
