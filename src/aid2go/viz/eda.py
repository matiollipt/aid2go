import matplotlib.pyplot as plt
from pathlib import Path

def plot_ppi_degree(degrees, save_path: Path):
    plt.figure()
    plt.hist(degrees, bins=50)
    plt.title("PPI Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.savefig(save_path)
    plt.close()

def plot_go_frequency(degrees, save_path: Path):
    plt.figure()
    plt.hist(degrees, bins=50)
    plt.title("GO Term Frequency")
    plt.xlabel("Degree (Annotations)")
    plt.ylabel("Count")
    plt.savefig(save_path)
    plt.close()
