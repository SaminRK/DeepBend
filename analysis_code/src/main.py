from chromosome.chromosome import Chromosome
from chromosome.nucleosomes import Nucleosomes

if __name__ == "__main__":
    Nucleosomes(Chromosome("VL")).plot_c0_vs_dist_from_dyad_spread()
