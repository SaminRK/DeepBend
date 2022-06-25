import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils.dinucleotide import mono_to_dinucleotide, dinucleotide_one_hot_encode
import pandas as pd


class preprocess:
    def __init__(self, file, shortened=False, seqns=None, c0=None):
        self.file = file
        self.shortened = shortened

        if file:
            self.df = pd.read_table(filepath_or_buffer=file, )
            
            self.fw_sequences = []
            if self.shortened:
                self.fw_sequences = self.df['Sequence']
            else:
                self.fw_sequences = [s[25:-25] for s in self.df['Sequence']]
        else:
            self.fw_sequences = seqns
            self.c0 = c0


    def read_fasta_into_list(self):
        return self.fw_sequences

    def read_fasta_forward(self):
        return self.read_fasta_into_list()

    # augment the samples with reverse complement
    def rc_comp2(self):

        def rc_comp(seq):
            rc_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
            rc_seq = ''.join([rc_dict[c] for c in seq[::-1]])
            return rc_seq

        seqn = self.read_fasta_into_list()
        all_sequences = []
        for seq in range(len(seqn)):
            all_sequences.append(rc_comp(seqn[seq]))

        return all_sequences

    # to augment on readout data
    def read_readout(self):
        if not self.file:
            return self.c0
        if self.shortened:
            if 'C0' in self.df.columns:
                return self.df['C0'].to_numpy()
            return None
        if ' C0' in self.df.columns:
            return self.df[' C0'].to_numpy()
        return None

    def augment(self):
        new_fasta = self.read_fasta_into_list()
        rc_fasta = self.rc_comp2()
        readout = self.read_readout()

        dict = {
            "new_fasta": new_fasta,
            "readout": readout,
            "rc_fasta": rc_fasta}
        return dict

    def without_augment(self):
        new_fasta = self.read_fasta_into_list()
        readout = self.read_readout()

        dict = {"new_fasta": new_fasta, "readout": readout}
        return dict

    def get_one_hot_encoded_dataset(self):
        dict = self.augment()
        dict["forward"] = np.asarray(one_hot_encode_sequences(dict["new_fasta"]))
        dict["reverse"] = np.asarray(one_hot_encode_sequences(dict["rc_fasta"]))

        return dict

    def get_dinucleotide_encoded_dataset(self):
        new_fasta = self.read_fasta_into_list()
        rc_fasta = self.rc_comp2()
        forward_sequences = mono_to_dinucleotide(new_fasta)
        reverse_sequences = mono_to_dinucleotide(rc_fasta)

        forward = np.asarray(dinucleotide_one_hot_encode(forward_sequences))
        reverse = np.asarray(dinucleotide_one_hot_encode(reverse_sequences))

        dict = {}
        dict["readout"] = self.read_readout()
        dict["forward"] = forward
        dict["reverse"] = reverse
        return dict

def one_hot_encode_sequences(seqs):
    # The LabelEncoder encodes a sequence of bases as a sequence of
    # integers.
    integer_encoder = LabelEncoder()
    # The OneHotEncoder converts an array of integers to a sparse matrix where
    # each row corresponds to one possible value of each feature.
    one_hot_encoder = OneHotEncoder(categories='auto')

    one_hot_encoded_seqs = []
    for seq in seqs:
        seq = seq + 'ACGT'
        integer_encoded = integer_encoder.fit_transform(list(seq))
        integer_encoded = np.array(integer_encoded).reshape(-1, 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
        one_hot_encoded_seqs.append(one_hot_encoded.toarray()[:-4])
    
    return one_hot_encoded_seqs


class Genome:
    def __init__(self, chromosome):
        valid_chromosomes = ['I', 'II', 'III', 'IV', 'V', 'VI']
        rel_file_path = ''
        if valid_chromosomes.count(chromosome) > 0:
            rel_file_path = f'../data/genome/chr{chromosome}.fsa'
            with open(rel_file_path, 'r') as f:
                segs = f.readlines()
            segs = [s.strip() for s in segs]
            self.chrom = ''.join(segs)
        elif all(c in "ATCG" for c in chromosome):
            self.chrom = chromosome
        else:
            raise Exception('Unknown chromosome number or sequence')

    def read_chromosome(self):
        return self.chrom
    
    def one_hot_encode(self):
        return one_hot_encode_sequences([self.chrom])[0]
        
    def rc_comp(self):
        rc_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
        rc_seq = ''.join([rc_dict[c] for c in self.chrom[::-1]])
        return rc_seq

    def augment(self):
        fw_one_hot_encoded = one_hot_encode_sequences([self.chrom])[0]
        rc_one_hot_encoded = one_hot_encode_sequences([self.rc_comp()])[0]

        return fw_one_hot_encoded, rc_one_hot_encoded


def get_dataset(dataset_filename, encoding):
    prep = preprocess(dataset_filename)
    if encoding == "dinucleotide":
        return prep.get_dinucleotide_encoded_dataset()
    return prep.get_one_hot_encoded_dataset()
