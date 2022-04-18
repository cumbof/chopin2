import os

metadata_path = "ThomasAM_2019_c__metadata.tsv"
dataset_path = "ThomasAM_2019_c__CMD.tsv"
taxa_level = "s__"

"""
# Group 14
selection = [
    "s__Bifidobacterium_longum",
    "s__Eubacterium_sp_CAG_38",
    "s__Mogibacterium_diversum",
    "s__Odoribacter_laneus",
    "s__Oscillibacter_sp_57_20",
    "s__Parabacteroides_distasonis",
    "s__Parabacteroides_johnsonii",
    "s__Prevotella_copri",
    "s__Pseudoflavonifractor_sp_An184",
    "s__Roseburia_sp_CAG_471",
    "s__Ruminococcus_bicirculans",
    "s__Ruminococcus_torques",
    "s__Ruthenibacterium_lactatiformans",
    "s__Streptococcus_oralis",
    "s__Streptococcus_salivarius"
]
"""

metadata = {line.strip().split("\t")[1]: line.strip().split("\t")[5] for line in open(metadata_path).readlines() if line.strip()}

matrix = {}
with open(dataset_path) as m:
    header = []
    for line in m:
        line = line.strip()
        if line:
            if not header:
                header = line.strip().split("\t")[1:]
            else:
                species = line.strip().split("\t")[0].split("|")[-1]
                if species.startswith(taxa_level):
                    profile = line.strip().split("\t")[1:]
                    for i, sample, in enumerate(header):
                        if sample not in matrix:
                            matrix[sample] = {}
                        matrix[sample][species] = profile[i]

len(matrix)

with open("{}__species.txt".format(os.path.splitext(dataset_path)[0]), "w+") as m:
#with open(os.path.join(os.path.dirname(dataset_path), "selection_14", "{}__species.txt".format(os.path.splitext(dataset_path)[0])), "w+") as m:
    species_list = sorted(list(matrix[list(matrix.keys())[0]].keys()))
    #species_list = sorted(selection)
    m.write("Profiles;{};Class\n".format(";".join(species_list)))
    for sample in matrix:
        m.write("{};".format(sample))
        for s in species_list:
            m.write("{};".format(float(matrix[sample][s])))
        m.write("{}\n".format(metadata[sample]))



# Generate random versions of the sub-dataset with the selected features only

import os, random

dataset_path = "./selection_14/ThomasAM_2019_c__species.txt"

classes = []
with open(dataset_path) as m:
    header = []
    for line in m:
        line = line.strip()
        if line:
            if not line.startswith("Profiles"):
                classes.append(line.split(";")[-1])

basepath = "./selection_14_rand/"
rand_count = 10000

while rand_count != 0:
    random.shuffle(classes)
    rand_folder = os.path.join(basepath, "rand_{}".format(rand_count))
    if not os.path.exists(rand_folder):
        os.mkdir(rand_folder)
    print(os.path.join(rand_folder, "ThomasAM_2019_c__species.txt"))
    with open(os.path.join(rand_folder, "ThomasAM_2019_c__species.txt"), "w+") as species:
        with open(dataset_path) as origin:
            sample_count = 0
            for line in origin:
                line = line.strip()
                if line:
                    if line.startswith("Profiles"):
                        species.write("{}\n".format(line))
                    else:
                        line_split = line.split(";")
                        line_split[-1] = classes[sample_count]
                        species.write("{}\n".format(";".join(line_split)))
                        sample_count += 1
    rand_count -= 1

