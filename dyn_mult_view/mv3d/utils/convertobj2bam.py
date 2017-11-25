import os
import sys

path = "../shapenet/test"

model_names = os.listdir(path)
for name in model_names:
    model_path = os.path.join(path, name, "models")
    os.system("obj2egg %s/model_normalized.obj %s/model.egg; egg2bam %s/model.egg %s/model.bam; cp %s/model.bam %s/.." % tuple([model_path] * 6))