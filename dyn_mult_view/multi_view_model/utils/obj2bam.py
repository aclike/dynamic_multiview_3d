import os, sys

"""
usage: python obj2bam.py <synset_dir>
"""
import dyn_mult_view
basedir = '/'.join(str.split(dyn_mult_view.__file__, '/')[:-1])

OBJ2EGG_CMD = 'python ' + basedir + '/multi_view_model/utils/obj2egg.py {1}/model_normalized.obj'
EGG2BAM_CMD = 'egg2bam -o {0}/model.bam {0}/model.egg'
CP_CMD = 'cp {0}/model.bam {0}/../{1}.bam'

full_cmd = '; '.join((OBJ2EGG_CMD, EGG2BAM_CMD, CP_CMD))

if __name__ == '__main__':
    synset_dir = sys.argv[1]
    for model_id in os.listdir(synset_dir):
        print('converting %s (output file: %s)' % (model_id, synset_dir + '/' + model_id + '.bam'))
        model_path = os.path.join(synset_dir, model_id, 'models')
        os.system(full_cmd.format(model_path, model_id))
