import os, sys

"""
usage: python obj2bam.py <synset_dir>
"""
import dyn_mult_view
basedir = '/'.join(str.split(dyn_mult_view.__file__, '/')[:-1])

OBJ2EGG_CMD = 'python ' + basedir + '/multi_view_model/utils/obj2egg.py {0}/model.obj'
# OBJ2EGG_CMD = 'python /Users/owenjow/dynamic_multiview_3d/dyn_mult_view/multi_view_model/utils/obj2egg.py {0}/model_normalized.obj'
MV_CMD = 'mv {0}/model.egg {0}/../{1}.egg'
EGG2BAM_CMD = 'egg2bam -o {0}/../{1}.bam {0}/../{1}.egg'
RM_CMD = 'rm {0}/../{1}.egg'

full_cmd = '; '.join((OBJ2EGG_CMD, MV_CMD, EGG2BAM_CMD))
# full_cmd = '; '.join((OBJ2EGG_CMD, MV_CMD, EGG2BAM_CMD, RM_CMD))

if __name__ == '__main__':
    synset_dir = sys.argv[1]
    for model_id in os.listdir(synset_dir):
        print('converting %s (output file: %s)' % (model_id, synset_dir + '/' + model_id + '.bam'))
        model_path = os.path.join(synset_dir, model_id)
        os.system(OBJ2EGG_CMD.format(model_path))
        os.system(MV_CMD.format(model_path, model_id))
        os.system(EGG2BAM_CMD.format(model_path, model_id))
        os.system(RM_CMD.format(model_path,model_id))
