import subprocess
import os

def main():
    folder = '/ceph/scratch/romanp/tiny-imagenet-200'

    subprocess.run(['mv %s/val %s/old_val' % (folder, folder)], shell=True)
    subprocess.run(['mv %s/train %s/old_train' % (folder, folder)], shell=True)

    with open('%s/old_val/val_annotations.txt' % folder, 'r') as annotations_file:
        for line in annotations_file.read().split('\n'):
            if len(line) > 0:
                image, class_id = line.split()[:2]
                subprocess.run(['mkdir -p %s/val/%s' % (folder, class_id)], shell=True)
                subprocess.run(['cp %s/old_val/images/%s %s/val/%s/' % (folder, image, folder, class_id)], shell=True)

                if not os.path.isdir('%s/train/%s' % (folder, class_id)):
                    subprocess.run(['mkdir -p %s/train/%s' % (folder, class_id)], shell=True)
                    subprocess.run(['cp %s/old_train/%s/images/* %s/train/%s/' %
                                    (folder, class_id, folder, class_id)], shell=True)



if __name__ == '__main__':
    main()
