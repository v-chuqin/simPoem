import cPickle as pkl
import gzip
import pandas as pd
import numpy as np


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

def remove_tags_used_char_mem(previous_source_seq, reference,
                     worddicts_r, 
                     reference_mask, reference_word_mask):
    new_reference_mask = np.copy(reference_mask)
    new_reference_word_mask = np.copy(reference_word_mask)
    for i in range(reference.shape[1]):
        previous_characters = previous_source_seq[1 : -2, i]
        for j in range(reference.shape[0]):
            flag = False
            for k in range(reference.shape[2]):
                if reference[j][i][k] in previous_characters or flag:
                    new_reference_word_mask[j][i][k] = 0.
                    new_reference_mask[j][i] = 0.
                    flag = True
    
    return new_reference_mask, new_reference_word_mask


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    continue

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target


class TestTextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source,
                 source_dict,
                 batch_size=1,
                 maxlen=100,
                 n_words_source=-1):
        self.source = fopen(source, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                temp = ss.strip().split('\t')
                for i in range(len(temp)):
                    ss = temp[i].split()
                    ss = [self.source_dict[w] if w in self.source_dict else 1
                          for w in ss]
                    if self.n_words_source > 0:
                        ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index

                    if len(ss) > self.maxlen:
                        continue

                    source.append(ss)

                if len(source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source


class TextKeywordIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target, keyword, 
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        self.keyword = fopen(keyword, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        self.keyword.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        keyword = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]
                    
                kk = self.keyword.readline()
                if kk == "":
                    raise IOError
                kk = kk.strip().split(' ')
                kk = [self.source_dict[k] if k in self.source_dict else 1
                        for k in kk]
                if self.n_words_source > 0:
                    kk = [k if k < self.n_words_source else 1 for k in kk]

                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    continue

                source.append(ss)
                target.append(tt)
                keyword.append(kk)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size or \
                        len(keyword) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0 or len(keyword) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target, keyword


class TestTextKeywordIterator:
    """Simple Bitext iterator."""
    def __init__(self, source,
                 source_dict,
                 batch_size=1,
                 maxlen=100,
                 n_words_source=-1):
        self.source = fopen(source, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        imageids = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                temp = ss.strip().split('@')
                imageid = temp[0]
                ss = temp[1]
                temp_list = []
                temp = ss.strip().split('\t')
                for i in range(len(temp)):
                    ss = temp[i].split(' ')
                    ss = [self.source_dict[w] if w in self.source_dict else 1
                          for w in ss]
                    if self.n_words_source > 0:
                        ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index

                    if len(ss) > self.maxlen:
                        continue

                    temp_list.append(ss)
                source.append(temp_list)
                imageids.append(imageid)

                if len(source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, imageids

class ImageTextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 feature_dir, fc7_dir,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1):
        self.fc7_dir = fc7_dir
        self.feature_dir = feature_dir
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        images = []
        fc7_images = []
        imageids = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                temp = ss.strip().split('@')
                image_id = temp[0]
                ss = temp[1].split()
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    continue

                source.append(ss)
                target.append(tt)
                imageids.append(image_id)

                image_file = self.feature_dir + '/' + image_id + '.npy'
                fc7_file = self.fc7_dir + '/' + image_id + '.npy'
                image = np.load(image_file)
                # image = image.reshape(image.shape[0], image.shape[1] * image.shape[2])
                fc7_image = np.load(fc7_file)
                images.append(image)
                fc7_images.append(fc7_image)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target, images, fc7_images, imageids
    

class TestImageTextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source,
                 feature_dir, fc7_dir,
                 source_dict,
                 batch_size=1,
                 maxlen=100,
                 n_words_source=-1):
        self.fc7_dir = fc7_dir
        self.feature_dir = feature_dir
        self.source = fopen(source, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        images = []
        fc7_images = []
        imageids = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                temp = ss.strip().split('@')
                image_id = temp[0]

                imageids.append(image_id)

                image_file = self.feature_dir + '/' + image_id + '.npy'
                fc7_file = self.fc7_dir + '/' + image_id + '.npy'
                image = np.load(image_file)
                # image = image.reshape(image.shape[0], image.shape[1] * image.shape[2])
                fc7_image = np.load(fc7_file)
                images.append(image)
                fc7_images.append(fc7_image)

                if len(images) >= self.batch_size and len(fc7_images) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(images) <= 0 or len(fc7_images) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return images, fc7_images, imageids

class ImageTextMemIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 reference,
                 feature_dir, fc7_dir,
                 source_dict, target_dict,
                 reference_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1, 
                 n_words_reference = -1):
        self.fc7_dir = fc7_dir
        self.feature_dir = feature_dir
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        self.reference = fopen(reference, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)
        with open(reference_dict, 'rb') as f:
            self.reference_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target
        self.n_words_reference = n_words_reference

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        self.reference.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        reference = []
        target = []
        images = []
        fc7_images = []
        imageids = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                temp = ss.strip().split('@')
                image_id = temp[0]
                ss = temp[1].split()
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]
                
                rr = self.reference.readline()
                if rr == "":
                    raise IOError
                rr = rr.strip().split('@')[1]
                rr = rr.strip().split()
                rr = [self.reference_dict[r] if r in self.reference_dict else 1
                        for r in rr]
                if self.n_words_reference > 0:
                    temp = []
                    for r in rr:
                        if r < self.n_words_reference:
                            temp.append(r)
                    rr = temp
                # r_list = []
                # for i in rr:
                #     temp = i.strip().split()
                #     temp = [self.source_dict[r] if r in self.source_dict else 1
                #             for r in temp]

                #     if self.n_words_source > 0:
                #         temp = [t if t < self.n_words_source else 1 for t in temp]
                #     r_list.append(temp)
                # if len(ss) > self.maxlen and len(tt) > self.maxlen:
                #     continue

                source.append(ss)
                target.append(tt)
                reference.append(rr)
                imageids.append(image_id)

                image_file = self.feature_dir + '/' + image_id + '.npy'
                fc7_file = self.fc7_dir + '/' + image_id + '.npy'
                image = np.load(image_file)
                # image = image.reshape(image.shape[0], image.shape[1] * image.shape[2])
                fc7_image = np.load(fc7_file)
                images.append(image)
                fc7_images.append(fc7_image)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target, images, fc7_images, imageids, reference


class TestImageTextMemIterator:
    """Simple Bitext iterator."""
    def __init__(self, source,
                 feature_dir, fc7_dir,
                 source_dict,
                 reference_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_reference = -1):
        self.fc7_dir = fc7_dir
        self.feature_dir = feature_dir
        self.source = fopen(source, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(reference_dict, 'rb') as f:
            self.reference_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_reference = n_words_reference

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        reference = []
        images = []
        fc7_images = []
        imageids = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                rr = self.source.readline()
                if rr == "":
                    raise IOError
                temp = rr.strip().split('@')
                image_id = temp[0]
                rr = temp[1].split()
                rr = [self.reference_dict[w] if w in self.reference_dict else 1
                      for w in rr]
                if self.n_words_reference > 0:
                    rr = [w if w < self.n_words_reference else 1 for w in rr]

                if self.n_words_reference > 0:
                    temp = []
                    for r in rr:
                        if r < self.n_words_reference:
                            temp.append(r)
                    rr = temp

                reference.append(rr)
                imageids.append(image_id)

                image_file = self.feature_dir + '/' + image_id + '.npy'
                fc7_file = self.fc7_dir + '/' + image_id + '.npy'
                image = np.load(image_file)
                # image = image.reshape(image.shape[0], image.shape[1] * image.shape[2])
                fc7_image = np.load(fc7_file)
                images.append(image)
                fc7_images.append(fc7_image)

                if len(reference) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(reference) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return images, fc7_images, imageids, reference

class ImageTextCharMemIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 reference,
                 feature_dir, fc7_dir,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1, skip = 0, phase = 'image'): 
        self.fc7_dir = fc7_dir
        self.feature_dir = feature_dir
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        self.reference = fopen(reference, 'r')
        for i in range(skip):
            self.source.readline()
            self.target.readline()
            self.reference.readline()
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.end_of_data = False
        self.phase = phase

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        self.reference.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        reference = []
        target = []
        images = []
        fc7_images = []
        imageids = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                temp = ss.strip().split('@')
                image_id = temp[0]
                ss = temp[1].split()
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]
                
                rr = self.reference.readline()
                if rr == "":
                    raise IOError
                rr = rr.strip().split('@')[1]
                rr = rr.strip().split('\t')
                r_list = []
                for r in rr:
                    temp_list = []
                    temp = r.split(' ')
                    flag = False
                    for t in temp:
                        if t in self.source_dict:
                            if self.source_dict[t] < self.n_words_source:
                                temp_list.append(self.source_dict[t])
                                flag = True
                            else:
                                temp_list.append(1)
                    if flag:
                        r_list.append(temp_list)

                source.append(ss)
                target.append(tt)
                reference.append(r_list)
                imageids.append(image_id)

                if 'image' == self.phase:
                    image_file = self.feature_dir + '/' + image_id + '.npy'
                    fc7_file = self.fc7_dir + '/' + image_id + '.npy'
                    image = np.load(image_file)
                    # image = image.reshape(image.shape[0], image.shape[1] * image.shape[2])
                    fc7_image = np.load(fc7_file)
                else:
                    image = np.zeros((512, 196)).astype('float32')
                    fc7_image = np.zeros((4096)).astype('float32')
                images.append(image)
                fc7_images.append(fc7_image)


                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target, images, fc7_images, imageids, reference
    

class TestImageTextCharMemIterator:
    """Simple Bitext iterator."""
    def __init__(self, source,
                 feature_dir, fc7_dir,
                 source_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1, phase = 'image'):
        self.fc7_dir = fc7_dir
        self.feature_dir = feature_dir
        self.source = fopen(source, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source

        self.end_of_data = False
        self.phase = phase

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        reference = []
        images = []
        fc7_images = []
        imageids = []

        try:

            # actual work here
            while True:
                rr = self.source.readline()
                if rr == "":
                    raise IOError
                temp = rr.strip().split('@')
                image_id = temp[0]
                rr = temp[1]
                rr = rr.strip().split('\t')
                r_list = []
                for r in rr:
                    temp_list = []
                    temp = r.split(' ')
                    flag = False
                    for t in temp:
                        if t in self.source_dict:
                            if self.source_dict[t] < self.n_words_source:
                                temp_list.append(self.source_dict[t])
                                flag = True
                            else:
                                temp_list.append(1)
                    if flag:
                        r_list.append(temp_list)

                reference.append(r_list)
                imageids.append(image_id)

                if 'image' == self.phase:
                    image_file = self.feature_dir + '/' + image_id + '.npy'
                    fc7_file = self.fc7_dir + '/' + image_id + '.npy'
                    image = np.load(image_file)
                    # image = image.reshape(image.shape[0], image.shape[1] * image.shape[2])
                    fc7_image = np.load(fc7_file)
                else:
                    image = np.zeros((512, 196)).astype('float32')
                    fc7_image = np.zeros(4096).astype('float32')
                images.append(image)
                fc7_images.append(fc7_image)

                if len(reference) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(reference) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return images, fc7_images, imageids, reference
    
class ImageTextCharMemToneIterator(ImageTextCharMemIterator):
    def __init__(self, source, target,
                 reference,
                 level, tone, 
                 feature_dir, fc7_dir,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1, skip = 0): 
        ImageTextCharMemIterator.__init__(self, source, target, 
                                          reference,
                                          feature_dir, fc7_dir, 
                                          source_dict, target_dict, 
                                          batch_size = batch_size, 
                                          maxlen = maxlen, 
                                          n_words_source = n_words_source, 
                                          n_words_target = n_words_target, 
                                          skip = skip)
        with open(level, 'rb') as f:
            self.level_word_dictionary = pkl.load(f)
        with open(tone, 'rb') as f:
            self.tone_word_dictionary = pkl.load(f)
        self.level_dictionary = {}
        self.n_levels = 0
        for word, level in self.level_word_dictionary.iteritems():
            if self.source_dict.has_key(word.encode('utf-8')):
                self.level_dictionary[self.source_dict[word.encode('utf-8')]] = level
                for l in level:
                    self.n_levels = self.n_levels > l and self.n_levels or l
        self.n_levels += 1
        
        self.tone_dictionary = {}
        self.n_tones = 0
        for word, tone in self.tone_word_dictionary.iteritems():
            if self.source_dict.has_key(word.encode('utf-8')):
                self.tone_dictionary[self.source_dict[word.encode('utf-8')]] = tone
                for t in tone:
                    self.n_tones = self.n_tones > t and self.n_tones or t
        self.n_tones += 1
        

class TestImageTextCharMemToneIterator(TestImageTextCharMemIterator):
    def __init__(self, source, level, tone, 
                 feature_dir, fc7_dir, source_dict, 
                 batch_size = 128, 
                 maxlen = 100, 
                 n_words_source = -1):
        TestImageTextCharMemIterator.__init__(self, source, 
                                              feature_dir, fc7_dir, 
                                              source_dict, 
                                              batch_size = batch_size, 
                                              maxlen = maxlen, 
                                              n_words_source = n_words_source)

        with open(level, 'rb') as f:
            self.level_word_dictionary = pkl.load(f)
        with open(tone, 'rb') as f:
            self.tone_word_dictionary = pkl.load(f)
        self.level_dictionary = {}
        self.n_levels = 0
        for word, level in self.level_word_dictionary.iteritems():
            if self.source_dict.has_key(word.encode('utf-8')):
                self.level_dictionary[self.source_dict[word.encode('utf-8')]] = level
                for l in level:
                    self.n_levels = self.n_levels > l and self.n_levels or l
        self.n_levels += 1

        self.tone_dictionary = {}
        self.n_tones = 0
        for word, tone in self.tone_word_dictionary.iteritems():
            if self.source_dict.has_key(word.encode('utf-8')):
                self.tone_dictionary[self.source_dict[word.encode('utf-8')]] = tone
                for t in tone:
                    self.n_tones = self.n_tones > t and self.n_tones or t
        self.n_tones += 1
    
class TextCharMemIterator(TextIterator):
    def __init__(self, source, target,
                 reference,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1, skip = 0): 
        TextIterator.__init__(self, source, target, 
                                          source_dict, target_dict, 
                                          batch_size = batch_size, 
                                          maxlen = maxlen, 
                                          n_words_source = n_words_source, 
                                          n_words_target = n_words_target) 
        self.reference = fopen(reference, 'r')
        
    def reset(self):
        TextIterator.reset(self)
        self.reference.seek(0)
        
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        reference = []
        target = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]
                
                rr = self.reference.readline()
                if rr == "":
                    raise IOError
                rr = rr.strip().split('\t')
                r_list = []
                for r in rr:
                    temp_list = []
                    temp = r.split(' ')
                    flag = False
                    for t in temp:
                        if t in self.source_dict:
                            if self.source_dict[t] < self.n_words_source:
                                temp_list.append(self.source_dict[t])
                                flag = True
                            else:
                                temp_list.append(1)
                    if flag:
                        r_list.append(temp_list)

                source.append(ss)
                target.append(tt)
                reference.append(r_list)

                # image = image.reshape(image.shape[0], image.shape[1] * image.shape[2])

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target, reference

class TestTextCharMemIterator:
    def __init__(self, source,
                 source_dict, 
                 batch_size = 128, 
                 maxlen = 100, 
                 n_words_source = -1):
        self.source = fopen(source, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source

        self.end_of_data = False

    def __iter__(self):
        return self
    
    def reset(self):
        self.source.seek(0)
    
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        reference = []

        try:

            # actual work here
            while True:
                rr = self.source.readline()
                if rr == "":
                    raise IOError
                rr = rr.strip().split('\t')
                r_list = []
                for r in rr:
                    temp_list = []
                    temp = r.split(' ')
                    flag = False
                    for t in temp:
                        if t in self.source_dict:
                            if self.source_dict[t] < self.n_words_source:
                                temp_list.append(self.source_dict[t])
                                flag = True
                            else:
                                temp_list.append(1)
                    if flag:
                        r_list.append(temp_list)

                reference.append(r_list)

                if len(reference) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(reference) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return reference

class TextCharMemToneIterator(TextCharMemIterator):
    def __init__(self, source, target,
                 reference,
                 level, tone, 
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1, skip = 0): 
        TextCharMemIterator.__init__(self, source, target, reference, 
                                          source_dict, target_dict, 
                                          batch_size = batch_size, 
                                          maxlen = maxlen, 
                                          n_words_source = n_words_source, 
                                          n_words_target = n_words_target) 
        with open(level, 'rb') as f:
            self.level_word_dictionary = pkl.load(f)
        with open(tone, 'rb') as f:
            self.tone_word_dictionary = pkl.load(f)
        self.level_dictionary = {}
        self.n_levels = 0
        for word, level in self.level_word_dictionary.iteritems():
            if self.source_dict.has_key(word.encode('utf-8')):
                self.level_dictionary[self.source_dict[word.encode('utf-8')]] = level
                for l in level:
                    self.n_levels = self.n_levels > l and self.n_levels or l
        self.n_levels += 1
        
        self.tone_dictionary = {}
        self.n_tones = 0
        for word, tone in self.tone_word_dictionary.iteritems():
            if self.source_dict.has_key(word.encode('utf-8')):
                self.tone_dictionary[self.source_dict[word.encode('utf-8')]] = tone
                for t in tone:
                    self.n_tones = self.n_tones > t and self.n_tones or t
        self.n_tones += 1


class TestTextCharMemToneIterator(TestTextCharMemIterator):
    def __init__(self, source, level, tone, 
                 source_dict, 
                 batch_size = 128, 
                 maxlen = 100, 
                 n_words_source = -1):
        TestTextCharMemIterator.__init__(self, source, source_dict, 
                                         batch_size = batch_size, 
                                         maxlen = maxlen, 
                                         n_words_source = n_words_source)
        with open(level, 'rb') as f:
            self.level_word_dictionary = pkl.load(f)
        with open(tone, 'rb') as f:
            self.tone_word_dictionary = pkl.load(f)
        self.level_dictionary = {}
        self.n_levels = 0
        for word, level in self.level_word_dictionary.iteritems():
            if self.source_dict.has_key(word.encode('utf-8')):
                self.level_dictionary[self.source_dict[word.encode('utf-8')]] = level
                for l in level:
                    self.n_levels = self.n_levels > l and self.n_levels or l
        self.n_levels += 1

        self.tone_dictionary = {}
        self.n_tones = 0
        for word, tone in self.tone_word_dictionary.iteritems():
            if self.source_dict.has_key(word.encode('utf-8')):
                self.tone_dictionary[self.source_dict[word.encode('utf-8')]] = tone
                for t in tone:
                    self.n_tones = self.n_tones > t and self.n_tones or t
        self.n_tones += 1

class TextCharMemSentiIterator(TextIterator):
    def __init__(self, source, target,
                 reference,
                 sentiment,
                 source_dict, target_dict,
                 sentiment_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1, skip = 0): 
        TextIterator.__init__(self, source, target, 
                                          source_dict, target_dict, 
                                          batch_size = batch_size, 
                                          maxlen = maxlen, 
                                          n_words_source = n_words_source, 
                                          n_words_target = n_words_target) 
        self.reference = fopen(reference, 'r')
        self.sentiment = fopen(sentiment, 'r')
        with open(sentiment_dict, 'rb') as f:
            self.sentiment_dict = pkl.load(f)
        
    def reset(self):
        TextIterator.reset(self)
        self.reference.seek(0)
        self.sentiment.seek(0)
        
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        reference = []
        sentiment = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.source_dict[w] if w in self.source_dict else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]
                
                rr = self.reference.readline()
                if rr == "":
                    raise IOError
                rr = rr.strip().split('\t')
                r_list = []
                for r in rr:
                    temp_list = []
                    temp = r.split(' ')
                    flag = False
                    for t in temp:
                        if t in self.source_dict:
                            if self.source_dict[t] < self.n_words_source:
                                temp_list.append(self.source_dict[t])
                                flag = True
                            else:
                                temp_list.append(1)
                    if flag:
                        r_list.append(temp_list)
                senti = self.sentiment.readline()
                senti = senti.strip().split(' ')
                senti = [self.sentiment_dict[s] if s in self.sentiment_dict else 0 for s in senti]
                

                source.append(ss)
                target.append(tt)
                reference.append(r_list)
                sentiment.append(senti)

                # image = image.reshape(image.shape[0], image.shape[1] * image.shape[2])

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target, reference, sentiment

class TestTextCharMemSentiIterator:
    def __init__(self, source, 
                 source_dict, 
                 sentiment_dict,
                 sentiment_mems,
                 batch_size = 128, 
                 maxlen = 100, 
                 n_words_source = -1, 
                 num_mems = 50):
        self.source = fopen(source, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(sentiment_dict, 'rb') as f:
            self.sentiment_dict = pkl.load(f)
        self.sentiment_mems = []
        for s in sentiment_mems:
            with open(s, 'rb') as f:
                s_dict = pkl.load(f) 
                self.sentiment_mems.append(
                        [self.sentiment_dict[w.encode('utf-8')] 
                            if w.encode('utf-8') in self.sentiment_dict 
                            else 0 
                        for w in s_dict])

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.num_mems = num_mems

        self.n_words_source = n_words_source

        self.end_of_data = False

    def __iter__(self):
        return self
    
    def reset(self):
        self.source.seek(0)
    
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        reference = []
        sentiment = []

        try:

            # actual work here
            while True:
                ss = self.source.readline()
                temp = ss.strip().split('@')
                senti = temp[0]
                ss = temp[1]
                if ss == "":
                    raise IOError
                ss = ss.strip().split('\t')
                r_list = []
                for r in ss:
                    temp_list = []
                    temp = r.split(' ')
                    flag = False
                    for t in temp:
                        if t in self.source_dict:
                            if self.source_dict[t] < self.n_words_source:
                                temp_list.append(self.source_dict[t])
                                flag = True
                            else:
                                temp_list.append(1)
                    if flag:
                        r_list.append(temp_list)

                reference.append(r_list)
                
                senti = senti.split(' ')
                temp_list = []
                for s in senti:
                    if '1' == s:
                        senti_mem = self.sentiment_mems[0]
                    else:
                        senti_mem = self.sentiment_mems[1]
                    t = []
                    random_mem = np.arange(len(senti_mem))
                    np.random.shuffle(random_mem)
                    for i in range(self.num_mems):
                        t.append(senti_mem[random_mem[i]])
                    temp_list.append(t)
                sentiment.append(temp_list)

                if len(reference) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(reference) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return reference, sentiment
    

class TextCharMemSentiToneIterator(TextCharMemSentiIterator):
    def __init__(self, source, target,
                 reference,
                 sentiment,
                 level, tone,  
                 source_dict, target_dict,
                 sentiment_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1, skip = 0): 
        TextCharMemSentiIterator.__init__(self, source, target, 
                                          reference, sentiment,
                                          source_dict, target_dict, 
                                          sentiment_dict, 
                                          batch_size = batch_size, 
                                          maxlen = maxlen, 
                                          n_words_source = n_words_source, 
                                          n_words_target = n_words_target) 
        with open(level, 'rb') as f:
            self.level_word_dictionary = pkl.load(f)
        with open(tone, 'rb') as f:
            self.tone_word_dictionary = pkl.load(f)
        self.level_dictionary = {}
        self.n_levels = 0
        for word, level in self.level_word_dictionary.iteritems():
            if self.source_dict.has_key(word.encode('utf-8')):
                self.level_dictionary[self.source_dict[word.encode('utf-8')]] = level
                for l in level:
                    self.n_levels = self.n_levels > l and self.n_levels or l
        self.n_levels += 1
        
        self.tone_dictionary = {}
        self.n_tones = 0
        for word, tone in self.tone_word_dictionary.iteritems():
            if self.source_dict.has_key(word.encode('utf-8')):
                self.tone_dictionary[self.source_dict[word.encode('utf-8')]] = tone
                for t in tone:
                    self.n_tones = self.n_tones > t and self.n_tones or t
        self.n_tones += 1
        
        
class TestTextCharMemSentiToneIterator(TestTextCharMemSentiIterator):
    def __init__(self, source, level, tone,
                 source_dict, 
                 sentiment_dict,
                 sentiment_mems,
                 batch_size = 128, 
                 maxlen = 100, 
                 n_words_source = -1, 
                 num_mems = 50):
        TestTextCharMemSentiIterator.__init__(self, source, source_dict, 
                                              sentiment_dict, 
                                              sentiment_mems, 
                                              batch_size = batch_size, 
                                              maxlen = maxlen, 
                                              n_words_source = n_words_source, 
                                              num_mems = num_mems)

        with open(level, 'rb') as f:
            self.level_word_dictionary = pkl.load(f)
        with open(tone, 'rb') as f:
            self.tone_word_dictionary = pkl.load(f)
        self.level_dictionary = {}
        self.n_levels = 0
        for word, level in self.level_word_dictionary.iteritems():
            if self.source_dict.has_key(word.encode('utf-8')):
                self.level_dictionary[self.source_dict[word.encode('utf-8')]] = level
                for l in level:
                    self.n_levels = self.n_levels > l and self.n_levels or l
        self.n_levels += 1
        
        self.tone_dictionary = {}
        self.n_tones = 0
        for word, tone in self.tone_word_dictionary.iteritems():
            if self.source_dict.has_key(word.encode('utf-8')):
                self.tone_dictionary[self.source_dict[word.encode('utf-8')]] = tone
                for t in tone:
                    self.n_tones = self.n_tones > t and self.n_tones or t
        self.n_tones += 1
