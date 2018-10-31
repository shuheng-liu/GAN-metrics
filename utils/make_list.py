import os


# generate a txt file containing image paths and labels
def make_list(folders, flags=None, ceils=None, mode='train', store_path='/output', verbose=False):
    suffices = ('jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG')
    if ceils is None: ceils = [-1] * len(folders)  # ceil constraint not imposed
    if flags is None: flags = list(range(len(folders)))  # flags = [0, 1, ..., n-1]
    assert len(folders) == len(flags) == len(ceils), (len(folders), len(flags), len(ceils))
    assert mode in ['train', 'val', 'test']
    folders_flags_ceils = [tup for tup in zip(folders, flags, ceils)
                           if isinstance(tup[0], str) and os.path.isdir(tup[0])]
    assert folders_flags_ceils

    if verbose:
        print('Making %s list' % mode)
        for tup in folders_flags_ceils:
            print('Folder {}: flag = {}, ceil = {}'.format(*tup))
    if not os.path.isdir(store_path): os.mkdir(store_path)
    out_list = os.path.join(store_path, mode + '.txt')
    list_length = 0
    with open(out_list, 'w') as fo:
        for (folder, flag, ceil) in folders_flags_ceils:
            count = 0
            for pic_name in os.listdir(folder):
                if pic_name.split('.')[-1] not in suffices:
                    print('Ignoring non-image file {} in folder {}.'.format(pic_name, folder),
                          'Legal suffices are', suffices)
                    continue
                count += 1
                list_length += 1
                fo.write("{} {}\n".format(os.path.join(folder, pic_name), flag))
                # if ceil is imposed (ceil > 0) and count exceeds ceil, break and write next flag
                if 0 < ceil <= count: break
    if verbose:
        print('%s list made\n' % mode)
    return out_list, list_length
