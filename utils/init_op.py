from datagenerator import ImageDataGenerator


def get_init_op(iterator, some_data: ImageDataGenerator):
    return iterator.make_initializer(some_data.data)
