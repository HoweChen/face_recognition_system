import pathlib


def register_to_db(ins, file, file_name, image_size="SMALL", num_jitters=1):
    # ins = Instance(mode="MTCNN", f_e_m="NORMAL", watermark=False)
    if isinstance(file, pathlib.PosixPath):
        try:
            ins.image_to_db(filepath=str(file.absolute()), name=str(file_name), image_size=image_size,
                            num_jitters=1)
        except Exception as e:
            raise e
    else:
        try:
            ins.image_to_db(filepath=file, name=file_name, image_size=image_size, num_jitters=num_jitters)
        except Exception as e:
            raise e
