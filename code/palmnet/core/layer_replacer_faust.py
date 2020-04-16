import os
import pickle
import zlib

from pyfaust import Faust

from palmnet.core.layer_replacer_sparse_facto import LayerReplacerSparseFacto


class LayerReplacerFaust(LayerReplacerSparseFacto):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.path_checkpoint_file is not None:
            assert os.path.isdir(str(self.path_checkpoint_file))
            self.dct_references_faust = dict()
            self.path_dct_references_faust = self.path_checkpoint_file / "dict_faust_references.pickle"

    def load_dct_name_compression(self):
        with open(str(self.path_dct_references_faust), 'rb') as rb_file:
            self.dct_references_faust = pickle.load(rb_file)

        for layer_name, tpl_faust_reference in self.dct_references_faust.items():
            if tpl_faust_reference is None:
                self.dct_name_compression[layer_name] = None
            else:
                self.dct_name_compression[layer_name] = {
                    "lambda": tpl_faust_reference[0],
                    "sparse_factors": Faust(filepath=tpl_faust_reference[1])
                }

    def save_dct_name_compression(self):
        if self.path_checkpoint_file is None:
            return

        for layer_name, dict_faust_obj in self.dct_name_compression.items():
            if layer_name in self.dct_references_faust:
                continue  # do not re-save multiple times the same thing

            if dict_faust_obj is None: # nothing to save
                self.dct_references_faust[layer_name] = None
            else:
                # create unique identifier for faust object
                id_faust = id(dict_faust_obj)
                hash_faust = hex(zlib.crc32(str.encode(str(id_faust))))
                # create path for faust object
                filename_faust = layer_name + "_" +str(hash_faust) + ".mat"
                path_faust = str(self.path_checkpoint_file / filename_faust)
                # save it
                lambda_ = dict_faust_obj["lambda"]
                faust_obj = dict_faust_obj["sparse_factors"]
                faust_obj.save(str(path_faust))
                # store reference to path
                self.dct_references_faust[layer_name] = (lambda_, path_faust)

        with open(str(self.path_dct_references_faust), 'wb') as wb_file:
            pickle.dump(self.dct_references_faust, wb_file)