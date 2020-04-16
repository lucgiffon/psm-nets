from palmnet.core.layer_replacer_sparse_facto_tucker import LayerReplacerSparseFactoTucker
from pyfaust import Faust


import os
import pickle
import zlib

class LayerReplacerSparseFactoTuckerFaust(LayerReplacerSparseFactoTucker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.path_checkpoint_file is not None:
            assert os.path.isdir(str(self.path_checkpoint_file))
            self.dct_references_faust = dict()  # this dict maps layer name to lambda value, in_rank, out_rank, and path to faust objects (in, core, out)
            self.path_dct_references_faust = self.path_checkpoint_file / "dict_faust_references.pickle"

    def load_dct_name_compression(self):
        with open(str(self.path_dct_references_faust), 'rb') as rb_file:
            self.dct_references_faust = pickle.load(rb_file)

        for layer_name, tpl_or_dct_faust_reference in self.dct_references_faust.items():
            if tpl_or_dct_faust_reference is None:
                self.dct_name_compression[layer_name] = None
            elif type(tpl_or_dct_faust_reference) == tuple:
                self.dct_name_compression[layer_name] = {
                    "lambda": tpl_or_dct_faust_reference[0],
                    "sparse_factors": Faust(filepath=tpl_or_dct_faust_reference[1])
                }
            else:
                # it is a dict
                self.dct_name_compression[layer_name] = dict()
                self.dct_name_compression[layer_name]["in_rank"] = tpl_or_dct_faust_reference["in_rank"]
                self.dct_name_compression[layer_name]["out_rank"] = tpl_or_dct_faust_reference["out_rank"]
                for tucker_part_name in self.lst_tucker_weights:
                    self.dct_name_compression[layer_name][tucker_part_name] = dict()
                    self.dct_name_compression[layer_name][tucker_part_name]["lambda"] = tpl_or_dct_faust_reference[tucker_part_name][0]
                    self.dct_name_compression[layer_name][tucker_part_name]["sparse_factors"] = Faust(filepath=tpl_or_dct_faust_reference[tucker_part_name][1])


    def save_dct_name_compression(self):
        if self.path_checkpoint_file is None:
            return

        for layer_name, dict_faust_obj in self.dct_name_compression.items():
            if layer_name in self.dct_references_faust:
                continue  # do not re-save multiple times the same thing

            if dict_faust_obj is None:  # nothing to save
                self.dct_references_faust[layer_name] = None
            else:
                if "in_rank" in dict_faust_obj:
                    self.dct_references_faust[layer_name] = dict()
                    self.dct_references_faust[layer_name]["in_rank"] = dict_faust_obj["in_rank"]
                    self.dct_references_faust[layer_name]["out_rank"] = dict_faust_obj["out_rank"]
                    for tucker_part_name in self.lst_tucker_weights:
                        # create unique identifier for tucker sparse facto faust object
                        id_faust = id(dict_faust_obj[tucker_part_name])
                        hash_faust = hex(zlib.crc32(str.encode(str(id_faust))))
                        # create path for faust object
                        filename_faust = layer_name + "_" + tucker_part_name + "_" + str(hash_faust) + ".mat"
                        path_faust = str(self.path_checkpoint_file / filename_faust)
                        # save it
                        lambda_ = dict_faust_obj[tucker_part_name]["lambda"]
                        faust_obj = dict_faust_obj[tucker_part_name]["sparse_factors"]
                        faust_obj.save(str(path_faust))
                        # store reference to path
                        self.dct_references_faust[layer_name][tucker_part_name] = (lambda_, path_faust)
                else:
                    # create unique identifier for tucker sparse facto faust object
                    id_faust = id(dict_faust_obj)
                    hash_faust = hex(zlib.crc32(str.encode(str(id_faust))))
                    # create path for faust object
                    filename_faust = layer_name + "_" + str(hash_faust) + ".mat"
                    path_faust = str(self.path_checkpoint_file / filename_faust)
                    # save it
                    lambda_ = dict_faust_obj["lambda"]
                    faust_obj = dict_faust_obj["sparse_factors"]
                    faust_obj.save(str(path_faust))
                    # store reference to path
                    self.dct_references_faust[layer_name] = (lambda_, path_faust)

        with open(str(self.path_dct_references_faust), 'wb') as wb_file:
            pickle.dump(self.dct_references_faust, wb_file)
