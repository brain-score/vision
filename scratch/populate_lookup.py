import os

import pandas as pd
from mkgu.assemblies import pwdb, AssemblyModel, AssemblyStoreMap, AssemblyStoreModel
from mkgu.stimuli import ImageModel, AttributeModel, ImageMetaModel, StimulusSetModel, ImageStoreModel, \
    StimulusSetImageMap, ImageStoreMap


pwdb.connect(reuse_if_open=True)


pwdb.create_tables(models=[ImageModel, AttributeModel, ImageMetaModel, StimulusSetModel, ImageStoreModel, StimulusSetImageMap, ImageStoreMap])


hvm_images = StimulusSetModel(name="dicarlo.hvm")
hvm_images.save()

hvm_image_store = ImageStoreModel(location_type="S3", store_type="zip",
                                  location="https://mkgu-dicarlolab-hvm.s3.amazonaws.com/HvM_with_discfade.zip")
hvm_image_store.save()

df_images = pd.read_pickle("image_meta_dataframe.pkl")
path_map = {"V0": "Variation00_20110203", "V3": "Variation03_20110128", "V6": "Variation06_20110131"}

eav_image_file_name = AttributeModel(name="image_file_name", type="str")
eav_object_name = AttributeModel(name="object_name", type="str")
eav_category_name = AttributeModel(name="category_name", type="str")
eav_background_id = AttributeModel(name="background_id", type="str")
eav_variation = AttributeModel(name="variation", type="int")
eav_ty = AttributeModel(name="ty", type="float")
eav_tz = AttributeModel(name="tz", type="float")
eav_rxy = AttributeModel(name="rxy", type="float")
eav_rxz = AttributeModel(name="rxz", type="float")
eav_ryz = AttributeModel(name="ryz", type="float")
eav_rxy_semantic = AttributeModel(name="rxy_semantic", type="float")
eav_rxz_semantic = AttributeModel(name="rxz_semantic", type="float")
eav_ryz_semantic = AttributeModel(name="ryz_semantic", type="float")
eav_size = AttributeModel(name="size", type="float")
eav_s = AttributeModel(name="s", type="float")

eav_image_file_name.save()
eav_object_name.save()
eav_category_name.save()
eav_background_id.save()
eav_variation.save()
eav_ty.save()
eav_tz.save()
eav_rxy.save()
eav_rxz.save()
eav_ryz.save()
eav_rxy_semantic.save()
eav_rxz_semantic.save()
eav_ryz_semantic.save()
eav_size.save()
eav_s.save()

for image in df_images.itertuples():
    pw_image = ImageModel(
        hash_id=image.image_id,
    )
    pw_stimulus_set_image_map = StimulusSetImageMap(stimulus_set=hvm_images, image=pw_image)
    pw_image_image_store_map = ImageStoreMap(image=pw_image, image_store=hvm_image_store,
                                             path=os.path.join(path_map[image.variation], image.image_file_name))
    pw_image.save()

    ImageMetaModel(image=pw_image, attribute=eav_image_file_name, value=str(image.image_file_name)).save()
    ImageMetaModel(image=pw_image, attribute=eav_object_name, value=str(image.object)).save()
    ImageMetaModel(image=pw_image, attribute=eav_category_name, value=str(image.category)).save()
    ImageMetaModel(image=pw_image, attribute=eav_background_id, value=str(image.background_id)).save()
    ImageMetaModel(image=pw_image, attribute=eav_variation, value=str(int(image.variation[-1]))).save()
    ImageMetaModel(image=pw_image, attribute=eav_ty, value=str(image.ty)).save()
    ImageMetaModel(image=pw_image, attribute=eav_tz, value=str(image.tz)).save()
    ImageMetaModel(image=pw_image, attribute=eav_rxy, value=str(image.rxy)).save()
    ImageMetaModel(image=pw_image, attribute=eav_rxz, value=str(image.rxz)).save()
    ImageMetaModel(image=pw_image, attribute=eav_ryz, value=str(image.ryz)).save()
    ImageMetaModel(image=pw_image, attribute=eav_rxy_semantic, value=str(image.rxy_semantic)).save()
    ImageMetaModel(image=pw_image, attribute=eav_rxz_semantic, value=str(image.rxz_semantic)).save()
    ImageMetaModel(image=pw_image, attribute=eav_ryz_semantic, value=str(image.ryz_semantic)).save()
    ImageMetaModel(image=pw_image, attribute=eav_size, value=str(image.size)).save()
    ImageMetaModel(image=pw_image, attribute=eav_s, value=str(image.s)).save()

    pw_stimulus_set_image_map.save()
    pw_image_image_store_map.save()


pwdb.create_tables(models=[AssemblyModel, AssemblyStoreMap, AssemblyStoreModel])


store = AssemblyStoreModel(assembly_type="netCDF",
                           location_type="S3",
                           location="https://mkgu-dicarlolab-hvm.s3.amazonaws.com/hvm_neuronal_features.nc")
store.save()


assy = AssemblyModel(name="dicarlo.Majaj2015", assembly_class="NeuronRecordingAssembly",
                     stimulus_set=hvm_images)
assy.save()


assy_store_map = AssemblyStoreMap(assembly_model=assy, assembly_store_model=store, role="dicarlo.Majaj2015")
assy_store_map.save()





