import pandas as pd
from mkgu.assemblies import pwdb, AssemblyModel, AssemblyStoreMap, AssemblyStoreModel
from mkgu.stimuli import ImageModel, StimulusSetModel, ImageStoreModel, StimulusSetImageMap, ImageStoreMap


pwdb.connect(reuse_if_open=True)


pwdb.create_tables(models=[ImageModel, StimulusSetModel, ImageStoreModel, StimulusSetImageMap, ImageStoreMap])


hvm_images = StimulusSetModel(name="dicarlo.hvm")
hvm_images.save()

hvm_image_store = ImageStoreModel(location_type="S3", store_type="zip",
                                  location="https://mkgu-dicarlolab-hvm.s3.amazonaws.com/HvM_with_discfade.zip")
hvm_image_store.save()

df_images = pd.read_pickle("image_meta_dataframe.pkl")
path_map = {"V0": "Variation00_20110203", "V3": "Variation03_20110128", "V6": "Variation06_20110131"}

for image in df_images.itertuples():
    pw_image = ImageModel(
        hash_id=image.image_id,
        object_name=image.object,
        category_name=image.category,
        background_id=image.background_id,
        image_file_name=image.image_file_name,
        variation=int(image.variation[-1]),
        ty=image.ty,
        tz=image.tz,
        rxy=image.rxy,
        rxz=image.rxz,
        ryz=image.ryz,
        rxy_semantic=image.rxy_semantic,
        rxz_semantic=image.rxz_semantic,
        ryz_semantic=image.ryz_semantic,
        size=image.size,
        s=image.s
    )
    pw_stimulus_set_image_map = StimulusSetImageMap(stimulus_set=hvm_images, image=pw_image)
    pw_image_image_store_map = ImageStoreMap(image=pw_image, image_store=hvm_image_store,
                                                  path=path_map[image.variation])
    pw_image.save()
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





