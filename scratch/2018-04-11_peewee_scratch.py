
# coding: utf-8

# In[1]:


from brainscore.lookup import pwdb, AssemblyModel, AssemblyStoreMap, AssemblyStoreModel
from brainscore.stimuli import ImageModel, StimulusSetModel, ImageStoreModel, StimulusSetImageMap, ImageStoreMap


# In[2]:


pwdb.connect(reuse_if_open=True)


# In[4]:


pwdb.database


# In[5]:


pwdb.get_tables()


# In[7]:


pwdb.create_tables(models=[AssemblyModel, AssemblyStoreMap, AssemblyStoreModel])


# In[8]:


store = AssemblyStoreModel(assembly_type="netCDF", 
                           location_type="S3", 
                           location="https://mkgu-dicarlolab-hvm.s3.amazonaws.com/hvm_neuronal_features.nc")


# In[9]:


store.save()


# In[10]:


assy = AssemblyModel(name="dicarlo.Hong2011", assembly_class="NeuronRecordingAssembly")


# In[11]:


assy.save()


# In[12]:


assy_store_map = AssemblyStoreMap(assembly_model=assy, assembly_store_model=store)


# In[13]:


assy_store_map.save()


# In[14]:


hvm = AssemblyModel.get(AssemblyModel.name == "dicarlo.Hong2011")
hvm


# In[22]:


[m.assembly_store_model.location for m in hvm.assembly_store_maps]


# In[7]:


pwdb.create_tables(models=[ImageModel, StimulusSetModel, ImageStoreModel, StimulusSetImageMap, ImageStoreMap])


# In[23]:


pwdb.get_tables()


# In[9]:


hvm_images = StimulusSetModel(name="dicarlo.hvm")
hvm_images.save()

