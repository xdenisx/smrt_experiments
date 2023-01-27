# SMRT tools

This repository contains a Python class and scripts aimed at to make it easier the SMRT model [Picard, 2018] run using a unified dictonary of parameters.
The configuration dictonary consits of several sections, each corresponds to atmoshpere, sea ice, soil and water content, respectively.
Once such dictonary is initialized, a simulation with the SMRT can be performed using a class ```SMRTtools```. To run the code, you also need to specify a path to your local SMRT code via ```smrt_path```.
If the dictonary is not specified a default will be used. In the follow the model is initialized with a dictonary ``model_parameters`` with solid ice and without snowpack:

```python
from SMRTtools import SMRTtools

smrt_setup = SMRTtools(model_parameters=model_parameters, smrt_path='../smrt', snowpack=None)
```

where ```model_parameters``` initialized as follows:

```python
# Model, snow and substrate parameters
model_parameters = {}

model_parameters['snow'] = {}
model_parameters['land'] = {}
model_parameters['ice'] = {}
model_parameters['model'] = {}
model_parameters['instrument'] = {}
model_parameters['atmosphere'] = {}

# Instrumental parameters
model_parameters['instrument']['fq_list'] = 37e9
model_parameters['instrument']['polarizations'] = ['h','v']
model_parameters['instrument']['theta'] = 55

# Model parameters          
model_parameters['model']['substrate'] = 'Fresh'

# Atmoshpere
model_parameters['atmosphere']['Td'] = 0
model_parameters['atmosphere']['Tbup'] = 0
model_parameters['atmosphere']['Transmissivity'] = 1

# Snow
model_parameters['snow']['sn_thickness'] = [0.]
model_parameters['snow']['sn_density'] = [100]
model_parameters['snow']['stickiness'] = 2
model_parameters['snow']['radius'] = 0.1e-3
model_parameters['snow']['sn_ms_model'] = 6
model_parameters['snow']['sn_temp'] = [273.15 - 20]

# Sea ice
total_ice_thickness = 5 # m
ice_layers = 2
bottom_layer_t_ice = 273.15 - 2
skin_layer_t_ice = 273.15 - 20
model_parameters['ice']['ice_temp'] = list(np.linspace(skin_layer_t_ice, bottom_layer_t_ice, ice_layers))
model_parameters['ice']['num_layers'] = ice_layers
model_parameters['ice']['thickness'] = total_ice_thickness
model_parameters['ice']['p_ex'] = np.array([1.0e-3] * (model_parameters['ice']['num_layers']))
model_parameters['ice']['porosity'] = 0.
model_parameters['ice']['layer_thickness'] = list(np.array([total_ice_thickness / ice_layers] * ice_layers))

```

then you can calculate and print brightness temperature for the model setup:

``` python
smrt_setup.calc_tb()
print('Brightness temperature at V polarization: {:.2f} K'.format(smrt_setup.tb.TbV()))
print('Brightness temperature at H polarization: {:.2f} K'.format(smrt_setup.tb.TbH()))
```


  