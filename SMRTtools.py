import sys
sys.path.append("../smrt")
from smrt import make_snowpack, make_model, make_soil, sensor_list, make_ice_column, PSU
from smrt.atmosphere.simple_isotropic_atmosphere import SimpleIsotropicAtmosphere
from smrt.microstructure_model.sticky_hard_spheres import StickyHardSpheres
import smrt
import numpy as np

class SMRTtools:
	def __init__(self, model_parameters=None, snowpack=None):

		# Instrumental parameters
		if model_parameters is None:
			raise ValueError('Please set the paramteters')
		else:
			self.model_parameters = model_parameters

		self.radiometer = sensor_list.passive(self.model_parameters['instrument']['fq_list'],
											  self.model_parameters['instrument']['theta'])

		# TB calculation is more accurate if number of streams is increased (currently: default = 32)
		n_max_stream = 128

		# Substrate parameters
		self.l_substrate = ['land', 'Fresh', 'MYI']

		# Initialize EM Model
		# !Phase normalization is not recommended but temprorary used here
		self.m = make_model('iba', 'dort',
							rtsolver_options={'n_max_stream': n_max_stream,
											  'phase_normalization': 'forced'})
		if not snowpack is None:
			# Snow and surface parameters
			print('\nAdding snowpack...')
			self.sn_ms_model_list = ['autocorrelation', 'exponential', 'gaussian_random_field',
									 'homogeneous', 'independent_sphere', 'sampled_autocorrelation',
									 'sticky_hard_spheres', 'test_autocorrelation', 'test_exponential',
									 'test_sticky_hard_spheres', 'teubner_strey', 'unified_autocorrelation',
									 'unified_scaled_exponential', 'unified_sticky_hard_spheres',
									 'unified_teubner_strey']

			print(f'Start initializing snowpack with:')
			print(f'''Microstructure model: {self.sn_ms_model_list[self.model_parameters['snow']['sn_ms_model']]}''')
			print(f'''Snow density: {self.model_parameters['snow']['sn_density']}''')
			print(f'''Snow depth: {self.model_parameters['snow']['sn_thickness']}''')
			print(f'''Snow temperature: {self.model_parameters['snow']['sn_temp']}''')
			print(f'''Sphere radius: {self.model_parameters['snow']['radius']}''')
			print(f'''Stickiness: {self.model_parameters['snow']['stickiness']}''')

			self.snowpack = make_snowpack(
				microstructure_model=self.sn_ms_model_list[self.model_parameters['snow']['sn_ms_model']],
				density=self.model_parameters['snow']['sn_density'],
				thickness=self.model_parameters['snow']['sn_thickness'],
				temperature=self.model_parameters['snow']['sn_temp'],
				radius=self.model_parameters['snow']['radius'],
				stickiness=self.model_parameters['snow']['stickiness'])
			print('Done\n')
		else:
			self.snowpack = None

		self.add_substrate()

	def add_substrate(self):
		'''
		Add substrate
		'''

		if self.model_parameters['model']['substrate'] in self.l_substrate:
			print(f'''Substrate: {self.model_parameters['model']['substrate']}''')
		else:
			raise ValueError(f'Please specify subtrate name in the model parameters ({self.l_substrate})')

		# Set atmospheric parameters
		print(f'''Init atmosphere with {self.model_parameters['atmosphere']['Td']} K''')
		self.atmosphere = SimpleIsotropicAtmosphere(tbdown=self.model_parameters['atmosphere']['Td'],
													tbup=self.model_parameters['atmosphere']['Tbup'],
													trans=self.model_parameters['atmosphere']['Transmissivity'])

		# Set substrate
		if self.model_parameters['model']['substrate'] == 'land':
			self.substrate = make_soil('soil_wegmuller',
									   permittivity_model=complex(10, 1),
									   roughness_rms=self.model_parameters['land']['roughness_rms'],
									   temperature=self.model_parameters['land']['surface_temp'])
			# Land surface temperature
			self.surface_temp = self.model_parameters['land']['surface_temp']

		if self.model_parameters['model']['substrate'] == 'MYI':
			print(f'''Ice porosity: {self.model_parameters['ice']['porosity']}''')
			print(f'''Ice thickness: {self.model_parameters['ice']['thickness']}''')
			print(f'''Ice surface T: {self.model_parameters['ice']['temp']}''')
			print(f'''Making model {self.model_parameters['model']['substrate']}''')
			# prepare inputs
			l = self.model_parameters['ice']['num_layers']
			self.self.model_parameters['ice']['layer_thickness'] = np.array(
				[self.model_parameters['ice']['thickness'] / l] * l)  # ice is N meters thick
			p_ex = np.array([1.0e-3] * (l))  # correlation length

			# temperature = np.linspace(self.model_parameters['ice']['temp'], 273.15 - 1.8,
			#                          l)  # temperature gradient in the ice from T deg K at top to freezing temperature of water at bottom (-1.8 deg C)

			salinity = np.linspace(2., 10.,
								   l) * PSU  # salinity profile ranging from salinity=2 at the top to salinity=10 at the bottom of the ice

			# create a multi-year sea ice column with assumption of spherical brine inclusions (brine_inclusion_shape=\spheres\), and 10% porosity:
			ice_type = 'multiyear'  # first-year or multi-year sea ice
			# porosity = model_parameters['ice']['porosity']  # ice porosity in fractions, [0..1]

			self.substrate = make_ice_column(ice_type=ice_type,
											 thickness=thickness,
											 temperature=self.model_parameters['ice']['temp'],
											 microstructure_model='exponential',
											 brine_inclusion_shape='spheres',
											 # brine_inclusion_shape can be \spheres\, \random_needles\ or \mix_spheres_needles\
											 salinity=salinity,
											 # either 'salinity' or 'brine_volume_fraction' should be given for sea ice; if salinity is given, brine volume fraction is calculated in the model; if none is given, ice is treated as fresh water ice
											 porosity=self.model_parameters['ice']['porosity'],
											 # either density or 'porosity' should be set for sea ice. If porosity is given, density is calculated in the model. If none is given, ice is treated as having a porosity of 0% (no air inclusions) corr_length=p_ex, add_water_substrate=\ocean\  # see comment below
											 corr_length=p_ex,
											 add_water_substrate=True
											 )

		if self.model_parameters['model']['substrate'] == 'Fresh':
			ice_type = 'fresh'
			self.substrate = make_ice_column(ice_type=ice_type,
											 thickness=self.model_parameters['ice']['layer_thickness'],
											 temperature=self.model_parameters['ice']['ice_temp'],
											 microstructure_model='homogeneous',
											 corr_length=self.model_parameters['ice']['p_ex'],
											 add_water_substrate=True
											 )

		# Add snowpack on top of substrate:
		if not self.snowpack is None:
			self.medium = self.snowpack + self.substrate
		else:
			self.medium = self.substrate

	def calc_tb(self):
		'''
		Calculate brighnness temperature
		'''

		self.medium.atmosphere = self.atmosphere
		sresult = self.m.run(self.radiometer, self.medium)
		self.tb = sresult

	def calc_e(self):
		'''
		Calculate Emissivity
		'''

		# Run model without atmsophere
		sresult_0 = self.m.run(self.radiometer, self.medium)
		# Run model with 1K atmosphere
		self.medium.atmosphere = self.atmosphere
		sresult_1 = self.m.run(self.radiometer, self.medium)

		# V-pol
		if self.model_parameters['model']['e_equation'] == 1:
			print(f'''Equation {self.model_parameters['model']['e_equation']}''')
			self.emissivity_V = 1 - (sresult_1.TbV() - sresult_0.TbV()) / self.model_parameters['atmosphere']['Td']
		elif self.model_parameters['model']['e_equation'] == 2:
			print(f'''Equation {self.model_parameters['model']['e_equation']}''')
			self.emissivity_V = (sresult_0.TbV()) / self.surface_temp
		else:
			raise ('Please specify correct equation number (1/2)')
		print(f'Ev={self.emissivity_V}')

		# H-pol
		if self.model_parameters['model']['e_equation'] == 1:
			print(f'''Equation {self.model_parameters['model']['e_equation']}''')
			# reflectivity_H = ( sresult_1.TbH() - sresult_0.TbH() ) / Tbdown
			self.emissivity_H = 1 - (sresult_1.TbH() - sresult_0.TbH()) / self.model_parameters['atmosphere']['Td']
		elif self.model_parameters['model']['e_equation'] == 2:
			print(f'''Equation {self.model_parameters['model']['e_equation']}''')
			self.emissivity_H = (sresult_0.TbH()) / self.surface_temp
		else:
			raise ('Please specify correct equation number (1/2)')
		print(f'Eh={self.emissivity_H}')

	# From old code (incorrect?)
	# reflectivity_H = (sresult_0.TbH() + sresult_1.TbH()) / 2 / T

	def layers_of_model(self,
						n_ice_layers=10,
						n_snow_layers=4,
						dz_snow_layer=0.1,
						dt=0,
						ice_th=2.5,
						t0=-4
						):
		'''
		Generate layers for snow and icepack
		'''

		t0 = 273.15 - (t0) * (-1)

		# Ice
		dz = ice_th / n_ice_layers
		z_ice = np.linspace(-dz, -1 * (ice_th) + dz, n_ice_layers)
		t_ice = -z_ice
		t_ice[t_ice > 2] = 2
		t_ice += t0 + dt

		# Snow
		dz = dz_snow_layer
		z_snow = np.linspace((n_snow_layers - 0.5) * dz_snow_layer, dz / 2,
							 n_snow_layers)
		t_snow = t0 + dt - z_snow

		return z_snow, t_snow, z_ice, t_ice