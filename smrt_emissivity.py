# Adopted from SMRT code: https://github.com/smrt-model/smrt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
sys.path.append("../smrt")
from smrt import make_snowpack, make_model, make_soil, sensor_list, make_ice_column, PSU
from smrt.atmosphere.simple_isotropic_atmosphere import SimpleIsotropicAtmosphere
from smrt.microstructure_model.sticky_hard_spheres import StickyHardSpheres
import smrt


def setup_snowpack(model='exponential', thickness_1=1, sn_density=320, T=265.,
                   radius=0.1e-3, stickiness=0.1, substrate=None, corr_length=None):
    # ### Make a snow layer

    print(f'\nMicrostructure model: {model}\n')
    print(f'\nSnow density: {sn_density}')
    print(f'Snow thickness: {thickness_1}')

    # Best?
    if model == 'sticky_hard_spheres':
        # 0.01, 0.05,
        # 100, 100,
        sp = make_snowpack(thickness_1, 'sticky_hard_spheres', density=sn_density,
                           temperature=T, radius=radius, stickiness=stickiness)

    if model == 'homogeneous':
        sp = make_snowpack(thickness_1, 'homogeneous', density=sn_density, radius=radius)

    if model == 'exponential':
        sp = make_snowpack(thickness_1, 'exponential', density=sn_density, corr_length=corr_length)

    if model == 'independent_sphere':
        sp = make_snowpack(thickness_1, 'independent_sphere', density=sn_density, radius=radius)

    if model == 'gaussian_random_field':
        sp = make_snowpack([thickness_1], 'gaussian_random_field', density=sn_density, corr_length=radius,
                           repeat_distance=1.0)

    return sp


def calc_emissivity_thickness(fq, cor_length=0.05e-3, snow_thickness=1, T=265,
                              snow_ms_model='sticky_hard_spheres',
                              theta=53.,
                              sn_density=320,
                              substrate='MYI',
                              roughness_rms=None):
    ''' Calculate emissivity for different thickness '''
    e1_h, e1_v = _do_test_kirchoff_law_thickness(fq, cor_length, snow_thickness, T,
                                                 snow_ms_model, theta, sn_density,
                                                 substrate, roughness_rms)
    return e1_h, e1_v


def calc_e(fq, thickness_1, T, model, theta, sn_density,
                                    substrate, roughness_rms):
    # roughness_rms=0.01,
    if substrate == 'land':
        print(f'roughness_rms: {roughness_rms}')
        substrate = make_soil('soil_wegmuller',
                              permittivity_model=complex(10, 1),
                              roughness_rms=roughness_rms,
                              temperature=T)

    if substrate == 'MYI':
        # prepare inputs
        l = 9  # 9 ice layers
        thickness = np.array([1.5 / l] * l)  # ice is 1.5m thick
        p_ex = np.array([1.0e-3] * (l))  # correlation length
        temperature = np.linspace(T, 273.15 - 1.8,
                                  l)  # temperature gradient in the ice from T deg K at top to freezing temperature of water at bottom (-1.8 deg C)
        salinity = np.linspace(2., 10.,
                               l) * PSU  # salinity profile ranging from salinity=2 at the top to salinity=10 at the bottom of the ice

        # create a multi-year sea ice column with assumption of spherical brine inclusions (brine_inclusion_shape="spheres"), and 10% porosity:
        ice_type = 'multiyear'  # first-year or multi-year sea ice
        porosity = 0.08  # ice porosity in fractions, [0..1]

        substrate = make_ice_column(ice_type=ice_type, thickness=thickness,
                                    temperature=T,
                                    microstructure_model="exponential",
                                    brine_inclusion_shape="spheres",
                                    # brine_inclusion_shape can be "spheres", "random_needles" or "mix_spheres_needles"
                                    salinity=salinity,
                                    # either 'salinity' or 'brine_volume_fraction' should be given for sea ice; if salinity is given, brine volume fraction is calculated in the model; if none is given, ice is treated as fresh water ice
                                    porosity=porosity,
                                    # either density or 'porosity' should be set for sea ice. If porosity is given, density is calculated in the model. If none is given, ice is treated as having a porosity of 0% (no air inclusions)
                                    corr_length=p_ex,
                                    add_water_substrate="ocean"  # see comment below
                                    )

    atmosphere1K = SimpleIsotropicAtmosphere(tbdown=T, tbup=0, trans=1)

    snowpack = setup_snowpack(T=T, model=model, thickness_1=thickness_1, sn_density=sn_density)

    # add snowpack on top of substrate:
    medium = snowpack + substrate

    # create the sensor
    # sensor = sensor_list.passive(1.4e9, 40.)
    radiometer = sensor_list.passive(fq, theta)

    n_max_stream = 128  # TB calculation is more accurate if number of streams is increased (currently: default = 32);

    # create the EM Model
    m = make_model("iba", "dort",
                   rtsolver_options={"n_max_stream": n_max_stream})  # , 'phase_normalization': "forced"})

    # run the model
    sresult_0 = m.run(radiometer, medium)
    snowpack.atmosphere = atmosphere1K
    sresult_1 = m.run(radiometer, medium)

    # V-pol
    # Picard equation for Emissivity
    emissivity_V = (sresult_0.TbV() + sresult_1.TbV()) / 2 / T
    # emissivity_V = sresult_1.TbV() / T
    # np.testing.assert_allclose(emissivity_V, 1 - reflectivity_V, atol=0.002)

    # H-pol
    emissivity_H = (sresult_0.TbH() + sresult_1.TbH()) / 2 / T
    # emissivity_H = sresult_1.TbH() / T
    # np.testing.assert_allclose(emissivity_H, 1 - reflectivity_H, atol=0.002)

    return emissivity_H, emissivity_V


def calc_emissivity_thickness(fq, snow_thickness=1, T=265,
                              snow_ms_model='sticky_hard_spheres',
                              theta=53.,
                              sn_density=320,
                              substrate='MYI',
                              roughness_rms=None):
    ''' Calculate emissivity for different thickness '''
    e1_h, e1_v = calc_e(fq, snow_thickness, T,
                        snow_ms_model, theta, sn_density,
                        substrate, roughness_rms)
    return e1_h, e1_v

####################################################################
# Calculate emissivity at fixed angle for different snow thicknesses
####################################################################

# Snow microstructure models
sn_ms_model_list = ['autocorrelation', 'exponential', 'gaussian_random_field',
                    'homogeneous', 'independent_sphere', 'sampled_autocorrelation',
                    'sticky_hard_spheres','test_autocorrelation', 'test_exponential',
                    'test_sticky_hard_spheres', 'teubner_strey', 'unified_autocorrelation',
                    'unified_scaled_exponential', 'unified_sticky_hard_spheres', 'unified_teubner_strey']

# Basic parameters
fq_list = [7e9, 11e9, 19e9, 24e9, 37e9, 89e9]
polarizations = ['h', 'v']
theta = 55

sn_th_max = 3.
sn_th_min = 0.01
sn_th_step = 0.1

sn_density = 600
substrate = 'land'
roughness_rms = 0.01
T = 265.

snow_ms_model = sn_ms_model_list[6]
densities_list = list(range(100, 700, 100))

d_res = {}

for sn_density in densities_list:
    d_res[sn_density]= {}
    for pol in polarizations:
        d_res[sn_density][pol]= {}
        d_res[sn_density][pol]= {}

ll_e1_h, ll_e1_v = [], []

for sn_density in densities_list:
    print(f'\nSnowpack density: {sn_density}')
    for idx, fq in enumerate(fq_list):
        print(fq/1e9)

        # Calculate E for different snow thickness
        for s_th in np.arange(sn_th_min, sn_th_max, sn_th_step):
            e1_h, e1_v = calc_emissivity_thickness(fq=fq,
                                                   snow_thickness=[s_th],
                                                   T=T,
                                                   snow_ms_model=snow_ms_model,
                                                   sn_density=sn_density,
                                                   substrate=substrate,
                                                   theta=theta,
                                                   roughness_rms=roughness_rms)
            ll_e1_h.append(e1_h)
            ll_e1_v.append(e1_v)

        fq_str = fq / 1e9

        d_res[sn_density]['h'][fq_str] = {}
        d_res[sn_density]['v'][fq_str] = {}

        d_res[sn_density]['h'][fq_str]['e1'] = ll_e1_h
        d_res[sn_density]['v'][fq_str]['e1'] = ll_e1_v

        ll_e1_h, ll_e1_v = [], []

ncols = 3
nrows = int(np.ceil(len(d_res.keys()) / ncols))

if substrate == 'land':
    substrate_title = substrate + ' (Wegmuller)'
else:
    substrate_title = substrate

plt.clf()

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15))

r_ch = 0
c_ch = 0

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

for i, density in enumerate(d_res.keys()):
    if r_ch >= ncols:
        r_ch = 0
        c_ch += 1

    pol = 'h'
    for i_color, ikey in enumerate(d_res[density][pol]):
        ax[c_ch, r_ch].plot(np.arange(sn_th_min, sn_th_max, sn_th_step), d_res[density][pol][ikey]['e1'],
                            label=f'{pol.upper()} {ikey} GHz',
                            linestyle='dashed', c=colors[i_color])

    ax[c_ch, r_ch].set_title(f'Snow density={density},'
                             f'$\Theta$={theta}$^\circ$\nSubstrate:'
                             f'{substrate_title}\n Surface roughnes={roughness_rms}',
                             fontsize='small')

    pol = 'v'
    for i_color, ikey in enumerate(d_res[density][pol]):
        ax[c_ch, r_ch].plot(np.arange(sn_th_min, sn_th_max, sn_th_step), d_res[density][pol][ikey]['e1'],
                            label=f'{pol.upper()} {ikey} GHz',
                            linewidth=3, c=colors[i_color])

    ax[c_ch, r_ch].legend(loc='lower right', prop={'size': 6})
    ax[c_ch, r_ch].grid()

    r_ch += 1

plt.xlabel('Snow thickness, m')
plt.ylabel('Emissivity')
plt.subplots_adjust(hspace=0.2)




