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
import argparse

def setup_snowpack(model='sticky_hard_spheres',
                   thickness_1=1,
                   sn_density=320,
                   T=265.,
                   radius=0.3e-3,
                   stickiness=0.2):
    '''
    Make a snow layer
    '''

    print(f'\nMicrostructure model: {model}')
    print(f'\nSnow density: {sn_density}')
    print(f'Snow thickness: {thickness_1}')

    # Best?
    if model == 'sticky_hard_spheres':
        sp = make_snowpack(thickness_1, model,
                           density=sn_density,
                           temperature=T,
                           radius=radius,
                           stickiness=stickiness)
    if model == 'independent_sphere':
        sp = make_snowpack(thickness_1, model, density=sn_density, radius=radius, stickiness=stickiness)
    if model == 'homogeneous':
        sp = make_snowpack(thickness_1, model, density=sn_density, radius=radius, stickiness=stickiness)
    if model == 'sampled_autocorrelation':
        sp = make_snowpack(thickness_1, model, density=sn_density, radius=radius, stickiness=stickiness)

    return sp

def calc_e(fq, thickness_1, T, model, theta, sn_density, substrate, roughness_rms, eq_num, radius, stickiness):
    '''
    Calculate emissivity over snowpack

    :param fq: frequency
    :param thickness_1: max snow thickness [m]
    :param T: temperature at surface [K]
    :param model: snow microstructure model
    :param theta: incidence angle
    :param sn_density: snow density [kg/m3]
    :param substrate: substrate name
    :param roughness_rms: roughness [mm]
    :param radius: radii of particels [mm]
    :param stickiness: stickiness
    :return: emissitivity at H- and V-polarization
    '''

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

    Tbdown = 1
    atmosphere1K = SimpleIsotropicAtmosphere(tbdown=Tbdown,
                                             tbup=0,
                                             trans=1)

    print(f'Crystal radius: {radius_um}')
    print(f'Crystal stickiness: {stickiness}')
    snowpack = setup_snowpack(T=T, model=model,
                              thickness_1=thickness_1,
                              sn_density=sn_density,
                              radius=radius,
                              stickiness=stickiness)

    # add snowpack on top of substrate:
    medium = snowpack + substrate

    # create the sensor
    radiometer = sensor_list.passive(fq, theta)

    n_max_stream = 128  # TB calculation is more accurate if number of streams is increased (currently: default = 32);

    # create the EM Model
    m = make_model("iba", "dort",
                   rtsolver_options={"n_max_stream": n_max_stream})  # , 'phase_normalization': "forced"})

    # run the model
    sresult_0 = m.run(radiometer, medium)
    medium.atmosphere = atmosphere1K
    sresult_1 = m.run(radiometer, medium)

    # V-pol
    if eq_num == '1':
        print(f'Equation {eq_num}')
        #reflectivity_V = ( sresult_1.TbV() - sresult_0.TbV() ) / Tbdown
        emissivity_V = 1 - (sresult_1.TbV() - sresult_0.TbV()) / Tbdown
    elif eq_num == '2':
        print(f'Equation {eq_num}')
        emissivity_V = (sresult_0.TbV()) / T
    else:
        raise('Please specify correct equation number (1/2)')

    # From old code (incorrect?)
    #emissivity_V = (sresult_0.TbV() + sresult_1.TbV()) / 2 / T

    print(f'sresult_1.TbV(),sresult_0.TbV: {sresult_1.TbV()},{sresult_0.TbV()}')
    print(f'Ev={emissivity_V}')

    # H-pol
    if eq_num == '1':
        print(f'Equation {eq_num}')
        #reflectivity_H = ( sresult_1.TbH() - sresult_0.TbH() ) / Tbdown
        emissivity_H = 1 - (sresult_1.TbH() - sresult_0.TbH()) / Tbdown
    elif eq_num=='2':
        print(f'Equation {eq_num}')
        emissivity_H = (sresult_0.TbH()) / T
    else:
        raise('Please specify correct equation number (1/2)')

    # From old code (incorrect?)
    #reflectivity_H = (sresult_0.TbH() + sresult_1.TbH()) / 2 / T

    print(f'Eh={emissivity_H}')

    return emissivity_H, emissivity_V

def calc_emissivity_thickness(fq, snow_thickness=1, T=265,
                              snow_ms_model='sticky_hard_spheres',
                              theta=53.,
                              sn_density=320,
                              substrate='MYI',
                              roughness_rms=None,
                              eq_num=1,
                              radius=0.3e-3,
                              stickiness=0.2):
    ''' Calculate emissivity for different thickness '''
    e1_h, e1_v = calc_e(fq, snow_thickness, T,
                        snow_ms_model, theta, sn_density,
                        substrate, roughness_rms, eq_num, radius, stickiness)
    return e1_h, e1_v

####################################################################
# Calculate emissivity at fixed angle for different snow thicknesses
####################################################################

parser = argparse.ArgumentParser(description='Calculate Emissivity')
parser.add_argument('-o', '--out_folder',
                    required=True,
                    help='Output folder')

parser.add_argument('-e', '--e_equation',
                    required=False,
                    default=1,
                    help='E-equation number')

parser.add_argument('-sub', '--substrate',
                    required=False,
                    default='land',
                    help='Substrate type')

parser.add_argument('-s', '--stickiness',
                    required=False,
                    default=0.2,
                    help='Stickiness')

parser.add_argument('-r', '--radius',
                    required=False,
                    default=0.4e-3,
                    help='Sphere radius')


args = parser.parse_args()
out_path = args.out_folder
if args.e_equation:
    eq_num = args.e_equation
else:
    eq_num = 1

##############################
# Snow microstructure models
##############################
sn_ms_model_list = ['autocorrelation', 'exponential', 'gaussian_random_field',
                    'homogeneous', 'independent_sphere', 'sampled_autocorrelation',
                    'sticky_hard_spheres','test_autocorrelation', 'test_exponential',
                    'test_sticky_hard_spheres', 'teubner_strey', 'unified_autocorrelation',
                    'unified_scaled_exponential', 'unified_sticky_hard_spheres', 'unified_teubner_strey']

snow_ms_model = sn_ms_model_list[6]

############################
# Instrument parameters
############################
fq_list = [7e9, 11e9, 19e9, 24e9, 37e9, 89e9]
polarizations = ['h', 'v']
theta = 55

############################
# Snowpack parameters
############################
sn_th_max = 3.
sn_th_min = 0.01
sn_th_step = 0.1
sn_density = 600
densities_list = list(range(100, 700, 100))

############################
# Substrate parameters
############################
substrate = args.substrate
roughness_rms = 0.01
T = 265.

############################
# Microstructure parameters
############################
# Crystal radius
radius = float(args.radius)
radius_um = int(radius*10**6)
# Stickiness of 0.2 is recommended as a first guess in SMRT [Loewe and Picard, 2015]
stickiness = float(args.stickiness)

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
        # Calculate E
        for s_th in np.arange(sn_th_min, sn_th_max, sn_th_step):
            e1_h, e1_v = calc_emissivity_thickness(fq=fq,
                                                   snow_thickness=[s_th],
                                                   T=T,
                                                   snow_ms_model=snow_ms_model,
                                                   sn_density=sn_density,
                                                   substrate=substrate,
                                                   theta=theta,
                                                   roughness_rms=roughness_rms,
                                                   eq_num=eq_num,
                                                   radius=radius,
                                                   stickiness=stickiness)
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

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12))

# Move plot title inside panel
plt.rcParams['axes.titley'] = 0.1

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

    '''
    ax[c_ch, r_ch].set_title(f'Sn. density={density}\ kg/m^3,$\ \Theta$={theta}$^\circ$'
                             f'\nSubstrate:$\ $ {substrate_title}'
                             f'\nSurface roughnes={roughness_rms}'
                             f'\nMicr. model={snow_ms_model}'
                             f'\nStickiness={stickiness}'
                             f'\nCrystal R={radius_um}$\ \mu m$',
                             fontsize='medium')
    '''

    text = f'Sn. density={density}$\ kg/m^3,\ \Theta$={theta}$^\circ$' \
        f'\nSubstrate:$\ $ {substrate_title}' \
        f'\nSurface roughnes={roughness_rms}' \
        f'\nMicr. model={snow_ms_model}' \
        f'\nStickiness={stickiness}' \
        f'\nCrystal R={radius_um}$\ \mu m$'

    ax[c_ch, r_ch].text(.01, .01, text, ha='left', va='bottom',
                        transform=ax[c_ch, r_ch].transAxes,
                        fontsize='medium')

    pol = 'v'
    for i_color, ikey in enumerate(d_res[density][pol]):
        ax[c_ch, r_ch].plot(np.arange(sn_th_min, sn_th_max, sn_th_step), d_res[density][pol][ikey]['e1'],
                            label=f'{pol.upper()} {ikey} GHz',
                            linewidth=3, c=colors[i_color])

    ax[c_ch, r_ch].legend(loc='lower right', prop={'size': 8})
    ax[c_ch, r_ch].grid(linewidth=0.15)

    ax[c_ch, r_ch].set_ylim([0.25, 1.01])

    if r_ch > 0:
        ax[c_ch, r_ch].axes.yaxis.set_ticklabels([])

    if r_ch == 0:
        ax[c_ch, r_ch].set_ylabel('Emissivity')

    r_ch += 1

plt.xlabel('Snow thickness, m')
#plt.ylabel('Emissivity')

plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.1)

plt.savefig(f'{out_path}/E_{substrate}_r{radius_um}_s{stickiness}_{snow_ms_model}.png', bbox_inches='tight', dpi=300)