import WLmap as wlm

# WLmap has 1 class/object aptly named WLmap. It requires the 5 input arguments given below 
# which are set at their default values here. It will read or calculate everything else automatically, such
# as the cosmology of the simulation.
sim_dir = '/example/simulation/data'
z_max = 3
opening_angle = 5 #in degrees
plane_sep = 50 #in Mpc
map_size =  1800 #in pixels per side

# Let's initialise WLmap
m = wlm.WLmap(sim_dir=sim_dir, z_max=z_max, opening_angle=opening_angle,
                             plane_sep = plane_sep, map_size=map_size)

# WLmap builds lightcones/lense planes by stacking snapshots out to a maximum redshift, z_max. The 
# stacked snapshots are then seperated into chunks with width of plane_sep. All the particles in a chunk that
# fall within the lightcone of given opening angle are then projected onto a plane along the line of sight of 
# the lightcone. It returns maps of mass, normalised by the DM particle mass, for each particle type.  
# Let's generate a lightcone.
# Note: This will take ~3 hours for a DM-only simulation with 1024^3 particles and box size of 400 Mpc
# and default initial parameters.
lightcone = m.gen_lightcone()

# The gen_lightcone() function will return the lense planes but they are also stored in the WLmap object.
lightcone = m.lightcone

# Next we can calculate the convergence and reduced shear maps which are also returned and stored.
# Different redshift distributions are available which are calculated in redshift_distribution().
m.gen_convergence_map()
kappa = m.convergence_map

m.gen_shear_maps()
shear1 = m.shear1_map
shear2 = m.shear2_map

# We can save the stored lightcone and/or the calculated maps as HDF5 files in the defined
# ouput directory. It will save them seperately as 'filename_lightcone.hdf5' and 'filename_maps.hdf5'.
# Each HDF5 file will also contain a 'Header' group that contains all important information WLmap needs 
# to reproduce a lightcone with the same input parameters and cosmology.
m.save_data(filename='./example')


# We can initialising WLmap with a 'Header' from a lightcone or maps HDF5 file.
m = wlm.WLmap(load_header='./example_lightcone.hdf5')

# We can also load a lightcone and convergence and shear maps. This will read the 'Header' and 
# re-initilaise by default.
# Note: Reading the 'Header' and re-initilaising deletes all previously stored variable from that class
# instance. To avoid this, set load_header=False.
m.load_lightcone(filename='./example_lightcone.hdf5')
m.load_map(filename='./example_maps.hdf5', map_name='convergence', load_header=False)
m.load_map(filename='./example_maps.hdf5', map_name='shear', load_header=False)

# Alternatively, initialise by loading a lightcone
m = wlm.WLmap(load_lightcone='./example_lightcone.hdf5')
