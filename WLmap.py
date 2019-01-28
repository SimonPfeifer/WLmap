import os
import h5py
import eagle as E
import numpy as np

def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True

class WLmap:
    def __init__(self, sim_dir=None, load_header=None, load_lightcone=None,
                        z_max=3, opening_angle=5, plane_sep=50, map_size=1800, seed=None):
        if load_header is not None:
            self.load_header(filename=load_header)
        elif load_lightcone is not None:
            self.load_header(filename=load_lightcone)
        else:
            self.sim_dir = sim_dir
            self.z_max = float(z_max)
            self.opening_angle = float(opening_angle) #in degrees
            self.plane_sep = int(plane_sep) #in Mpc h^-1
            self.map_size = int(map_size)
            if seed is None:
                self.seed = np.random.randint(1E6)
            else:
                self.seed = seed
        # Make sure strings are not in bytes
        if type(self.sim_dir) is bytes:
                self.sim_dir = self.sim_dir.decode('UTF-8')
        # Check paths exist
        if not os.path.isdir(self.sim_dir):
            raise ValueError('Simulation directory does not exist: {}'.format(sim_dir))
                
        # Simulation data
        self.snap_dir = [d for d in next(os.walk(self.sim_dir))[1] if d.startswith('snap')] # check available snapshots
        if len(self.snap_dir) is 0:
            raise ValueError('No snapshots found in directory: {}'.format(self.sim_dir))
        self.snap_dir.sort()
        self.n_snap = int(self.snap_dir[-1][-3:]) #snapshot number at z=0
        self.box_size = E.readAttribute('SNAPSHOT', self.sim_dir, format(self.n_snap, '03d'), '/Header/BoxSize')
        if self.plane_sep > self.box_size:
            raise ValueError('Maximum plane seperation exceeds simulation box size: reduce plane seperation')
        self.n_particles = int(E.readAttribute('SNAPSHOT', self.sim_dir, format(self.n_snap, '03d'), '/Header/NumPart_Total')[1])
        if np.sum(E.readAttribute('SNAPSHOT', self.sim_dir, format(self.n_snap, '03d'), '/Header/NumPart_Total') > 0) == 1:
            self.dmonly = True
        else:
            self.dmonly = False

        # Cosmology information
        self.h0 = E.readAttribute('SNAPSHOT', self.sim_dir, format(self.n_snap, '03d'), '/Header/HubbleParam')
        self.omega_m = E.readAttribute('SNAPSHOT', self.sim_dir, format(self.n_snap, '03d'), '/Header/Omega0')
        self.omega_b = E.readAttribute('SNAPSHOT', self.sim_dir, format(self.n_snap, '03d'), '/Header/OmegaBaryon')
        self.omega_l = E.readAttribute('SNAPSHOT', self.sim_dir, format(self.n_snap, '03d'), '/Header/OmegaLambda')
        self.w = -1.0
        self.wa = 0.0

        # Local class variable
        # Calculate the distance and width of the last plane
        self.d_max = float(np.ceil(self.redshift2mpc(self.z_max) / self.plane_sep) * self.plane_sep)
        self.w_max = float(2 * self.d_max * np.tan(self.opening_angle / 2.0 * np.pi/180.0))
        if self.w_max > self.box_size:
            raise ValueError('Maximum light cone width exceeds simulation box size: reduce opening angle and/or redshift')

        # Calculate the distance and redshift to all planes and mid-points between them
        self.d_planes = np.arange(0.0, self.d_max+self.plane_sep, self.plane_sep)
        self.z_planes = np.asarray([self.mpc2redshift(d) for d in self.d_planes])
        self.d_midplanes = (self.d_planes[1:] + self.d_planes[:-1]) / 2.0
        self.z_midplanes = np.asarray([self.mpc2redshift(d) for d in self.d_midplanes])
        
        # Read redshifts for available snapshots and calculate lookback distances for them
        self.z_snap_output = [E.readAttribute('SNAPSHOT', self.sim_dir, format(tag, '03d'), '/Header/Redshift') for tag in range(self.n_snap+1-len(self.snap_dir), self.n_snap+1)[::-1]]
        if np.max(self.z_snap_output) < self.z_max:
            raise ValueError('Maximum light cone redshift exceeds maximum snapshot redshift: reduce redshift value')
        self.d_snap_output = np.asarray([self.redshift2mpc(z) for z in self.z_snap_output])

        # Analytically calculate the mean density (10E10 M_sun Mpc-3) and read in DM particle mass (10E10 M_sun h-1) 
        self.density_critical = ((3.0 * (self.h0*100)**2) / (8.0 * np.pi * 6.67408E-11)) * 1.54463E-12 /self.h0**2 #1.5523E-12
        self.density_mean = self.density_critical * self.omega_m
        self.dm_particle_mass = E.readAttribute('SNAPSHOT', self.sim_dir, format(self.n_snap, '03d'), '/Header/MassTable')[1]

        # Store results
        self.lightcone = None
        self.convergence_map = None
        self.shear1_map = None
        self.shear2_map = None
        if load_lightcone is not None:
            self.load_lightcone(filename=load_lightcone, load_header=False)

    def H(self, z, n=1000):
        '''Calculate H(z) for a given cosmology.'''
        a = 1.0/(1 + z)
        H = self.omega_m * np.power(a, -3)
        H += self.omega_l * np.exp(-3.0 * (self.wa*(1-a) + (1+self.wa+self.w)*np.log(a)))
        H = self.h0 * 100 * np.sqrt(H)
        return H

    def redshift2mpc(self, z, n=1000):
        '''Convert a redshift to a distance in Mpc h^-1 for a given cosmology.'''
        x = np.linspace(0, z, n, endpoint=True)
        d = np.trapz(2.998E5 / self.H(x, n), x) * self.h0
        return d 

    def mpc2redshift(self, mpc, n=1000, dz=0.1):
        '''Convert a distance in Mpc h^-1 to a redshift for a given cosmology.'''
        z_list = [0]
        mpc_list = [0]
        while mpc_list[-1] < mpc:
            z_list.append(z_list[-1]+dz)
            mpc_list.append(self.redshift2mpc(z_list[-1], n))
        z = np.interp(mpc, mpc_list, z_list)
        return z

    def read_snapshot(self, sim_dir, tag, part_type):
        '''Read  in and return snapshot data for different particle type: dm=0, gas=1, stars=2'''
        if part_type is 0:
            pos = E.readArray('SNAPSHOT', sim_dir, tag, '/PartType1/Coordinates', noH=False, physicalUnits=False)
            mass = np.ones(self.n_particles)
        # If gas and stars are included, normalise their mass by the dark matter particle mass
        elif part_type is 1:
            pos = E.readArray('SNAPSHOT', sim_dir, tag, '/PartType0/Coordinates', noH=False, physicalUnits=False)
            mass = E.readArray('SNAPSHOT', sim_dir, tag, '/PartType0/Mass', noH=False, physicalUnits=False) / self.dm_particle_mass
        elif part_type is 2:
            pos = E.readArray('SNAPSHOT', sim_dir, tag, '/PartType4/Coordinates', noH=False, physicalUnits=False)
            mass = E.readArray('SNAPSHOT', sim_dir, tag, '/PartType4/Mass', noH=False, physicalUnits=False) / self.dm_particle_mass
        else:
            raise ValueError('Invalid particle type: {}. Allowed values are dm=0, gas=1, stars=2'.format(part_type))
        return pos, mass

    def translate_coordinates(self, coord, offset):
        ''' Translate coordinates by an offset.'''
        return np.asarray(coord)+np.asarray(offset)

    def wrap_coordinates(self, coord):
        '''Wrap coordinates that are outside square/cube boundaries.'''
        coord = np.mod(coord, self.box_size)
        return coord

    def rot90(self, coord, axis, clockwise=True):
        '''Rotate coordinates 90 degrees around axis x, y or z.'''
        axes = [0, 1, 2]
        axes.pop(axis)
        axes = axes[::-1]
        axes.insert(axis, axis)
        coord = np.asarray(coord)[:, axes]
        if clockwise:
            coord[:, (axis+1)%3] *= -1
        else:
            coord[:, (axis-1)%3] *= -1
        return coord

    def rot180(self, coord, axis):
        '''Rotate coordinates 180 degrees around axis x, y or z.'''
        axes = [0, 1, 2]
        axes.pop(axis)
        coord[:, axes] *= -1
        return coord
    
    def randomise_coordinates(self, coord, n_rot=3):
        '''Randomly translate and rotate an array of 3D coordinates.'''
        # Random coordinate translation with maximum size of the simulation box size
        coord = self.translate_coordinates(coord, (np.random.random(3)-0.5)*self.box_size)
        coord = self.wrap_coordinates(coord)

        # Translate origin to center of box, rotate randomly by factors of 90 degrees and translate back
        coord -= self.box_size/2
        rotations = [np.random.randint(0,3,2) for i in range(n_rot)]
        for rotation in rotations:
            r, axis = rotation
            if r == 0:
                coord = self.rot90(coord, axis=axis)
            if r == 1:
                coord = self.rot90(coord, axis=axis, clockwise=False)
            if r == 2:
                coord = self.rot180(coord, axis=axis)
        coord += self.box_size/2
        return coord

    def redshift_distribution(self, z, survey='cfhtlens_fit'):
        '''Return the probability from a normalised source redshift distribution for a particular survey for a given redshift.'''
        z = np.asarray(z)
        if survey == 'cfhtlens_fit':
            # arxiv.org/pdf/1303.1806.pdf
            n_s = 1.5 * np.exp(- (z-0.7)**2 / 0.32**2) + 0.2 * np.exp(-(z-1.2)**2 / 0.46**2)
        else:
            raise ValueError('Invalid survey argument: {}'.format(survey))
        return n_s

    def lensing_kernel(self, survey='cfhtlens_fit'):
        '''Calculate the lensing kernel for a set of lens planes and given source redshift distribution.'''
        z_width = self.z_planes[1:] - self.z_planes[:-1]
        n_s = self.redshift_distribution(self.z_midplanes, survey=survey)
        g = np.zeros(len(self.d_midplanes))
        for i, d in enumerate(self.d_midplanes[:-1]):
            g[i] = self.d_midplanes[i] * np.sum(n_s[i+1:] * (1.0 - self.d_midplanes[i]/self.d_midplanes[i+1:]) * z_width[i+1:])
        return g

    def gen_projection_plane(self, pos, mass=None):
        '''Generate a single 2D projection plane from a 3D particle distribution along a square cone.'''
        x_s = (self.w_max*(pos[:,2]-self.d_max) + 2*self.d_max*pos[:,0]) / (2*self.w_max*pos[:,2])
        y_s = (self.w_max*(pos[:,2]-self.d_max) + 2*self.d_max*pos[:,1]) / (2*self.w_max*pos[:,2])
        n, xedges, yedges = np.histogram2d(x_s, y_s, bins=[self.map_size, self.map_size], range=[[0,1],[0,1]], weights=mass)
        return n

    def gen_lightcone(self):
        '''Generate a set of equally spaced (in distance), 2D projection planes from a set of simulation snapshot particle distributions.'''
        if module_exists('tqdm'):
            from tqdm import tqdm
            tqdm_exists = True

        # Set up an empty array to store the projection planes
        if self.dmonly:
            part_types = [0]
            self.lightcone = np.zeros([1, len(self.d_midplanes), self.map_size, self.map_size])
        else:
            part_types = [0, 1, 2]
            self.lightcone = np.zeros([3, len(self.d_midplanes), self.map_size, self.map_size])

        # Calculate the tag for the snapshot that the center of the 
        # particle distribution used to make each projection plane is closest to
        plane_snap_tag = [np.argmin(abs(self.d_snap_output - d)) for d in self.d_midplanes]
        plane_snap_tag = [format(self.n_snap-i, '03d') for i in plane_snap_tag]
        for part_type in part_types: # Generate a lightcone/projection planes for each particle type
            if tqdm_exists:
                pbar = tqdm(total=len(self.d_midplanes)-1)
            d = 0.0
            d_box = 0.0
            np.random.seed(self.seed)
            snap_tag = plane_snap_tag[0]
            pos, mass = self.read_snapshot(self.sim_dir, snap_tag, part_type)
            pos = self.randomise_coordinates(pos)
            pos = self.translate_coordinates(pos, [0,0,d])
            for i, _ in enumerate(self.d_midplanes):
                if plane_snap_tag[i] == snap_tag: # Check if the new plane uses the same snapshot redshift
                    if d_box+self.plane_sep > self.box_size: #Check if new plane is outside of the current snapshot box boundary
                        # Rotate and translate current snapshot again to get a new realisation
                        pos = self.randomise_coordinates(pos)
                        pos = self.translate_coordinates(pos, [0,0,d])
                        d_box = 0
                else: # If new plane has a different redshift to current snapshot, load new snapshot
                    pos, mass = self.read_snapshot(self.sim_dir, plane_snap_tag[i], part_type)
                    pos = self.randomise_coordinates(pos)
                    pos = self.translate_coordinates(pos, [0,0,d])
                    snap_tag = plane_snap_tag[i]
                    d_box = 0

                # Mask particles outside of lightcone boundaries
                mask = (pos[:,2]>d) & (pos[:,2]<d+self.plane_sep)
                pos_tmp = pos[mask]
                mass_tmp = mass[mask]

                # Collapse particles onto a plane
                self.lightcone[part_type, i] = self.gen_projection_plane(pos_tmp, mass_tmp)
                d_box += self.plane_sep
                d += self.plane_sep

                if tqdm_exists:
                    pbar.update(1)
            if tqdm_exists:
                pbar.close()
        return self.lightcone

    def gen_overdensity_maps(self):
        '''Generate overdensity maps from a set of projected 2D particle planes.'''
        # Calculate the collapsed volume for each plane corresponding to a square truncated pyramid
        plane_width = self.w_max * self.d_planes/self.d_max
        plane_volume = np.asarray([self.plane_sep/3.0 * (plane_width[i]**2 + plane_width[i]*plane_width[i+1] + plane_width[i+1]**2) for i, pw in enumerate(plane_width[:-1])]) # V=1/3*(x1**2 + x1*x2 + x2**2)*h

        # Calculate the mean collapsed volume per pixel (voxel) and from that it's projected density
        voxel_volume = plane_volume / (self.map_size**2)
        voxel_density = self.dm_particle_mass / voxel_volume # this adds a zero for the first/observers plane

        # Calculate the overdensity with respect to the mean density of the simulation
        if self.dmonly:
            overdensity_maps= (self.lightcone * voxel_density[:, None, None] - self.density_mean) / self.density_mean
        else:
            overdensity_maps= (np.sum(self.lightcone, axis=0) * voxel_density[:, None, None] - self.density_mean) / self.density_mean
        return overdensity_maps

    def gen_convergence_map(self, survey='cfhtlens_fit'):
        '''Generate convergence maps from a set of overdensity maps.'''
        if self.lightcone is None:
            self.gen_lightcone()
        overdensity_maps = self.gen_overdensity_maps()
        g = self.lensing_kernel(survey=survey)
        constant = (3.0 * self.omega_m * (self.h0*100)**2) / (2 * 2.9979E5**2) / (self.h0)**2 * (1 + self.z_midplanes) * g * self.plane_sep
        self.convergence_map = np.sum(overdensity_maps * constant[:, None, None], axis=0)
        return self.convergence_map

    def gen_shear_maps(self, survey='cfhtlens_fit'):
        '''Generate the 2 component shear maps from a convergence map.'''
        # Fourier transform convergence map and pad edges with zeros
        if self.convergence_map is None:
            self.gen_convergence_map(survey=survey)
        kappa_fft = np.fft.fft2(self.convergence_map)
        kappa_fft[:,[0,-1]] = kappa_fft[[0,-1]] = 0.0

        # Set up wave vectors and scale by the map in radians
        grid_range = np.arange(self.map_size, dtype='float') - (int(self.map_size) - 1) / 2
        l1, l2 = np.meshgrid(grid_range, grid_range)
        l1 = np.roll(l1, int(self.map_size/2), axis=1) * 2.0 * np.pi / (np.pi/180.0 * self.opening_angle)
        l2 = np.roll(l2, int(self.map_size/2), axis=0) * 2.0 * np.pi / (np.pi/180.0 * self.opening_angle)
        
        # Remove bottom-right corner zero to stop division by zero on next 2 lines
        # The zero edge padding for the kappa map removes this effect later
        l1[-1, -1] = l2[-1, -1] = 1 

        # Calculate shear components
        g1 = (np.square(l1) - np.square(l2)) / (np.square(l1) + np.square(l2)) * kappa_fft
        g2 = 2.0 * (l1 * l2) / (np.square(l1) + np.square(l2)) * kappa_fft
        g1 = np.fft.ifft2(g1).real 
        g2 = np.fft.ifft2(g2).real
        self.shear1_map = g1 / (1.0 - self.convergence_map)
        self.shear2_map = g2 / (1.0 - self.convergence_map)
        return self.shear1_map, self.shear2_map

    def gen_header(self, f):
        '''Create header file to store variables from __init__'''
        header = f.require_group('Header')
        header.attrs['sim_dir'] = self.sim_dir
        header.attrs['redshift'] = self.z_max
        header.attrs['opening_angle'] = self.opening_angle
        header.attrs['plane_seperation'] = self.plane_sep
        header.attrs['map_size'] = self.map_size
        header.attrs['seed'] = self.seed
        header.attrs['h0'] = self.h0
        header.attrs['omega_m'] = self.omega_m
        header.attrs['omega_b'] = self.omega_b
        header.attrs['omega_l'] = self.omega_l
        header.attrs['w'] = self.w
        header.attrs['wa'] = self.wa

    def save_data(self, filename, lightcone=True, maps=True):
        '''Save all important data from current class instance.'''
        if lightcone:
            # Create a file for the lightcone
            with h5py.File(filename + '_lightcone.hdf5', 'w') as f:
                self.gen_header(f)
                # Lightcone
                lc = f.require_group('Lightcone')
                if self.lightcone is not None:
                    if self.dmonly:
                        lc.create_dataset('lightcone_dm', shape=np.shape(self.lightcone),
                                                    dtype='float64', data=self.lightcone)
                    else:
                        lc.create_dataset('lightcone_dm', shape=np.shape(self.lightcone[0]),
                                                    dtype='float64', data=self.lightcone[0])
                        lc.create_dataset('lightcone_gas', shape=np.shape(self.lightcone[1]),
                                                    dtype='float64', data=self.lightcone[1])
                        lc.create_dataset('lightcone_stars', shape=np.shape(self.lightcone[2]),
                                                    dtype='float64', data=self.lightcone[2])
        if maps:
            # Create a file for the maps and any other data
            with h5py.File(filename + '_maps.hdf5', 'w') as f:
                self.gen_header(f)
                # Cosmic shear
                if self.convergence_map is not None or self.shear1_map is not None or self.shear2_map is not None:
                    cs = f.require_group('CosmicShear')
                    if self.convergence_map is not None:
                        cs.create_dataset('convergence', shape=np.shape(self.convergence_map),
                                                    dtype='float64', data=self.convergence_map)
                    if self.shear1_map is not None and self.shear2_map is not None:
                        cs.create_dataset('shear1', shape=np.shape(self.shear1_map),
                                                    dtype='float64', data=self.shear1_map)
                        cs.create_dataset('shear2', shape=np.shape(self.shear2_map),
                                                    dtype='float64', data=self.shear2_map)

    def load_header(self, filename):
        '''Load important header attributes required by __init__.'''
        with h5py.File(filename, 'r') as f:
            header = f['/Header']
            self.sim_dir = header.attrs.get('sim_dir')
            self.z_max = header.attrs.get('redshift')
            self.opening_angle = header.attrs.get('opening_angle')
            self.plane_sep = header.attrs.get('plane_seperation')
            self.map_size = header.attrs.get('map_size')
            self.seed = header.attrs.get('seed')

    def load_lightcone(self, filename, load_header=True):
        '''Load a lightcone and associated header.'''
        if load_header:
            self.__init__(load_header=filename)
        with h5py.File(filename, 'r') as f:
            lc = f['/Lightcone']
            if self.dmonly:
                self.lightcone = np.array(lc.get('lightcone_dm'))
            else:
                self.lightcone = [None, None, None]
                self.lightcone[0] = np.array(lc.get('lightcone_dm'))
                self.lightcone[1] = np.array(lc.get('lightcone_gas'))
                self.lightcone[2] = np.array(lc.get('lightcone_stars'))

    def load_map(self, filename, map_name, load_header=True):
        '''Load a map and associated header.'''
        if load_header:
            self.__init__(load_header=filename)
        if map_name == 'convergence':
            with h5py.File(filename, 'r') as f:
                cs = f['/CosmicShear']
                self.convergence_map = np.array(cs.get('convergence'))
        if map_name == 'shear':
            with h5py.File(filename, 'r') as f:
                cs = f['/CosmicShear']
                self.shear1_map = np.array(cs.get('shear1'))
                self.shear2_map = np.array(cs.get('shear2'))
