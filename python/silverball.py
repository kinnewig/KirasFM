#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#    Copyright (C) 2009-2015 Ovidio Peña Rodríguez <ovidio@bytesfall.com>
#    Copyright (C) 2013-2015  Konstantin Ladutenko <kostyfisik@gmail.com>
#
#    This file is part of python-scattnlay
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    The only additional remark is that we expect that all publications
#    describing work using this software, or all commercial products
#    using it, cite the following reference:
#    [1] O. Pena and U. Pal, "Scattering of electromagnetic radiation by
#        a multilayered sphere," Computer Physics Communications,
#        vol. 180, Nov. 2009, pp. 2348-2354.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle

import scattnlay
from scattnlay import fieldnlay
from scattnlay import scattnlay

# Define the material parameters
radius       = 100         # radius in nm
index_silver = 0.0 + 4.0j  # complex refractiv index of the silver ball

wavelength   = 600 # wavelength of the incoming planar wave in nm

k0 = 2.0 * np.pi / wavelength

nm = 1.0

x = np.array([k0 * radius], dtype = np.float64 )
m = np.array([index_silver], dtype = np.complex128 )

print("x =", x)
print("m =", m)

# Create the Grid
npts = 281

factor = 2.0
scan = np.linspace( -factor * k0 * radius, factor * k0 * radius, npts)

coordX, coordZ = np.meshgrid(scan, scan)
coordX.resize(npts*npts)
coordZ.resize(npts*npts)
coordY = np.zeros(npts*npts, dtype = np.float64)

# Compute the near field with the help of Mie's theory:
terms, E, H = fieldnlay(x, m, coordX, coordY, coordZ)

# |E|/|Eo|
Er   = np.absolute(E)
Eabs = np.sqrt(Er[:, 0]**2 + Er[:, 1]**2 + Er[:, 2]**2)

# If needed save the result:
#result  = np.vstack((coordX, coordY, coordZ, Eabs)).transpose()
#np.savetxt("field.txt", result, fmt = "%.5f")
#print(result)


# Create the Plot
Eabs_data   = np.resize(Eabs, (npts, npts)).T

# Rescale to better show the axes
scale_x = np.linspace( min(coordX) / k0, max(coordX) / k0, npts)
scale_z = np.linspace( min(coordZ) / k0, max(coordZ) / k0, npts)

# Interpolation can be 'nearest', 'bilinear' or 'bicubic'
plt.imshow(Eabs_data, interpolation = 'nearest', cmap = cm.viridis, origin = 'lower' , extent = (min(scale_x), max(scale_x), min(scale_z), max(scale_z)) )
plt.title('Eabs')
plt.axis("image")


# This part draws the nano shell
# create a circle to indicate the radius of the nano particle
center = (0, 0)
circle = Circle(center, radius, fill=False, edgecolor='white', linewidth=2)

# add the nano shell to the plot
plt.gca().add_artist(circle)

plt.savefig("Silverball.png")
plt.draw()

plt.show()

plt.close()

