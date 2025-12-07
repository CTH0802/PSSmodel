# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
This is a module of image fits data.
'''
__author__ = "Masayuki Yamaguchi"
#-------------------------------------------------------------------------
# Modules
#-------------------------------------------------------------------------
# standard packages
import os
import copy
import datetime as dt

# numerical packages
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, interp1d
import scipy.ndimage as sn
from scipy.interpolate import interp1d
from scipy import signal
import  random
from decimal import *

# asttronomical packages
import astropy.coordinates as coord
import astropy.io.fits as pyfits
import astropy.units as unit
from astropy.convolution import convolve_fft
#from astroquery.gaia import Gaia

# design packages
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm,rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import *
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
from matplotlib.patches import Ellipse
from matplotlib import ticker
from matplotlib import rc
import seaborn as sns

#del matplotlib.font_manager.weight_dict['roman']
#matplotlib.font_manager._rebuild()

# original packages
import util
from astrounit import * # import astronomical unit

#-------------------------------------------------------------------------
# IMAGEFITS (Manupulating FITS FILES)
#-------------------------------------------------------------------------
class IMFITS(object):
    angunit = "mas"

    # Initialization
    def __init__(self, fitsfile=None, uvfitsfile=None, source=None,
                 dx=2., dy=None, nx=100, ny=None, nxref=None, nyref=None,
                 angunit="arcsec", **args):
        '''
        This is a class to handle image data, in particular, a standard image
        FITS data sets.

        The order of priority for duplicated parameters is
            1 uvfitsfile (strongest),
            2 source
            3 fitsfile
            4 dx, dy, nx, nys
            5 other parameters (weakest).

        Args:
            fitsfile (string):
                input FITS file
            uvfitsfile (string):
                input uv-fits file
            source (string):
               The source of the image RA and Dec will be obtained from CDS
            dx (float):
               The pixel size along the RA axis. If dx > 0, the sign of dx
               will be switched.
            dy (float; default=abs(dx)):
               The pixel size along the Dec axis.
            nx (integer):
                the number of pixels in the RA axis.
            ny (integer; default=ny):
                the number of pixels in the Dec axis.
                Default value is same to nx.
            nxref (float, default=(nx+1)/2.):
                The reference pixel in RA direction.
                "1" will be the left-most pixel.
            nyref (float, default=(ny+1)/2.):
                the reference pixel of the DEC axis.
                "1" will be the bottom pixel.
            angunit (string):
                angular unit for fov, x, y, dx and dy.

            **args: you can also specify other header information.
                x (float):
                    The source RA at the reference pixel.
                y (float):
                    The source DEC at the reference pixel.
                dy (float):
                    the grid size in the DEC axis.
                    MUST BE POSITIVE for the astronomical image.
                f (float):
                    the reference frequency in Hz
                nf (integer):
                    the number of pixels in the Frequency axis
                nfref (float):
                    the reference pixel of the Frequency axis
                df (float):
                    the grid size of the Frequency axis
                s (float):
                    the reference Stokes parameter
                ns (integer):
                    the number of pixels in the Stokes axis
                nsref (float):
                    the reference pixel of the Stokes axis
                ds (float):
                    the grid size of the Stokes axis

                observer (string)
                telescope (string)
                instrument (string)
                object (string)
                dateobs (string)

        Returns:
            imdata.IMFITS object
        '''
        # get conversion factor for angular scale
        angconv = self.angconv(unit1 = angunit, unit2 = "deg")
        self.angunit = angunit

        # get keys of Args
        argkeys = args.keys()

        # Set header and data
        self.init_header()
        self.data = None

        # set pixel size
        if dy is None:
            dy = np.abs(dx)
        self.header["dx"] = -np.abs(dx)
        self.header["dy"] = dy

        # set pixel size
        if ny is None:
            ny = nx
        self.header["nx"] = nx
        self.header["ny"] = ny

        # ref pixel
        if nxref is None:
            nxref = (nx+1.)/2
        if nyref is None:
            nyref = (ny+1.)/2
        self.header["nxref"] = nxref
        self.header["nyref"] = nyref

        # read header from Args
        for argkey in argkeys:
            headerkeys = self.header.keys()
            if argkey in headerkeys:
                self.header[argkey] = self.header_dtype[argkey](args[argkey])

        self.header["x"] *= angconv
        self.header["y"] *= angconv
        self.header["dx"] *= angconv
        self.header["dy"] *= angconv
        self.data = np.zeros([self.header["ns"], self.header["nf"],
                              self.header["ny"], self.header["nx"]])

        # Initialize from fitsfile
        if fitsfile is not None:
            self.read_fits(fitsfile)

        # Set source coordinates
        if source is not None:
            self.set_source(source)

        # copy headers from uvfits file
        if uvfitsfile is not None:
            self.read_uvfitsheader(uvfitsfile)

        # initialize fitsdata
        self.update_fits()

    # Definition of Headers and their datatypes
    def init_header(self):
        header = {}
        header_dtype = {}

        # Information
        header["object"] = "NONE"
        header_dtype["object"] = str
        header["telescope"] = "NONE"
        header_dtype["telescope"] = str
        header["instrument"] = "NONE"
        header_dtype["instrument"] = str
        header["observer"] = "NONE"
        header_dtype["observer"] = str
        header["dateobs"] = "NONE"
        header_dtype["dateobs"] = str

        # RA information
        header["x"] = np.float64(0.)
        header_dtype["x"] = np.float64
        header["dx"] = np.float64(-1.)
        header_dtype["dx"] = np.float64
        header["nx"] = np.int64(1)
        header_dtype["nx"] = np.int64
        header["nxref"] = np.float64(1.)
        header_dtype["nxref"] = np.float64

        # Dec information
        header["y"] = np.float64(0.)
        header_dtype["y"] = np.float64
        header["dy"] = np.float64(1.)
        header_dtype["dy"] = np.float64
        header["ny"] = np.int64(1)
        header_dtype["ny"] = np.int64
        header["nyref"] = np.float64(1.)
        header_dtype["nyref"] = np.float64



        # BEAM Information
        header["BMAJ"] = np.float64(0.)
        header_dtype["BMAJ"] = np.float64
        header["BMIN"] = np.float64(0.)
        header_dtype["BMIN"] = np.float64
        header["BPA"] = np.float64(0.)
        header_dtype["BPA"] = np.float64

        # Third Axis Information
        header["f"] = np.float64(0.)
        header_dtype["f"] = np.float64
        header["df"] = np.float64(4e9)
        header_dtype["df"] = np.float64
        header["nf"] = np.int64(1)
        header_dtype["nf"] = np.int64
        header["nfref"] = np.float64(1.)
        header_dtype["nfref"] = np.float64
        header["restfreq"] = np.float64(0.)
        header_dtype["restfreq"] = np.float64

        # Stokes Information
        header["s"] = np.int64(1)
        header_dtype["s"] = np.int64
        header["ds"] = np.int64(1)
        header_dtype["ds"] = np.int64
        header["ns"] = np.int64(1)
        header_dtype["ns"] = np.int64
        header["nsref"] = np.int64(1)
        header_dtype["nsref"] = np.int64

        self.header = header
        self.header_dtype = header_dtype

    # set source name and source coordinates
    def set_source(self, source="SgrA*"):
        srccoord = coord.SkyCoord.from_name(source)

        # Information
        self.header["object"] = source
        self.header["x"] = srccoord.ra.deg
        self.header["y"] = srccoord.dec.deg
        self.update_fits()

    # Read data from an image fits file
    def read_fits(self, fitsfile):
        '''
        Read data from the image FITS file

        Args:
          fitsfile (string): input image FITS file
        '''
        hdulist = pyfits.open(fitsfile)
        self.hdulist = hdulist

        keyname = "RESTFRQ"
        try:
            self.header["restfreq"] = self.header_dtype["restfreq"](
                hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'"
                  % (keyname))

        keyname = "BMAJ"
        try:
            self.header["BMAJ"] = self.header_dtype["BMAJ"](
                hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'"
                  % (keyname))

        keyname = "BMIN"
        try:
            self.header["BMIN"] = self.header_dtype["BMIN"](
                hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'"
                  % (keyname))


        keyname = "BPA"
        try:
            self.header["BPA"] = self.header_dtype["BPA"](
                hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'"
                  % (keyname))

        keyname = "OBJECT"
        try:
            self.header["object"] = self.header_dtype["object"](
                hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'"
                  % (keyname))

        keyname = "TELESCOP"
        try:
            self.header["telescope"] = self.header_dtype["telescope"](
                hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'"
                  % (keyname))

        keyname = "INSTRUME"
        try:
            self.header["instrument"] = self.header_dtype["instrument"](
                hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'"
                  % (keyname))

        keyname = "OBSERVER"
        try:
            self.header["observer"] = self.header_dtype["observer"](
                hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'"
                  % (keyname))

        keyname = "DATE-OBS"
        try:
            self.header["dateobs"] = \
                self.header_dtype["dateobs"](hdulist[0].header.get(keyname))
        except:
            print("warning: FITS file doesn't have a header info of '%s'" % (keyname))

        isx = False
        isy = False
        isf = False
        iss = False
        naxis = hdulist[0].header.get("NAXIS")
        for i in range(naxis):
            ctype = hdulist[0].header.get("CTYPE%d" % (i + 1))
            if ctype is None:
                continue
            if ctype[0:2] == "RA":
                isx = i + 1
            elif ctype[0:3] == "DEC":
                isy = i + 1
            elif ctype[0:4] == "FREQ":
                isf = i + 1
            elif ctype[0:6] == "STOKES":
                iss = i + 1

        if isx != False:
            self.header["nx"] = \
                self.header_dtype["nx"](hdulist[0].header.get("NAXIS%d" % (isx)))
            self.header["x"] = \
                self.header_dtype["x"](hdulist[0].header.get("CRVAL%d" % (isx)))
            self.header["dx"] = \
                self.header_dtype["dx"](hdulist[0].header.get("CDELT%d" % (isx)))
            self.header["nxref"] = \
                self.header_dtype["nxref"](hdulist[0].header.get("CRPIX%d" % (isx)))
        else:
            print("Warning: No image data along RA axis.")

        if isy != False:
            self.header["ny"] = self.header_dtype["ny"](
                hdulist[0].header.get("NAXIS%d" % (isy)))
            self.header["y"] = self.header_dtype["y"](
                hdulist[0].header.get("CRVAL%d" % (isy)))
            self.header["dy"] = self.header_dtype["dy"](
                hdulist[0].header.get("CDELT%d" % (isy)))
            self.header["nyref"] = self.header_dtype["nyref"](
                hdulist[0].header.get("CRPIX%d" % (isy)))
        else:
            print("Warning: No image data along DEC axis.")

        if isf != False:
            self.header["nf"] = self.header_dtype["nf"](
                hdulist[0].header.get("NAXIS%d" % (isf)))
            self.header["f"] = self.header_dtype["f"](
                hdulist[0].header.get("CRVAL%d" % (isf)))
            self.header["df"] = self.header_dtype["df"](
                hdulist[0].header.get("CDELT%d" % (isf)))
            self.header["nfref"] = self.header_dtype["nfref"](
                hdulist[0].header.get("CRPIX%d" % (isf)))
        else:
            print("Warning: No image data along STOKES axis.")

        if iss != False:
            self.header["ns"] = self.header_dtype["ns"](
                hdulist[0].header.get("NAXIS%d" % (iss)))
            self.header["s"] = self.header_dtype["s"](
                hdulist[0].header.get("CRVAL%d" % (iss)))
            self.header["ds"] = self.header_dtype["ds"](
                hdulist[0].header.get("CDELT%d" % (iss)))
            self.header["nsref"] = self.header_dtype["nsref"](
                hdulist[0].header.get("CRPIX%d" % (iss)))
        else:
            print("Warning: No image data along STOKES axis.")

        self.data = np.nan_to_num(hdulist[0].data.reshape([self.header["ns"], self.header["nf"], self.header["ny"], self.header["nx"]]))

    def read_uvfitsheader(self, infits):
        '''
        Read header information from uvfits file

        Args:
          infits (string): input uv-fits file
        '''
        hdulist = pyfits.open(infits)
        hduinfos = hdulist.info(output=False)
        for hduinfo in hduinfos:
            idx = hduinfo[0]
            if hduinfo[1] == "PRIMARY":
                grouphdu = hdulist[idx]
        if not 'grouphdu' in locals():
            print("[Error] %s does not contain the Primary HDU" % (infits))

        if 'OBJECT' in grouphdu.header:
            self.header["object"] = self.header_dtype["object"](
                grouphdu.header.get('OBJECT'))
        else:
            self.header["object"] = self.header_dtype["object"]('None')

        if 'TELESCOP' in grouphdu.header:
            self.header["telescope"] = self.header_dtype["telescope"](
                grouphdu.header.get('TELESCOP'))
        else:
            self.header["telescope"] = self.header_dtype["telescope"]('None')

        if 'INSTRUME' in grouphdu.header:
            self.header["instrument"] = self.header_dtype["instrument"](
                grouphdu.header.get('INSTRUME'))
        else:
            self.header["instrument"] = self.header_dtype["instrument"]('None')

        if 'OBSERVER' in grouphdu.header:
            self.header["observer"] = self.header_dtype["observer"](
                grouphdu.header.get('OBSERVER'))
        else:
            self.header["observer"] = self.header_dtype["observer"]('None')

        if 'DATE-OBS' in grouphdu.header:
            self.header["dateobs"] = self.header_dtype["dateobs"](
                grouphdu.header.get('DATE-OBS'))
        else:
            self.header["dateobs"] = self.header_dtype["dateobs"]('None')

        naxis = grouphdu.header.get("NAXIS")
        for i in range(naxis):
            ctype = grouphdu.header.get("CTYPE%d" % (i + 1))
            if ctype is None:
                continue
            elif ctype[0:2] == "RA":
                isx = i + 1
            elif ctype[0:3] == "DEC":
                isy = i + 1
            elif ctype[0:4] == "FREQ":
                isf = i + 1

        if isx != False:
            self.header["x"] = self.header_dtype["x"](
                hdulist[0].header.get("CRVAL%d" % (isx)))
        else:
            print("Warning: No RA info.")

        if isy != False:
            self.header["y"] = self.header_dtype["y"](
                hdulist[0].header.get("CRVAL%d" % (isy)))
        else:
            print("Warning: No Dec info.")

        if isf != False:
            self.header["f"] = self.header_dtype["f"](
                hdulist[0].header.get("CRVAL%d" % (isf)))
            self.header["df"] = self.header_dtype["df"](hdulist[0].header.get(
                "CDELT%d" % (isf)) * hdulist[0].header.get("NAXIS%d" % (isf)))
            self.header["nf"] = self.header_dtype["nf"](1)
            self.header["nfref"] = self.header_dtype["nfref"](1)
        else:
            print("Warning: No image data along the Frequency axis.")

        '''
        This is for EHT imaging Library
        for axistype in ["x,y,f"]:
            if np.isnan(self.header[axistype]):
                self.header[axistype] = 0.
            if np.isnan(self.header[axistype]):
                self.header["d%s"%(axistype)] = 0.
            if np.isnan(self.header[axistype]):
                self.header["n%sref"%(axistype)] = self.header["n%d"%(axistype)]/2+1
        '''

        self.update_fits()

    def update_fits(self,cctab=True,threshold=None, relative=True,
                    istokes=0, ifreq=0):
        '''
        Reflect current self.data / self.header info to the image FITS data.
        Args:
            cctab (boolean): If True, AIPS CC table is attached to fits file.
            istokes (integer): index for Stokes Parameter at which the image will be used for CC table.
            ifreq (integer): index for Frequency at which the image will be used for CC table.
            threshold (float): pixels with the absolute intensity smaller than this value will be ignored in CC table.
            relative (boolean): If true, theshold value will be normalized with the peak intensity of the image.
        '''

        # CREATE HDULIST
        hdu = pyfits.PrimaryHDU(self.data)
        hdulist = pyfits.HDUList([hdu])

        # GET Current Time
        dtnow = dt.datetime.now()

        # FILL HEADER INFO
        hdulist[0].header.set("OBJECT",   self.header["object"])
        hdulist[0].header.set("TELESCOP", self.header["telescope"])
        hdulist[0].header.set("INSTRUME", self.header["instrument"])
        hdulist[0].header.set("OBSERVER", self.header["observer"])
        hdulist[0].header.set("DATE",     "%04d-%02d-%02d" %
                              (dtnow.year, dtnow.month, dtnow.day))
        hdulist[0].header.set("DATE-OBS", self.header["dateobs"])
        hdulist[0].header.set("DATE-MAP", "%04d-%02d-%02d" %
                              (dtnow.year, dtnow.month, dtnow.day))
        hdulist[0].header.set("BSCALE",   np.float64(1.))
        hdulist[0].header.set("BZERO",    np.float64(0.))
        hdulist[0].header.set("BUNIT",    "JY/PIXEL")
        hdulist[0].header.set("EQUINOX",  np.float64(2000.))
        hdulist[0].header.set("OBSRA",    np.float64(self.header["x"]))
        hdulist[0].header.set("OBSDEC",   np.float64(self.header["y"]))
        hdulist[0].header.set("DATAMAX",  np.nan_to_num(self.data.max()))
        hdulist[0].header.set("DATAMIN",  np.nan_to_num(self.data.min()))
        hdulist[0].header.set("CTYPE1",   "RA---SIN")
        hdulist[0].header.set("CRVAL1",   np.float64(self.header["x"]))
        hdulist[0].header.set("CDELT1",   np.float64(self.header["dx"]))
        hdulist[0].header.set("CRPIX1",   np.float64(self.header["nxref"]))
        hdulist[0].header.set("CROTA1",   np.float64(0.))
        hdulist[0].header.set("CTYPE2",   "DEC---SIN")
        hdulist[0].header.set("CRVAL2",   np.float64(self.header["y"]))
        hdulist[0].header.set("CDELT2",   np.float64(self.header["dy"]))
        hdulist[0].header.set("CRPIX2",   np.float64(self.header["nyref"]))
        hdulist[0].header.set("CROTA2",   np.float64(0.))
        hdulist[0].header.set("CTYPE3",   "FREQ")
        hdulist[0].header.set("RESTFRQ",  np.float64(self.header["restfreq"]))
        hdulist[0].header.set("CRVAL3",   np.float64(self.header["f"]))
        hdulist[0].header.set("CDELT3",   np.float64(self.header["df"]))
        hdulist[0].header.set("CRPIX3",   np.float64(self.header["nfref"]))
        hdulist[0].header.set("CROTA3",   np.float64(0.))
        hdulist[0].header.set("CTYPE4",   "STOKES")
        hdulist[0].header.set("CRVAL4",   np.int64(self.header["s"]))
        hdulist[0].header.set("CDELT4",   np.int64(self.header["ds"]))
        hdulist[0].header.set("CRPIX4",   np.int64(self.header["nsref"]))
        hdulist[0].header.set("CROTA4",   np.int64(0))


        # Add AIPS CC Table
        if cctab:
            aipscctab = self._aipscc(threshold=threshold, relative=relative,
                    istokes=istokes, ifreq=ifreq)

            hdulist.append(hdu=aipscctab)

            next = len(hdulist)
            hdulist[next-1].name = 'AIPS CC'

        self.hdulist = hdulist

    def _aipscc(self, threshold=None, relative=True,
                    istokes=0, ifreq=0):
        '''
        Make AIPS CC table

        Args:
            istokes (integer): index for Stokes Parameter at which the image will be saved
            ifreq (integer): index for Frequency at which the image will be saved
            threshold (float): pixels with the absolute intensity smaller than this value will be ignored.
            relative (boolean): If true, theshold value will be normalized with the peak intensity of the image.
        '''
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        xg, yg = self.get_xygrid(angunit="deg")
        X, Y = np.meshgrid(xg, yg)
        X = X.reshape(Nx * Ny)
        Y = Y.reshape(Nx * Ny)
        flux = self.data[istokes, ifreq]
        flux = flux.reshape(Nx * Ny)

        # threshold
        if threshold is None:
            thres = np.finfo(np.float32).eps
        else:
            if relative:
                thres = self.peak(istokes=istokes, ifreq=ifreq) * threshold
            else:
                thres = threshold
        thres = np.abs(thres)

        # adopt threshold
        X = X[flux >= thres]
        Y = Y[flux >= thres]
        flux = flux[flux >= thres]

        # make table columns
        c1 = pyfits.Column(name='FLUX', array=flux, format='1E',unit='JY')
        c2 = pyfits.Column(name='DELTAX', array=X, format='1E',unit='DEGREES')
        c3 = pyfits.Column(name='DELTAY', array=Y, format='1E',unit='DEGREES')
        c4 = pyfits.Column(name='MAJOR AX', array=np.zeros(len(flux)), format='1E',unit='DEGREES')
        c5 = pyfits.Column(name='MINOR AX', array=np.zeros(len(flux)), format='1E',unit='DEGREES')
        c6 = pyfits.Column(name='POSANGLE', array=np.zeros(len(flux)), format='1E',unit='DEGREES')
        c7 = pyfits.Column(name='TYPE OBJ', array=np.zeros(len(flux)), format='1E',unit='CODE')

        # make CC table
        tab = pyfits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7])
        return tab

    def save_fits(self, outfitsfile, overwrite=True):
        '''
        save the image(s) to the image FITS file.

        Args:
            outfitsfile (string): file name
            overwrite (boolean): It True, an existing file will be overwritten.
        '''
        if os.path.isfile(outfitsfile):
            if overwrite:
                os.system("rm -f %s" % (outfitsfile))
                self.hdulist.writeto(outfitsfile)
            else:
                print("Warning: does not overwrite %s" % (outfitsfile))
        else:
            self.hdulist.writeto(outfitsfile)


    def winmod(self, imagewin, save_totalflux=False):
        # create output fits
        outfits = copy.deepcopy(self)

        for idxs in np.arange(self.header["ns"]):
            for idxf in np.arange(self.header["nf"]):
                image = outfits.data[idxs, idxf]
                masked = imagewin == False
                image[np.where(masked)] = 0
                outfits.data[idxs, idxf] = image
                if save_totalflux:
                    totalflux = self.totalflux(istokes=idxs, ifreq=idxf)
                    outfits.data[idxs, idxf] *= totalflux / image.sum()
        # Update and Return
        outfits.update_fits()
        return outfits

    def angconv(self, unit1="deg", unit2="deg"):
        '''
        return a conversion factor from unit1 to unit2
        Available angular units are uas, mas, asec or arcsec, amin or arcmin and degree.
        '''
        return util.angconv(unit1,unit2)

    #-------------------------------------------------------------------------
    # Getting Some information about images
    #-------------------------------------------------------------------------
    def get_xygrid(self, twodim=False, angunit=None):
        '''
        calculate the grid of the image

        Args:
          angunit (string): Angular unit (uas, mas, asec or arcsec, amin or arcmin, degree)
          twodim (boolean): It True, the 2D grids will be returned. Otherwise, the 1D arrays will be returned
        '''
        if angunit is None:
            angunit = self.angunit

        dx = self.header["dx"]
        dy = self.header["dy"]
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        Nxref = self.header["nxref"]
        Nyref = self.header["nyref"]
        xg = dx * (np.arange(Nx) - Nxref + 1) * self.angconv("deg", angunit)
        yg = dy * (np.arange(Ny) - Nyref + 1) * self.angconv("deg", angunit)
        if twodim:
            xg, yg = np.meshgrid(xg, yg)
        return xg, yg

    def get_imextent(self, angunit=None):
        '''
        calculate the field of view of the image

        Args:
          angunit (string): Angular unit (uas, mas, asec or arcsec, amin or arcmin, degree)
        '''
        if angunit is None:
            angunit = self.angunit

        dx = self.header["dx"]
        dy = self.header["dy"]
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        Nxref = self.header["nxref"]
        Nyref = self.header["nyref"]
        xmax = (1 - Nxref - 0.5) * dx
        xmin = (Nx - Nxref + 0.5) * dx
        ymax = (Ny - Nyref + 0.5) * dy
        ymin = (1 - Nyref - 0.5) * dy
        return np.asarray([xmax, xmin, ymin, ymax]) * self.angconv("deg", angunit)

    def peak(self, absolute=False, istokes=0, ifreq=0):
        '''
        calculate the peak intensity of the image

        Args:
          istokes (integer): index for Stokes Parameter at which the peak intensity will be calculated
          ifreq (integer): index for Frequency at which the peak intensity will be calculated
        '''
        if absolute:
            t = np.argmax(np.abs(self.data[istokes, ifreq]))
            t = np.unravel_index(t, [self.header["ny"], self.header["nx"]])
            return self.data[istokes, ifreq][t]
        else:
            return self.data[istokes, ifreq].max()

    def min(self,istokes=0, ifreq=0):
        '''
        calculate the minimum intensity of the image

        Args:
          istokes (integer): index for Stokes Parameter at which the peak intensity will be calculated
          ifreq (integer): index for Frequency at which the peak intensity will be calculated
        '''
        return self.data[istokes, ifreq].min()

    def jysr2jypixel(self, istokes=0, ifreq=0):
        """
        convert image unit [Jy/sr] into [Jy/pixel]
        Arg:
          istokes (integer): index for Stokes Parameter at which the peak intensity will be calculated
          ifreq (integer): index for Frequency at which the peak intensity will be calculated
        """
        # step-0: convert pixel scale
        im =  copy.deepcopy(self)
        imdx = im.header["dx"]*im.angconv("deg","asec") # unit: arcsec
        imdy = im.header["dy"]*im.angconv("deg","asec") # unit: arcsec
        const = (np.pi/(180*3600))**2
        im.data[istokes,ifreq] = im.data[istokes,ifreq]*abs(imdx*imdy)*const
        print("convert image unit [Jy/sr] into [Jy/pixel]...")
        print("inputted cell size [asec] = ", imdy)
        im.update_fits()
        return im

    def jypixel2jyasec(self, istokes=0, ifreq=0):
        """
        convert image unit [Jy/pixel] into [Jy/asec^2]
        Arg:
          istokes (integer): index for Stokes Parameter at which the peak intensity will be calculated
          ifreq (integer): index for Frequency at which the peak intensity will be calculated
        """
        # step-0: convert pixel scale
        #ref = IMFITS(dx=abs(self.header["dx"]*3600),nx= self.header["nx"], angunit="asec")
        #im = ref.cpimage(self, save_totalflux=True)
        im =  copy.deepcopy(self)
        # step-1:we convert deg/pixel into asec/pixel
        imdx = im.header["dx"]*im.angconv("deg","asec") # unit: arcsec
        imdy = im.header["dy"]*im.angconv("deg","asec") # unit: arcsec
        # step-2:we devide Jy/pixel by arcsec^2/pixel, we get Jy/arcsec^2
        im.data[istokes,ifreq] = im.data[istokes,ifreq]/abs(imdx*imdy)
        #Reflect current self.data / self.header info to the image FITS data
        im.update_fits()
        return im

    def jyasec2jypixel(self, istokes=0, ifreq=0):
        """
        convert image unit [Jy/asec^2] into [Jy/pixel]
        Arg:
          istokes (integer): index for Stokes Parameter at which the peak intensity will be calculated
          ifreq (integer): index for Frequency at which the peak intensity will be calculated
        """
        # step-0: convert pixel scale
        ref = IMFITS(dx=abs(self.header["dx"]*3600),nx= self.header["nx"], angunit="asec")
        im = ref.cpimage(self, save_totalflux=True)
        # step-1:we convert deg/pixel into asec/pixel
        imdx = im.header["dx"]*im.angconv("deg","asec") # unit: arcsec
        imdy = im.header["dy"]*im.angconv("deg","asec") # unit: arcsec
        # step-2: Jy/asec^2 multiply arcsec^2/pixel is equal to Jy/pixel
        im.data[istokes,ifreq] = im.data[istokes,ifreq]*abs(imdx*imdy)
        #Reflect current self.data / self.header info to the image FITS data
        im.update_fits()
        return im

    def jybeam2jypixel(self, istokes=0, ifreq=0):
        """
        convert image unit [Jy/beam] into [Jy/pixel]
        Arg:
          istokes (integer): index for Stokes Parameter at which the peak intensity will be calculated
          ifreq (integer): index for Frequency at which the peak intensity will be calculated
        """
        cellsize          = (np.pi/180.0)**2*np.abs(self.header["dx"])*np.abs(self.header["dy"]) # radian
        beam_unit         = (np.pi/(4.0*np.log(2.0)))*((np.pi/180.0)**2.0)*self.header["BMAJ"]*self.header["BMIN"] # sr/beam
        climfac           = copy.deepcopy(self)
        climfac.data[istokes,ifreq] = np.nan_to_num(self.data[istokes,ifreq])*(cellsize/beam_unit)#Jy/pixel
        return climfac

    def jybeam2jyasec(self, istokes=0, ifreq=0):
        """
        #  image [Jy/beam] -> [Jy/asec**2]
        """
        cellsize              = (np.pi/180.0)**2*np.abs(self.header["dx"])*np.abs(self.header["dx"]) # radian
        beam_unit             = (np.pi/(4.0*np.log(2.0)))*((np.pi/180.0)**2.0)*self.header["BMAJ"]*self.header["BMIN"] # sr/beam
        climfac               = copy.deepcopy(self)
        climfac.data[0,0] = np.nan_to_num(self.data[istokes,ifreq])*((cellsize)/beam_unit) # Jy/pixel

        # step-0: convert pixel scale
        ref = IMFITS(dx=abs(climfac.header["dx"]*3600),nx= climfac.header["nx"], angunit="asec")
        im = ref.cpimage(climfac, save_totalflux=True)
        # step-1:we convert deg/pixel into asec/pixel
        imdx = im.header["dx"]*im.angconv("deg","asec") # unit: arcsec
        imdy = im.header["dy"]*im.angconv("deg","asec") # unit: arcsec

        # step-2:we devide Jy/pixel by arcsec^2/pixel, we get Jy/arcsec^2
        im.data[istokes,ifreq] = (im.data[istokes,ifreq])/abs(imdx*imdy)
        #Reflect current self.data / self.header info to the image FITS data
        im.update_fits()
        return im

    def jypixel2jybeam(self, istokes=0, ifreq=0, BMAJ = 0., BMIN = 0.):
        """
        #  intensity [Jy/pixel] -> [Jy/beam]
        cellsize_asec : arcsec
        BMAJ : arcsec
        BMIN : arcsec
        """
        # step-0
        ref = IMFITS(dx=abs(self.header["dx"]*3600),nx= self.header["nx"], angunit="asec")
        im = ref.cpimage(self, save_totalflux=True)
        # step-1
        imdx = im.header["dx"]*im.angconv("deg","asec") # unit: arcsec
        imdy = im.header["dy"]*im.angconv("deg","asec") # unit: arcsec
        # step-2
        im.data[istokes,ifreq] = im.data[istokes,ifreq]*(1.132*BMAJ*BMIN)/abs(imdx*imdy)
        #Reflect current self.data / self.header info to the image FITS data
        im.update_fits()
        return im

    def plt_offsourcearea(self, radius = 1.0, unit = "jypixel-1", bins = 25, savefigname="noise_hist",  istokes=0, ifreq=0):
        """
        plot off-sorce area on the imput image domain.
        Args:
         radius (arcsec): off-source radius.
         unit: input image unit, please select "jypixel-1" or "jyasec-2"
         bins: specify the number of bins for a histgram.
         istokes (integer): index for Stokes Parameter at which l1-norm will be calculated
         ifreq (integer): index for Frequency at which l1-norm will be calculated
        """
        if unit == "jypixel-1":
            image = self.jypixel2jyasec()
        elif unit == "jyasec-2":
            image = self

        # caluculate data of off-source area.
        offsorce = image.offsource_area(radius)
        hist = np.histogram(offsorce,bins)
        vmax = hist[0][0]

        # caluculate confidence interval.
        conflevel = image.confidence_level(radius, istokes, ifreq)

        # plot data
        matplotlibrc(nrows=1,ncols=1,width=300,height=300) # set size of figure
        fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False) # gerenete figure and panels
        plt.subplots_adjust(wspace=0.15)

        # setup the scale in the graph
        majorLocator = ticker.MultipleLocator(0.02)
        majorFormatter = ticker.FormatStrFormatter('%.2f')
        minorLocator = ticker.MultipleLocator(0.01)

        plt.sca(ax) # stands for "set current axis")
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.gca().yaxis.set_tick_params(which='both', direction='in',left=True,right = True)
        plt.gca().xaxis.set_tick_params(which='both', direction='in', top = True,  bottom = True)


        plt.xlabel("Intensity (Jy arcsec$^{-2}$)",fontsize = 15)
        plt.ylabel('Number of Pixels ',fontsize = 15)
        plt.ylim(0.1,vmax*10)
        plt.xlim(0,conflevel[0]*1.2)
        plt.hist(offsorce,
                label="Sample values in off-source area",
                 range=(min(offsorce),max(offsorce)),
                 alpha=0.3,
                 bins=bins,
                 color="k",
                 align="mid",
                 rwidth=1,
                 log = True,
                 histtype="stepfilled",
                 normed = False)

        plt.plot([conflevel[0],conflevel[0]],[0,vmax*10],'k--',lw=1, alpha = 0.5)
        plt.text(conflevel[0]+0.005,vmax,'100%',ha='left',va='center',fontsize=12,color="k", rotation='vertical', alpha = 0.5)

        plt.plot([conflevel[2],conflevel[2]],[0,vmax*10],'k--',lw=1, alpha = 0.5)
        plt.text(conflevel[2]+0.005,vmax,'99.7%',ha='left',va='center',fontsize=12,color="k",rotation='vertical', alpha = 0.5)

        plt.plot([conflevel[3],conflevel[3]],[0,vmax*10],'k--',lw=1, alpha = 0.5)
        plt.text(conflevel[3]+0.005,vmax,'95%',ha='left',va='center',fontsize=12,color="k",rotation='vertical', alpha = 0.5)

        plt.savefig(savefigname+".png",bbox_inches='tight',pad_inches=0)


    #-------------------------------------------------------------------------
    # Plotting
    #-------------------------------------------------------------------------
    def imshow(self, logscale=False, angunit=None, dyrange=100, axs = None,flame_lw = 1,
               tick_list = np.arange(-100,100,0.5), tick_color = "w", cmap = "inferno",
               vmin=None, istokes=0, ifreq=0, tick_width = 1.2,labelsize= 18, **imshow_args):
        '''
        plot contours of the image

        Args:
          logscale (boolean):
            If True, the color contour will be on log scales. Otherise,
            the color contour will be on linear scales.
          angunit (string):
            Angular Unit for the axis labels (pixel, uas, mas, asec or arcsec,
            amin or arcmin, degree)
          vmin (string):
            minimum value of the color contour in Jy/pixel
          istokes (integer):
            index for Stokes Parameter at which the image will be plotted
          ifreq (integer):
            index for Frequency at which the image will be plotted
          **imshow_args: Args will be input in matplotlib.pyplot.imshow
        '''
        # thickness of flame
        plt.gca().spines["top"].set_linewidth(flame_lw)
        plt.gca().spines["left"].set_linewidth(flame_lw)
        plt.gca().spines["bottom"].set_linewidth(flame_lw)
        plt.gca().spines["right"].set_linewidth(flame_lw)

        plt.minorticks_on()
        plt.yticks(tick_list, font="times")
        plt.xticks(tick_list, font="times")
        plt.tick_params(which='minor',direction='in', bottom=True, left=True, right=True, top=True,
                        length=4, width=tick_width, labelsize=labelsize, colors = tick_color, labelcolor='k')
        plt.tick_params(which='major',direction='in', bottom=True, left=True, right=True, top=True,
                        length=8, width=tick_width, labelsize=labelsize, colors = tick_color, labelcolor='k')
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(4))
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator(4))

        if angunit is None:
            angunit = self.angunit

        # Get Image Axis
        if angunit == "pixel":
            imextent = None
        else:
            imextent = self.get_imextent(angunit)

        if logscale:
            plotdata = np.log10(self.data[istokes, ifreq])
            if vmin is None:
                vmin_scaled = np.log10(self.peak(istokes, ifreq)/dyrange)
            else:
                vmin_scaled = np.log10(vmin)
            plotdata[np.where(plotdata < vmin_scaled)] = vmin_scaled
            if axs != None:
                image = axs.imshow(plotdata, extent=imextent, origin="lower",cmap=cmap,
                           vmin=vmin_scaled, **imshow_args)
            else:
                image = plt.imshow(plotdata, extent=imextent, origin="lower", cmap=cmap,
                           vmin=vmin_scaled, **imshow_args)
        else:
            if axs != None:
                image = axs.imshow(self.data[istokes, ifreq], extent=imextent, origin="lower",
                           vmin=vmin,cmap=cmap, **imshow_args)
            else:
                image = plt.imshow(self.data[istokes, ifreq], extent=imextent, origin="lower",
                           vmin=vmin, cmap=cmap,**imshow_args)


        #image.axes.tick_params(axis='both',which='both',direction='in')

        # Axis Label
        if angunit.lower().find("pixel") == 0:
            unit = "pixel"
        elif angunit.lower().find("uas") == 0:
            unit = r"$\rm \mu$as"
        elif angunit.lower().find("mas") == 0:
            unit = "mas"
        elif angunit.lower().find("arcsec") * angunit.lower().find("asec") == 0:
            unit = "arcsec"
        elif angunit.lower().find("arcmin") * angunit.lower().find("amin") == 0:
            unit = "arcmin"
        elif angunit.lower().find("deg") == 0:
            unit = "deg"
        else:
            unit = "mas"
        plt.xlabel("Relative RA (%s)" % (unit), fontsize = labelsize, font="times")
        plt.ylabel("Relative Dec (%s)" % (unit), fontsize = labelsize, font="times")

        return(image)



    def contour(self, cmul=None, levels=None, angunit=None,
                colors="white",
                istokes=0, ifreq=0,
                **contour_args):
        '''
        plot contours of the image

        Args:
          istokes (integer): index for Stokes Parameter at which the image will be plotted
          ifreq (integer): index for Frequency at which the image will be plotted
          angunit (string): Angular Unit for the axis labels (pixel, uas, mas, asec or arcsec, amin or arcmin, degree)
          colors (string, array-like): colors of contour levels
          cmul: The lowest contour level. Default value is 1% of the peak intensity.
          levels: contour level. This will be multiplied with cmul.
          **contour_args: Args will be input in matplotlib.pyplot.contour
        '''
        if angunit is None:
            angunit = self.angunit

        # Get Image Axis
        if angunit == "pixel":
            imextent = None
        else:
            imextent = self.get_imextent(angunit)

        # Get image
        image = self.data[istokes, ifreq]
        
        clevels = cmul * np.asarray(levels)

        plt.contour(image, extent=imextent, origin="lower",
                    colors=colors, levels=clevels, **contour_args)
        # plt.contour(image,extent=imextent,origin="lower",
        #            colors=colors,levels=-levels,ls="--",**contour_args)
        # Axis Label
        if angunit.lower().find("pixel") == 0:
            unit = "pixel"
        elif angunit.lower().find("uas") == 0:
            unit = r"$\rm \mu$as"
        elif angunit.lower().find("mas") == 0:
            unit = "mas"
        elif angunit.lower().find("arcsec") * angunit.lower().find("asec") == 0:
            unit = "arcsec"
        elif angunit.lower().find("arcmin") * angunit.lower().find("amin") == 0:
            unit = "arcmin"
        elif angunit.lower().find("deg") == 0:
            unit = "deg"
        else:
            unit = "mas"

        plt.xlabel("Relative RA (%s)" % (unit))
        plt.ylabel("Relative Dec (%s)" % (unit))

    def self_contour(self, rms=None, step=None, angunit=None,
                colors="white", 
                istokes=0, ifreq=0,
                **contour_args):
        '''
        plot contours of the image

        Args:
          istokes (integer): index for Stokes Parameter at which the image will be plotted
          ifreq (integer): index for Frequency at which the image will be plotted
          angunit (string): Angular Unit for the axis labels (pixel, uas, mas, asec or arcsec, amin or arcmin, degree)
          colors (string, array-like): colors of contour levels
          cmul: The lowest contour level. Default value is 1% of the peak intensity.
          levels: contour level. This will be multiplied with cmul.
          **contour_args: Args will be input in matplotlib.pyplot.contour
        '''
        if angunit is None:
            angunit = self.angunit

        # Get Image Axis
        if angunit == "pixel":
            imextent = None
        else:
            imextent = self.get_imextent(angunit)

        # Get image
        image = self.data[istokes, ifreq]

        if step is None:
            clevels = np.arange(3, 100, 3)
        else:
            clevels = np.arange(step, 100, step)
        clevels = rms * np.asarray(clevels)

        plt.contour(image, extent=imextent, origin="lower",
                    colors=colors, levels=clevels, **contour_args)
        # plt.contour(image,extent=imextent,origin="lower",
        #            colors=colors,levels=-levels,ls="--",**contour_args)
        # Axis Label
        if angunit.lower().find("pixel") == 0:
            unit = "pixel"
        elif angunit.lower().find("uas") == 0:
            unit = r"$\rm \mu$as"
        elif angunit.lower().find("mas") == 0:
            unit = "mas"
        elif angunit.lower().find("arcsec") * angunit.lower().find("asec") == 0:
            unit = "arcsec"
        elif angunit.lower().find("arcmin") * angunit.lower().find("amin") == 0:
            unit = "arcmin"
        elif angunit.lower().find("deg") == 0:
            unit = "deg"
        else:
            unit = "mas"

        plt.xlabel("Relative RA (%s)" % (unit))
        plt.ylabel("Relative Dec (%s)" % (unit))


    #-------------------------------------------------------------------------
    # DS9
    #-------------------------------------------------------------------------
    def open_ds9(self):
        pass

    def read_ds9reg(self):
        pass

    #-------------------------------------------------------------------------
    # Output some information to files
    #-------------------------------------------------------------------------
    def to_difmapmod(self, outfile, threshold=None, relative=True,
                     istokes=0, ifreq=0):
        '''
        Save an image into a difmap model file

        Args:
          istokes (integer): index for Stokes Parameter at which the image will be saved
          ifreq (integer): index for Frequency at which the image will be saved
          threshold (float): pixels with the absolute intensity smaller than this value will be ignored.
          relative (boolean): If true, theshold value will be normalized with the peak intensity of the image.
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        xg, yg = self.get_xygrid(angunit="mas")
        X, Y = np.meshgrid(xg, yg)
        R = np.sqrt(X * X + Y * Y)
        theta = np.rad2deg(np.arctan2(X, Y))
        flux = self.data[istokes, ifreq]

        R = R.reshape(Nx * Ny)
        theta = theta.reshape(Nx * Ny)
        flux = flux.reshape(Nx * Ny)

        if threshold is None:
            thres = np.finfo(np.float32).eps
        else:
            if relative:
                thres = self.peak(istokes=istokes, ifreq=ifreq) * threshold
            else:
                thres = threshold
        thres = np.abs(thres)

        f = open(outfile, "w")
        for i in np.arange(Nx * Ny):
            if np.abs(flux[i]) < thres:
                continue
            line = "%20e %20e %20e\n" % (flux[i], R[i], theta[i])
            f.write(line)
        f.close()

    #-------------------------------------------------------------------------
    # Editing images
    #-------------------------------------------------------------------------
    def cpimage(self, fitsdata, save_totalflux=False, order=3):
        '''
        Copy the first image into the image grid specified in the secondaly input image.

        Args:
          fitsdata: input imagefite.imagefits object. This image will be copied into self.
          self: input imagefite.imagefits object specifying the image grid where the orgfits data will be copied.
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        # generate output imfits object
        outfits = copy.deepcopy(self)

        dx0 = fitsdata.header["dx"]
        dy0 = fitsdata.header["dy"]
        Nx0 = fitsdata.header["nx"]
        Ny0 = fitsdata.header["ny"]
        Nxr0 = fitsdata.header["nxref"]
        Nyr0 = fitsdata.header["nyref"]

        dx1 = outfits.header["dx"]
        dy1 = outfits.header["dy"]
        Nx1 = outfits.header["nx"]
        Ny1 = outfits.header["ny"]
        Nxr1 = outfits.header["nxref"]
        Nyr1 = outfits.header["nyref"]

        coord = np.zeros([2, Nx1 * Ny1])
        xgrid = (np.arange(Nx1) + 1 - Nxr1) * dx1 / dx0 + Nxr0 - 1
        ygrid = (np.arange(Ny1) + 1 - Nyr1) * dy1 / dy0 + Nyr0 - 1
        x, y = np.meshgrid(xgrid, ygrid)
        coord[0, :] = y.reshape(Nx1 * Ny1)
        coord[1, :] = x.reshape(Nx1 * Ny1)

        for idxs in np.arange(outfits.header["ns"]):
            for idxf in np.arange(outfits.header["nf"]):
                outfits.data[idxs, idxf] = sn.map_coordinates(
                    fitsdata.data[idxs, idxf], coord, order=order,
                    mode='constant', cval=0.0, prefilter=True).reshape([Ny1, Nx1]
                                                                       ) * dx1 * dy1 / dx0 / dy0
                # Flux Scaling
                if save_totalflux:
                    totalflux = fitsdata.totalflux(istokes=idxs, ifreq=idxf)
                    outfits.data[idxs, idxf] *= totalflux / \
                        outfits.totalflux(istokes=idxs, ifreq=idxf)

        outfits.update_fits()
        return outfits

    def gauss_convolve(self, majsize = None, minsize=None, x0=None, y0=None,
                       pa=0., scale=1., angunit=None, pos="rel", save_totalflux=False):
        '''
        Gaussian Convolution

        Args:
          self: input imagefite.imagefits object
          majsize (float): Major Axis Size
          minsize (float): Minor Axis Size. If None, it will be same to the Major Axis Size (Circular Gaussian)
          angunit (string): Angular Unit for the sizes (uas, mas, asec or arcsec, amin or arcmin, degree)
          pa (float): Position Angle of the Gaussian
          scale (float): The sizes will be multiplied by this value.
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        if minsize is None:
            minsize = majsize

        if angunit is None:
            angunit = self.angunit

        # Create outputdata
        outfits = copy.deepcopy(self)

        # Create Gaussian
        imextent = outfits.get_imextent(angunit)
        Imxref = (imextent[0] + imextent[1]) / 2.
        Imyref = (imextent[2] + imextent[3]) / 2.
        if x0 is None:
            x0 = 0.
        if y0 is None:
            y0 = 0.
        if pos=="rel":
            x0 += Imxref
            y0 += Imyref

        X, Y = outfits.get_xygrid(angunit=angunit, twodim=True)
        cospa = np.cos(np.deg2rad(pa))
        sinpa = np.sin(np.deg2rad(pa))
        X1 = (X - x0) * cospa - (Y - y0) * sinpa
        Y1 = (X - x0) * sinpa + (Y - y0) * cospa
        majsig = majsize / np.sqrt(2 * np.log(2)) / 2 * scale
        minsig = minsize / np.sqrt(2 * np.log(2)) / 2 * scale
        gauss = np.exp(-X1 * X1 / 2 / minsig / minsig - Y1 * Y1 / 2 / majsig / majsig)
        gauss/= 2*np.pi*majsig*minsig

        # Replace nan with zero
        gauss[np.isnan(gauss)] = 0
        # Convolusion (except:gauss is zero array)
        if np.any(gauss != 0):
            for idxs in np.arange(outfits.header["ns"]):
                for idxf in np.arange(outfits.header["nf"]):
                    newimage = convolve_fft(outfits.data[idxs, idxf], gauss)
                    outfits.data[idxs, idxf] = newimage
                    # Flux Scaling
                    if save_totalflux:
                        totalflux = self.totalflux(istokes=idxs, ifreq=idxf)
                        outfits.data[idxs, idxf] *= totalflux / \
                            outfits.totalflux(istokes=idxs, ifreq=idxf)

        # Update and Return
        outfits.update_fits()
        return outfits

    def ds9flag(self, regfile, save_totalflux=False):
        '''
        Flagging the image with DS9region file

        Args:
          self: input imagefite.imagefits object
          regfile (string): input DS9 region file
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        # create output fits
        outfits = copy.deepcopy(self)

        # original file
        xgrid = np.arange(self.header["nx"])
        ygrid = np.arange(self.header["ny"])
        X, Y = np.meshgrid(xgrid, ygrid)

        # Check which grids should be flagged
        pixels = get_flagpixels(regfile, X, Y)
        pixels = (pixels == False)
        pixels = np.where(pixels)

        for idxs in np.arange(self.header["ns"]):
            for idxf in np.arange(self.header["nf"]):
                image = outfits.data[idxs, idxf]
                image[pixels] = 0.
                outfits.data[idxs, idxf] = image
                # Flux Scaling
                if save_totalflux:
                    totalflux = self.totalflux(istokes=idxs, ifreq=idxf)
                    outfits.data[idxs, idxf] *= totalflux / image.sum()

        # Update and Return
        outfits.update_fits()
        return outfits

    def read_cleanbox(self, regfile):
        # Read DS9-region file
        f = open(regfile)
        lines = f.readlines()
        f.close()

        # original file
        xgrid = np.arange(self.header["nx"])
        ygrid = np.arange(self.header["ny"])
        X, Y = np.meshgrid(xgrid, ygrid)
        area = np.zeros(X.shape, dtype="Bool")

        # Read each line
        for line in lines:
            # Skipping line
            if line[0] == "#":
                continue
            if "image" in line == True:
                continue
            if "(" in line == False:
                continue
            if "global" in line:
                continue

            # Replacing many characters to empty spaces
            line = line.replace("(", " ")
            line = line.replace(")", " ")
            while "," in line:
                line = line.replace(",", " ")

            # split line to elements
            elements = line.split(" ")
            while "" in elements:
                elements.remove("")
            while "\n" in elements:
                elements.remove("\n")

            if len(elements) < 4:
                continue

            # Check whether the box is for "inclusion" or "exclusion"
            if elements[0][0] == "-":
                elements[0] = elements[0][1:]
                exclusion = True
            else:
                exclusion = False

            if elements[0] == "box":
                tmparea = region_box(X, Y,
                                      x0=np.float64(elements[1]),
                                      y0=np.float64(elements[2]),
                                      width=np.float64(elements[3]),
                                      height=np.float64(elements[4]),
                                      angle=np.float64(elements[5]))
            elif elements[0] == "circle":
                tmparea = region_circle(X, Y,
                                         x0=np.float64(elements[1]),
                                         y0=np.float64(elements[2]),
                                         radius=np.float64(elements[3]))
            elif elements[0] == "ellipse":
                tmparea = region_ellipse(X, Y,
                                          x0=np.float64(elements[1]),
                                          y0=np.float64(elements[2]),
                                          radius1=np.float64(elements[3]),
                                          radius2=np.float64(elements[4]),
                                          angle=np.float64(elements[5]))
            else:
                print("[WARNING] The shape %s is not available." %
                      (elements[0]))

            if not exclusion:
                area += tmparea
            else:
                area[np.where(tmparea)] = False

        return area

    def comshift(self, save_totalflux=False, ifreq=0, istokes=0):
        '''
        Shift the image so that its center-of-mass position coincides with the reference pixel.

        Args:
          istokes (integer):
            index for Stokes Parameter at which the image will be edited
          ifreq (integer):
            index for Frequency at which the image will be edited
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.

        Returns:
          imdata.IMFITS object
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        image = outfits.data[istokes, ifreq]
        nxref = outfits.header["nxref"]
        nyref = outfits.header["nyref"]

        # move the center of mass to the actual center of the self
        pix = sn.measurements.center_of_mass(image)
        outfits.data[istokes, ifreq] = sn.interpolation.shift(
            image, np.asarray([nyref - 1, nxref - 1]) - pix)

        # scale total flux
        if save_totalflux:
            totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
            outfits.data[istokes, ifreq] *= totalflux / \
                outfits.totalflux(istokes=istokes, ifreq=ifreq)

        # update FITS
        outfits.update_fits()
        return outfits

    def imageshift(self, shift=[0,0], unit = "arcsec", save_totalflux=False, ifreq=0, istokes=0):
        '''
        Shift the image.

        Arg:
        shift ([float, float]):
            image shift on the arcsec scale (x,y).
          istokes (integer):
            index for Stokes Parameter at which the image will be edited
          ifreq (integer):
            index for Frequency at which the image will be edited
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.

        Returns:
          imdata.IMFITS object
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        image = np.nan_to_num(outfits.data[istokes, ifreq])
        nxref = outfits.header["nxref"]
        nyref = outfits.header["nyref"]
        cellx = np.abs(outfits.header["dx"]*3600) # arcsec/pixel
        celly = np.abs(outfits.header["dy"]*3600) # arcsec/pixel

        if unit == "arcsec":
            pixelx = int(shift[0]/cellx) 
            pixely = int(shift[1]/celly) 
            print("# Image shift is performed ###################")
            print("(RA, Dec) = (%1d, %1d) [pixel] is shifted"%(pixelx, pixely))
            print("(RA, Dec) = (%0.3f, %0.3f) [arcsec] is shifted"%(shift[0], shift[1]))

        elif unit == "pixel":
            pixelx = int(shift[0])
            pixely = int(shift[1])
            print("# Image shift is performed ###################")
            print("(RA, Dec) = (%1d, %1d) [pixel] is shifted"%(pixelx, pixely))
            print("(RA, Dec) = (%0.3f, %0.3f) [arcsec] is shifted"%(pixelx*cellx, pixely*celly))


        # move the center of mass to the actual center of the self
        #outfits.data[istokes, ifreq] = sn.interpolation.shift(image,np.asarray([nyref - 1, nxref - 1]) - [nyref - pixely , nxref + pixelx])
        outfits.data[istokes, ifreq] = sn.interpolation.shift(image,np.asarray([nyref - 1, nxref - 1]) - [nyref - pixely , nxref + pixelx])

        # scale total flux
        if save_totalflux:
            totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
            outfits.data[istokes, ifreq] *= totalflux /outfits.totalflux(istokes=istokes, ifreq=ifreq)
        # update FITS
        outfits.update_fits()

        return outfits

    def peakshift(self,save_totalflux=False,ifreq=0,istokes=0):
        '''
        Shift the image so that its peak position coincides with the reference pixel.

        Arg:
          istokes (integer):
            index for Stokes Parameter at which the image will be edited
          ifreq (integer):
            index for Frequency at which the image will be edited
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.

        Returns:
          imdata.IMFITS object
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        image_ = outfits.data[istokes, ifreq]
        image = np.nan_to_num(image_)
        nxref = outfits.header["nxref"]
        nyref = outfits.header["nyref"]
        cellsize = outfits.header["dy"]*3600 # arcsec

        # move the center of mass to the actual center of the self
        pix = np.unravel_index(np.argmax(image), image.shape) # find peak position on the 2D dimention. ex (1,2)
        offset = np.array([nyref-1, nxref-1]) - pix
        print("Offset [arcsec]: (Dec, RA)= ",offset*cellsize)
        outfits.data[istokes, ifreq] = sn.interpolation.shift(image, np.asarray([nyref - 1, nxref - 1]) - pix)
        # scale total flux
        if save_totalflux:
            totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
            outfits.data[istokes, ifreq] *= totalflux / \
                outfits.totalflux(istokes=istokes, ifreq=ifreq)
        # update FITS
        outfits.update_fits()
        return outfits


    def zeropad(self, Mx, My):
        '''
        Uniformly pad zero and extend fov of the image to (My, Mx) pixels.

        Args:
          self: input imagefite.imagefits object
          Mx (integer): Number of pixels in RA (x) direction for the padded image.
          My (integer): Number of pixels in Dec(y) direction for the padded image.

        Returns:
          imdata.IMFITS object
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        Nx = outfits.header["nx"]
        Ny = outfits.header["ny"]
        Nf = outfits.header["nf"]
        Ns = outfits.header["ns"]
        if (Nx > Mx):
            print("[Error] please set a pixel size for RA  larger than original one!")
            return -1
        if (Ny > My):
            print("[Error] please set a pixel size for Dec larger than original one!")
            return -1
        newdata = np.zeros([Ns, Nf, My, Mx])
        for istokes in np.arange(Ns):
            for ifreq in np.arange(Nf):
                # update data
                newdata[istokes, ifreq, np.around(My / 2 - Ny / 2):np.around(My / 2 - Ny / 2) + Ny, np.around(
                    Mx / 2 - Nx / 2):np.around(Mx / 2 - Nx / 2) + Nx] = outfits.data[istokes, ifreq]
        outfits.data = newdata

        # update pixel info
        outfits.header["nx"] = Mx
        outfits.header["ny"] = My
        outfits.header["nxref"] += Mx / 2 - Nx / 2
        outfits.header["nyref"] += My / 2 - Ny / 2
        outfits.update_fits()

        return outfits

    def cut_image(self, savename = None , radius = 0, savefits = False, istokes = 0, ifreq = 0):
        """
        radius : arcsec
        """

        # cut image size #######################################################
        self.data[istokes,ifreq]= np.nan_to_num(self.data[istokes,ifreq])

        if int(np.round(2*radius/(self.header["dy"]*3600)))%2 != 0:
            imagesize = int(np.round(2*radius/(self.header["dy"]*3600)))+1
        else:
            imagesize = int(np.round(2*radius/(self.header["dy"]*3600)))

        ref = IMFITS(dx=np.abs(self.header["dx"])*3600,nx= imagesize, angunit="asec")

        modelimage = ref.cpimage(self, save_totalflux=False)
        modelimage.header       = self.header
        modelimage.header["nx"] = imagesize
        modelimage.header["ny"] = imagesize
        modelimage.header["nxref"] = 1+imagesize/2
        modelimage.header["nyref"] = 1+imagesize/2
        if savefits == True:
            os.system("rm -rf "+savename+".fits")
            modelimage.save_fits(savename+".fits")
            print("You saved "+savename+".fits")
        ###################################################################

        return modelimage



    def rotate(self, angle=0, deg=True, save_totalflux=False):
        '''
        Rotate the input image

        Args:
          self: input imagefite.imagefits object
          angle (float): Rotational Angle. Anti-clockwise direction will be positive (same to the Position Angle).
          deg (boolean): It true, then the unit of angle will be degree. Otherwise, it will be radian.
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        if deg:
            degangle = -angle
            radangle = -np.deg2rad(angle)
        else:
            degangle = -np.rad2deg(angle)
            radangle = -angle
        #cosa = np.cos(radangle)
        #sina = np.sin(radangle)
        Nx = outfits.header["nx"]
        Ny = outfits.header["ny"]
        for istokes in np.arange(self.header["ns"]):
            for ifreq in np.arange(self.header["nf"]):
                image = outfits.data[istokes, ifreq]
                # rotate data
                newimage = sn.rotate(image, degangle,order=1)
                # get the size of new data
                My = newimage.shape[0]
                Mx = newimage.shape[1]
                # take the center of the rotated image
                outfits.data[istokes, ifreq] = newimage[My // 2 - Ny // 2:My // 2 - Ny // 2 + Ny,
                                                        Mx // 2 - Nx // 2:Mx // 2 - Nx // 2 + Nx]
                # Flux Scaling
                if save_totalflux:
                    totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
                    outfits.data[istokes, ifreq] *= totalflux / \
                        outfits.totalflux(istokes=istokes, ifreq=ifreq)
        outfits.update_fits()
        return outfits


    def deproject(self, angle = 0, inclination = 0, save_totalflux = True):
        '''
        deproject the input image

        Args:
          self: input imagefite.imagefits object
          angle (Unit:deg, Type:float): Rotational Angle. Anti-clockwise direction will be positive (same to the Position Angle).
          inclination (Unit:deg, Type:float): Inclination.
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        imrot = self.rotate(angle= angle, deg=True, save_totalflux=save_totalflux)
        imrot.header["dx"] = imrot.header["dx"]/np.cos(np.deg2rad(inclination))
        ref = copy.deepcopy(self)
        imrotinc = ref.cpimage(imrot, save_totalflux=save_totalflux)
        #imrotinc.data[0,0] = imrotinc.data[0,0]
        imrotinc = imrotinc.rotate(angle= - angle, deg=True, save_totalflux=save_totalflux)
        imrotinc.update_fits()
        return(imrotinc)

    def offsource_area(self, radius= 1.0, inputunit = "jypixel-1", outputunit = "jypixel-1", istokes=0, ifreq=0):
        """
        specify off-source area on the input image domain
        Args:
         radius (unit: arcsec): radius of the offsource area
        """
        if inputunit == "jybeam-1" and outputunit == "jypixel-1":
            self = self.jybeam2jypixel()

        fov = self.header["nx"]*abs(self.header["dx"]*3600) # arcsec
        xg,yg = self.get_xygrid(angunit="arcsec")
        xg,yg = np.meshgrid(xg,yg)
        t = ((xg**2+yg**2 >radius**2) & (xg**2+yg**2  < (fov*0.4)**2))
        region = np.nan_to_num(self.data[istokes, ifreq][t])
        return region

    def onsource_area(self, radius= 1.0, inner_radius = 0.0, istokes=0, ifreq=0):
        """
        specify on-source area on the input image domain
        Args:
         radius (unit: arcsec): radius of the off-source area
        """
        fov = self.header["nx"]*abs(self.header["dx"]*3600) # arcsec
        xg,yg = self.get_xygrid(angunit="arcsec")
        xg,yg = np.meshgrid(xg,yg)
        if inner_radius == 0.0:
            t = (xg**2+yg**2) < radius**2
        elif inner_radius > 0.0:
            t = ((xg**2+yg**2) < radius**2) & (xg**2+yg**2 > inner_radius**2)

        region = np.nan_to_num(self.data[istokes, ifreq][t])
        return region

    def totalflux(self, unit = "jypixel-1", radius = 1.0, istokes=0, ifreq=0):
        '''
        calculate the total flux of the input-image [jy/pixel].
        The unit of the input image must be [jy/pixel].

        Args:
          istokes (integer): index for Stokes Parameter at which the total flux will be calculated
          ifreq (integer): index for Frequency at which the total flux will be calculated
          radius (unit: arcsec): radius from central position (x,y = 0,0)
        '''
        if unit == "jybeam-1":
            data = self.jybeam2jypixel()
        elif unit == "jypixel-1":
            data = self
        region = data.onsource_area(radius = radius, istokes=istokes, ifreq = ifreq)
        return region.sum()


    def convert_BrightnessTemp(self, frequency = 0.0, unit = "jybeam-1", istokes=0, ifreq=0):
        """
        Convert Intensity [Jy/pixel] or [Jy/beam] into brightness temperature [K]
        Args:
          istokes (integer): index for Stokes Parameter at which the total flux will be calculated
          ifreq (integer): index for Frequency at which the total flux will be calculated
        """
        if unit == "jybeam-1":
            self = self.jybeam2jypixel()
        elif unit == "jypixel-1":
            pass

        self.data[istokes, ifreq] = np.nan_to_num(self.data[istokes, ifreq])# nan to zero
        self.data[istokes, ifreq] = np.where(self.data[0,0] < 0.0, 0.0, self.data[istokes, ifreq]) # minus to zero

        # SI unit
        if frequency == 0.0:
            freq = self.header['f'] #Hz
        elif frequency >0.0:
            freq = frequency
        dx = np.abs(self.header["dx"]*self.angconv("deg","asec"))
        nx = self.header["nx"]
        xcellsize = abs(self.header["dx"]*3600) # cellsize
        ycellsize = abs(self.header["dy"]*3600) # cellsize
        # step-0: convert pixel scale
        ref = IMFITS(dx=dx,nx= nx, angunit="asec")
        ima = ref.cpimage(self, save_totalflux=True)
        I = 1.0e-23*self.data[istokes, ifreq]*((3600.*180./np.pi)**2.)/(xcellsize*ycellsize)# jy -> cgs unit
        I +=0.0 # minus zero -> plus zero, avoiding error.
        ima.data[istokes, ifreq] = (hp*freq/kb)/np.log(1.+((2.*hp*freq**3.)/(I*clight**2.)))
        ima.data[istokes, ifreq] = np.where(ima.data[0,0] == np.inf,0.0, ima.data[istokes, ifreq]) # (case, True, False)
        #Reflect current self.data / self.header info to the image FITS data
        ima.update_fits()

        print("#### Calculation of the Brightness Temperature ###")
        print("Cell Size = %1.4f [arcsec]"%(xcellsize))
        print("Observing Frequency = %1.3f [GHz]"%(freq*1e-9))
        print("Peak Brightness Temperature = %1.3f [K]"%(ima.peak()))
        print("##################################################")

        return ima

    def DiskTemp(radius, luminosity = 1.0,flaring_angle = 0.05):
        '''
        calculate the disk midplane temperature [K] in radial profile, as in Kenyon & Hartmann 1987
        Args:
         luminosity (unit : Lsun): default value is 1.0.
         flaring angle: default value is 0.05 (Dullemond & Dominik 2004).
        '''
        const = (1/(4*np.pi))**(0.25)*(1.496*1e13)**-(0.5)*(3.83*1e33)**(0.25)*(5.670*1e-5)**(-0.25)
        temp = const*(radius)**(-0.5)*(luminosity)**(0.25)*flaring_angle**(0.25)
        return temp


    def planck_func2D(self, frequency=0.0, istokes=0, ifreq=0):
        """
        planck function
        note. not considering approximation.
        Args:
         self: temperature in 2D image [K/pixel]
         frequency: freqency [Hz]
        """
        v = frequency#Hz
        outfits = copy.deepcopy(self)
        fterm = (2.0*hp*(v**3))/(clight**2)
        exp   = np.exp(hp*v/(kb*self.data[istokes, ifreq]))-1.0
        outfits.data[0,0] = fterm/exp
        return outfits

    def planck_func(self, T):
        """
        planck function
        note. not considering approximation.
        Args:
         T: temperature in 2D image [K]
        """
        v = self.header['f'] #Hz
        fterm =(2.0*hp*(v**3))/(clight**2)
        exp   = np.exp((hp*v)/(kb*T))-1.0
        Bv    = fterm/exp
        return Bv

    def calc_diskmass(self, dpc=140.0, Tdust= 20.0, radius =1.0, istokes=0, ifreq=0):
        """
        calculate mass from total flux.
        dust opacity kappa is drived in Andrews et al. (2013).
        instensity unit of the input image must be [jy/pixel].

        Args:
         freq (unit: Hz)
         dpc (unit: arcsec) : object distance
         Tdust  (unit: K): Dust Temperature
         radius (unit: arcsec): radius from central position (x,y = 0,0)
        """
        freq = self.header["f"]
        ### setting constant ###
        Msun = 1.9884e33 # [g]
        Mjup = 1.8986e30 # [g]
        Mearth =  5.9724e27 # [g]

        ### interactiove section
        print('### Calculation of Disk Mass from Dust Continuum ###')
        dcm = dpc*3.08568e18 # pc --> cm

        #Dust opacity kappa drived in Andrews et al. (2013).
        kappa = 2.3*((freq*1.e-9)/230.0)**0.4 # [cm2 g-1]

        # total flux
        Fv = self.totalflux(radius, istokes, ifreq) # Jy
        Fv_cgs=float(Fv)*1.0e-23 # Jy --> cgs

        # luminosity  : unit : Lsun
        lum = 4*np.pi*(dcm**2)*self.header["f"]*Fv_cgs/Lsun

        # planck function
        Bv=self.planck_func(Tdust)

        # solution
        Mtotal = Fv_cgs*dcm*dcm/(kappa*Bv) # [g]
        Mtotal_sun = Mtotal/Msun           # g --> Msun
        Mtotal_jup = Mtotal/Mjup           # g --> Msun
        Mtotal_earth = Mtotal/Mearth       # g --> Msun

        print('freqency = {} GHz'.format(freq*1.0e-9))
        print('Total Flux = {} mJy'.format(Fv*1e3))
        print('distance = {} pc'.format(dpc))
        print('Luminosity = {} Lsun'.format(lum))
        print('dust opacity kappa = {} cm2/g'.format(kappa))
        print('Tdsut = {} K'.format(Tdust))
        print('Dust Mass Mdust = {} Msun'.format(Mtotal_sun))
        print('Dust Mass Mdust = {} Mjup'.format(Mtotal_jup))
        print('Dust Mass Mdust = {} Mearth'.format(Mtotal_earth))

        return {"Msun":Mtotal_sun, "Mjup":Mtotal_jup, "Mearth":Mtotal_earth}


    def confidence_level(self, radius = 1.0, istokes=0, ifreq=0):
        """
        specify confidence levels of the off-source area on the input image domain
        return [0]:confidence level: 100%,[1]: confidence level in 99.0%, [2]:confidence level in 99.7%
        Args:
         radius (unit: arcsec): radius of the offsource area
        """
        #print ("# caluculate confidence interval ###############")
        offarea = self.offsource_area(radius, istokes=0, ifreq=0)
        noise = np.sort(offarea)
        #confidence level: 100%
        n999 = int(round(len(offarea)))
        noise999 = noise[n999-1]
        #print ("Number of confidence level in 100%:  ",int(n999),"/", len(offarea))
        #print ("The Intensity value of confidence level in 100% :", noise999)
        # confidence level : 99%
        n99 = int(round(len(offarea)*0.99))
        noise99 = noise[n99-1]
        #print ("Number of confidence level in 99.0%:  ",int(n99),"/", len(offarea))
        #print ("The Intensity value of confidence level in 99.0% :", noise99)
        # confidence level : 99.7%
        n997 = int(round(len(offarea)*0.997))
        noise997 = noise[n997-1]
        #print ("Number of confidence level in 99.7%:  ",int(n997),"/", len(offarea))
        #print ("The Intensity value of confidence level in 99.7% :", noise997)

        n95 = int(round(len(offarea)*0.95))
        noise95 =  noise[n95-1]
        #print ("Number of confidence level in 95%:  ",int(noise95),"/", len(offarea))
        #print ("######################################")

        return (noise999,noise99,noise997, noise95)


    def plt_spmimage(self, name = None,offsource_radius = 1.0, scale = "linear",unit = "jyasec-2", orientation = "vertical", ticklocation = 'right', labelsize = 10, colorbar_tick_size = 5.0,
                    gamma =1.0,  mapcolor = "inferno" , pa = 0 , inclination = 0, Dodeproject =False, nameon = False, object_distance = 0.0, onsource_radius = 0.5, normalize = False, lowestcont_value = None,
                    vmax = None, vmin = None, contour = True, contourlevel = [1,3,6,9,12], tickcolor = "w", range = 0.5, beam_maj = 0, beam_min =0, beam_pa=0,istokes=0, ifreq=0):
        """
        Parameters
        ----------
        fitsfile: unit : Jy/pixel
            fitsfile
        name: str
            object name
        offsource_radius: float
            position matching in arseconds.
        orientation: str
        	vertical or horizontal
        ticklocation: str
            right or top
        """

        if object_distance > 0.:
            obj_distance = object_distance * 0.1*2.06265*1e5*(np.pi/(3600*180)) # au @ 0.1 arcsec
            obj_distance = np.round(obj_distance,1)
        elif (object_distance ==0.0) and (name != None):
            object_distance  = object_propaty(name)["distance"] # pc
            obj_distance = object_distance * 0.1*2.06265*1e5*(np.pi/(3600*180)) # au @ 0.1 arcsec
            obj_distance = np.round(obj_distance,1)
        else:
            obj_distance = 0

        if unit == "jyasec-2":
            spmim = self.jypixel2jyasec(istokes, ifreq) # Jy/asec2
        if unit == "mjyasec-2":
            spmim = self.jypixel2jyasec(istokes, ifreq) # Jy/asec2
            spmim.data[0,0] = spmim.data[0,0]*1e3  # mJy/asec2

        if normalize == True:
            spmim.data[0,0] = spmim.data[0,0]/spmim.peak()
        if Dodeproject == True:
            spmim = spmim.deproject(angle = pa, inclination = inclination, save_totalflux = True)

        # get header infermation
        #fov = self.header["ny" ]*self.header["dy"]*3600 # arcsec
        conflevel = spmim.confidence_level(offsource_radius, istokes, ifreq)

        # estimate disk mass
        #diskmass = self.calc_diskmass(dpc= object_distance, Tdust= 20.0, radius =offsource_radius, istokes= istokes, ifreq= ifreq)

        # set a colorbar
        my_cmap = copy.copy(cm.get_cmap(mapcolor)) # copy the default cmap
        my_cmap.set_bad((0,0,0)) # <-  important !
        # plot  images
        matplotlibrc(nrows=1,ncols=1,width=330,height=300) # set size of figure
        fig, axs = plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False) # gerenete figure and panels
        plt.subplots_adjust(wspace=0.4)

        ## input
        if vmax != None:
            vmax = vmax
        else:
            vmax = spmim.peak()#*np.cos(np.deg2rad(inclination))
        if vmin != None:
            vmin = vmin
        else:
            vmin = 0

        levels = np.array(contourlevel)

        ax = axs
        plt.sca(ax) # indicate "set current axis
        tickcolor = tickcolor
        labelcolor = "k"
        plt.tick_params(which='both',direction='in',bottom =True,top = True,left = True,right =True,
                        colors=tickcolor,labelrotation=0, labelcolor=labelcolor)
        plt.tick_params(labelsize = labelsize)

        params = {'axes.labelsize': labelsize,
                  'axes.titlesize': labelsize,
                  'font.size' : labelsize,
                  'legend.fontsize': labelsize,
                  'xtick.labelsize': labelsize,
                  'ytick.labelsize': labelsize,
                  'axes.linewidth' :2.0,
                  'xtick.major.width' : 1.0,
                  'ytick.major.width' : 1.0,
                  'xtick.minor.width' : 1.0,
                  'ytick.minor.width' : 1.0,
                   'xtick.major.size' : 6,
                   'ytick.major.size' : 6,
                    'xtick.minor.size' : 4.,
                    'ytick.minor.size' : 4.}
        rcParams.update(params)

        #plt.rcParams["xtick.direction"] = "out"
        #plt.rcParams["ytick.direction"] = "out"

        if contour == True:
            if lowestcont_value != None:
                lowestcont_value = lowestcont_value
            else:
                lowestcont_value= conflevel[0]

            spwimagecont = spmim.contour(angunit='arcsec',
                                        cmul=lowestcont_value,
                                        levels=levels,
                                        relative=False,
                                        linewidths = 0.8)
        if scale     == "log":
            spwimage = spmim.imshow(angunit='arcsec', interpolation='nearest',norm = LogNorm(vmin = vmin, vmax =vmax),cmap=my_cmap)
        elif  scale == "linear":
            spwimage = spmim.imshow(angunit='arcsec',vmin = vmin, vmax =vmax,cmap=my_cmap)
        elif  scale == "gamma":
            spwimage = spmim.imshow(angunit='arcsec',vmin = vmin, vmax =vmax,   norm = PowerNorm(gamma), cmap=my_cmap)

        plt.xlim(range,-range)
        plt.ylim(-range,range)

        if obj_distance >0.:
            plt.text(range*(-0.1), range*(-0.9), '0".1  (= '+str(obj_distance)+' au)', ha = 'center', va = 'bottom',color ="w", fontsize=12)
            plt.plot([-0.05,0.05],[range*(-0.7),range*(-0.7)],color="w",lw=2)
            
        if nameon == True:
            plt.text(0,range*(0.75),name, ha = 'center', va = 'bottom',color ="w", fontsize=18)
        #if Dodeproject ==True:
            #plt.xlabel("Major Axis (arcsec)")
            #plt.plot([0,0],[-range,range],'w--', linewidth=1.0)
            #plt.text( 0.1, 0.3, 'Major Axis',ha = 'right', va = 'bottom',color ="w", fontsize=12)

        if beam_maj > 0:
            bmin_plot, bmaj_plot = ax.transLimits.transform((beam_min,beam_maj)) - ax.transLimits.transform((0,0))
            beam = patches.Ellipse(xy=(0.15, 0.15), width=bmin_plot, height=bmaj_plot, fc="white", angle= beam_pa, transform=ax.transAxes)
            ax.add_patch(beam)
            plt.text(range*(0.65), range*(-0.9),
                    np.str(int(np.round(beam_maj*1e3,0)))+r"$\times$"+np.str(int(np.round(beam_min*1e3,0)))+" mas",
                    ha = 'center', va = 'bottom',color ="w", fontsize=12)


        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes(ticklocation,size="7%", pad=0.05)

        if Dodeproject ==True:
            if unit == "mjyasec-2":
                if normalize == True:
                    cb =plt.colorbar(spwimage,cax=ax_cb, orientation=orientation, ticklocation = ticklocation,
                                    label= r"Normalized Intensity $\times$ cos$i$)")
                else:
                    cb =plt.colorbar(spwimage,cax=ax_cb, orientation=orientation, ticklocation = ticklocation,
                                    label= r"Intensity $\times$ cos$i$ (mJy arcsec$^{-2}$)")
                cb.ax.tick_params(labelsize=labelsize, size=colorbar_tick_size)

            elif unit == "jyasec-2":
                if normalize == True:
                    cb =plt.colorbar(spwimage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation,
                                    label= r"Normalized Intensity $\times$ cos$i$)")
                else:
                    cb =plt.colorbar(spwimage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation,
                                    label= r"Intensity $\times$ cos$i$ (Jy arcsec$^{-2}$)")
                cb.ax.tick_params(labelsize=labelsize, size=colorbar_tick_size)

        elif Dodeproject ==False:
            if unit == "mjyasec-2":
                if normalize == True:
                    cb = plt.colorbar(spwimage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation,
                                    label= r"Normalized Intensity")
                else:
                    cb = plt.colorbar(spwimage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation,
                                    label= r"Intensity (mJy arcsec$^{-2}$)")
                cb.ax.tick_params(labelsize=labelsize,size=colorbar_tick_size)

            elif unit == "jyasec-2":
                if normalize == True:
                    cb = plt.colorbar(spwimage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation,
                                    label= r"Normalized Intensity")
                else:
                    cb = plt.colorbar(spwimage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation,
                                    label= r"Intensity (Jy arcsec$^{-2}$)")
                cb.ax.tick_params(labelsize=labelsize,size=colorbar_tick_size)

        # calculate total flux above detection threshold [Jy pixel-1]
        Idt     = self.confidence_level(offsource_radius, istokes, ifreq)[0]
        threshold = np.where(self.data[0,0] > Idt, self.data[0,0], 0) # (case, True, False)
        image_threshold = copy.deepcopy(self)
        image_threshold.data[0,0] = threshold # Jy pixel-1
        totalflux  = image_threshold.onsource_area(radius= onsource_radius, istokes=istokes, ifreq=ifreq).sum() # Jy

        print("## SpM Image  ##########################")
        im_jyasec = self.jypixel2jyasec()
        conflevel_jypixel     = self.confidence_level(offsource_radius, istokes, ifreq)
        conflevel_jyasec      = im_jyasec.confidence_level(offsource_radius, istokes, ifreq)
        print("Peak   = %1.3f [mJy/asec2]"%(im_jyasec.peak()*1e3))
        print("I_100  = %1.3f [mJy/asec2]"%(conflevel_jyasec[0]*1e3))
        print("I_100  = %1.6f [mJy/pixel]"%(conflevel_jypixel[0]*1e3))
        print("SNR    = %1.1f"%(im_jyasec.peak()/conflevel_jyasec[0]))
        print("Total Flux = %1.2f [mJy]"%(totalflux*1e3))
        print("Contour = ",contourlevel,"*I100")
        print("########################################")


    def plt_climage(self, name, outputunit = "jybeam-1", factor = 1e20, onsource_radius = 1.0, offsource_radius = 1.0, scale = "linear", gamma =1.0,
                    orientation = "vertical", ticklocation = 'right', vmin = 0, vmax = 0, step = 0, object_distance = 0, pa =0, inclination = 0,
                    mapcolor = "inferno",contlevels=[], Dodeproject = False, title_color = "white", nameon = False, nametype = "moment", sigmalevel = 5, tick_list =  np.arange(-100,100,0.5),
                    label_name = None, range =1.0, beam_color = "white", cont_colors="white", scale_color="k", scale_range=[-15,-16], contour = True, imageon = True, istokes=0, ifreq=0, maj_cut = False, min_cut = False, length = [1, 1, 1, 1],
                    nancolor="w"):
        """
        Note that input unit must be Jy beam-1.
        Parameters
        ----------
        fitsfile: str
            fitsfile
        name: str
            object name
        outtputunit: str
            output unit : select "jybeam-1" or "jyasec-2"
        offsource_radius: float
            position matching in arseconds.
        """
        self = copy.deepcopy(self)
        xx, yy = self.get_xygrid(twodim=True)
        mask = (xx**2 + yy**2 <= 20**2)
        self.data[0, 0][~mask] = np.nan
        # preparation #########################################
        cellsize   = (np.pi/180.0)**2*np.abs(self.header["dx"])*np.abs(self.header["dx"]) # radian
        beam_unit  = (np.pi/(4.0*np.log(2.0)))*((np.pi/180.0)**2.0)*self.header["BMAJ"]*self.header["BMIN"] # sr/beam
        imdx = self.header["dx"]*self.angconv("deg","asec") # unit: arcsec
        imdy = self.header["dy"]*self.angconv("deg","asec") # unit: arcsec


        if object_distance > 0.:
            obj_distance = object_distance * 0.1*2.06265*1e5*(np.pi/(3600*180)) # au @ 0.1 arcsec
            obj_distance = np.round(obj_distance,1)
        else :
            object_distance      = object_propaty(name)["distance"] # pc
            obj_distance = object_distance * 0.1*2.06265*1e5*(np.pi/(3600*180)) # au @ 0.1 arcsec
            obj_distance = np.round(obj_distance,1)

        # calculate peak value [Jy beam-1]
        peak = self.onsource_area(radius= onsource_radius, istokes=0, ifreq=0).max()
        # calculate total flux above 5 sigma [Jy beam-1]
        cl_offsource_area = self.offsource_area(radius= offsource_radius,
                                    inputunit = "jybeam-1", outputunit = "jybeam-1",
                                    istokes= istokes, ifreq = ifreq)

        rmsnoise = cl_offsource_area.std()
        threshold = np.where(self.data[0,0] > sigmalevel*rmsnoise, self.data[0,0], 0) # (case, True, False)
        climage_threshold = copy.deepcopy(self)
        climage_threshold.data[0,0] = threshold
        climage_threshold_jypixel = climage_threshold.jybeam2jypixel()
        totalflux  = climage_threshold_jypixel.onsource_area(radius = onsource_radius, istokes=istokes, ifreq=ifreq).sum()

        # calculate peak value [Jy asec-2]
        peak_jyasec2 = peak*(cellsize/beam_unit)/(np.abs(imdx*imdy))
        # calculate rms value [Jy asec-2]
        rms_jyasec2 = rmsnoise*(cellsize/beam_unit)/(np.abs(imdx*imdy))

        if outputunit == "jybeam-1":
            print ("## CLEAN Image  ###################")
            print ("BEAM MAJ   = %1.3f [arcsec]"%(self.header["BMAJ"]*3600))
            print ("BEAM MIN   = %1.3f [arcsec]"%(self.header["BMIN"]*3600))
            print ("BEAM PA    = %1.3f [deg]"%(self.header["BPA"]))
            print ("Peak       = %1.3f [mJy/asec2]"%(peak_jyasec2*1e3))
            print ("RMS noise  = %1.3f [mJy/asec2]"%(rms_jyasec2*1e3))
            print ("Peak       = %1.3f [mJy/beam]"%(peak*1e3))
            print ("RMS noise  = %1.3f [mJy/beam]"%(rmsnoise*1e3))
            print ("SNR        = %1.3f"%(peak/rmsnoise))
            print("Total Flux = %0.1f mJy"%(totalflux*1e3))
            print ("###################################")
        elif outputunit == "conticm2":
            print ("## CLEAN Image  ###################")
            print ("BEAM MAJ   = %1.3f [arcsec]"%(self.header["BMAJ"]*3600))
            print ("BEAM MIN   = %1.3f [arcsec]"%(self.header["BMIN"]*3600))
            print ("BEAM PA    = %1.3f [deg]"%(self.header["BPA"]))
            print ("Peak       = %1.3f [cm2]"%(peak/factor))
            print ("RMS noise  = %1.3f [cm2]"%(rmsnoise))
            print("Total Flux = %1.3f [cm2]"%(totalflux))
            print ("###################################")
        elif outputunit == "cm2":
            print ("## CLEAN Image  ###################")
            print ("BEAM MAJ   = %1.3f [arcsec]"%(self.header["BMAJ"]*3600))
            print ("BEAM MIN   = %1.3f [arcsec]"%(self.header["BMIN"]*3600))
            print ("BEAM PA    = %1.3f [deg]"%(self.header["BPA"]))
            print ("Peak       = %1.3f [cm2]"%(peak))
            print ("###################################")
        elif outputunit == "ratio":
            print ("## CLEAN Image  ###################")
            print ("BEAM MAJ   = %1.3f [arcsec]"%(self.header["BMAJ"]*3600))
            print ("BEAM MIN   = %1.3f [arcsec]"%(self.header["BMIN"]*3600))
            print ("BEAM PA    = %1.3f [deg]"%(self.header["BPA"]))
            print ("Peak       = %1.3f [cm2]"%(peak*1e3))
            print ("###################################")  
        elif outputunit == "factor":
            print ("## CLEAN Image  ###################")
            print ("BEAM MAJ   = %1.3f [arcsec]"%(self.header["BMAJ"]*3600))
            print ("BEAM MIN   = %1.3f [arcsec]"%(self.header["BMIN"]*3600))
            print ("BEAM PA    = %1.3f [deg]"%(self.header["BPA"]))
            print ("Peak       = %1.3f [cm2]"%(peak))
            print ("###################################")       
            
        # set a colorbar
        my_cmap = copy.copy(cm.get_cmap(mapcolor)) # copy the default cmap
        my_cmap.set_bad(color=nancolor) # <-  important !
        # plot  images
        fig = plt.figure(figsize=(8.27, 8.27))
        ax  = fig.add_subplot(111)
        font = {'family' : 'times',
                'size'   : 18}
        matplotlib.rc('font', **font)
        
        # setup of unit ######################
        if outputunit == "jybeam-1":
            image = self
            rms_vaule = rmsnoise
        
        elif outputunit == "ratio":
            image = self
            rms_vaule = step

        elif outputunit == "factor":
            image = self
            rms_vaule = step

        elif outputunit == "mjybeam-1":
            image = self
            image.data[0,0] = self.data[0,0]*1e3 # jy -> mjy
            rms_vaule = rmsnoise*1e3

        elif outputunit == "conticm2":
            image = self
            rms_vaule = step

        elif outputunit == "cm2":
            image = self
            rms_vaule = step
            
        elif outputunit == "jyasec-2":
            image = self.jybeam2jyasec(istokes= istokes, ifreq = ifreq) # Jy/asec2
            rms_vaule = rms_jyasec2

        elif outputunit == "mjyasec-2":
            image = self.jybeam2jyasec(istokes= istokes, ifreq = ifreq) # Jy/asec2
            image.data[0,0] = image.data[0,0]*1e3 # jy -> mjy/asec2
            rms_vaule = rms_jyasec2*1e3

        ########################################

        if Dodeproject ==True:
            image = image.deproject(angle = pa, inclination = inclination, save_totalflux = True)

        ## input
        if vmax != 0:
            vmax = vmax
        else:
            im_revised = copy.deepcopy(image)
            im_revised.data[istokes,ifreq] = np.nan_to_num(image.data[istokes,ifreq])
            vmax = im_revised.peak()

        if vmin != 0:
            vmin = vmin
        else:
            vmin = 0
        levels = np.array(contlevels)
        plt.sca(ax) # indicate "set current axis
        tickcolor = "w"
        labelcolor = "k"
        plt.tick_params(which='both',direction='in',bottom =True,top = True,left = True,right =True,
                        colors=tickcolor,labelrotation=0, labelcolor=labelcolor)
        #Input
        if contour == True:
            imagecont = image.contour(angunit='arcsec',
                                        logscale=False,
                                        cmul=rms_vaule,
                                        levels=levels,
                                        colors=cont_colors,
                                        linewidths = 0.8)
        if imageon == True:
            if scale     == "log":
                climage = image.imshow(angunit='arcsec', interpolation='nearest',vmin = vmin, vmax =vmax, norm = LogNorm(vmin = vmin, vmax =vmax),cmap=my_cmap, tick_list = tick_list)
                
            elif  scale == "linear":
                climage = image.imshow(angunit='arcsec',vmin = vmin, vmax =vmax,cmap=my_cmap, tick_list = tick_list, tick_color='k', tick_width='1',labelsize= 18)
                
            elif  scale == "gamma":
                climage = image.imshow(angunit='arcsec',vmin = vmin, vmax =vmax, norm = PowerNorm(gamma), cmap=my_cmap, tick_list = tick_list)
                

        plt.xlim(range,-range)
        plt.ylim(-range,range)


        plt.text(range*(-0.8), range*(-0.9), str(int(obj_distance)*10)+' AU', ha = 'center', va = 'bottom',color =scale_color, fontsize=18, family = 'times')
        plt.plot(scale_range,[range*(-0.75),range*(-0.75)],color=scale_color,lw=2)
        
        ang = self.header["BPA"]
        dis_maj = (self.header["BMAJ"] * 3600)
        dis_min = (self.header["BMIN"] * 3600)
        tran = ax.transLimits.transform((0,0))
        #tran = [0,0]
        beam_center = [0,0]
        Major_x = abs(dis_maj * np.sin(ang) / 2)
        Major_y = abs(dis_maj * np.cos(ang) / 2)
        Minor_x = abs(dis_min * np.cos(ang) / 2)
        Minor_y = abs(dis_min * np.sin(ang) / 2)
        Major = [beam_center[0] - Major_x + tran[0], beam_center[1] + Major_y + tran[1], beam_center[0] + Major_x - tran[0], beam_center[1] - Major_y - tran[1]]
        Minor = [beam_center[0] - Minor_x - tran[0], beam_center[1] - Minor_y + tran[1], beam_center[0] + Minor_x + tran[0], beam_center[1] + Minor_y - tran[1]]
                
        if maj_cut == True:
            plt.plot([beam_center[0], Major[0] * length[0]],[beam_center[1], Major[1] * length[0]], color="c",lw=2)
            plt.plot([beam_center[0], Major[2] * length[1]],[beam_center[1], Major[3] * length[1]], color="c",lw=2)
            
        if min_cut == True:
            plt.plot([beam_center[0], Minor[0] * length[2]],[beam_center[1], Minor[1] * length[2]], color="c", lw=2)
            plt.plot([beam_center[0], Minor[2] * length[3]],[beam_center[1], Minor[3] * length[3]], color="c", lw=2)
        
        if nameon == True:
            if nametype == "continuum":
                plt.text(range*(0.73),range*(0.87),name, ha = 'center', va = 'bottom',color = title_color, fontsize=30, family = 'times')
            elif nametype == "moment":
                plt.text(range*(0.95),range*(0.85),name, ha = 'left', va = 'bottom',color = title_color, fontsize=20, family = 'times')

        bmin_plot, bmaj_plot = ax.transLimits.transform((self.header["BMIN"]*3600, self.header["BMAJ"]*3600)) - ax.transLimits.transform((0,0))
        beam = patches.Ellipse(xy=(0.12, 0.1), width=bmin_plot, height=bmaj_plot, fc=beam_color, angle= self.header["BPA"], transform=ax.transAxes)
        ax.add_patch(beam)
        
        # plt.text(range*(0.6), range*(-0.9),
        #             str(int(np.round(self.header["BMAJ"]*3600*1e3,0)))+r"$\times$"+str(int(np.round(self.header["BMIN"]*3600*1e3,0)))+" mas",
        #             ha = 'center', va = 'bottom',color ="w", fontsize=18, family = 'times')

        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

        if imageon == True:
            divider = make_axes_locatable(ax)
            ax_cb = divider.append_axes(ticklocation,size="7%", pad=0.05)

            if outputunit == "jybeam-1":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation)
                cb.set_label(label= r"Intensity (Jy beam$^{-1}$)", family = 'times')
            
            elif outputunit == "ratio":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation)
                cb.set_label(label= name+r' $(10^{-10})$', family = 'times')
                
            elif outputunit == "factor":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation)
                if label_name == None:
                    cb.set_label(label= name, family = 'times')
                else :
                    cb.set_label(label= label_name, family = 'times')

            elif outputunit == "conticm2":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation)
                cb.set_label(label= r"Column density (cm$^{2}$)", family = 'times')
                                
            elif outputunit == "cm2":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation)
                cb.set_label(label= r"Column density (cm$^{2}$)", family = 'times')
                
            elif outputunit == "mjybeam-1":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation)
                cb.set_label(label= r"Intensity (mJy beam$^{-1}$)", family = 'times')
                
            elif outputunit == "jyasec-2":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation)
                cb.set_label(label= r"Intensity (Jy arcsec$^{-2}$)", family = 'times')
                cb.ax.locator_params(nbins=5)

            elif outputunit == "mjyasec-2":
                cb =plt.colorbar(climage,cax=ax_cb, orientation=orientation, ticklocation = ticklocation)
                cb.set_label(label= r"Intensity (mJy arcsec$^{-2}$)", family = 'times')

        return{"offsource_area":cl_offsource_area}
    
    def plt_image(self, name, outputunit = "jybeam-1",  scale = "linear", gamma =1.0,
                    orientation = "vertical", ticklocation = 'right', vmin = 0, vmax = 0, rms = 0, step = 0, object_distance = 0, pa =0, inclination = 0,
                    mapcolor = "inferno",contlevels=[5,10,15], Dodeproject = False, nameon = False, sigmalevel = 5, tick_list =  np.arange(-100,100,5),
                    range =1.0, cont_colors="white", contour = True, imageon = True, istokes=0, ifreq=0):
        """
        Note that input unit must be Jy beam-1.
        Parameters
        ----------
        fitsfile: str
            fitsfile
        name: str
            object name
        outtputunit: str
            output unit : select "jybeam-1" or "jyasec-2"
        offsource_radius: float
            position matching in arseconds.
        """
        # preparation #########################################
        cellsize   = (np.pi/180.0)**2*np.abs(self.header["dx"])*np.abs(self.header["dx"]) # radian
        beam_unit  = (np.pi/(4.0*np.log(2.0)))*((np.pi/180.0)**2.0)*self.header["BMAJ"]*self.header["BMIN"] # sr/beam
        imdx = self.header["dx"]*self.angconv("deg","asec") # unit: arcsec
        imdy = self.header["dy"]*self.angconv("deg","asec") # unit: arcsec


        if object_distance > 0.:
            obj_distance = object_distance * 0.1*2.06265*1e5*(np.pi/(3600*180)) # au @ 0.1 arcsec
            obj_distance = np.round(obj_distance,1)
        else :
            object_distance      = object_propaty(name)["distance"] # pc
            obj_distance = object_distance * 0.1*2.06265*1e5*(np.pi/(3600*180)) # au @ 0.1 arcsec
            obj_distance = np.round(obj_distance,1)

        # calculate peak value [Jy beam-1]
        peak = self.onsource_area(radius= onsource_radius, istokes=0, ifreq=0).max()
        # calculate total flux above 5 sigma [Jy beam-1]
        cl_offsource_area = self.offsource_area(radius= offsource_radius,
                                    inputunit = "jybeam-1", outputunit = "jybeam-1",
                                    istokes= istokes, ifreq = ifreq)
        rmsnoise = cl_offsource_area.std()
        threshold = np.where(self.data[0,0] > sigmalevel*rmsnoise, self.data[0,0], 0) # (case, True, False)
        climage_threshold = copy.deepcopy(self)
        climage_threshold.data[0,0] = threshold
        climage_threshold_jypixel = climage_threshold.jybeam2jypixel()
        totalflux  = climage_threshold_jypixel.onsource_area(radius= onsource_radius, istokes=istokes, ifreq=ifreq).sum()

        # calculate peak value [Jy asec-2]
        peak_jyasec2 = peak*(cellsize/beam_unit)/(np.abs(imdx*imdy))
        # calculate rms value [Jy asec-2]
        rms_jyasec2 = rmsnoise*(cellsize/beam_unit)/(np.abs(imdx*imdy))


        print ("## CLEAN Image  ###################")
        print ("BEAM MAJ   = %1.3f [arcsec]"%(self.header["BMAJ"]*3600))
        print ("BEAM MIN   = %1.3f [arcsec]"%(self.header["BMIN"]*3600))
        print ("BEAM PA    = %1.3f [deg]"%(self.header["BPA"]))
        print ("Peak       = %1.3f [mJy/asec2]"%(peak_jyasec2*1e3))
        print ("RMS noise  = %1.3f [mJy/asec2]"%(rms_jyasec2*1e3))
        print ("Peak       = %1.3f [mJy/beam]"%(peak*1e3))
        print ("RMS noise  = %1.3f [mJy/beam]"%(rmsnoise*1e3))
        print ("SNR        = %1.3f"%(peak/rmsnoise))
        print("Total Flux = %0.1f mJy"%(totalflux*1e3))
        print ("###################################")

        # set a colorbar
        my_cmap = copy.copy(cm.get_cmap(mapcolor)) # copy the default cmap
        my_cmap.set_bad((0,0,0)) # <-  important !
        # plot  images
        matplotlibrc(nrows=1,ncols=1,width=330,height=300) # set size of figure
        fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False) # gerenete figure and panels
        plt.subplots_adjust(wspace=0.4)

        # setup of unit ######################
        if outputunit == "jybeam-1":
            image = self
            rms_vaule = rmsnoise

        elif outputunit == "mjybeam-1":
            image = self
            image.data[0,0] = self.data[0,0]*1e3 # jy -> mjy
            rms_vaule = rmsnoise*1e3


        elif outputunit == "jyasec-2":
            image = self.jybeam2jyasec(istokes= istokes, ifreq = ifreq) # Jy/asec2
            rms_vaule = rms_jyasec2

        elif outputunit == "mjyasec-2":
            image = self.jybeam2jyasec(istokes= istokes, ifreq = ifreq) # Jy/asec2
            image.data[0,0] = image.data[0,0]*1e3 # jy -> mjy/asec2
            rms_vaule = rms_jyasec2*1e3

        ########################################

        if Dodeproject ==True:
            image = image.deproject(angle = pa, inclination = inclination, save_totalflux = True)

        ## input
        if vmax != 0:
            vmax = vmax
        else:
            im_revised = copy.deepcopy(image)
            im_revised.data[istokes,ifreq] = np.nan_to_num(image.data[istokes,ifreq])
            vmax = im_revised.peak()

        if vmin != 0:
            vmin = vmin
        else:
            vmin = rms_vaule
        levels = np.array(contlevels)
        plt.sca(ax) # indicate "set current axis
        tickcolor = "w"
        labelcolor = "k"
        plt.tick_params(which='both',direction='in',bottom =True,top = True,left = True,right =True,
                        colors=tickcolor,labelrotation=0, labelcolor=labelcolor)
        #Input
        if contour == True:
            imagecont = image.self_contour(angunit='arcsec',
                                        rms = rms,
                                        step = step,
                                        colors=cont_colors,
                                        linewidths = 0.8)
        if imageon == True:
            if scale     == "log":
                climage = image.imshow(angunit='arcsec', interpolation='nearest',vmin = vmin, vmax =vmax, norm = LogNorm(vmin = vmin, vmax =vmax),cmap=my_cmap, tick_list = tick_list)
            elif  scale == "linear":
                climage = image.imshow(angunit='arcsec',vmin = vmin, vmax =vmax,cmap=my_cmap, tick_list = tick_list)
            elif  scale == "gamma":
                climage = image.imshow(angunit='arcsec',vmin = vmin, vmax =vmax, norm = PowerNorm(gamma), cmap=my_cmap, tick_list = tick_list)

        plt.xlim(range,-range)
        plt.ylim(-range,range)


        plt.text(range*(-0.1), range*(-0.9), '0".1  (= '+str(obj_distance)+' au)', ha = 'center', va = 'bottom',color ="w", fontsize=12)
        plt.plot([-0.05,0.05],[range*(-0.7),range*(-0.7)],color="w",lw=2)
        if nameon == True:
            plt.text(0,range*(0.75),name, ha = 'center', va = 'bottom',color ="w", fontsize=18)

        bmin_plot, bmaj_plot = ax.transLimits.transform((self.header["BMIN"]*3600, self.header["BMAJ"]*3600)) - ax.transLimits.transform((0,0))
        beam = patches.Ellipse(xy=(0.15, 0.15), width=bmin_plot, height=bmaj_plot, fc="white", angle= self.header["BPA"], transform=ax.transAxes)
        ax.add_patch(beam)
        plt.text(range*(0.65), range*(-0.9),
                    str(int(np.round(self.header["BMAJ"]*3600*1e3,0)))+r"$\times$"+str(int(np.round(self.header["BMIN"]*3600*1e3,0)))+" mas",
                    ha = 'center', va = 'bottom',color ="w", fontsize=12)

        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

        if imageon == True:
            divider = make_axes_locatable(ax)
            ax_cb = divider.append_axes(ticklocation,size="7%", pad=0.05)

            if outputunit == "jybeam-1":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation,
                                label= r"Intensity (Jy beam$^{-1}$)")
            elif outputunit == "mjybeam-1":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation,
                                label= r"Intensity (mJy beam$^{-1}$)")
            elif outputunit == "jyasec-2":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation,
                                label= r"Intensity (Jy arcsec$^{-2}$)")
                cb.ax.locator_params(nbins=5)

            elif outputunit == "mjyasec-2":
                cb =plt.colorbar(climage,cax=ax_cb, orientation=orientation, ticklocation = ticklocation,
                                label= r"Intensity (mJy arcsec$^{-2}$)")

        return{"offsource_area":cl_offsource_area}    
    
    def plt_cleanimage(self, name, outputunit = "jybeam-1", onsource_radius = 1.0, offsource_radius = 1.0, scale = "linear", gamma =1.0,
                    orientation = "vertical", ticklocation = 'right', vmin = 0, vmax = 0,  object_distance = 0, pa =0, inclination = 0,
                    mapcolor = "inferno",contlevels=[5,10,15], Dodeproject = False, nameon = False, sigmalevel = 5, tick_list =  np.arange(-100,100,0.5),
                    range =1.0, cont_colors="white", contour = True, imageon = True, istokes=0, ifreq=0):
        """
        Note that input unit must be Jy beam-1.
        Parameters
        ----------
        fitsfile: str
            fitsfile
        name: str
            object name
        outtputunit: str
            output unit : select "jybeam-1" or "jyasec-2"
        offsource_radius: float
            position matching in arseconds.
        """
        # preparation #########################################
        cellsize   = (np.pi/180.0)**2*np.abs(self.header["dx"])*np.abs(self.header["dx"]) # radian
        beam_unit  = (np.pi/(4.0*np.log(2.0)))*((np.pi/180.0)**2.0)*self.header["BMAJ"]*self.header["BMIN"] # sr/beam
        imdx = self.header["dx"]*self.angconv("deg","asec") # unit: arcsec
        imdy = self.header["dy"]*self.angconv("deg","asec") # unit: arcsec


        if object_distance > 0.:
            obj_distance = object_distance * 0.1*2.06265*1e5*(np.pi/(3600*180)) # au @ 0.1 arcsec
            obj_distance = np.round(obj_distance,1)
        else :
            object_distance      = object_propaty(name)["distance"] # pc
            obj_distance = object_distance * 0.1*2.06265*1e5*(np.pi/(3600*180)) # au @ 0.1 arcsec
            obj_distance = np.round(obj_distance,1)

        # calculate peak value [Jy beam-1]
        peak = self.onsource_area(radius= onsource_radius, istokes=0, ifreq=0).max()
        # calculate total flux above 5 sigma [Jy beam-1]
        cl_offsource_area = self.offsource_area(radius= offsource_radius,
                                    inputunit = "jybeam-1", outputunit = "jybeam-1",
                                    istokes= istokes, ifreq = ifreq)
        rmsnoise = cl_offsource_area.std()
        threshold = np.where(self.data[0,0] > sigmalevel*rmsnoise, self.data[0,0], 0) # (case, True, False)
        climage_threshold = copy.deepcopy(self)
        climage_threshold.data[0,0] = threshold
        climage_threshold_jypixel = climage_threshold.jybeam2jypixel()
        totalflux  = climage_threshold_jypixel.onsource_area(radius= onsource_radius, istokes=istokes, ifreq=ifreq).sum()

        # calculate peak value [Jy asec-2]
        peak_jyasec2 = peak*(cellsize/beam_unit)/(np.abs(imdx*imdy))
        # calculate rms value [Jy asec-2]
        rms_jyasec2 = rmsnoise*(cellsize/beam_unit)/(np.abs(imdx*imdy))


        print ("## CLEAN Image  ###################")
        print ("BEAM MAJ   = %1.3f [arcsec]"%(self.header["BMAJ"]*3600))
        print ("BEAM MIN   = %1.3f [arcsec]"%(self.header["BMIN"]*3600))
        print ("BEAM PA    = %1.3f [deg]"%(self.header["BPA"]))
        print ("Peak       = %1.3f [mJy/asec2]"%(peak_jyasec2*1e3))
        print ("RMS noise  = %1.3f [mJy/asec2]"%(rms_jyasec2*1e3))
        print ("Peak       = %1.3f [mJy/beam]"%(peak*1e3))
        print ("RMS noise  = %1.3f [mJy/beam]"%(rmsnoise*1e3))
        print ("SNR        = %1.3f"%(peak/rmsnoise))
        print("Total Flux = %0.1f mJy"%(totalflux*1e3))
        print ("###################################")

        # set a colorbar
        my_cmap = copy.copy(cm.get_cmap(mapcolor)) # copy the default cmap
        my_cmap.set_bad((0,0,0)) # <-  important !
        # plot  images
        matplotlibrc(nrows=1,ncols=1,width=330,height=300) # set size of figure
        fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False) # gerenete figure and panels
        plt.subplots_adjust(wspace=0.4)

        # setup of unit ######################
        if outputunit == "jybeam-1":
            image = self
            rms_vaule = rmsnoise

        elif outputunit == "mjybeam-1":
            image = self
            image.data[0,0] = self.data[0,0]*1e3 # jy -> mjy
            rms_vaule = rmsnoise*1e3


        elif outputunit == "jyasec-2":
            image = self.jybeam2jyasec(istokes= istokes, ifreq = ifreq) # Jy/asec2
            rms_vaule = rms_jyasec2

        elif outputunit == "mjyasec-2":
            image = self.jybeam2jyasec(istokes= istokes, ifreq = ifreq) # Jy/asec2
            image.data[0,0] = image.data[0,0]*1e3 # jy -> mjy/asec2
            rms_vaule = rms_jyasec2*1e3

        ########################################

        if Dodeproject ==True:
            image = image.deproject(angle = pa, inclination = inclination, save_totalflux = True)

        ## input
        if vmax != 0:
            vmax = vmax
        else:
            im_revised = copy.deepcopy(image)
            im_revised.data[istokes,ifreq] = np.nan_to_num(image.data[istokes,ifreq])
            vmax = im_revised.peak()

        if vmin != 0:
            vmin = vmin
        else:
            vmin = rms_vaule
        levels = np.array(contlevels)
        plt.sca(ax) # indicate "set current axis
        tickcolor = "w"
        labelcolor = "k"
        plt.tick_params(which='both',direction='in',bottom =True,top = True,left = True,right =True,
                        colors=tickcolor,labelrotation=0, labelcolor=labelcolor)
        #Input
        if contour == True:
            imagecont = image.contour(angunit='arcsec',
                                        logscale=True,
                                        cmul=vmin,
                                        levels=levels,
                                        colors=cont_colors,
                                        relative=False,
                                        linewidths = 0.8)
        if imageon == True:
            if scale     == "log":
                climage = image.imshow(angunit='arcsec', interpolation='nearest',vmin = vmin, vmax =vmax, norm = LogNorm(vmin = vmin, vmax =vmax),cmap=my_cmap, tick_list = tick_list)
            elif  scale == "linear":
                climage = image.imshow(angunit='arcsec',vmin = vmin, vmax =vmax,cmap=my_cmap, tick_list = tick_list)
            elif  scale == "gamma":
                climage = image.imshow(angunit='arcsec',vmin = vmin, vmax =vmax, norm = PowerNorm(gamma), cmap=my_cmap, tick_list = tick_list)

        plt.xlim(range,-range)
        plt.ylim(-range,range)


        plt.text(range*(-0.1), range*(-0.9), '0".1  (= '+str(obj_distance)+' au)', ha = 'center', va = 'bottom',color ="w", fontsize=12)
        plt.plot([-0.05,0.05],[range*(-0.7),range*(-0.7)],color="w",lw=2)
        if nameon == True:
            plt.text(0,range*(0.75),name, ha = 'center', va = 'bottom',color ="w", fontsize=18)

        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

        if imageon == True:
            divider = make_axes_locatable(ax)
            ax_cb = divider.append_axes(ticklocation,size="7%", pad=0.05)

            if outputunit == "jybeam-1":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation,
                                label= r"Intensity (Jy beam$^{-1}$)")
            elif outputunit == "mjybeam-1":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation,
                                label= r"Intensity (mJy beam$^{-1}$)")
            elif outputunit == "jyasec-2":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation,
                                label= r"Intensity (Jy arcsec$^{-2}$)")
                cb.ax.locator_params(nbins=5)

            elif outputunit == "ratio":
                cb =plt.colorbar(climage, cax=ax_cb,orientation=orientation, ticklocation = ticklocation,
                                label= name+r' $(10^{-10})$')            

            elif outputunit == "mjyasec-2":
                cb =plt.colorbar(climage,cax=ax_cb, orientation=orientation, ticklocation = ticklocation,
                                label= r"Intensity (mJy arcsec$^{-2}$)")

        return{"offsource_area":cl_offsource_area}

    def disk_radius(self, pa = 0, inclination = 0,  threshold_int = 0, disk_radius = 0.5, approxi_num = 3,res_factor = 4, check_grad = False,
                    percentage = 0.95, totalflux = 0, ploton = True, objectname = None, object_distance = 0, expected_radius = 0):
        """
        Note that input image must be a SpM image [Jy pixel-1].
        parameters:
        threshold_int: detection threshold intensity [Jy pixel-1]
        disk_radius: arcsec
        percentage: percentage of total flux [float]
        """

        image = self.deproject(angle = pa, inclination = inclination, save_totalflux=True)
        fig = plt.figure(figsize=(10.0, 10.0))
        ax = fig.add_subplot(111)
        ax1 = ax
        plt.sca(ax1) # stands for "set current axis"
        plt.gca().yaxis.set_tick_params(which='both', direction='in',left=True, right = True)
        plt.gca().xaxis.set_tick_params(which='both', direction='in')

        flux_list =  np.array([])
        fluxerror_list =  np.array([])

        radius = np.arange(image.header["dy"]*3600, disk_radius, image.header["dy"]*3600/20)

        for r in radius:
            flux_list = np.append(flux_list, image.totalflux(radius = r, unit='jypixel-1'))
            fluxerror_list = np.append(fluxerror_list, len(image.onsource_area(radius = r))*threshold_int)


        if totalflux > 0:
            print("# case 1: The total continuum flux is taken to be the user specified value.")
            thresfold = None
            while thresfold == None:
                print("approxi_num is set to be :",approxi_num)
                thresfold = np.where(np.round(flux_list,approxi_num)==np.round(totalflux*percentage,approxi_num))[0][0]
                approxi_num += -1
            print("approxi_num reached :",approxi_num)
            thresfold_radius = radius[thresfold] # arcsec

        else:
            print("# case 2: The total continuum flux is taken to be the asymptotic value of the flux curve.")
            print("# Resolution factor is set to be %0.1f"%(res_factor))

            f = interp1d(radius, flux_list, kind = 'cubic')
            u = np.arange(radius[0], radius[-1], np.diff(radius).mean()/res_factor)

            round_num = 2
            grad = np.array([])
            zero_grad_radius = np.array([])
            #while len(zero_grad_radius) == 0:
            while len(grad) == 0:
                #print("searcing gradient zero at gradient number : ",round_num)
                grad = np.where(np.round(np.gradient(f(u),u), round_num) == 0)[0]
                print("## round number = %d"%(round_num))
                round_num -= 1

            zero_grad_radius_candiate = np.array(u[grad])
            #print("zero gradient radius candiate [arcsec] ->", zero_grad_radius_candiate)
            zero_grad_radius = [x for x in zero_grad_radius_candiate if x > expected_radius]

            totalflux = f([zero_grad_radius[0]]) # Jy
            print("Total flux is found to be %0.2f mJy"%(totalflux*1e3))
            print("Zero gradient radius is found to be %0.2f arcsec ", zero_grad_radius[0])

            thresfold = None
            while thresfold == None:
                print("approxi_num is set to be :",approxi_num)
                thresfold = np.where(np.round(flux_list,approxi_num)==np.round(totalflux*percentage, approxi_num))[0][0]
                approxi_num += -1
            #print("approxi_num reached :",approxi_num)
            thresfold_radius = radius[thresfold] # arcsec
            print("Dust disk radius = %0.3f [arcsec]"%(thresfold_radius))

        flux_error = threshold_int*len(image.onsource_area(radius= thresfold_radius)) # Jy
        thresfold_error = np.where(np.round(flux_list,approxi_num)==np.round((totalflux-flux_error)*percentage, approxi_num))[0][0]
        thresfold_radius_error = radius[thresfold_error] # arcsec

        diskradius_error = thresfold_radius-thresfold_radius_error # arsec

        # distance unit
        if object_distance > 1.0:
            distance = object_distance # pc
            distance_au = distance*2.062*1e5*(np.pi/(3600*180))  # au/asec
        else:
            object_prop = object_propaty(objectname)
            distance = object_prop["distance"] #pc
            distance_au = distance*2.062*1e5*(np.pi/(3600*180))  # au/asec

        print("Dust disk radius = %0.3f±%0.3f [arcsec]"%(thresfold_radius,diskradius_error))
        print("Dust disk radius = %0.3f±%0.3f [au]"%(thresfold_radius*distance_au,diskradius_error*distance_au))

        if ploton == True:
            plt.scatter(radius, flux_list, s= 5, c ="#280a69", alpha= 0.5)
            plt.axhline(y= totalflux, xmin=0.0, xmax=np.abs(image.header["nx"])*0.5, color = "#808080", linestyle = "--", linewidth = 1.0)
            plt.axvline(ymin= 0, ymax=10, x = thresfold_radius, color = "#808080", linestyle = "--", linewidth = 1.0)
            plt.xlim(0, disk_radius)
            plt.ylim(0, totalflux*1.5)
            plt.ylabel("$Flux (Jy)$")
            plt.xlabel(r"Radius [arcsec]")

        return{"radius":thresfold_radius, "radius_error":diskradius_error, "flux_list":flux_list,"radius": radius}

    def hard_threshold(self, threshold=0.01, relative=True, save_totalflux=False,
                       istokes=0, ifreq=0):
        '''
        Do hard-threshold the input image

        Args:
          istokes (integer): index for Stokes Parameter at which the image will be edited
          ifreq (integer): index for Frequency at which the image will be edited
          threshold (float): threshold
          relative (boolean): If true, theshold value will be normalized with the peak intensity of the image
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        if relative:
            thres = np.abs(threshold * self.peak(istokes=istokes, ifreq=ifreq))
        else:
            thres = np.abs(threshold)
        # thresholding
        image = outfits.data[istokes, ifreq]
        t = np.where(np.abs(self.data[istokes, ifreq]) < thres)
        image[t] = 0
        outfits.data[istokes, ifreq] = image
        # flux scaling
        if save_totalflux:
            totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
            outfits.data[istokes, ifreq] *= totalflux / \
                outfits.totalflux(istokes=istokes, ifreq=ifreq)
        outfits.update_fits()
        return outfits

    def soft_threshold(self, threshold=0.01, relative=True, save_totalflux=False,
                       istokes=0, ifreq=0):
        '''
        Do soft-threshold the input image

        Args:
          istokes (integer): index for Stokes Parameter at which the image will be edited
          ifreq (integer): index for Frequency at which the image will be edited
          threshold (float): threshold
          relative (boolean): If true, theshold value will be normalized with the peak intensity of the image
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        if relative:
            thres = np.abs(threshold * self.peak(istokes=istokes, ifreq=ifreq))
        else:
            thres = np.abs(threshold)
        # thresholding
        image = outfits.data[istokes, ifreq]
        t = np.where(np.abs(self.data[istokes, ifreq]) < thres)
        image[t] = 0
        t = np.where(self.data[istokes, ifreq] >= thres)
        image[t] -= thres
        t = np.where(self.data[istokes, ifreq] <= -thres)
        image[t] += thres
        outfits.data[istokes, ifreq] = image
        if save_totalflux:
            totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
            outfits.data[istokes, ifreq] *= totalflux / \
                outfits.totalflux(istokes=istokes, ifreq=ifreq)
        outfits.update_fits()
        return outfits


    def from_geomodel(self, geomodel, istokes=0, ifreq=0, overwrite=True, usetheano=False):
        '''
        Args:
            modelfunc (func, must be lambda x,y):
                a function to compute brightness at x, y (in rad).
                The unit of brightness must be Jy/rad^2
        '''
        # copy self (for output)
        outfits = copy.deepcopy(self)

        # compile funcitons
        x, y = sp.symbols("x y", real=True)
        expr = geomodel.I(x, y).simplify()
        if usetheano:
            func = theano_function([x,y], [expr], dims={x:2, y:2}, dtypes={x: 'float64', y: 'float64'})
        else:
            func = sp.lambdify([x, y], expr, "numpy")

        # change headers
        dx=np.abs(self.header["dx"])*util.angconv("deg", "rad")
        dy=np.abs(self.header["dy"])*util.angconv("deg", "rad")
        nx=self.header["nx"]
        ny=self.header["ny"]
        x,y = self.get_xygrid(twodim=True, angunit="rad")
        I = func(x,y)*dx*dy

        if overwrite:
            outfits.data[istokes,ifreq,:,:] = I
        else:
            outfits.data[istokes,ifreq,:,:] += I
        return outfits


    def add_gauss(self, x0=0., y0=0., totalflux=1., majsize=1., minsize=None,
                  pa=0., istokes=0, ifreq=0, angunit=None):
        if angunit is None:
            angunit = self.angunit

        # copy self (for output)
        outfits = copy.deepcopy(self)

        # get size
        thmaj = majsize
        if minsize is None:
            thmin = thmaj
        else:
            thmin = minsize

        # Calc X,Y grid
        #yg = (np.arange(self.header["ny"]) - self.header["nyref"] +
        #      1) * self.header["dy"] * 3600e3  # (mas)
        #xg = (np.arange(self.header["nx"]) - self.header["nxref"] +
        #      1) * self.header["dx"] * 3600e3  # (mas)
        #X, Y = np.meshgrid(xg, yg)
        X, Y = self.get_xygrid(twodim=True, angunit=angunit)

        # Calc Gaussian Distribution
        X1 = X - x0
        Y1 = Y - y0
        cospa = np.cos(np.deg2rad(pa))
        sinpa = np.sin(np.deg2rad(pa))
        X2 = X1 * cospa - Y1 * sinpa
        Y2 = X1 * sinpa + Y1 * cospa
        majsig = thmaj / np.sqrt(2 * np.log(2)) / 2
        minsig = thmin / np.sqrt(2 * np.log(2)) / 2
        gauss = np.exp(-X2 * X2 / 2 / minsig / minsig -
                       Y2 * Y2 / 2 / majsig / majsig)
        gauss /= gauss.sum()
        gauss *= totalflux

        # add to original FITS file
        outfits.data[istokes, ifreq] += gauss

        return outfits

    def edge_detect(self, method="prewitt", mask=None, sigma=1,
                    low_threshold=0.1, high_threshold=0.2):
        '''
        Output edge-highlighted images.

        Args:
          method (string, default="prewitt"):
            Type of edge filters to be used.
            Availables are ["prewitt","sobel","scharr","roberts","canny"].
          mask (array):
            array for masking
          sigma (integer):
            index for canny
          low_threshold (float):
            index for canny
          high_threshold (float):
            index for canny

        Returns:
          imdata.IMFITS object
        '''
        from skimage.filters import prewitt, sobel, scharr, roberts
        from skimage.feature import canny

        # copy self (for output)
        outfits = copy.deepcopy(self)

        # get information
        nstokes = outfits.header["ns"]
        nif = outfits.header["nf"]
        # detect edge
        # prewitt
        if method == "prewitt":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = prewitt(
                            outfits.data[idxs, idxf])
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = prewitt(
                            outfits.data[idxs, idxf], mask=mask)
        # sobel
        if method == "sobel":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = sobel(
                            outfits.data[idxs, idxf])
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = sobel(
                            outfits.data[idxs, idxf], mask=mask)
        # scharr
        if method == "scharr":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = scharr(
                            outfits.data[idxs, idxf])
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = scharr(
                            outfits.data[idxs, idxf], mask=mask)
        # roberts
        if method == "roberts":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = roberts(
                            outfits.data[idxs, idxf])
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = roberts(
                            outfits.data[idxs, idxf], mask=mask)
        # canny
        if method == "canny":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = canny(
                            outfits.data[idxs, idxf], sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold, use_quantiles=True)
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = canny(outfits.data[idxs, idxf], mask=mask, sigma=sigma,
                                                         low_threshold=low_threshold, high_threshold=high_threshold, use_quantiles=True)

        outfits.update_fits()
        return outfits

    def circle_hough(self, radius, ntheta=360,
                     angunit=None, istokes=0, ifreq=0):
        '''
        A function calculates the circle Hough transform (CHT) of the input image

        Args:
          radius (array):
            array for radii for which the circle Hough transform is
            calculated. The unit of the radius is specified with angunit.
          Ntheta (optional, integer):
            The number of circular shifts to be used in the circle Hough transform.
            For instance, ntheta=360 (default) gives circular shifts of every 1 deg.
          angunit (optional, string):
            The angular unit for radius and also the output peak profile
          istokes (integer): index for Stokes Parameter at which the CHT to be performed
          ifreq (integer): index for Frequency at which the CHT to be performed

        Returns:
          H (ndarray):
            The Circle Hough Accumulator. This is a three dimensional array of which
            shape is [Nx, Ny, Nr] in *Fortran Order*.
          profile (pd.DataFrame):
            The table for the peak profile Hr(r)=max_r(H(x,y,r)).
        '''
        if angunit is None:
            angunit = self.angunit

        Nr = len(radius)
        Nx = self.header["nx"]
        Ny = self.header["ny"]

        # get xy-coordinates
        xgrid, ygrid = self.get_xygrid(angunit=angunit)
        if self.header["dx"] < 0:
            sgnx = -1
        else:
            sgnx = 1
        if self.header["dy"] < 0:
            sgny = -1
        else:
            sgny = 1

        # calculate circle hough transform
        H = fortlib.houghlib.circle_hough(self.data[istokes, ifreq],
                                           sgnx * xgrid, sgny * ygrid,
                                           radius, np.int32(ntheta))
        isfort = np.isfortran(H)

        # make peak profile
        profile = pd.DataFrame()
        profile["ir"] = np.arange(Nr)
        profile["r"] = radius
        profile["xpeak"] = np.zeros(Nr)
        profile["ypeak"] = np.zeros(Nr)
        profile["ixpeak"] = np.zeros(Nr, dtype=np.int64)
        profile["iypeak"] = np.zeros(Nr, dtype=np.int64)
        profile["hpeak"] = np.zeros(Nr)
        if isfort:
            for i in np.arange(Nr):
                profile.loc[i, "hpeak"] = np.max(H[:, :, i])
                peakxyidx = np.unravel_index(
                    np.argmax(H[:, :, i]), dims=[Ny, Nx])
                profile.loc[i, "xpeak"] = xgrid[peakxyidx[1]]
                profile.loc[i, "ypeak"] = ygrid[peakxyidx[0]]
                profile.loc[i, "ixpeak"] = peakxyidx[1]
                profile.loc[i, "iypeak"] = peakxyidx[0]
        else:
            for i in np.arange(Nr):
                profile.loc[i, "hpeak"] = np.max(H[i, :, :])
                peakxyidx = np.unravel_index(
                    np.argmax(H[i, :, :]), dims=[Ny, Nx])
                profile.loc[i, "xpeak"] = xgrid[peakxyidx[1]]
                profile.loc[i, "ypeak"] = ygrid[peakxyidx[0]]
                profile.loc[i, "ixpeak"] = peakxyidx[1]
                profile.loc[i, "iypeak"] = peakxyidx[0]
        return H, profile


#-------------------------------------------------------------------------
# Calculate Matrix Among Images
#-------------------------------------------------------------------------
def calc_metric(fitsdata, reffitsdata, metric="NRMSE", istokes1=0, ifreq1=0, istokes2=0, ifreq2=0, edgeflag=False):
    '''
    Calculate metrics between two images

    Args:
      fitsdata (imdata.IMFITS object):
        input image

      reffitsdata (imdata.IMFITS object):
        reference image

      metric (string):
        type of a metric to be calculated.
        Availables are ["NRMSE","MSE","SSIM","DSSIM"]

      istokes1 (integer):
        index for the Stokes axis of the input image

      ifreq1 (integer):
        index for the frequency axis of the input image

      istokes2 (integer):
        index for the Stokes axis of the reference image

      ifreq2 (integer):
        index for the frequency axis of the reference image

      edgeflag (boolean):
        calculation of metric on image domain or image gradient domain

    Returns:
      ???
    '''
    from skimage.filters import prewitt

    # adjust resolution and FOV
    fitsdata2 = copy.deepcopy(fitsdata)
    reffitsdata2 = copy.deepcopy(reffitsdata)
    fitsdata2 = reffitsdata2.cpimage(fitsdata2)
    # edge detection
    if edgeflag:
        fitsdata2 = fitsdata2.edge_detect(method="sobel")
        reffitsdata2 = reffitsdata2.edge_detect(method="sobel")
    # get image data
    inpimage = fitsdata2.data[istokes1, ifreq1]
    refimage = reffitsdata2.data[istokes2, ifreq2]
    # calculate metric
    if metric == "NRMSE" or metric == "MSE":
        metrics = np.sum((inpimage - refimage)**2)
        metrics /= np.sum(refimage**2)
    if metric == "SSIM" or metric == "DSSIM":
        meanI = np.mean(inpimage)
        meanK = np.mean(refimage)
        stdI = np.std(inpimage, ddof=1)
        stdK = np.std(refimage, ddof=1)
        cov = np.sum((inpimage - meanI) * (refimage - meanK)) / \
            (inpimage.size - 1)
        metrics = (2 * meanI * meanK / (meanI**2 + meanK**2)) * \
            (2 * stdI * stdK / (stdI**2 + stdK**2)) * (cov / (stdI * stdK))
    if metric == "NRMSE":
        metrics = np.sqrt(metrics)
    if metric == "DSSIM":
        metrics = 1 / abs(metrics) - 1

    return metrics


#-------------------------------------------------------------------------
# Fllowings are subfunctions for ds9flag and read_cleanbox
#-------------------------------------------------------------------------
def get_flagpixels(regfile, X, Y):
    # Read DS9-region file
    f = open(regfile)
    lines = f.readlines()
    f.close()
    keep = np.zeros(X.shape, dtype="Bool")
    # Read each line
    for line in lines:
        # Skipping line
        if line[0] == "#":
            continue
        if "image" in line == True:
            continue
        if "(" in line == False:
            continue
        if "global" in line:
            continue
        # Replacing many characters to empty spaces
        line = line.replace("(", " ")
        line = line.replace(")", " ")
        while "," in line:
            line = line.replace(",", " ")
        # split line to elements
        elements = line.split(" ")
        while "" in elements:
            elements.remove("")
        while "\n" in elements:
            elements.remove("\n")
        if len(elements) < 4:
            continue
        # Check whether the box is for "inclusion" or "exclusion"
        if elements[0][0] == "-":
            elements[0] = elements[0][1:]
            exclusion = True
        else:
            exclusion = False
        if elements[0] == "box":
            tmpkeep = region_box(X, Y,
                                  x0=np.float64(elements[1]),
                                  y0=np.float64(elements[2]),
                                  width=np.float64(elements[3]),
                                  height=np.float64(elements[4]),
                                  angle=np.float64(elements[5]))
        elif elements[0] == "circle":
            tmpkeep = region_circle(X, Y,
                                     x0=np.float64(elements[1]),
                                     y0=np.float64(elements[2]),
                                     radius=np.float64(elements[3]))
        elif elements[0] == "ellipse":
            tmpkeep = region_ellipse(X, Y,
                                      x0=np.float64(elements[1]),
                                      y0=np.float64(elements[2]),
                                      radius1=np.float64(elements[3]),
                                      radius2=np.float64(elements[4]),
                                      angle=np.float64(elements[5]))
        else:
            print("[WARNING] The shape %s is not available." % (elements[0]))
        if not exclusion:
            keep += tmpkeep
        else:
            keep[np.where(tmpkeep)] = False
    return keep


def region_circle(X, Y, x0, y0, radius):
    return (X - x0) * (X - x0) + (Y - y0) * (Y - y0) <= radius * radius


def region_ellipse(X, Y, x0, y0, radius1, radius2, angle):
    cosa = np.cos(np.deg2rad(angle))
    sina = np.sin(np.deg2rad(angle))
    dX = X - x0
    dY = Y - y0
    X1 = dX * cosa + dY * sina
    Y1 = -dX * sina + dY * cosa
    return X1 * X1 / radius1 / radius1 + Y1 * Y1 / radius2 / radius2 <= 1

def gaia_query(ra, dec, radius=5 * unit.arcsec):
    '''Return an ADQL query string for Gaia DR2 + geometric distances
        from Bailer-Jones et al. 2018 '''
    query_string = '''SELECT *, DISTANCE(POINT('ICRS',g.ra, g.dec),
                    POINT('ICRS', %s, %s)) as r
                    FROM gaiadr2.gaia_source AS g, external.gaiadr2_geometric_distance AS d
                    WHERE g.source_id = d.source_id AND CONTAINS(POINT('ICRS',g.ra, g.dec),
                    CIRCLE('ICRS',%15.10f, %15.10f,%15.10f))=1 ORDER BY r ASC''' % \
        (ra, dec, ra, dec, radius.to(unit.degree).value)
    return(query_string)

def object_distance(name):
    """ Query VizieR Photometry
    The VizieR photometry tool extracts photometry points around a given position
    or object name from photometry-enabled catalogs in VizieR.

    The VizieR photometry tool is developed by Anne-Camille Simon and Thomas Boch
    .. url:: http://vizier.u-strasbg.fr/vizier/sed/doc/

    Parameters
    ----------
    name: str
        object name
    radius: float
        position matching in arseconds.
    """

    # Physical parameters #################
    clight = 2.99792458e10 # velosity of light  [cm s-1]
    pcTOcm = 3.09e18       # pc --> cm

    # Query of  Gaia   ###################
    coords = coord.SkyCoord.from_name(name)
    print ("Object Name is  ", name)
    print ("(RA, Dec) = " , (coords.ra.to_string(unit.hour), coords.dec.to_string(unit.deg)))

    query_string = gaia_query(coords.ra.to_value(), coords.dec.to_value(), radius = 1.0*unit.arcsec)
    job = Gaia.launch_job(query_string, verbose=False)
    a = job.get_results()
    if len(a)==0:
        print("No results for star %s." % name)

    elif len(a) ==1:
        object_distance         = 1000/a['parallax'][0]     #  object distance [pc]
        obj_distance_au = objprop["distance"] * 0.1*2.06265*1e5*(np.pi/(3600*180)) # au @ 0.1 arcsec
        object_distance_cm = object_distance*pcTOcm #  object distance [cm]
        pmra    = a['pmra']
        pmdec = a['pmdec']
        print("\nThe distance to %s is %0.2f pc. The proper motion is %0.1f, %0.1f mas/yr." \
          % (name, object_distance, pmra, pmdec))

    elif len(a) >=2:
        object_distance         = 1000/a[0]['parallax']     #  object distance [pc]
        object_distance_cm = object_distance*pcTOcm #  object distance [cm]
        pmra    = a[0]['pmra']
        pmdec = a[0]['pmdec']
        print("\nThe distance to %s is %0.2f pc. The proper motion is %0.1f, %0.1f mas/yr." \
          % (object_name, object_distance, pmra, pmdec))
        print ("Note that GAIA has two data for determing the object's distance.")

    return {"distance": object_distance, "distance_au_0.1arcsec":obj_distance_au, "RA": coords.ra.to_string(unit.hour), "Dec": coords.dec.to_string(unit.deg)}


def object_propaty(name):
    """ Query VizieR Photometry
    The VizieR photometry tool extracts photometry points around a given position
    or object name from photometry-enabled catalogs in VizieR.

    The VizieR photometry tool is developed by Anne-Camille Simon and Thomas Boch
    .. url:: http://vizier.u-strasbg.fr/vizier/sed/doc/

    Parameters
    ----------
    name: str
        object name
    radius: float
        position matching in arseconds.
    """

    # Physical parameters ########################
    clight = 2.99792458e10        # velosity of light  [cm s-1]

    # Query of  Gaia   ###################
    coords = coord.SkyCoord.from_name(name)
    print("Object Name is  ", name)
    print("(RA, Dec) = " , (coords.ra.to_string(unit.hour), coords.dec.to_string(unit.deg)))

    query_string = gaia_query(coords.ra.to_value(), coords.dec.to_value(), radius = 1.0*unit.arcsec)
    job = Gaia.launch_job(query_string, verbose=False)
    a = job.get_results()
    if len(a)==0:
        print("No results for star %s." % name)

    elif len(a) ==1:
        object_distance         = 1000/a['parallax'][0]     #  object distance [pc]
        object_distance_cm = object_distance*pcTOcm #  object distance [cm]
        pmra    = a['pmra']
        pmdec = a['pmdec']
        print("\nThe distance to %s is %0.2f pc. The proper motion is %0.1f, %0.1f mas/yr." \
          % (name, object_distance, pmra, pmdec))

    elif len(a) >=2:
        object_distance         = 1000/a[0]['parallax']     #  object distance [pc]
        object_distance_cm = object_distance*pcTOcm #  object distance [cm]
        pmra    = a[0]['pmra']
        pmdec = a[0]['pmdec']
        print("\nThe distance to %s is %0.2f pc. The proper motion is %0.1f, %0.1f mas/yr." \
          % (object_name, object_distance, pmra, pmdec))
        print ("Note that GAIA has two data for determing the object's distance.")

    return {"distance": object_distance, "RA": coords.ra.to_string(unit.hour), "Dec": coords.dec.to_string(unit.deg)}

def matplotlibrc(nrows=1,ncols=1,width=250,height=250):
    letterratio = 1.294
    # Get this from LaTeX using \showthe\columnwidth
    fig_width_pt  = width*ncols
    fig_height_pt = height*nrows
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width     = fig_width_pt*inches_per_pt  # width in inches
    fig_height    = fig_height_pt*inches_per_pt # height in inches
    fig_size      = [fig_width,fig_height]
    params = {'axes.labelsize': 14,
              'axes.titlesize': 15,
              'font.size' : 15,
              'legend.fontsize': 14,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'xtick.top'           : True,   # draw ticks on the top side
              'xtick.major.top' : True,
              'figure.figsize': fig_size,
              'figure.dpi'    : 600,
              'font.family': 'Times New Roman',
              "mathtext.fontset" : 'stix', #"Times New Roman"
              'mathtext.tt':'Times New Roman',
              'axes.linewidth' :2.5,
              'xtick.major.width' : 1.0,
              'ytick.major.width' : 1.0,
              'xtick.minor.width' : 1.0,
              'ytick.minor.width' : 1.0,
               'xtick.major.size' : 6,
               'ytick.major.size' : 6,
                'xtick.minor.size' : 4.,
                'ytick.minor.size' : 4.,

    }
    rcParams.update(params)


def calc_jybeam2jyasec(intensity = 0., cellsize_asec = 0., BMAJ = 0., BMIN = 0.):
    """
    #  intensity [Jy/beam] -> [Jy/asec**2]
    cellsize_asec : arcsec
    BMAJ : arcsec
    BMIN : arcsec
    """
    cellsize              = ((cellsize_asec*np.pi)/(180.0*3600.0))**2 # radian
    #beam_unit             = (np.pi/(4.0*np.log(2.0)))*((np.pi/(180.0*3600.0))**2.0)*BMAJ*BMIN # sr/beam
    beam_unit             = (np.pi/(4.0*np.log(2.0)))*((np.pi/(180.0*3600.0))**2.0)*BMAJ*BMIN # sr/beam
    intensity_jypixel = intensity*((cellsize)/beam_unit) # Jy/pixel
    # step-1:we convert deg/pixel into asec/pixel
    # step-2:we devide Jy/pixel by arcsec^2/pixel, we get Jy/arcsec^2
    intensity_jyasec = intensity_jypixel/(cellsize_asec**2)
    return intensity_jyasec

def calc_jypixel2jybeam(intensity = 0., cellsize_asec = 0., BMAJ = 0., BMIN = 0.):
    """
    #  intensity [Jy/pixel] -> [Jy/beam]
    cellsize_asec : arcsec
    BMAJ : arcsec
    BMIN : arcsec
    """
    cellsize          = ((cellsize_asec*np.pi)/(180.0*3600.0))**2.0 # radian
    beam_unit    = (np.pi/(4.0*np.log(2.0)))*((np.pi/(180.0*3600.0))**2.0)*BMAJ*BMIN # sr/beam
    # step-2:we devide Jy/pixel by beam/pixel, we get Jy/beam
    intensity_jybeam = intensity/(beam_unit/cellsize)
    return intensity_jybeam


def change_pixelscale(fitsfile, fitsname, pixelscale = 0.1):
    """
    Caution : It could cause a offset of ~ 0.0025 asec on the resultant image domain.
    fitsfile (str): fitsfile
    fitname (str): "~~.fits"
    pixelscale (float): pixelscale * cell size on the image domain.
    """
    image = IMFITS(fitsfile)
    ref = IMFITS(dx=pixelscale*image.header["dy"]*3600.,nx= int(np.round(image.header["ny"]/pixelscale,0)), angunit="asec")
    im = ref.cpimage(image, save_totalflux=True)
    im.save_fits(fitsname, overwrite =True)
    # arange the header of the model image
    hdulist_ref                 = pyfits.open(fitsfile)
    hdulist                     = pyfits.open(fitsname)
    hdulist[0].header           = hdulist_ref[0].header
    hdulist[0].header["CDELT1"] = pixelscale*image.header["dx"]
    hdulist[0].header["CDELT2"] = pixelscale*image.header["dy"]
    # CRVAL1 and CRVAL2 give the center coordinate as right ascension and declination or longitude and latitude in decimal degrees.
    # CRVAL1 : X-axis reference pixel value in coordinates
    # CRVAL2 : Y-axis reference pixel value in coordinates
    hdulist[0].header["CRPIX1"] = image.header["nxref"]/pixelscale
    hdulist[0].header["CRPIX2"] = image.header["nyref"]/pixelscale
    hdulist[0].header["NAXIS1"] = image.header["nx"]/pixelscale
    hdulist[0].header["NAXIS2"] = image.header["ny"]/pixelscale
    hdulist.writeto(fitsname, overwrite=True)
    final_image = IMFITS(fitsname)
    return(final_image)


def export_ms(msfile, outfile='uv.npy'):
    '''Export an ms file to a numpy save file.

    Direct copy of Luca's export.

    Everything is exported, so generally the ms file would already be
    averaged to a small number of channels, e.g. one per spw.

    For this to run smoothly we need to have the same number of channels
    for ALL scans. So no spectral windows with different number of
    channels, otherwise it gets complicated. See:
    https://safe.nrao.edu/wiki/pub/Main/RadioTutorial/BandwidthSmearing.pdf
    to choose how much to smooth a dataset in frequency.

    Errors are taken into account when time averaging in split:
    https://casa.nrao.edu/casadocs/casa-5.1.1/uv-manipulation/time-average

    And when channel averaging:
    https://casa.nrao.edu/casadocs/casa-5.1.1/uv-manipulation/channel-average

    Parameters
    ----------
    outfile : str
        File with model visibilities.
    ms : ms object
        Pass casac.ms()
    '''

    cc=2.9979e10 #cm/s

    # Use CASA table tools to get columns of UVW, DATA, WEIGHT, etc.
    outputfilename=outfile
    msfilename=msfile
    tb.open(msfilename)

    data    = tb.getcol("DATA")
    uvw     = tb.getcol("UVW")
    weight  = tb.getcol("WEIGHT")
    ant1    = tb.getcol("ANTENNA1")
    ant2    = tb.getcol("ANTENNA2")
    flags   = tb.getcol("FLAG")
    spwid   = tb.getcol("DATA_DESC_ID")
    tb.close()
    if np.any(flags):
        print("Note: some of the data is FLAGGED")
    print("Found data with "+str(data.shape[-1])+" uv points")


    #Use CASA ms tools to get the channel/spw info
    #ms.open(msfilename)
    #spw_info = ms.getspectralwindowinfo()
    #nchan = spw_info["0"]["NumChan"]
    #npol = spw_info["0"]["NumCorr"]
    #ms.close()
    #print("with "+str(nchan)+" channels per SPW and "+str(npol)+" polarizations,")

    # Use CASA table tools to get frequencies, which are needed to
    # calculate u-v points from baseline lengths
    tb.open(msfilename+"/SPECTRAL_WINDOW")
    freqs = tb.getcol("CHAN_FREQ")
    rfreq = tb.getcol("REF_FREQUENCY")
    tb.close()
    print(str(freqs.shape[1])+" SPWs and Channel 0 frequency of 1st SPW of "+str(rfreq[0]/1e9)+" GHz")
    print("corresponding to "+str(2.9979e8/rfreq[0]*1e3)+" mm")
    print("Average wavelength is "+str(2.9979e8/np.average(rfreq)*1e3)+" mm")

    print("Datasets has baselines between "+str(np.min(np.sqrt(uvw[0,:]**2.0+uvw[1,:]**2.0)))+" and "+str(np.max(np.sqrt(uvw[0,:]**2.0+uvw[1,:]**2.0)))+" m")

    #Initialize u and v arrays (coordinates in Fourier space)
    uu=np.zeros((freqs.shape[0],uvw[0,:].size))
    vv=np.zeros((freqs.shape[0],uvw[0,:].size))

    #Fill u and v arrays appropriately from data values.
    for i in np.arange(freqs.shape[0]):
        for j in np.arange(uvw.shape[1]):
            uu[i,j]=uvw[0,j]*freqs[i,spwid[j]]/(cc/100.0)
            vv[i,j]=uvw[1,j]*freqs[i,spwid[j]]/(cc/100.0)

    # Extract real and imaginary part of the visibilities at all u-v
    # coordinates, for both polarization states (XX and YY), extract
    # weights which correspond to 1/(uncertainty)^2
    Re_xx = data[0,:,:].real
    Re_yy = data[1,:,:].real
    Im_xx = data[0,:,:].imag
    Im_yy = data[1,:,:].imag
    weight_xx = weight[0,:]
    weight_yy = weight[1,:]

    # Since we don't care about polarization, combine polarization states
    # (average them together) and fix the weights accordingly. Also if
    # any of the two polarization states is flagged, flag the outcome of
    # the combination.
    flags = flags[0,:,:]*flags[1,:,:]
    ## weighted average : Re , Im
    Re = np.where((weight_xx + weight_yy) != 0, (Re_xx*weight_xx + Re_yy*weight_yy) / (weight_xx + weight_yy), 0.)
    Im = np.where((weight_xx + weight_yy) != 0, (Im_xx*weight_xx + Im_yy*weight_yy) / (weight_xx + weight_yy), 0.)
    wgts = (weight_xx + weight_yy)

    # Find which of the data represents cross-correlation between two
    # antennas as opposed to auto-correlation of a single antenna.
    # We don't care about the latter so we don't want it.
    xc = np.where(ant1 != ant2)[0]

    # Select only cross-correlation data
    data_real = Re[:,xc]
    data_imag = Im[:,xc]
    flags = flags[:,xc]
    data_wgts = wgts[xc]
    data_uu = uu[:,xc]
    data_vv = vv[:,xc]
    # Yamaguchi comments
    ## np.repeat(1, 5) -> [1 1 1 1 1]
    ## np.reshape(row, column)
    data_wgts=np.reshape(np.repeat(wgts[xc], uu.shape[0]), data_uu.shape)

    # Delete previously used (and not needed) variables (to free up some memory?)
    del Re
    del Im
    del wgts
    del uu
    del vv

    # Select only data that is NOT flagged
    data_real = data_real[np.logical_not(flags)]
    data_imag = data_imag[np.logical_not(flags)]
    flagss = flags[np.logical_not(flags)]
    data_wgts = data_wgts[np.logical_not(flags)]
    data_uu = data_uu[np.logical_not(flags)]
    data_vv = data_vv[np.logical_not(flags)]

    # Wrap up all the arrays/matrices we need, (u-v coordinates, complex
    # visibilities, and weights for each visibility) and save them all
    # together in a numpy file
    u, v, Re, Im, w = data_uu, data_vv, data_real, data_imag, data_wgts
    np.save(outputfilename, [u, v, Re, Im, w])
