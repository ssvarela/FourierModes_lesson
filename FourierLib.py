import numpy as np

class Asymmetry:
    """
    Class for analyzing galaxy asymmetries using Fourier modes.
    Computes radial Fourier coefficients and reconstructs 2D Fourier maps.

    Parameters
    ----------
    stars_data : ndarray
        Stellar data array with columns [x, y, z, vx, vy, vz, m, ...].
    """

    def __init__(self, stars_data):
        # Store stellar data
        self.fstar = stars_data

    def Mfourier(self, Ropt, r_min=0.4, r_max=1.2, r_int=0.1, GetRadialDistr=False):
        """
        Compute radial Fourier coefficients (m=1 to m=6) in annuli.

        Parameters
        ----------
        Ropt : float
            Optical radius of the galaxy.
        r_min, r_max : float
            Radial range (in units of Ropt).
        r_int : float
            Width of radial bins.
        GetRadialDistr : bool
            If True, returns radial coefficients and phase angles.

        Returns
        -------
        Amean : ndarray
            Mean Am coefficients for each mode.
        dictRadDistr : dict (optional)
            Dictionary with radial coefficients and phase angles.
        """
        # Select stars within disk height
        ind = (np.abs(self.fstar[:, 2]) <= 10)
        sel = self.fstar[ind]

        # Extract positions and masses
        a = sel[:, :3]
        am = sel[:, 6]

        # Compute polar angle and normalized radius
        theta = np.arctan2(a[:, 1], a[:, 0])
        r2d = np.sqrt(a[:, 0]**2 + a[:, 1]**2) / Ropt

        # Initialize lists for coefficients and phases
        n_ = round((r_max - r_min) / r_int)
        B = [[], [], [], [], [], [], []]
        PhAng = [[], [], [], [], [], [], []]
        Rb = []

        # Loop over radial bins
        rmin = r_min
        rmax = r_min + r_int
        for i in range(n_):
            Rb.append(0.5 * (rmax + rmin))
            ind = (rmin < r2d) & (r2d <= rmax)
            selm = am[ind]
            seltheta = theta[ind]
            # Compute coefficients for each Fourier mode
            for m in range(1, 7):
                an = np.sum(selm * np.cos(m * seltheta))
                bn = np.sum(selm * np.sin(m * seltheta))
                B[m].append(np.sqrt(an**2 + bn**2))
                PhAng[m].append(np.arctan2(bn, an))
            # Store total mass in the bin (m=0)
            B[0].append(np.sum(selm))
            rmax += r_int
            rmin += r_int

        # Convert lists to arrays
        B = np.array(B)
        Rb = np.array(Rb)

        # Normalize modes by total mass
        An = B[1:] / B[0]
        dictRadDistr = {'radial_Am': An,
                        'radial_PhaseAngle': PhAng[1:],
                        'rbins': Rb}
        # Compute mean value for each mode
        Amean = np.array([np.mean(Ai) for Ai in An])
        if not GetRadialDistr:
            return Amean
        else:
            return Amean, dictRadDistr

    def Mfourier2D(self, Ropt, grid_size=100, modes=(1, 2, 3, 4, 5, 6),
                   r_min=0.0, r_max=1.2, n_radios=30, zmax=10, normalize=True):
        """
        Reconstructs 2D Fourier maps using interpolated modulus and phase.

        Parameters
        ----------
        Ropt : float
            Optical radius of the galaxy.
        grid_size : int
            XY grid size.
        modes : tuple
            Fourier modes to compute.
        r_min, r_max : float
            Radial range (in units of Ropt).
        n_radios : int
            Number of radial bins.
        zmax : float
            Disk height cut.
        normalize : bool
            If True, normalizes coefficients by total mass in each bin.

        Returns
        -------
        maps2D : dict
            Dictionary {mode: 2D map (XY)}.
        xi, yi : ndarray
            XY grid coordinates.
        """
        # Select stars within disk height
        mask = np.abs(self.fstar[:, 2]) <= zmax
        x = self.fstar[mask, 0]
        y = self.fstar[mask, 1]
        masa = self.fstar[mask, 6]

        # Compute radius and polar angle
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        rmax = Ropt * r_max
        rmin = Ropt * r_min
        rbins = np.linspace(rmin, rmax, n_radios + 1)
        rcenters = 0.5 * (rbins[:-1] + rbins[1:])

        # Compute radial coefficients for each mode
        Bm = {m: np.zeros(n_radios) for m in modes}
        phim = {m: np.zeros(n_radios) for m in modes}
        for i in range(n_radios):
            sel = (r >= rbins[i]) & (r < rbins[i + 1])
            if np.sum(sel) < 5:
                for m in modes:
                    Bm[m][i] = np.nan
                    phim[m][i] = np.nan
                continue
            mi, thetai = masa[sel], theta[sel]
            m0 = np.sum(mi)
            for m in modes:
                am = np.sum(mi * np.cos(m * thetai))
                bm = np.sum(mi * np.sin(m * thetai))
                Bm[m][i] = np.sqrt(am**2 + bm**2) / m0 if normalize else np.sqrt(am**2 + bm**2)
                phim[m][i] = np.arctan2(bm, am)

        # Build XY grid
        lim = rmax
        xi = np.linspace(-lim, lim, grid_size)
        yi = np.linspace(-lim, lim, grid_size)
        Xc, Yc = np.meshgrid(xi, yi, indexing='ij')
        Rxy = np.sqrt(Xc**2 + Yc**2)
        Thxy = np.arctan2(Yc, Xc)

        # Reconstruct 2D map for each mode
        maps2D = {}
        for m in modes:
            # Interpolate Bm and phim on XY grid
            Bm_interp = np.interp(Rxy, rcenters, Bm[m], left=0, right=0)
            phim_interp = np.interp(Rxy, rcenters, phim[m], left=0, right=0)
            # 2D map: A_m(r) * cos(m*theta - phi_m(r))
            maps2D[m] = Bm_interp * np.cos(m * Thxy - phim_interp)

        return maps2D, xi, yi

    def Mfourier2D_v2(self, Ropt, grid_size=200, modes=(1, 2, 3, 4, 5, 6),
                      r_min=0.0, r_max=1.2, n_radios=30, zmax=10, normalize=True):
        """
        Reconstructs 2D Fourier maps using Cartesian coefficients (Wc, Ws).
        This version is numerically more stable.

        Parameters
        ----------
        Ropt : float
            Optical radius of the galaxy.
        grid_size : int
            XY grid size.
        modes : tuple
            Fourier modes to compute.
        r_min, r_max : float
            Radial range (in units of Ropt).
        n_radios : int
            Number of radial bins.
        zmax : float
            Disk height cut.
        normalize : bool
            If True, normalizes coefficients by total mass in each bin.

        Returns
        -------
        maps2D : dict
            Dictionary {mode: 2D map (XY)}.
        xi, yi : ndarray
            XY grid coordinates.
        """
        # Select stars within disk height
        mask = np.abs(self.fstar[:, 2]) <= zmax
        x = self.fstar[mask, 0]
        y = self.fstar[mask, 1]
        masa = self.fstar[mask, 6]

        # Compute radius and polar angle
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        rmax = Ropt * r_max
        rmin = Ropt * r_min
        rbins = np.linspace(rmin, rmax, n_radios + 1)
        rcenters = 0.5 * (rbins[:-1] + rbins[1:])

        # Initialize arrays for Wc and Ws
        Wc = {m: np.full(n_radios, np.nan) for m in modes}
        Ws = {m: np.full(n_radios, np.nan) for m in modes}

        # Compute Cartesian coefficients in each bin
        for i in range(n_radios):
            sel = (r >= rbins[i]) & (r < rbins[i + 1])
            if np.sum(sel) < 5:
                continue
            mi = masa[sel]
            thetai = theta[sel]
            m0 = np.sum(mi)
            for m in modes:
                a_m = np.sum(mi * np.cos(m * thetai))
                b_m = np.sum(mi * np.sin(m * thetai))
                if normalize and m0 > 0:
                    Wc[m][i] = a_m / m0
                    Ws[m][i] = b_m / m0
                else:
                    Wc[m][i] = a_m
                    Ws[m][i] = b_m

        # Build XY grid
        lim = rmax
        xi = np.linspace(-lim, lim, grid_size)
        yi = np.linspace(-lim, lim, grid_size)
        Xc, Yc = np.meshgrid(xi, yi, indexing='ij')
        Rxy = np.sqrt(Xc**2 + Yc**2)
        Thxy = np.arctan2(Yc, Xc)

        # Reconstruct 2D map for each mode
        maps2D = {}
        for m in modes:
            # Interpolate Wc and Ws on XY grid
            Wc_interp = np.interp(Rxy, rcenters, np.nan_to_num(Wc[m], 0.0), left=0.0, right=0.0)
            Ws_interp = np.interp(Rxy, rcenters, np.nan_to_num(Ws[m], 0.0), left=0.0, right=0.0)
            # 2D map: Wc*cos(m*theta) + Ws*sin(m*theta)
            maps2D[m] = Wc_interp * np.cos(m * Thxy) + Ws_interp * np.sin(m * Thxy)

        return maps2D, xi, yi