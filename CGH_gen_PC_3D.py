"""
Binary intensity CGH generation for a 3D point-cloud target (DMD projection).

A wireframe 3D box is sampled into a set of point sources; each point
contributes a spherical-wave phase pattern on the DMD. Three methods
generate the binary intensity hologram (off-axis interference + mean
thresholding):
  - RS  : random superposition of the point patterns, single step
  - GS  : weighted compressive-sampling GS, binarization only at the end
  - ITR : weighted compressive-sampling GS with binarization in the loop

If enabled, the astigmatism of the projection path is pre-compensated
in the point phase patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Short aliases for the FFT routines used throughout
fft2 = np.fft.fft2
fftshift = np.fft.fftshift


def points_btwn3D(p0, p1, N):
    """Return p0, N equally spaced intermediate points, and p1."""
    points = [p0]
    for i in range(1, N + 1):
        t = i / (N + 1)
        points.append((p0[0] + t * (p1[0] - p0[0]),
                       p0[1] + t * (p1[1] - p0[1]),
                       p0[2] + t * (p1[2] - p0[2])))
    points.append(p1)
    return points


def rs(x, y, z, d, ref_field, lam, res):
    """Random-superposition (RS) binary hologram of a set of point sources.

    Uses module-level global: Phase_abb.
    Returns the binary intensity hologram (bool, res x res).
    """
    # DMD pixel coordinates [m]
    slm_xcoord, slm_ycoord = np.meshgrid(np.linspace(-1.0, 1.0, res), np.linspace(-1.0, 1.0, res))
    slm_xcoord = slm_xcoord * d * float(res) / 2.0
    slm_ycoord = slm_ycoord * d * float(res) / 2.0

    # Random pixel ordering and initial phase (piston) of each point source
    coordslist = np.asarray(range(res**2))
    np.random.shuffle(coordslist)
    pists = np.random.random(x.shape[0]) * 2 * np.pi

    # Spherical-wave phase pattern of each point source, with aberration
    # pre-compensation
    slm_p_phase = np.zeros((x.shape[0], res**2))
    for i in range(x.shape[0]):
        slm_p_phase[i, :] = (2.0 * np.pi / lam) * np.sqrt(
            (x[i] - slm_xcoord.flatten())**2
            + (y[i] - slm_ycoord.flatten())**2
            + z[i]**2)
    slm_p_phase = (slm_p_phase - Phase_abb.flatten()) % (2 * np.pi)

    # Hologram phase: superposition of all patterns with their pistons
    slm_total_field = np.sum(1.0 / float(res**2) * np.exp(1j * (slm_p_phase + pists[:, None])), axis=0)
    slm_total_phase = np.angle(slm_total_field)

    # Binary intensity hologram: interference + mean thresholding
    interference = np.abs(ref_field + np.exp(1j * slm_total_phase))**2
    interference_binary = (interference > np.mean(interference))

    return interference_binary.reshape(res, res)


def wcsgs_phaseonly(x, y, z, d, ref_field, lam, res, iters, sub):
    """Weighted compressive-sampling GS: the hologram stays phase-only
    during the iterations and is binarized only at the end (GS method).

    sub is the fraction of DMD pixels used at each iteration.
    Uses module-level global: Phase_abb.
    Returns the binary intensity hologram (bool, res x res).
    """
    # DMD pixel coordinates [m]
    slm_xcoord, slm_ycoord = np.meshgrid(np.linspace(-1.0, 1.0, res), np.linspace(-1.0, 1.0, res))
    slm_xcoord = slm_xcoord * d * float(res) / 2.0
    slm_ycoord = slm_ycoord * d * float(res) / 2.0

    # Random pixel ordering, initial pistons and uniform weights
    coordslist = np.asarray(range(res**2))
    np.random.shuffle(coordslist)
    pists = np.random.random(x.shape[0]) * 2 * np.pi
    weights = np.ones(x.shape[0]) / float(x.shape[0])

    # Spherical-wave phase pattern of each point source, with aberration
    # pre-compensation
    slm_p_phase = np.zeros((x.shape[0], res**2))
    for i in range(x.shape[0]):
        slm_p_phase[i, :] = (2.0 * np.pi / lam) * np.sqrt(
            (x[i] - slm_xcoord.flatten())**2
            + (y[i] - slm_ycoord.flatten())**2
            + z[i]**2)
    slm_p_phase = (slm_p_phase - Phase_abb.flatten()) % (2 * np.pi)

    ## Main loop, on a rolling random subset of the DMD pixels
    for n in range(iters - 1):
        coordslist = np.roll(coordslist, int(coordslist.shape[0] * sub))
        coordslist_sparse = coordslist[:int(coordslist.shape[0] * sub)]

        # Hologram phase from the weighted superposition
        slm_total_field = np.sum(weights[:, None] / float(res**2) * np.exp(1j * (slm_p_phase[:, coordslist_sparse] + pists[:, None])), axis=0)
        slm_total_phase = np.angle(slm_total_field)

        # Back-propagation to the points: update pistons and weights
        spot_fields = np.sum(1.0 / float(res**2) * np.exp(1j * (slm_total_phase[None, :] - slm_p_phase[:, coordslist_sparse])), axis=1)
        pists = np.angle(spot_fields)
        ints = np.abs(spot_fields)**2
        weights = weights * (np.mean(np.sqrt(ints)) / np.sqrt(ints))
        weights = weights / np.sum(weights)

        print("GS iter", n + 1)

    ## Final full-resolution hologram phase, then binarization
    slm_total_field = np.sum(weights[:, None] / float(res**2) * np.exp(1j * (slm_p_phase + pists[:, None])), axis=0)
    slm_total_phase = np.angle(slm_total_field)
    interference = np.abs(ref_field + np.exp(1j * slm_total_phase))**2
    interference_binary = (interference > np.mean(interference))

    return interference_binary.reshape(res, res)


def wcsgs_binary(x, y, z, d, ref_field, lam, res, iters, sub):
    """Weighted compressive-sampling GS with binarization inside the loop:
    pistons and weights are updated from the binary hologram (ITR method).

    sub is the fraction of DMD pixels used at each iteration.
    Uses module-level global: Phase_abb.
    Returns the binary intensity hologram (bool, res x res).
    """
    # DMD pixel coordinates [m]
    slm_xcoord, slm_ycoord = np.meshgrid(np.linspace(-1.0, 1.0, res), np.linspace(-1.0, 1.0, res))
    slm_xcoord = slm_xcoord * d * float(res) / 2.0
    slm_ycoord = slm_ycoord * d * float(res) / 2.0

    # Random pixel ordering, initial pistons and uniform weights
    coordslist = np.asarray(range(res**2))
    np.random.shuffle(coordslist)
    pists = np.random.random(x.shape[0]) * 2 * np.pi
    weights = np.ones(x.shape[0]) / float(x.shape[0])

    # Spherical-wave phase pattern of each point source, with aberration
    # pre-compensation
    slm_p_phase = np.zeros((x.shape[0], res**2))
    for i in range(x.shape[0]):
        slm_p_phase[i, :] = (2.0 * np.pi / lam) * np.sqrt(
            (x[i] - slm_xcoord.flatten())**2
            + (y[i] - slm_ycoord.flatten())**2
            + z[i]**2)
    slm_p_phase = (slm_p_phase - Phase_abb.flatten()) % (2 * np.pi)

    ## Main loop, on a rolling random subset of the DMD pixels
    for n in range(iters - 1):
        coordslist = np.roll(coordslist, int(coordslist.shape[0] * sub))
        coordslist_sparse = coordslist[:int(coordslist.shape[0] * sub)]

        # Binary hologram from the weighted superposition
        slm_total_field = np.sum(weights[:, None] / float(res**2) * np.exp(1j * (slm_p_phase[:, coordslist_sparse] + pists[:, None])), axis=0)
        slm_total_phase = np.angle(slm_total_field)
        interference = np.abs(ref_field[coordslist_sparse] + np.exp(1j * slm_total_phase))**2
        interference_binary = (interference > np.mean(interference))

        # Back-propagation of the binary hologram to the points: update
        # pistons and weights
        spot_fields = np.sum(1.0 / float(res**2) * interference_binary[None, :] * np.exp(1j * (np.angle(ref_field)[None, coordslist_sparse] - slm_p_phase[:, coordslist_sparse])), axis=1)
        pists = np.angle(spot_fields)
        ints = np.abs(spot_fields)**2
        weights = weights * (np.mean(np.sqrt(ints)) / np.sqrt(ints))
        weights = weights / np.sum(weights)

        print("ITR iter", n + 1)

    ## Final full-resolution hologram phase, then binarization
    slm_total_field = np.sum(weights[:, None] / float(res**2) * np.exp(1j * (slm_p_phase + pists[:, None])), axis=0)
    slm_total_phase = np.angle(slm_total_field)
    interference = np.abs(ref_field + np.exp(1j * slm_total_phase))**2
    interference_binary = (interference > np.mean(interference))

    return interference_binary.reshape(res, res)


if __name__ == "__main__":

    ##### Target: wireframe 3D box (two squares at different depths) #####

    p1 = np.asarray([0.006, 0.006, 0.6])
    p2 = np.asarray([-0.006, 0.006, 0.6])
    p3 = np.asarray([-0.006, -0.006, 0.6])
    p4 = np.asarray([0.006, -0.006, 0.6])

    p5 = np.asarray([0, 0, 0.4])
    p6 = np.asarray([-0.012, 0, 0.4])
    p7 = np.asarray([-0.012, -0.012, 0.4])
    p8 = np.asarray([0, -0.012, 0.4])

    num = 15                      # Intermediate points per segment

    # Front square, back square, connecting edges
    segments = [(p1, p2), (p2, p3), (p3, p4), (p4, p1),
                (p5, p6), (p6, p7), (p7, p8), (p8, p5),
                (p1, p5), (p2, p6), (p3, p7), (p4, p8)]
    pts = np.vstack([np.array(points_btwn3D(q0, q1, num)) for q0, q1 in segments])
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    ##### Physical and numerical parameters #####

    d = 10.8e-6                   # DMD pixel pitch [m]
    lam = 532e-9                  # Wavelength [m]
    res = 1080                    # Hologram resolution (pixels per side)
    Mg = 2                        # Magnification of the projection path
    d = d / Mg                    # Effective pixel pitch [m]

    sub = 0.1                     # Compressive-sampling fraction per iteration
    iters = 5                    # Number of iterations (GS and ITR)

    # Spatial grid in the hologram plane
    cc = np.linspace(-d * res / 2, d * res / 2 - d, res)
    X, Y = np.meshgrid(cc, cc)

    ##### Flags #####

    # Aberration pre-compensation: the astigmatism coefficients below were
    # measured for our specific projection setup. When False the aberration
    # phase is zero: ideal hologram generation.
    compensate_aberration = False

    # Save the binary holograms as 8-bit BMP files
    save_holograms = False

    # Astigmatism aberration phase of the projection path
    a = 15e-2                     # Astigmatism coefficient (0/90 deg)
    b = 8e-2                      # Astigmatism coefficient (45 deg)
    if compensate_aberration:
        Phase_abb = 2 * np.pi * (a * (X**2 - Y**2) + b * 2 * X * Y) / lam
    else:
        Phase_abb = np.zeros_like(X)

    # Off-axis reference field
    theta = 0.67 * np.pi / 12
    phase_reference = 2 * np.pi * np.sin(theta) * X / lam + 2 * np.pi * np.sin(theta) * Y / lam
    reference = np.exp(1j * phase_reference)

    # Fresnel back-propagation factor of the reconstruction plane (front face)
    z_rec = 0.6                   # Reconstruction distance [m]
    k = 2 * np.pi / lam
    propagation_factor = np.exp(-1j * k * (X**2 + Y**2) / (2 * z_rec))

    ##### CGH generation and numerical reconstructions #####

    binary_hologram_bgs = wcsgs_binary(x, y, z, d, reference.flatten(), lam, res, iters, sub)
    image_bgs = np.abs(fftshift(fft2(fftshift(binary_hologram_bgs * reference * propagation_factor))))**2

    binary_hologram_pgs = wcsgs_phaseonly(x, y, z, d, reference.flatten(), lam, res, iters, sub)
    image_pgs = np.abs(fftshift(fft2(fftshift(binary_hologram_pgs * reference * propagation_factor))))**2

    binary_hologram_rs = rs(x, y, z, d, reference.flatten(), lam, res)
    image_rs = np.abs(fftshift(fft2(fftshift(binary_hologram_rs * reference * propagation_factor))))**2

    ##### Display of the numerical reconstructions at z_rec #####
    # The exponent 1e-3 compresses the dynamic range for visualization

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(image_bgs**1e-3, cmap='gray')
    ax[0].set_title('ITR')
    ax[1].imshow(image_pgs**1e-3, cmap='gray')
    ax[1].set_title('GS')
    ax[2].imshow(image_rs**1e-3, cmap='gray')
    ax[2].set_title('RS')
    plt.show()

    if save_holograms:
        Image.fromarray((255 * binary_hologram_rs).astype(np.uint8)).save('Hologram_RS_Box.bmp')
        Image.fromarray((255 * binary_hologram_bgs).astype(np.uint8)).save('Hologram_ITR_Box.bmp')
        Image.fromarray((255 * binary_hologram_pgs).astype(np.uint8)).save('Hologram_GS_Box.bmp')
