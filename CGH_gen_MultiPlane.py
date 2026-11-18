"""
Binary intensity CGH generation for DMD projection - multi-plane version.

Two targets are placed at two different depths through Fresnel
propagation factors and encoded in a single binary hologram.
Three hologram generation methods are compared:
  - RS  : single-step random-superposition phase hologram
  - GS  : Gerchberg-Saxton with target resampling
  - ITR : iterative binary-hologram estimation (Fienup-like)

The phase holograms (RS, GS) are converted to binary intensity
holograms by interference with an off-axis reference wave and
mean-value thresholding. If enabled, the astigmatism of the projection
path is pre-compensated in the hologram phase and re-applied in the
numerical reconstruction to emulate the physical system.
"""

import time

import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Short aliases for the FFT routines used throughout
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

get_time = time.perf_counter


def rs_minimal(target):
    """Random-superposition (RS) hologram: single FFT of the target
    amplitude with a random initial phase. Returns the hologram phase."""
    # Target intensity -> normalized amplitude
    target = (target - np.min(target)) / (np.max(target) - np.min(target))
    ampl = np.sqrt(target)

    # Random initial phase in [0, 2*pi)
    phase_0 = 2 * np.pi * np.random.rand(*target.shape)
    target_complex_field = ampl * np.exp(1j * phase_0)

    # Back-propagate to the hologram plane and keep the phase only
    holo = np.angle(fftshift(ifft2(fftshift(target_complex_field))))
    return holo

def gs_resampled(target, n_iter):
    """Gerchberg-Saxton on a 2x resampled target (double-size grid).

    Returns:
        kinoform  : hologram phase, cropped back to the original size
        Campo_rec : normalized reconstructed intensity (last iteration)
        RMSE      : per-iteration reconstruction error
    """
    target = np.array(target)

    # Normalize the target to [0, 255]
    target_scaled = 255 * (target - np.min(target)) / (np.max(target) - np.min(target))

    n = target_scaled.shape[0]

    ## Resampling: zero-pad the target spectrum to double the grid size
    xx = fft2(target_scaled)
    xx = fftshift(xx)
    xx = np.pad(xx, [(n // 2, n // 2), (n // 2, n // 2)], mode='constant')
    xx = ifftshift(xx)
    target_resampled = np.abs(ifft2(xx)) * 4

    # Normalize the resampled target to [0, 255]
    target_resampled_scaled = 255 * (target_resampled - np.min(target_resampled)) \
        / (np.max(target_resampled) - np.min(target_resampled))

    ## Initialization: target amplitude with random phase
    A0 = np.sqrt(target_resampled_scaled)
    fi_0 = 2 * np.pi * np.random.rand(*target_resampled_scaled.shape)
    A = A0 * np.exp(1j * fi_0)

    ## Fixed amplitude in the hologram plane: uniform on the central n x n
    # aperture, zero outside, with total energy matching the target
    H_ampl = np.ones((n, n))
    H_ampl = np.pad(H_ampl, [(n // 2, n // 2), (n // 2, n // 2)], mode='constant', constant_values=0)
    H_ampl = H_ampl * np.sum(target_resampled_scaled) / np.sum(H_ampl)
    H_ampl = np.sqrt(H_ampl)

    ## Iterations
    RMSE = np.zeros(n_iter)
    progress_bar = tqdm(total=n_iter, desc="GS Iterating", unit="iteration")

    for f in range(n_iter):
        # Back-propagate to the hologram plane and apply the amplitude constraint
        H = fftshift(ifft2(fftshift(A)))
        kino = np.angle(H)
        H = H_ampl * np.exp(1j * kino)

        # Propagate to the object plane
        A = fftshift(fft2(fftshift(H)))

        # Normalized reconstructed intensity (for RMSE monitoring)
        A_int = np.abs(A) ** 2
        A_int = (A_int - np.min(A_int)) / (np.max(A_int) - np.min(A_int))

        # Apply the target amplitude constraint, keeping the estimated phase
        Phase_out = np.angle(A)
        A = A0 * np.exp(1j * Phase_out)

        RMSE[f] = np.sum((A_int - target_resampled) ** 2) / (np.sum(target_resampled) ** 2)

        progress_bar.update(1)

    progress_bar.close()

    ## Crop the hologram phase back to the original (central) n x n window
    start = kino.shape[0] // 4
    kinoform = kino[start:start + n, start:start + n]

    Campo_rec = A_int

    return kinoform, Campo_rec, RMSE


def iterativeEstMulti(target1, target2, object_distance1, object_distance2, thetax, thetay, n_iter):
    """Iterative binary-hologram estimation (ITR) for two target planes.

    Same scheme as the single-plane version: object-plane amplitude
    constraint, then binary intensity constraint in the hologram plane
    (off-axis interference + mean thresholding). Fresnel propagation
    factors place the two targets at different depths. The aberration
    phase (zero if compensation is disabled) is pre-compensated at the end.

    Uses module-level globals: X, Y, lam, N, Phase_abb.
    Returns the binary intensity hologram (values 0 / 255).
    """
    # Initial random intensity in the hologram plane
    O = np.abs(-1 + 2 * np.random.rand(*target1.shape))

    # Off-axis reference field
    R_p = 2 * np.pi * np.sin(thetax) * X / lam + 2 * np.pi * np.sin(thetay) * Y / lam
    R_f = np.exp(1j * R_p)

    # Fresnel diffraction factors of the two object planes
    k = 2 * np.pi / lam
    propagation_factor1 = np.exp(1j * k * (X**2 + Y**2) / (2 * object_distance1))
    propagation_factor2 = np.exp(1j * k * (X**2 + Y**2) / (2 * object_distance2))

    for it in range(n_iter):
        # Propagate the binary hologram to the two object planes
        a1 = fftshift(fft2(fftshift(O * propagation_factor1.conj())))
        a2 = fftshift(fft2(fftshift(O * propagation_factor2.conj())))

        # Strong amplitude compression to flatten the dynamic range
        a1 = a1**1e-3
        a2 = a2**1e-3

        # Object constraint: target amplitude with the estimated phase
        a12 = (np.sqrt(np.abs(target1)) * a1) / np.abs(a1)
        a22 = (np.sqrt(np.abs(target2)) * a2) / np.abs(a2)

        # Normalize and scale to [0, 255]
        a12 = (a12 - np.min(a12)) / (np.max(a12) - np.min(a12))
        a22 = (a22 - np.min(a22)) / (np.max(a22) - np.min(a22))
        a1 = 255 * a12
        a2 = 255 * a22

        ## Resampling: zero-pad the spectra to double the grid size
        xx1 = fftshift(fft2(a1))
        xx2 = fftshift(fft2(a2))
        xx1 = np.pad(xx1, [(N // 2, N // 2), (N // 2, N // 2)], mode='constant')
        xx2 = np.pad(xx2, [(N // 2, N // 2), (N // 2, N // 2)], mode='constant')
        target_resampled1 = np.abs(ifft2(ifftshift(xx1))) * 4
        target_resampled2 = np.abs(ifft2(ifftshift(xx2))) * 4

        # Normalize the resampled targets to [0, 255]
        target_resampled_scaled1 = 255 * (target_resampled1 - np.min(target_resampled1)) \
            / (np.max(target_resampled1) - np.min(target_resampled1))
        target_resampled_scaled2 = 255 * (target_resampled2 - np.min(target_resampled2)) \
            / (np.max(target_resampled2) - np.min(target_resampled2))

        # New object fields: resampled amplitude with fresh random phase
        A01 = np.sqrt(target_resampled_scaled1)
        A02 = np.sqrt(target_resampled_scaled2)
        fi_01 = 2 * np.pi * np.random.rand(*target_resampled_scaled1.shape)
        fi_02 = 2 * np.pi * np.random.rand(*target_resampled_scaled2.shape)
        A1 = A01 * np.exp(1j * fi_01)
        A2 = A02 * np.exp(1j * fi_02)

        ## Back-propagate to the hologram plane, keep the phases
        H1 = fftshift(ifft2(fftshift(A1)))
        H2 = fftshift(ifft2(fftshift(A2)))
        kino1 = np.angle(H1)
        kino2 = np.angle(H2)

        # Crop the phases back to the central N x N window
        start = kino1.shape[0] // 4
        O_p1 = kino1[start:start + N, start:start + N]
        O_p2 = kino2[start:start + N, start:start + N]

        # Binary intensity hologram: off-axis interference + thresholding
        O_f1 = np.exp(1j * O_p1)
        O_f2 = np.exp(1j * O_p2)
        I = np.abs(R_f + O_f1 * propagation_factor1 + O_f2 * propagation_factor2)**2
        I = (I > np.mean(I))
        O = 255 * I

    # Final hologram with astigmatism pre-compensation
    O_p1 = (O_p1 - Phase_abb) % (2 * np.pi)
    O_p2 = (O_p2 - Phase_abb) % (2 * np.pi)
    O_f1 = np.exp(1j * O_p1)
    O_f2 = np.exp(1j * O_p2)
    I = np.abs(R_f + O_f1 * propagation_factor1 + O_f2 * propagation_factor2)**2
    I = (I > np.mean(I))
    O = 255 * I

    return O


##### Physical and numerical parameters #####

d = 10.8e-6                   # DMD pixel pitch [m]
lam = 532e-9                  # Wavelength [m]
res = 1080                    # Hologram resolution (pixels per side)
Mg = 0.5                      # Magnification of the projection path
d *= Mg                       # Effective pixel pitch [m]
L = d * res                   # Hologram lateral size [m]

# Spatial grid in the hologram plane
x = np.linspace(-L/2, L/2 - d, res)
X, Y = np.meshgrid(x, x)

##### Flags #####

# Aberration pre-compensation: the astigmatism coefficients below were
# measured for our specific projection setup. When False the aberration
# phase is zero: ideal hologram generation and ideal propagation.
compensate_aberration = False

# Save the binary holograms as 8-bit TIFF files
save_holograms = False

# Astigmatism aberration phase of the projection path
a = 15e-2                     # Astigmatism coefficient (0/90 deg)
b = 8e-2                      # Astigmatism coefficient (45 deg)
if compensate_aberration:
    Phase_abb = 2 * np.pi * (a * (X**2 - Y**2) + b * 2 * X * Y) / lam
else:
    Phase_abb = np.zeros_like(X)

N = res                       # Hologram size (pixels)
P = 540                       # Effective object size (pixels)


##### Target image preparation #####

input_image_path1 = "synthetic_Polimi.png"
input_image_path2 = "synthetic_LUX.png"

# Load, drop transparency, convert to grayscale (ITU-R 601 luma weights)
img1 = Image.open(input_image_path1).convert("RGBA")
img2 = Image.open(input_image_path2).convert("RGBA")
rgb_img1 = np.array(img1)[:, :, :3]
rgb_img2 = np.array(img2)[:, :, :3]
gray_img1 = np.dot(rgb_img1, [0.2989, 0.5870, 0.1140])
gray_img2 = np.dot(rgb_img2, [0.2989, 0.5870, 0.1140])

# Resize to the effective object size
image1 = np.array(Image.fromarray(gray_img1).resize((P, P)))
image2 = np.array(Image.fromarray(gray_img2).resize((P, P)))
image2 = 255 - image2         # Invert the second target's contrast

# Place the two P x P targets side by side on the N x N grid
target1 = np.zeros((N, N))
target2 = np.zeros((N, N))
target1[(N - P) // 2:(N + P) // 2, 0:N // 2] = image1
target2[(N - P) // 2:(N + P) // 2, N // 2:N] = image2

plt.imshow(target1, cmap='gray')
plt.title('Target 1')
plt.show()
plt.imshow(target2, cmap='gray')
plt.title('Target 2')
plt.show()

##### Object planes and reference wave #####

d1 = 0.3                      # Distance of object plane 1 [m]
d2 = 0.9                      # Distance of object plane 2 [m]

# Fresnel diffraction factors of the two object planes
k = 2 * np.pi / lam
propagation_factor1 = np.exp(1j * k * (X**2 + Y**2) / (2 * d1))
propagation_factor2 = np.exp(1j * k * (X**2 + Y**2) / (2 * d2))

# Off-axis reference angles: no horizontal carrier is needed, the
# lateral offsets of the two targets are already encoded in the grid
thetax = 0
thetay = 0.65 * np.pi / 12

itr = 5                       # Number of iterations (GS and ITR)


##### CGH generation: ITR #####

t1 = get_time()
hologram = iterativeEstMulti(target1, target2, d1, d2, thetax, thetay, itr)
hologram_ITR = 255 * hologram
t1 = get_time() - t1
print("ITR computation time:", t1)

# Numerical reconstructions at the two planes (projection aberration re-applied)
Rec1 = fftshift(fft2(fftshift(hologram_ITR * propagation_factor1.conj() * np.exp(1j * Phase_abb))))
Rec2 = fftshift(fft2(fftshift(hologram_ITR * propagation_factor2.conj() * np.exp(1j * Phase_abb))))

if save_holograms:
    # 'hologram' (not 'hologram_ITR') holds the 0/255 values fitting uint8
    Image.fromarray(hologram.astype(np.uint8)).save('Hologram_ITR_Multi.tiff')

##### CGH generation: GS + off-axis binarization #####

t2 = get_time()

# Hologram phases from GS (aberration pre-compensation if enabled)
Op1, _, _ = gs_resampled(target1, itr)
Op2, _, _ = gs_resampled(target2, itr)
Op1 = (Op1 - Phase_abb) % (2 * np.pi)
Op2 = (Op2 - Phase_abb) % (2 * np.pi)
Of1 = np.exp(1j * Op1)
Of2 = np.exp(1j * Op2)

# Off-axis reference field
Rp = 2 * np.pi * np.sin(thetax) * X / lam + 2 * np.pi * np.sin(thetay) * Y / lam
Rf = np.exp(1j * Rp)

# Binary intensity hologram: interference + mean thresholding
Ip = np.abs(Rf + Of1 * propagation_factor1 + Of2 * propagation_factor2)**2
Ip = (Ip > np.mean(Ip))
hologram_GS = 255 * Ip.astype(int)

t2 = get_time() - t2
print("GS computation time:", t2)

# Numerical reconstructions
Recg1 = fftshift(fft2(fftshift(hologram_GS * propagation_factor1.conj() * np.exp(1j * Phase_abb))))
Recg2 = fftshift(fft2(fftshift(hologram_GS * propagation_factor2.conj() * np.exp(1j * Phase_abb))))

if save_holograms:
    Image.fromarray(hologram_GS.astype(np.uint8)).save('Hologram_GS_Multi.tiff')

##### CGH generation: RS + off-axis binarization #####

t3 = get_time()

# Hologram phases from RS (aberration pre-compensation if enabled)
Or1 = rs_minimal(target1)
Or2 = rs_minimal(target2)
Or1 = (Or1 - Phase_abb) % (2 * np.pi)
Or2 = (Or2 - Phase_abb) % (2 * np.pi)
Orf1 = np.exp(1j * Or1)
Orf2 = np.exp(1j * Or2)

# Off-axis reference field
Rp1 = 2 * np.pi * np.sin(thetax) * X / lam + 2 * np.pi * np.sin(thetay) * Y / lam
Rf1 = np.exp(1j * Rp1)

# Binary intensity hologram: interference + mean thresholding
Ip1 = np.abs(Rf1 + Orf1 * propagation_factor1 + Orf2 * propagation_factor2)**2
Ip1 = (Ip1 > np.mean(Ip1))
hologram_RS = 255 * Ip1.astype(int)

t3 = get_time() - t3
print("RS computation time:", t3)

# Numerical reconstructions
Recr1 = fftshift(fft2(fftshift(hologram_RS * propagation_factor1.conj() * np.exp(1j * Phase_abb))))
Recr2 = fftshift(fft2(fftshift(hologram_RS * propagation_factor2.conj() * np.exp(1j * Phase_abb))))

if save_holograms:
    Image.fromarray(hologram_RS.astype(np.uint8)).save('Hologram_RS_Multi.tiff')

##### Display of the numerical reconstructions #####
# The exponent 1e-2 compresses the dynamic range for visualization

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow(np.abs(Rec1) ** 1e-2, cmap='gray')
ax[1].imshow(np.abs(Rec2) ** 1e-2, cmap='gray')
fig.suptitle('Reconstruction ITR')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow(np.abs(Recg1) ** 1e-2, cmap='gray')
ax[1].imshow(np.abs(Recg2) ** 1e-2, cmap='gray')
fig.suptitle('Reconstruction GS')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow(np.abs(Recr1) ** 1e-2, cmap='gray')
ax[1].imshow(np.abs(Recr2) ** 1e-2, cmap='gray')
fig.suptitle('Reconstruction RS')
plt.show()
