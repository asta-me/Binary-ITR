"""
Binary intensity CGH generation for DMD projection.

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


def iterativeEst(target, n_iter):
    """Iterative binary-hologram estimation (ITR).

    Alternates between object-plane amplitude constraint and binary
    intensity constraint in the hologram plane (off-axis interference
    + mean thresholding). The aberration phase (zero if compensation
    is disabled) is pre-compensated at the end.

    Uses module-level globals: X, Y, lam, N, P, image, Phase_abb.
    Returns the binary intensity hologram (values 0 / 255).
    """
    # Initial random intensity in the hologram plane
    O = np.abs(-1 + 2 * np.random.rand(*target.shape))

    # Off-axis reference field
    theta = 0.662 * np.pi / 12
    R_p = 2 * np.pi * np.sin(theta) * X / lam + 2 * np.pi * np.sin(theta) * Y / lam
    R_f = np.exp(1j * R_p)

    for k in range(n_iter):
        # Propagate the binary hologram to the object plane
        a = fftshift(fft2(fftshift(O)))

        # Strong amplitude compression to flatten the dynamic range
        a1 = a**1e-3

        # Object constraint on the first diffraction order: target
        # amplitude with the estimated phase
        a2 = (np.sqrt(np.abs(image)) * a1[P:N, P:N]) / np.abs(a1[P:N, P:N])

        # First-order filtering: re-embed the P x P order at the center
        a3 = np.zeros((N, N), dtype=np.complex128)
        a3[((N - P) // 2):((N + P) // 2), ((N - P) // 2):((N + P) // 2)] = a2

        # Normalize and scale to [0, 255]
        a3 = (a3 - np.min(a3)) / (np.max(a3) - np.min(a3))
        a = 255 * a3

        ## Resampling: zero-pad the spectrum to double the grid size
        xx = fft2(a)
        xx = fftshift(xx)
        xx = np.pad(xx, [(N // 2, N // 2), (N // 2, N // 2)], mode='constant')
        xx = ifftshift(xx)
        target_resampled = np.abs(ifft2(xx)) * 4

        # Normalize the resampled target to [0, 255]
        target_resampled_scaled = 255 * (target_resampled - np.min(target_resampled)) \
            / (np.max(target_resampled) - np.min(target_resampled))

        # New object field: resampled amplitude with fresh random phase
        A0 = np.sqrt(target_resampled_scaled)
        fi_0 = 2 * np.pi * np.random.rand(*target_resampled_scaled.shape)
        A = A0 * np.exp(1j * fi_0)

        ## Back-propagate to the hologram plane, keep the phase
        H = fftshift(ifft2(fftshift(A)))
        kino = np.angle(H)

        # Crop the phase back to the central N x N window
        start = kino.shape[0] // 4
        O_p = kino[start:start + N, start:start + N]

        # Binary intensity hologram: off-axis interference + thresholding
        O_f = np.exp(1j * O_p)
        I = np.abs(R_f + O_f)**2
        I = (I > np.mean(I))
        O = 255 * I

    # Final hologram with astigmatism pre-compensation
    O_p = (O_p - Phase_abb) % (2 * np.pi)
    O_f = np.exp(1j * O_p)
    I = np.abs(R_f + O_f)**2
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

itr = 5                       # Number of iterations (GS and ITR)

N = res                       # Hologram size (pixels)
P = 540                       # Effective object size (pixels)

# Spatial grid in the hologram plane
x = np.linspace(-L/2, L/2 - d, res)
X, Y = np.meshgrid(x, x)

##### Flags #####

# Aberration pre-compensation: the astigmatism coefficients below were
# measured for our specific projection setup. When False the aberration
# phase is zero: ideal hologram generation and ideal propagation.
compensate_aberration = True

# Save the binary holograms as 8-bit TIFF files
save_holograms = True

# Astigmatism aberration phase of the projection path
a = 15e-2                     # Astigmatism coefficient (0/90 deg)
b = 8e-2                      # Astigmatism coefficient (45 deg)
if compensate_aberration:
    Phase_abb = 2 * np.pi * (a * (X**2 - Y**2) + b * 2 * X * Y) / lam
else:
    Phase_abb = np.zeros_like(X)

##### Target image preparation #####

input_image_path = "pcl.png"

# Load, drop transparency, convert to grayscale (ITU-R 601 luma weights)
img = Image.open(input_image_path).convert("RGBA")
rgb_img = np.array(img)[:, :, :3]
gray_img = np.dot(rgb_img, [0.2989, 0.5870, 0.1140])

# Resize to the effective object size
image = np.array(Image.fromarray(gray_img).resize((P, P)))

# Embed the P x P image at the center of the N x N hologram grid
m, n = image.shape
target = np.zeros((N, N))
target[((N - m) // 2):((N + m) // 2), ((N - n) // 2):((N + n) // 2)] = image


##### CGH generation: ITR #####

t1 = get_time()
hologram = iterativeEst(target, itr)
hologram_ITR = 255 * hologram
t1 = get_time() - t1
print("ITR computation time:", t1)

# Numerical reconstruction (projection aberration re-applied)
Rec = fftshift(fft2(fftshift(hologram_ITR * np.exp(1j * Phase_abb))))

if save_holograms:
    # 'hologram' (not 'hologram_ITR') holds the 0/255 values fitting uint8
    Image.fromarray(hologram.astype(np.uint8)).save('Hologram_ITR_pcl.tiff')

##### CGH generation: GS + off-axis binarization #####

# t2 = get_time()

# # Hologram phase from GS (aberration pre-compensation if enabled)
# Op, _, _ = gs_resampled(target, itr)
# Op = (Op - Phase_abb) % (2 * np.pi)
# Of = np.exp(1j * Op)

# # Off-axis reference field
# theta = 0.662 * np.pi / 12
# Rp = 2 * np.pi * np.sin(theta) * X / lam + 2 * np.pi * np.sin(theta) * Y / lam
# Rf = np.exp(1j * Rp)

# # Binary intensity hologram: interference + mean thresholding
# Ip = np.abs(Rf + Of)**2
# Ip = (Ip > np.mean(Ip))
# hologram_GS = 255 * Ip.astype(int)

# t2 = get_time() - t2
# print("GS computation time:", t2)

# # Numerical reconstruction
# Rec_gs = fftshift(fft2(fftshift(hologram_GS * np.exp(1j * Phase_abb))))

# if save_holograms:
#     Image.fromarray(hologram_GS.astype(np.uint8)).save('Hologram_GS_cow.tiff')


# ##### CGH generation: RS + off-axis binarization #####

# t3 = get_time()

# # Hologram phase from RS (aberration pre-compensation if enabled)
# Op1 = rs_minimal(target)
# Op1 = (Op1 - Phase_abb) % (2 * np.pi)
# Of1 = np.exp(1j * Op1)

# # Off-axis reference field
# theta = 0.662 * np.pi / 12
# Rp1 = 2 * np.pi * np.sin(theta) * X / lam + 2 * np.pi * np.sin(theta) * Y / lam
# Rf1 = np.exp(1j * Rp1)

# # Binary intensity hologram: interference + mean thresholding
# Ip1 = np.abs(Rf1 + Of1)**2
# Ip1 = (Ip1 > np.mean(Ip1))
# hologram_RS = 255 * Ip1.astype(int)

# t3 = get_time() - t3
# print("RS computation time:", t3)

# # Numerical reconstruction
# Rec_rs = fftshift(fft2(fftshift(hologram_RS * np.exp(1j * Phase_abb))))

# if save_holograms:
#     Image.fromarray(hologram_RS.astype(np.uint8)).save('Hologram_RS_cow.tiff')


# ##### Display of the numerical reconstructions (first order region) #####
# # The exponent 1e-3 compresses the dynamic range for visualization

# plt.figure()
# plt.imshow(np.abs(Rec[P:N, P:N]) ** 1e-3, cmap='gray')
# plt.colorbar()
# plt.title('Reconstruction ITR')
# plt.show()

# plt.figure()
# plt.imshow(np.abs(Rec_gs[P:N, P:N]) ** 1e-3, cmap='gray')
# plt.colorbar()
# plt.title('Reconstruction GS')
# plt.show()

# plt.figure()
# plt.imshow(np.abs(Rec_rs[P:N, P:N]) ** 1e-3, cmap='gray')
# plt.colorbar()
# plt.title('Reconstruction RS')
# plt.show()


# if __name__=="__main__":
#     image= np.zeros((540,540))
#     xc=(np.random.random(20)*540).astype(int)
#     yc=(np.random.random(20)*540).astype(int)
#     image[yc, xc] = 1
#     m, n = image.shape
#     N = 1080
#     target = np.zeros((N, N))
#     target[((N - m) // 2):((N + m) // 2), ((N - n) // 2):((N + n) // 2)] = image

#     itr_holo= iterativeEst(image,10)
#     plt.imshow(itr_holo)
#     plt.show()