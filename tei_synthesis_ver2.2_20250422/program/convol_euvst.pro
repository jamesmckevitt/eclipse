function convol_euvst, image, resolution=resolution, psf=g

if not (keyword_set(resolution)) then resolution=0.4

resolution_km = 720 * resolution

fwhm_pixel = (resolution_km)/192.
sigma_pixel = fwhm_pixel / (2 * sqrt(2 * alog(2) ) )

; g_temp = gaussian_function([sigma_pixel,sigma_pixel], /normalize, width = 11)

g = gaussian_function([sigma_pixel,sigma_pixel], /normalize, width = 10)

image_out = convol(image, g, /edge_mirror)

return, image_out
end
